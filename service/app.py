"""Modal deployment of the /v1 standee-pipeline contract.

Deploy sequence:

    modal secret create standee-api \
        STANDEE_API_KEY=<key> \
        CORS_ALLOW_ORIGINS=https://tools.example.tw
    modal run -m service.weights          # once: fill the weights Volume
    modal deploy -m service.app

Endpoints (see service.schemas for the contract):
    POST /v1/jobs        async: validate, spawn GPU job, return job_id
    GET  /v1/jobs/{id}   poll: running / succeeded (+result) / failed
    POST /v1/embed       sync: MobileSAM image embedding
    GET  /v1/health      open (no key): capability probe; ?warm=1 also
                         boots a GPU engine container in the background

Job IDs are Modal function-call IDs — no database in phase 1. A job that
was spawned reports "running" until it resolves (Modal does not expose a
queued/started distinction cheaply); POST returns it as "pending".
"""

import os

import modal

from service.version import PIPELINE_VERSION

app = modal.App("standee-pipeline")

volume = modal.Volume.from_name("standee-weights", create_if_missing=True)
WEIGHTS_DIR = "/weights"

# torchvision is pinned <0.17 because basicsr 1.4.2 imports
# torchvision.transforms.functional_tensor, removed in 0.17.
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.1.2",
        "torchvision==0.16.2",
        "transformers==4.47.1",
        "timm==1.0.13",
        "kornia==0.7.4",
        "einops==0.8.0",
        "basicsr==1.4.2",
        "realesrgan==0.3.0",
        "opencv-python-headless==4.10.0.84",
        "numpy<2",
        "pillow==10.4.0",
        "pydantic==2.10.4",
        "huggingface_hub==0.27.1",
        "git+https://github.com/ChaoningZhang/MobileSAM.git",
    )
    .add_local_python_source("service")
)

web_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("fastapi[standard]==0.115.6", "pydantic==2.10.4")
    .add_local_python_source("service")
)


@app.cls(
    image=gpu_image,
    gpu="T4",
    volumes={WEIGHTS_DIR: volume},
    timeout=600,
    scaledown_window=300,
)
class StandeeEngine:
    """One container holds all three models, loaded once per cold start."""

    @modal.enter()
    def load(self) -> None:
        from service.ops.embed import load_sam_predictor
        from service.ops.segment import load_birefnet
        from service.ops.upscale import load_upsampler

        self.models = {
            "birefnet": load_birefnet(f"{WEIGHTS_DIR}/birefnet"),
            "upsampler": load_upsampler(f"{WEIGHTS_DIR}/RealESRGAN_x4plus.pth"),
            "sam": load_sam_predictor(f"{WEIGHTS_DIR}/mobile_sam.pt"),
        }

    @modal.method()
    def run_job(self, request: dict) -> dict:
        import base64

        from service.pipeline import run_pipeline
        from service.schemas import JobRequest

        req = JobRequest.model_validate(request)
        image_bytes = base64.b64decode(req.image_b64)
        result = run_pipeline(self.models, image_bytes, req.ops)
        return result.model_dump()

    @modal.method()
    def embed(self, image_b64: str) -> dict:
        import base64
        import io

        from PIL import Image, ImageOps

        from service.ops.embed import compute_embedding

        image = Image.open(io.BytesIO(base64.b64decode(image_b64)))
        image = ImageOps.exif_transpose(image)
        return compute_embedding(self.models["sam"], image)

    @modal.method()
    def warm(self) -> bool:
        return True  # @modal.enter already did the work


@app.function(
    image=web_image,
    secrets=[modal.Secret.from_name("standee-api")],
    min_containers=0,
)
@modal.asgi_app()
def api():
    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse

    from service.schemas import (EmbedRequest, EmbedResponse, HealthResponse,
                                 JobRequest, JobResponse, JobResult)

    web_app = FastAPI(title="standee-pipeline", version=PIPELINE_VERSION)

    origins = [o.strip()
               for o in os.environ.get("CORS_ALLOW_ORIGINS", "").split(",")
               if o.strip()]
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @web_app.middleware("http")
    async def require_api_key(request: Request, call_next):
        open_paths = ("/v1/health",)
        if request.method == "OPTIONS" or request.url.path in open_paths:
            return await call_next(request)
        if request.headers.get("x-api-key") != os.environ["STANDEE_API_KEY"]:
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return await call_next(request)

    engine = StandeeEngine()

    @web_app.post("/v1/jobs", response_model=JobResponse,
                  response_model_exclude_none=True)
    def create_job(req: JobRequest):
        call = engine.run_job.spawn(req.model_dump())
        return JobResponse(job_id=call.object_id, status="pending")

    @web_app.get("/v1/jobs/{job_id}", response_model=JobResponse,
                 response_model_exclude_none=True)
    def get_job(job_id: str):
        try:
            call = modal.FunctionCall.from_id(job_id)
        except Exception:
            return JSONResponse({"error": "unknown job id"}, status_code=404)
        try:
            result = call.get(timeout=0)
        except TimeoutError:
            return JobResponse(job_id=job_id, status="running")
        except modal.exception.RemoteError as exc:
            return JobResponse(job_id=job_id, status="failed", error=str(exc))
        except Exception as exc:  # user-code exception re-raised by Modal
            return JobResponse(job_id=job_id, status="failed", error=str(exc))
        return JobResponse(job_id=job_id, status="succeeded",
                           result=JobResult.model_validate(result))

    @web_app.post("/v1/embed", response_model=EmbedResponse)
    def embed(req: EmbedRequest):
        return EmbedResponse(**engine.embed.remote(req.image_b64))

    @web_app.get("/v1/health", response_model=HealthResponse)
    def health(warm: int = 0):
        if warm:
            engine.warm.spawn()
        return HealthResponse(warm=bool(warm))

    return web_app
