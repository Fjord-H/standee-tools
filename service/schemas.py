"""The /v1 API contract, shared by every backend that implements it.

Ops are named by CAPABILITY ("segment", "upscale"), never by model, so
backends stay swappable. Every response model carries pipeline_version.

There is deliberately no face-restoration capability in this contract:
children's faces must never be invented (see CLAUDE.md hard rule). Adding
one requires an explicit contract change, not a config flag.
"""

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field

from service.version import PIPELINE_VERSION

JobStatus = Literal["pending", "running", "succeeded", "failed"]


# ---------------------------------------------------------------- requests

class SegmentOp(BaseModel):
    """Background removal. Output picks up a soft 8-bit alpha channel."""
    op: Literal["segment"]
    include_matte: bool = False  # also return the raw matte as grayscale PNG


class UpscaleOp(BaseModel):
    """Upscale toward a target print size.

    The crop's longer side is the driving dimension and is mapped to
    target_print_cm (standees are sized by their tallest measurement).
    """
    op: Literal["upscale"]
    target_print_cm: float = Field(gt=0, le=300)
    dpi: int = Field(default=100, gt=0, le=1200)


OpSpec = Annotated[Union[SegmentOp, UpscaleOp], Field(discriminator="op")]


class JobRequest(BaseModel):
    image_b64: str = Field(min_length=1)  # source photo (JPEG/PNG), base64
    ops: list[OpSpec] = Field(min_length=1)


class EmbedRequest(BaseModel):
    image_b64: str = Field(min_length=1)


# ---------------------------------------------------------------- reports

class SegmentReport(BaseModel):
    op: Literal["segment"] = "segment"
    matte_included: bool


class UpscaleReport(BaseModel):
    op: Literal["upscale"] = "upscale"
    skipped: bool          # crop already met the target; op was a no-op
    raw_scale: float       # target_px / crop_px as received by the op
    target_px: int         # pixels the print needs (driving dimension)
    output_px: int         # pixels actually produced (driving dimension)
    capped: bool           # needed > x4; output falls short of target


OpReport = Annotated[Union[SegmentReport, UpscaleReport],
                     Field(discriminator="op")]


# --------------------------------------------------------------- responses

class JobResult(BaseModel):
    image_b64: str                        # PNG; soft alpha if segmented
    matte_b64: Optional[str] = None       # grayscale PNG, on include_matte
    reports: list[OpReport] = Field(default_factory=list)


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    pipeline_version: str = PIPELINE_VERSION
    result: Optional[JobResult] = None    # set when status == "succeeded"
    error: Optional[str] = None           # set when status == "failed"


class EmbedResponse(BaseModel):
    """SAM image embedding for the (future) browser-side ONNX decoder."""
    embedding_b64: str                    # raw little-endian float32, C-order
    shape: list[int]                      # e.g. [1, 256, 64, 64]
    dtype: str = "float32"
    pipeline_version: str = PIPELINE_VERSION


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    warm: bool                            # a GPU engine warm-up was triggered
    pipeline_version: str = PIPELINE_VERSION
