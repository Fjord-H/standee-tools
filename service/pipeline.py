"""Op dispatcher: runs a validated ops list in order over one image.

Capability names map to op implementations here and nowhere else. Op
modules import torch and friends, so they are imported lazily — this
module stays importable in GPU-less environments (tests, web container).
"""

import base64
import io
from dataclasses import dataclass, field
from typing import Optional

from service.schemas import JobResult, OpSpec


@dataclass
class PipelineState:
    image: "object"                     # PIL.Image, RGB or RGBA
    matte: Optional["object"] = None    # PIL.Image mode L, set by segment
    include_matte: bool = False
    reports: list = field(default_factory=list)


def _registry(models: dict):
    """Capability name -> op callable. Lazy imports keep torch off the
    import path of this module. There is intentionally no face-restoration
    entry (CLAUDE.md hard rule)."""
    from service.ops.segment import run_segment
    from service.ops.upscale import run_upscale
    return {
        "segment": lambda state, op: run_segment(models, state, op),
        "upscale": lambda state, op: run_upscale(models, state, op),
    }


def _png_b64(image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def run_pipeline(models: dict, image_bytes: bytes,
                 ops: list[OpSpec]) -> JobResult:
    from PIL import Image, ImageOps

    image = Image.open(io.BytesIO(image_bytes))
    image = ImageOps.exif_transpose(image)  # phone photos carry orientation
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGB")

    state = PipelineState(image=image)
    registry = _registry(models)
    for op in ops:
        registry[op.op](state, op)  # ops mutate state, append their report

    matte_b64 = None
    if state.include_matte and state.matte is not None:
        matte = state.matte
        if matte.size != state.image.size:  # upscale ran after segment
            matte = matte.resize(state.image.size, Image.LANCZOS)
        matte_b64 = _png_b64(matte)

    return JobResult(
        image_b64=_png_b64(state.image),
        matte_b64=matte_b64,
        reports=state.reports,  # ops append SegmentReport / UpscaleReport
    )
