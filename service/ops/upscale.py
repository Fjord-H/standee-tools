"""Capability "upscale": Real-ESRGAN x4plus toward a target print size.

The scale decision lives in service.scale (pure, unit-tested). This op
only executes the plan: skip, or one x4 pass followed by a Lanczos
resample down to the exact target. RGBA input is supported — RealESRGANer
upsamples the alpha channel alongside the color (alpha_upsampler
"realesrgan"), preserving the soft matte through the upscale.
"""

import numpy as np
from PIL import Image

from service.pipeline import PipelineState
from service.scale import plan_upscale
from service.schemas import UpscaleOp, UpscaleReport


def load_upsampler(model_path: str, device: str = "cuda"):
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    rrdb = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23,
                   num_grow_ch=32, scale=4)
    return RealESRGANer(scale=4, model_path=model_path, model=rrdb,
                        tile=512, tile_pad=16, half=True, device=device)


def run_upscale(models: dict, state: PipelineState, op: UpscaleOp) -> None:
    crop_px = max(state.image.size)  # longer side drives the print size
    plan = plan_upscale(crop_px, op.target_print_cm, op.dpi)

    if not plan.skip:
        # RealESRGANer works in BGR(A) numpy, like cv2.
        rgb_a = np.array(state.image)
        bgr_a = rgb_a[:, :, [2, 1, 0]] if rgb_a.shape[2] == 3 \
            else rgb_a[:, :, [2, 1, 0, 3]]
        out_bgr_a, _ = models["upsampler"].enhance(bgr_a, outscale=4)
        out = out_bgr_a[:, :, [2, 1, 0]] if out_bgr_a.shape[2] == 3 \
            else out_bgr_a[:, :, [2, 1, 0, 3]]
        image = Image.fromarray(out)

        if max(image.size) > plan.output_px:  # x4 overshot: settle on target
            ratio = plan.output_px / max(image.size)
            new_size = (round(image.size[0] * ratio),
                        round(image.size[1] * ratio))
            image = image.resize(new_size, Image.LANCZOS)
        state.image = image

    state.reports.append(UpscaleReport(
        skipped=plan.skip,
        raw_scale=plan.raw_scale,
        target_px=plan.target_px,
        output_px=max(state.image.size),
        capped=plan.capped,
    ))
