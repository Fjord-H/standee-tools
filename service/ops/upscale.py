"""Capability "upscale": Real-ESRGAN x4plus toward a target print size.

Plain-torch implementation. The released RealESRGAN_x4plus.pth checkpoint
loads into the minimal RRDBNet (service.ops.rrdbnet) with strict=True, so
an architecture mismatch fails loudly at container start instead of
producing silently wrong output. Inference is tiled (512 px tiles, 16 px
overlap context) to bound VRAM on T4; the soft alpha matte is upsampled
bicubically alongside the RGB pass — tile seams must be checked per
docs/smoke-test.md after any change here.

The scale decision lives in service.scale (pure, unit-tested). This op
only executes the plan: skip, or one x4 pass followed by a Lanczos
resample down to the exact target.
"""

import numpy as np
import torch
from PIL import Image

from service.ops.rrdbnet import RRDBNet
from service.pipeline import PipelineState
from service.scale import MODEL_SCALE, plan_upscale
from service.schemas import UpscaleOp, UpscaleReport


class TiledUpsampler:
    """Fixed x4 RRDBNet inference over overlapping tiles.

    Each tile is cut with `tile_pad` pixels of context on every interior
    edge and the padded margin is discarded from the output, so tiles are
    stitched from interior-only pixels.
    """

    def __init__(self, model: RRDBNet, device: str = "cuda",
                 tile: int = 512, tile_pad: int = 16):
        self.model = model
        self.device = device
        self.tile = tile
        self.tile_pad = tile_pad
        self.scale = MODEL_SCALE

    @torch.no_grad()
    def upscale_rgb(self, rgb: np.ndarray) -> np.ndarray:
        """uint8 HxWx3 in, uint8 (4H)x(4W)x3 out."""
        h, w = rgb.shape[:2]
        s = self.scale
        inp = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        inp = inp.to(self.device).half().div_(255.0)
        out = torch.empty((3, h * s, w * s), dtype=torch.float32)

        for y0 in range(0, h, self.tile):
            y1 = min(y0 + self.tile, h)
            py0 = max(y0 - self.tile_pad, 0)
            py1 = min(y1 + self.tile_pad, h)
            for x0 in range(0, w, self.tile):
                x1 = min(x0 + self.tile, w)
                px0 = max(x0 - self.tile_pad, 0)
                px1 = min(x1 + self.tile_pad, w)
                tile_out = self.model(inp[:, :, py0:py1, px0:px1])
                tile_out = tile_out.float().cpu()[0]
                out[:, y0 * s:y1 * s, x0 * s:x1 * s] = tile_out[
                    :,
                    (y0 - py0) * s:(y1 - py0) * s,
                    (x0 - px0) * s:(x1 - px0) * s,
                ]

        out = out.permute(1, 2, 0).clamp_(0, 1).mul_(255).round_()
        return out.to(torch.uint8).numpy()


def load_upsampler(model_path: str, device: str = "cuda") -> TiledUpsampler:
    model = RRDBNet()
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    state = state.get("params_ema") or state.get("params") or state
    # strict=True: an architecture mismatch must fail the cold start, not
    # degrade into silently wrong prints.
    model.load_state_dict(state, strict=True)
    model.eval()
    model.to(device)
    model.half()
    return TiledUpsampler(model, device=device)


def run_upscale(models: dict, state: PipelineState, op: UpscaleOp) -> None:
    crop_px = max(state.image.size)  # longer side drives the print size
    plan = plan_upscale(crop_px, op.target_print_cm, op.dpi)

    if not plan.skip:
        src = state.image
        rgb = np.array(src.convert("RGB"))
        image = Image.fromarray(models["upsampler"].upscale_rgb(rgb))
        if src.mode == "RGBA":  # carry the soft matte through the upscale
            image.putalpha(src.getchannel("A").resize(image.size, Image.BICUBIC))

        if max(image.size) > plan.output_px:  # x4 overshot: settle on target
            ratio = plan.output_px / max(image.size)
            image = image.resize((round(image.size[0] * ratio),
                                  round(image.size[1] * ratio)), Image.LANCZOS)
        state.image = image

    state.reports.append(UpscaleReport(
        skipped=plan.skip,
        raw_scale=plan.raw_scale,
        target_px=plan.target_px,
        output_px=max(state.image.size),
        capped=plan.capped,
    ))
