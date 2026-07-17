"""Capability "segment": BiRefNet background removal with soft alpha.

Model: BiRefNet original weights (MIT, trained on DIS5K), loaded from the
weights Volume via transformers with trust_remote_code. Output contract:
the state's image becomes RGBA with a soft 8-bit alpha — byte-format
compatible with what the embedded u2netp produces, so the downstream
distance-transform / outline / cut-line pipeline is engine-agnostic.
"""

import torch
from PIL import Image
from torchvision import transforms

from service.schemas import SegmentOp, SegmentReport
from service.pipeline import PipelineState

INPUT_SIZE = (1024, 1024)
_NORMALIZE = transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225])


def load_birefnet(weights_dir: str, device: str = "cuda"):
    from transformers import AutoModelForImageSegmentation
    model = AutoModelForImageSegmentation.from_pretrained(
        weights_dir, trust_remote_code=True)
    model.to(device)
    model.eval()
    model.half()
    return model


def predict_matte(model, image: Image.Image, device: str = "cuda") -> Image.Image:
    """Run BiRefNet on a PIL image, return a soft matte (mode L) at the
    image's original size."""
    rgb = image.convert("RGB")
    tensor = transforms.functional.to_tensor(
        rgb.resize(INPUT_SIZE, Image.BILINEAR))
    tensor = _NORMALIZE(tensor).unsqueeze(0).to(device).half()
    with torch.no_grad():
        pred = model(tensor)[-1].sigmoid()
    matte = transforms.functional.to_pil_image(pred[0, 0].float().cpu())
    return matte.resize(image.size, Image.LANCZOS)


def run_segment(models: dict, state: PipelineState, op: SegmentOp) -> None:
    matte = predict_matte(models["birefnet"], state.image)
    rgba = state.image.convert("RGB").convert("RGBA")
    rgba.putalpha(matte)
    state.image = rgba
    state.matte = matte
    state.include_matte = state.include_matte or op.include_matte
    state.reports.append(SegmentReport(matte_included=op.include_matte))
