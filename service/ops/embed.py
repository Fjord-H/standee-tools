"""Sync capability behind POST /v1/embed: MobileSAM image embedding.

The heavy ViT encoder runs here once per photo; the tiny ONNX decoder in
the browser (a later phase) consumes this embedding for per-click
interactive masking. Wire format: raw little-endian float32 bytes in
C-order plus an explicit shape, so the JS side can feed it straight into
onnxruntime-web without a numpy dependency.
"""

import base64

import numpy as np
from PIL import Image


def load_sam_predictor(checkpoint_path: str, device: str = "cuda"):
    from mobile_sam import SamPredictor, sam_model_registry
    sam = sam_model_registry["vit_t"](checkpoint=checkpoint_path)
    sam.to(device)
    sam.eval()
    return SamPredictor(sam)


def compute_embedding(predictor, image: Image.Image) -> dict:
    """Returns {embedding_b64, shape, dtype} for EmbedResponse."""
    rgb = np.array(image.convert("RGB"))
    predictor.set_image(rgb)
    emb = predictor.get_image_embedding().cpu().numpy()  # (1, 256, 64, 64)
    emb = np.ascontiguousarray(emb.astype("<f4"))
    return {
        "embedding_b64": base64.b64encode(emb.tobytes()).decode("ascii"),
        "shape": list(emb.shape),
        "dtype": "float32",
    }
