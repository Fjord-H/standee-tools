"""One-time weight download into the shared Modal Volume.

Run once before the first deploy (and again only to refresh weights):

    modal run -m service.weights

Cold starts never download anything — the engine containers mount the
"standee-weights" Volume read-only and load from disk.

Licenses (why these exact weights — see CLAUDE.md):
  - BiRefNet original weights: MIT (NOT BRIA RMBG-2.0, which is CC BY-NC)
  - MobileSAM: Apache-2.0
  - Real-ESRGAN x4plus: BSD-3-Clause
"""

import modal

app = modal.App("standee-weights-setup")

volume = modal.Volume.from_name("standee-weights", create_if_missing=True)
WEIGHTS_DIR = "/weights"

BIREFNET_REPO = "ZhengPeng7/BiRefNet"
MOBILE_SAM_URL = ("https://github.com/ChaoningZhang/MobileSAM/raw/master/"
                  "weights/mobile_sam.pt")
REALESRGAN_URL = ("https://github.com/xinntao/Real-ESRGAN/releases/download/"
                  "v0.1.0/RealESRGAN_x4plus.pth")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "huggingface_hub==0.27.1", "requests==2.32.3")


@app.function(image=image, volumes={WEIGHTS_DIR: volume}, timeout=1800)
def download_weights(force: bool = False) -> None:
    import os

    import requests
    from huggingface_hub import snapshot_download

    def fetch(url: str, dest: str) -> None:
        if os.path.exists(dest) and not force:
            print(f"exists, skipping: {dest}")
            return
        print(f"downloading {url}")
        resp = requests.get(url, timeout=600, allow_redirects=True)
        resp.raise_for_status()
        tmp = dest + ".part"
        with open(tmp, "wb") as f:
            f.write(resp.content)
        os.replace(tmp, dest)
        print(f"wrote {dest} ({os.path.getsize(dest)} bytes)")

    birefnet_dir = os.path.join(WEIGHTS_DIR, "birefnet")
    if force or not os.path.isdir(birefnet_dir):
        print(f"downloading {BIREFNET_REPO}")
        snapshot_download(BIREFNET_REPO, local_dir=birefnet_dir)
    else:
        print(f"exists, skipping: {birefnet_dir}")

    fetch(MOBILE_SAM_URL, os.path.join(WEIGHTS_DIR, "mobile_sam.pt"))
    fetch(REALESRGAN_URL, os.path.join(WEIGHTS_DIR, "RealESRGAN_x4plus.pth"))

    volume.commit()
    print("weights volume ready")


@app.local_entrypoint()
def main(force: bool = False) -> None:
    download_weights.remote(force=force)
