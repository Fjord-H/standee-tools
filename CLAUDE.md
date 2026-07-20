# CLAUDE.md — AI Image-Processing Service (Standee Pipeline)

## Purpose
Self-hosted AI image processing for a party decoration business (Taiwan).
Prepares parents' phone photos of children for large-format print standees
(50–90 cm at 100 dpi): background removal / cutout and upscaling.
Replaces the quality ceiling of the embedded u2netp (320x320) in the
existing browser tools, without abandoning it as offline fallback.

## Workflow rules (important)
- Ideas and architecture are discussed and explicitly approved BEFORE any
  code is written. Never generate application code without a go-ahead.
- Detailed written specs from Fjord count as approval to proceed.
- All significant algorithms get Node.js unit tests.
- UI-facing content: Traditional Chinese (zh-Hant), warm tone, no emoji.

## Model decisions
- Automatic segmentation: BiRefNet (original weights, MIT — trained on
  DIS5K). Do NOT use BRIA RMBG-2.0 (CC BY-NC, commercial license needed).
  Benchmark BiRefNet-HR and BEN2 as alternates; hair edges are the
  quality-critical case.
- Interactive tap-to-select: MobileSAM / SAM 2.1 small.
  Encoder/decoder split: heavy encoder runs once server/sidecar-side and
  returns an embedding; tiny (~8 MB) ONNX decoder runs in the browser for
  per-click interactive masking. One roundtrip per photo, clicks are local.
- Optional refinement (deferred): trimap + ViTMatte, only if real prints
  show hair fringing.
- Upscaling: Real-ESRGAN x4plus. Required scale computed from
  crop-size vs. target-print-size (need is crop-driven, not sensor-driven;
  90 cm @ 100 dpi = ~3540 px). Skip upscaling when unnecessary.
- HARD RULE: no face-hallucinating restoration (GFPGAN, CodeFormer, SUPIR)
  by default. Children's faces must not be invented. Any face restoration
  is opt-in with human review. See "Generative face restoration — status"
  below for the licensing investigation and current status.

## Generative face restoration — status (do not build yet)
Trigger: some parent photos are too blurry/low-res for BiRefNet + Real-ESRGAN
x4plus to save (identified by Uta in production use).

Investigated: CodeFormer and GFPGAN, the standard options. Both are
non-commercial-only licenses (CodeFormer: NTU S-Lab License 1.0;
GFPGAN carries NVIDIA/DFDNet non-commercial restrictions). This business
sells prints for revenue, so using either as-is on paid orders is
commercial use under these licenses, regardless of whether AI use is
disclosed to clients. Training an in-house model is out of scope: multi-
month research effort, needs separately-licensed face training data, and
does not remove the actual concern — a self-trained model would still
invent facial detail on a blurry photo.

Action taken: emailed CodeFormer's author (contact listed on the
official repo) to ask about a commercial license for our volume.
Awaiting reply as of 2026-07-19.

If a commercial license or license-clean alternative is ever obtained,
the HARD RULE above is unchanged: any face-restoration op is opt-in per
photo and requires human review before use. Never automatic, never
default-on. Licensing and hallucination risk are separate problems, and
only the first is resolved by a license.

Caution on paid third-party "CodeFormer API" services: the authors'
own repo lists most public CodeFormer-branded API hosts as unauthorized,
non-official deployments. Paying one does not confer a legitimate
commercial license. A genuine paid option would need to be an
independently, commercially-licensed model, not a wrapped CodeFormer
endpoint.

Also investigated: RestoreFormer++ (Tencent ARC lineage, transformer-based).
Code is genuinely Apache 2.0 (verified against the actual LICENSE file).
Weights are trained on FFHQ, which is CC BY-NC-SA 4.0 (non-commercial) and
additionally restricted from use in facial recognition development. README
does not restate any separate commercial grant for the released weights.
Same conclusion as CodeFormer/GFPGAN: structural, not a licensing label to
shop around — the free academic face-restoration ecosystem converges on
FFHQ/CelebA-HQ, both non-commercial. Emailed the author (contact listed on
the official repo) in parallel with the CodeFormer email. Awaiting reply
as of 2026-07-19.

## Serving architecture
- Primary: desktop-local sidecar. The future desktop app runs inference
  on Uta's machine (BiRefNet fp16 ~200 MB, MobileSAM ~40 MB,
  Real-ESRGAN ~65 MB). Zero monthly cost, zero cold start, client photos
  never leave the machine (個資法-friendly selling point).
- Optional secondary: serverless GPU endpoint (Modal preferred; RunPod
  Serverless as alternate) if server-grade cutouts are later wanted in the
  client-facing HTML tools. Effectively free at current volume
  (hundreds of orders/year); cache model weights on persistent storage to
  keep cold starts ~15 s.
- Rejected: dedicated self-hosted GPU box (ops burden + internet-exposed
  children's photos, no cost advantage at this volume).
- Rejected: commercial cutout APIs as the core (no interactive selection,
  no soft-matte pipeline integration, privacy).

## API contract (shared by browser tools, desktop app, any backend)
- Async job model: POST /v1/jobs with image (or presigned upload) and an
  operations pipeline; poll GET /v1/jobs/{id} or webhook.
- Ops named by CAPABILITY, never by model ("segment", "upscale") so
  backends are swappable. Every response carries pipeline_version.
- Sync endpoint POST /v1/embed returns SAM image embedding for the
  browser-side decoder. Only latency-sensitive call in the system.
- Output contract: PNG with soft alpha (optional raw matte channel) —
  byte-format-identical to u2netp output so the downstream
  distance-transform / outline / cut-line pipeline is engine-agnostic.
- Auth: simple API keys + CORS allowlist.

## Fallback strategy
- Capability detection per session: probe /v1/embed with short timeout.
  Success = enhanced mode (BiRefNet + SAM). Failure = silent fallback to
  embedded u2netp, unchanged pipeline. Never block on the network.
- UI badge indicates active engine: 標準模式 / 增強模式.

## Decisions resolved
1. Serving order: MODAL-FIRST, sidecar later. Rationale: zero-install for
   Uta (browser tools just gain 增強模式 when the endpoint is reachable),
   all updates ship server-side with no version drift, and her Mac stays
   untouched — no background inference process or model weights until the
   desktop app can bundle them properly with an auto-updater.
   - Accepted tradeoff: ~15 s cold start after idle. Mitigation: fire a
     warm-up probe on tool open + 「AI 引擎啟動中…」 status message.
   - Accepted tradeoff: photos transit Modal during this phase. Do NOT
     market 「照片不離開電腦」 until the sidecar era. Sidecar remains the
     end-state for the privacy story.
2. Still open (defer, non-blocking): whether client-facing tools get
   server-grade cutouts or stay u2netp-only. Modal-first makes adding
   this later a base-URL change.

## Phase 1 scope — Modal deployment of the /v1 contract
- Modal app exposing: POST /v1/jobs (async: segment via BiRefNet,
  upscale via Real-ESRGAN x4plus, pipeline ops by capability name),
  GET /v1/jobs/{id}, POST /v1/embed (sync SAM image embedding),
  GET /v1/health (used by the capability probe / warm-up).
- Model weights cached on a Modal Volume (never downloaded on cold start).
- Output contract: PNG with soft alpha, byte-format-compatible with
  u2netp output. pipeline_version in every response.
- Auth: API key header + CORS allowlist.
- Browser-side SAM decoder (ONNX) is a SEPARATE later step — phase 1
  only serves the embedding.
- Unit tests (Node.js where applicable, pytest for the Python service)
  for contract shapes and the scale-computation logic
  (crop size ÷ target print size, skip when ≤1x).