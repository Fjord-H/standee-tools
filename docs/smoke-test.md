# Deploy smoke test

Run after every deploy that touches the GPU image, the ops, or the
weights Volume. Needs the deployed base URL and the API key.

## 1. Health and warm-up
- `GET /v1/health` with no key → 200, `pipeline_version` matches the
  deployed `service/version.py`.
- `GET /v1/health?warm=1` → 200; a GPU container should appear in the
  Modal dashboard and finish its cold start (~15 s budget).

## 2. Auth
- `POST /v1/jobs` without `X-API-Key` → 401.
- With a wrong key → 401. With the right key → 200.

## 3. Embed (sync)
- `POST /v1/embed` with a test photo → 200, `shape == [1, 256, 64, 64]`,
  decoded byte length equals `1*256*64*64*4`.

## 4. Segment job
- Photo of a person with fine hair detail against a busy background.
- Ops `[{"op": "segment", "include_matte": true}]` → poll to `succeeded`.
- Result PNG has soft (not binary) alpha; hair edges keep partial
  transparency. Matte PNG matches image dimensions.

## 5. Upscale job — normal and capped
- Crop with longer side ~1000 px, ops
  `[{"op": "upscale", "target_print_cm": 90}]` → `succeeded`, report
  `skipped=false, capped=false, output_px=3543`; output longer side is
  3543 px.
- Crop with longer side ~500 px, same op → `capped=true,
  output_px=2000`; downstream UI must surface this for human judgment.
- Segment→upscale chained: alpha survives the upscale (soft edges, no
  halo).

## 6. Tile-seam check (required — tiling is our own code)
The upsampler processes 512 px input tiles with 16 px overlap
(`TiledUpsampler` in service/ops/upscale.py). A crop-math bug shows up as
visible seams on the tile grid.

- Input: a large, densely textured image — longer side ≥ 1500 px so the
  x4 pass crosses multiple tile boundaries. Good subjects: foliage,
  fabric weave, confetti, grass.
- Run ops `[{"op": "upscale", "target_print_cm": 90}]`.
- Open the result at 100% zoom (no browser scaling) and inspect along the
  tile grid: input-space boundaries at 512 px multiples land at output
  x = 2048, 4096, … px and the same rows in y.
- Look for brightness steps, texture discontinuities, or ghost lines
  running exactly on those lines, in both directions.
- Any visible seam = bug in the overlap-crop arithmetic in
  `TiledUpsampler.upscale_rgb` — do not ship; fix and re-run this check.
