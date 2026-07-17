"""Upscale planning: crop size vs. target print size.

The need for upscaling is crop-driven, not sensor-driven: what matters is
how many pixels the cropped subject has along the print's driving
dimension (the standee's height, i.e. the longer image side), compared to
the pixels the print needs (90 cm @ 100 dpi ~= 3543 px).

Real-ESRGAN x4plus runs at a fixed x4. The plan is therefore:
  - raw_scale <= 1        -> skip the op entirely
  - 1 < raw_scale <= 4    -> one x4 pass, then Lanczos down to exact target
  - raw_scale > 4         -> one x4 pass only, output stays below target and
                             the plan is flagged `capped` (chaining passes
                             degrades faces; a human decides what to do)
"""

from dataclasses import dataclass

CM_PER_INCH = 2.54
MODEL_SCALE = 4  # Real-ESRGAN x4plus is a fixed x4 model


def required_pixels(print_cm: float, dpi: int = 100) -> int:
    """Pixels needed along the driving dimension for a given print size."""
    if print_cm <= 0:
        raise ValueError("print_cm must be > 0")
    if dpi <= 0:
        raise ValueError("dpi must be > 0")
    return round(print_cm / CM_PER_INCH * dpi)


@dataclass(frozen=True)
class ScalePlan:
    target_px: int    # pixels the print needs on the driving dimension
    raw_scale: float  # target_px / crop_px
    skip: bool        # crop already has enough pixels
    capped: bool      # needed more than x4; output will fall short of target
    output_px: int    # driving-dimension size the pipeline will produce


def plan_upscale(crop_px: int, print_cm: float, dpi: int = 100) -> ScalePlan:
    """Decide whether and how to upscale a crop for a given print size.

    crop_px is the crop's pixel count along the driving dimension
    (longer side). Pure function; unit-tested without any model.
    """
    if crop_px <= 0:
        raise ValueError("crop_px must be > 0")
    target_px = required_pixels(print_cm, dpi)
    raw_scale = target_px / crop_px

    if raw_scale <= 1.0:
        return ScalePlan(target_px, raw_scale, skip=True, capped=False,
                         output_px=crop_px)
    if raw_scale > MODEL_SCALE:
        return ScalePlan(target_px, raw_scale, skip=False, capped=True,
                         output_px=crop_px * MODEL_SCALE)
    return ScalePlan(target_px, raw_scale, skip=False, capped=False,
                     output_px=target_px)
