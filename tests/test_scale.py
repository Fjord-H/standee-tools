"""Unit tests for the pure upscale-planning math (service.scale)."""

import pytest

from service.scale import MODEL_SCALE, ScalePlan, plan_upscale, required_pixels


class TestRequiredPixels:
    def test_90cm_at_100dpi_is_about_3540(self):
        # The spec's headline number: 90 cm @ 100 dpi ~= 3540 px.
        assert required_pixels(90) == 3543

    def test_50cm_at_100dpi(self):
        assert required_pixels(50) == 1969

    def test_dpi_scales_linearly(self):
        assert required_pixels(90, dpi=200) == 7087

    def test_rejects_nonpositive_print_size(self):
        with pytest.raises(ValueError):
            required_pixels(0)
        with pytest.raises(ValueError):
            required_pixels(-30)

    def test_rejects_nonpositive_dpi(self):
        with pytest.raises(ValueError):
            required_pixels(90, dpi=0)


class TestPlanUpscale:
    def test_skips_when_crop_already_large_enough(self):
        plan = plan_upscale(4000, 90)  # 4000 px > 3543 px needed
        assert plan.skip is True
        assert plan.capped is False
        assert plan.output_px == 4000  # untouched

    def test_skips_at_exact_target(self):
        target = required_pixels(90)
        plan = plan_upscale(target, 90)
        assert plan.skip is True
        assert plan.raw_scale == 1.0

    def test_upscales_and_settles_on_exact_target(self):
        plan = plan_upscale(1000, 90)  # needs 3.543x -> one x4 pass + resample
        assert plan.skip is False
        assert plan.capped is False
        assert plan.target_px == 3543
        assert plan.output_px == 3543

    def test_exactly_4x_is_not_capped(self):
        # 100 px crop with a print size that needs exactly 400 px
        plan = plan_upscale(100, 400 * 2.54 / 100)
        assert plan.target_px == 400
        assert plan.raw_scale == pytest.approx(4.0)
        assert plan.capped is False
        assert plan.output_px == 400

    def test_caps_beyond_4x(self):
        plan = plan_upscale(500, 90)  # needs 7.086x
        assert plan.skip is False
        assert plan.capped is True
        assert plan.output_px == 500 * MODEL_SCALE  # 2000, short of 3543
        assert plan.output_px < plan.target_px

    def test_rejects_nonpositive_crop(self):
        with pytest.raises(ValueError):
            plan_upscale(0, 90)

    def test_plan_is_frozen(self):
        plan = plan_upscale(1000, 90)
        with pytest.raises(AttributeError):
            plan.skip = True

    def test_raw_scale_reported_even_when_skipped(self):
        plan = plan_upscale(7086, 90)
        assert plan.raw_scale == pytest.approx(0.5, abs=0.001)


class TestScalePlanShape:
    def test_fields(self):
        plan = plan_upscale(1000, 90)
        assert isinstance(plan, ScalePlan)
        assert set(plan.__dataclass_fields__) == {
            "target_px", "raw_scale", "skip", "capped", "output_px"}
