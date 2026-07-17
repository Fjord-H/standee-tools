"""Unit tests for the /v1 contract shapes (service.schemas)."""

import pytest
from pydantic import ValidationError

from service.schemas import (EmbedResponse, HealthResponse, JobRequest,
                             JobResponse, JobResult, SegmentOp, UpscaleOp,
                             UpscaleReport)
from service.version import PIPELINE_VERSION


class TestJobRequest:
    def test_accepts_segment_then_upscale(self):
        req = JobRequest.model_validate({
            "image_b64": "aGVsbG8=",
            "ops": [
                {"op": "segment", "include_matte": True},
                {"op": "upscale", "target_print_cm": 90},
            ],
        })
        assert isinstance(req.ops[0], SegmentOp)
        assert isinstance(req.ops[1], UpscaleOp)
        assert req.ops[1].dpi == 100  # default print resolution

    def test_ops_are_named_by_capability_not_model(self):
        # "birefnet" is a model, not a capability — must be rejected.
        with pytest.raises(ValidationError):
            JobRequest.model_validate({
                "image_b64": "aGVsbG8=",
                "ops": [{"op": "birefnet"}],
            })

    def test_no_face_restoration_capability_exists(self):
        # CLAUDE.md hard rule: face restoration must not be reachable
        # through the contract at all.
        for forbidden in ("face_restore", "gfpgan", "codeformer", "restore"):
            with pytest.raises(ValidationError):
                JobRequest.model_validate({
                    "image_b64": "aGVsbG8=",
                    "ops": [{"op": forbidden}],
                })

    def test_rejects_empty_ops(self):
        with pytest.raises(ValidationError):
            JobRequest.model_validate({"image_b64": "aGVsbG8=", "ops": []})

    def test_rejects_empty_image(self):
        with pytest.raises(ValidationError):
            JobRequest.model_validate(
                {"image_b64": "", "ops": [{"op": "segment"}]})

    def test_upscale_rejects_nonpositive_print_size(self):
        with pytest.raises(ValidationError):
            UpscaleOp.model_validate({"op": "upscale", "target_print_cm": 0})


class TestResponses:
    def test_job_response_carries_pipeline_version(self):
        resp = JobResponse(job_id="fc-123", status="pending")
        assert resp.pipeline_version == PIPELINE_VERSION

    def test_embed_response_carries_pipeline_version(self):
        resp = EmbedResponse(embedding_b64="AAAA",
                             shape=[1, 256, 64, 64])
        assert resp.pipeline_version == PIPELINE_VERSION
        assert resp.dtype == "float32"

    def test_health_response_carries_pipeline_version(self):
        resp = HealthResponse(warm=False)
        assert resp.pipeline_version == PIPELINE_VERSION
        assert resp.status == "ok"

    def test_succeeded_response_roundtrips_result(self):
        payload = {
            "job_id": "fc-123",
            "status": "succeeded",
            "result": {
                "image_b64": "aW1n",
                "matte_b64": "bWF0dGU=",
                "reports": [
                    {"op": "segment", "matte_included": True},
                    {"op": "upscale", "skipped": False, "raw_scale": 3.5,
                     "target_px": 3543, "output_px": 3543, "capped": False},
                ],
            },
        }
        resp = JobResponse.model_validate(payload)
        assert isinstance(resp.result, JobResult)
        assert isinstance(resp.result.reports[1], UpscaleReport)
        assert resp.result.reports[1].target_px == 3543

    def test_failed_response_has_error_no_result(self):
        resp = JobResponse(job_id="fc-123", status="failed", error="boom")
        assert resp.result is None
        assert resp.error == "boom"

    def test_status_is_a_closed_set(self):
        with pytest.raises(ValidationError):
            JobResponse(job_id="fc-123", status="exploded")

    def test_matte_is_optional(self):
        result = JobResult(image_b64="aW1n")
        assert result.matte_b64 is None
        assert result.reports == []
