"""Single source of truth for the pipeline version string.

Carried in every /v1 response so clients can detect backend changes.
Bump when a model, weight file, or op behavior changes.
"""

PIPELINE_VERSION = "p1.0-birefnet-esrganx4"
