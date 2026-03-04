"""Backend app package ownership map.

Module responsibilities:
- `app.main`: FastAPI app and HTTP endpoint handlers.
- `app.pipeline.pipeline`: segmentation pipeline execution (`run_pipeline`).
- `app.util.config`: shared runtime configuration and mutable process state.
- `app.util.network`: external network/model download and predictor loading logic.
- `app.util.util`: utility helpers, TIFF conversions, and shared request models.
"""
