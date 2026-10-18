"""FastAPI app: health, predict and model-info. The model is loaded once at startup."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Body, FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from credit_risk.api.examples import OPENAPI_EXAMPLES
from credit_risk.api.schemas import (
    ClientFeatures,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)
from credit_risk.api.service import LoadedModel, load_model, predict
from credit_risk.config import load_config

NO_MODEL_DETAIL = "El modelo no está disponible. Entrénalo con `uv run poe train`."


def create_app(model_dir: Path | None = None) -> FastAPI:
    """Build the app; tests pass a directory with a synthetic model."""
    directory = model_dir if model_dir is not None else load_config().paths.models

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model = load_model(directory)  # None if the artifact is missing
        yield

    app = FastAPI(
        title="API de riesgo de crédito",
        description=(
            "Probabilidad de mora severa (90+ días en los próximos 2 años), "
            "decisión de crédito y las 3 razones principales de cada cliente."
        ),
        lifespan=lifespan,
    )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # The default 422 body echoes ``input``, which cannot be serialised for NaN/Infinity.
        detail = [
            {k: v for k, v in error.items() if k not in {"input", "ctx"}} for error in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": detail})

    def get_model(request: Request) -> LoadedModel:
        loaded = getattr(request.app.state, "model", None)
        if loaded is None:
            raise HTTPException(status_code=503, detail=NO_MODEL_DETAIL)
        return loaded

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request, response: Response) -> HealthResponse:
        loaded = getattr(request.app.state, "model", None)
        if loaded is None:
            response.status_code = 503
            return HealthResponse(status="unavailable", model_loaded=False, model_version=None)
        return HealthResponse(status="ok", model_loaded=True, model_version=loaded.version)

    @app.post("/predict", response_model=PredictionResponse)
    def predict_endpoint(
        request: Request,
        client: Annotated[ClientFeatures, Body(openapi_examples=OPENAPI_EXAMPLES)],
    ) -> dict:
        return predict(get_model(request), client.to_frame())

    @app.get("/model-info", response_model=ModelInfoResponse)
    def model_info(request: Request) -> dict:
        loaded = get_model(request)
        metadata = loaded.metadata
        return {
            "model_version": loaded.version,
            "champion": metadata["champion"],
            "thresholds": metadata["thresholds"],
            "costs": metadata["costs"],
            "review_capacity": metadata["review_capacity"],
            "test_metrics": metadata["test_metrics"],
            "features": metadata["features"],
        }

    return app


app = create_app()
