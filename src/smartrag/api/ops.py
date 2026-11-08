"""Ops endpoints (13), unauthenticated so probes and load balancers can call them.

    GET /healthz   liveness: the process answers
    GET /readyz    readiness: the index and the embedder can serve (503 otherwise)
    GET /version   package and API version

They reveal no data: `/readyz` reports check names and short details only.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from smartrag.__about__ import __version__
from smartrag.models import ReadinessResponse, VersionResponse

router = APIRouter()


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    """liveness"""
    return {"status": "ok"}


@router.get("/readyz", response_model=ReadinessResponse,
            responses={503: {"model": ReadinessResponse}})
async def readyz(request: Request) -> JSONResponse:
    """readiness of the index and the embedder"""
    readiness: ReadinessResponse = await request.app.state.engine.readiness()
    return JSONResponse(readiness.model_dump(mode="json"),
                        status_code=200 if readiness.ready else 503)


@router.get("/version", response_model=VersionResponse)
async def version() -> VersionResponse:
    """package and API version"""
    return VersionResponse(version=__version__)
