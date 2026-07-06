from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    version: str


class ComponentStatus(BaseModel):
    ready: bool
    required: bool
    details: str = ""


class ReadinessResponse(BaseModel):
    ready: bool
    components: dict[str, ComponentStatus]

