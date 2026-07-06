from __future__ import annotations

from pydantic import BaseModel, Field


class WebSocketMessageDoc(BaseModel):
    type: str
    description: str
    payload_schema: str


class WebSocketEndpointDoc(BaseModel):
    path: str
    protocol: str = "websocket"
    accepts: list[WebSocketMessageDoc] = Field(default_factory=list)
    emits: list[WebSocketMessageDoc] = Field(default_factory=list)
    errors: list[WebSocketMessageDoc] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ApiIndexResponse(BaseModel):
    swagger_url: str
    openapi_url: str
    redoc_url: str
    websocket_docs_url: str

