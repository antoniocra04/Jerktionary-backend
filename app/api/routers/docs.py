from __future__ import annotations

from fastapi import APIRouter

from app.api.schemas.docs import ApiIndexResponse, WebSocketEndpointDoc, WebSocketMessageDoc

router = APIRouter(prefix="/api/docs", tags=["documentation"])


@router.get(
    "",
    response_model=ApiIndexResponse,
    summary="API documentation links",
    description="Returns the documentation entry points exposed by this backend.",
)
async def api_docs_index() -> ApiIndexResponse:
    return ApiIndexResponse(
        swagger_url="/docs",
        openapi_url="/openapi.json",
        redoc_url="/redoc",
        websocket_docs_url="/api/docs/websocket/audio",
    )


@router.get(
    "/websocket/audio",
    response_model=WebSocketEndpointDoc,
    summary="Audio WebSocket contract",
    description=(
        "Documents the /ws/audio WebSocket endpoint. WebSocket routes are not native "
        "OpenAPI operations, so this HTTP endpoint makes the contract visible in Swagger."
    ),
)
async def audio_websocket_docs() -> WebSocketEndpointDoc:
    return WebSocketEndpointDoc(
        path="/ws/audio",
        accepts=[
            WebSocketMessageDoc(
                type="binary",
                description="PCM audio chunk: 16 kHz, mono, signed int16 little-endian.",
                payload_schema="bytes",
            )
        ],
        emits=[
            WebSocketMessageDoc(
                type="transcript_update",
                description="Partial transcript with extracted terms.",
                payload_schema="TranscriptUpdateEvent",
            ),
            WebSocketMessageDoc(
                type="terms_update",
                description="Extracted terms emitted when the latest transcript contains terms.",
                payload_schema="TermsUpdateEvent",
            ),
        ],
        errors=[
            WebSocketMessageDoc(
                type="error",
                description="Sent when a message does not contain a binary audio chunk.",
                payload_schema='{"type":"error","code":"INVALID_AUDIO_CHUNK"}',
            )
        ],
        notes=[
            "Send binary frames only.",
            "The backend buffers audio and emits updates after ASR_MIN_AUDIO_SECONDS.",
            "The endpoint uses the same readiness dependencies as the HTTP API.",
        ],
    )

