from __future__ import annotations

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from loguru import logger
from pydantic import ValidationError

from app.api.schemas.terms import TermItem
from app.api.schemas.transcript import TermsUpdateEvent, TranscriptUpdateEvent, WsConfigMessage
from app.core.dependencies import get_app_state
from app.core.errors import AsrApiError, AsrUnavailableError
from app.core.state import AppState
from app.services.transcript_service import TranscriptSession

router = APIRouter(tags=["audio"])


@router.websocket("/ws/audio")
async def audio_ws(websocket: WebSocket, state: AppState = Depends(get_app_state)) -> None:
    await websocket.accept()
    session: TranscriptSession | None = None
    asr_error_sent = False
    logger.info("Audio WebSocket connected")
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break

            text = message.get("text")
            if text is not None:
                # Optional handshake: a JSON config choosing the ASR provider. Must
                # arrive before the first audio chunk; later configs are ignored.
                if session is not None:
                    continue
                try:
                    config = WsConfigMessage.model_validate_json(text)
                except ValidationError:
                    await websocket.send_json({"type": "error", "code": "INVALID_CONFIG"})
                    continue
                try:
                    session = await state.transcript_service.create_session(config.asr)
                    logger.info("ASR session created (provider={})", config.asr.provider)
                except AsrUnavailableError:
                    asr_error_sent = True
                    await websocket.send_json({"type": "error", "code": "ASR_UNAVAILABLE"})
                continue

            chunk = message.get("bytes")
            if chunk is None:
                await websocket.send_json({"type": "error", "code": "INVALID_AUDIO_CHUNK"})
                continue

            if session is None:
                # Clients that skip the handshake get the local model, as before.
                try:
                    session = await state.transcript_service.create_session()
                except AsrUnavailableError:
                    if not asr_error_sent:
                        asr_error_sent = True
                        await websocket.send_json({"type": "error", "code": "ASR_UNAVAILABLE"})
                    continue

            try:
                update = await session.process_audio(chunk)
            except AsrApiError as exc:
                logger.warning("API transcription rejected: {}", exc.message)
                await websocket.send_json({"type": "error", "code": "ASR_API_ERROR"})
                continue
            if update is None:
                continue

            items = [
                TermItem(
                    text=term.text,
                    normalized=term.normalized,
                    start=term.start,
                    end=term.end,
                    type=term.type,
                    confidence=term.confidence,
                )
                for term in update.terms
            ]
            transcript_event = TranscriptUpdateEvent(
                text=update.text,
                is_final=update.is_final,
                terms=items,
            )
            await websocket.send_json(transcript_event.model_dump(mode="json"))
            if items:
                terms_event = TermsUpdateEvent(items=items)
                await websocket.send_json(terms_event.model_dump(mode="json"))
    except WebSocketDisconnect:
        logger.info("Audio WebSocket disconnected")
    finally:
        if session is not None:
            await session.close()
