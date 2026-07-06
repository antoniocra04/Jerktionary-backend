from __future__ import annotations

from typing import cast

from starlette.requests import HTTPConnection

from app.core.state import AppState


def get_app_state(connection: HTTPConnection) -> AppState:
    return cast(AppState, connection.app.state.app_state)
