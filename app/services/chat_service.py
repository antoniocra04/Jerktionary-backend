from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

from app.core.errors import LlmResponseError, LlmUnavailableError
from app.domain.entities.chat import ChatMessage
from app.domain.interfaces.llm import LlmProvider


class ChatService:
    """Free-form conversation with the provider the backend was started with.

    Unlike explain/answer there is no JSON contract and no prompt of our own: the
    client owns the whole message list, and this only guards the request and turns
    provider deltas into a stream.
    """

    def __init__(
        self,
        llm_provider: LlmProvider,
        *,
        llm_enabled: bool,
        reasoning_levels: Sequence[str] = (),
    ) -> None:
        self._llm_provider = llm_provider
        self._llm_enabled = llm_enabled
        self._reasoning_levels = tuple(reasoning_levels)

    @property
    def reasoning_levels(self) -> tuple[str, ...]:
        return self._reasoning_levels

    async def chat_stream(
        self,
        *,
        messages: Sequence[ChatMessage],
        system: str = "",
        model: str = "",
        reasoning_effort: str = "",
    ) -> AsyncIterator[str]:
        if not self._llm_enabled:
            raise LlmUnavailableError("LLM is unavailable")
        if reasoning_effort and reasoning_effort not in self._reasoning_levels:
            # Silently dropping it would look like the control does nothing;
            # forwarding it can fail the whole request on a strict endpoint.
            raise LlmResponseError(
                f"Provider does not support reasoning effort '{reasoning_effort}'"
            )

        produced = False
        async for delta in self._llm_provider.chat_stream(
            messages,
            system=system,
            model=model,
            reasoning_effort=reasoning_effort,
        ):
            produced = True
            yield delta

        if not produced:
            raise LlmResponseError("LLM produced no output")
