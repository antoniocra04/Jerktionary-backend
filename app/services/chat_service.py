from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

from app.core.errors import LlmResponseError, LlmUnavailableError
from app.domain.entities.chat import ChatMessage, ModelInfo
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

    async def model_info(self, model: str) -> ModelInfo | None:
        """Per-model capabilities, when the provider can report them."""
        lookup = getattr(self._llm_provider, "model_info", None)
        if lookup is None:
            return None
        return await lookup(model)

    async def levels(self, model: str) -> tuple[str, ...]:
        """Reasoning levels for a specific model, narrowing the provider-wide set
        whenever the provider is precise about it."""
        info = await self.model_info(model)
        if info is not None and info.reasoning_levels is not None:
            return info.reasoning_levels
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
        if reasoning_effort:
            # Checked against the model's own levels where the provider reports
            # them: on makora these differ per model, so the provider-wide set
            # would let through efforts that fail the request.
            allowed = await self.levels(model)
            if reasoning_effort not in allowed:
                raise LlmResponseError(
                    f"Модель не поддерживает уровень ризонинга «{reasoning_effort}». "
                    f"Доступно: {', '.join(allowed) if allowed else '—'}"
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
