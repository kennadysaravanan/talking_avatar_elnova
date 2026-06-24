"""
GPT streaming LLM provider (OpenAI chat completions).

Streams tokens so the turn manager can start TTS on the first complete sentence/clause to cut
latency. Cancellable: when the consuming asyncio task is cancelled mid-stream, the async
generator is closed and the underlying HTTP stream is aborted.
"""

from __future__ import annotations

import os
from typing import AsyncIterator

from openai import AsyncOpenAI

SYSTEM_PROMPT = os.environ.get(
    "AVATAR_SYSTEM_PROMPT",
    "You are a friendly, concise talking avatar. Reply in short, natural spoken-style "
    "sentences. Avoid markdown, lists, or emoji — your text is spoken aloud.",
)


class OpenAILLM:
    def __init__(self, model: str = None):
        self._client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = model or os.environ.get("OPENAI_LLM_MODEL", "gpt-4o-mini")

    async def stream_reply(self, user_text: str, history: list[dict]) -> AsyncIterator[str]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history,
                    {"role": "user", "content": user_text}]
        stream = await self._client.chat.completions.create(
            model=self.model, messages=messages, stream=True, temperature=0.7,
        )
        try:
            async for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield delta
        finally:
            # On cancellation/close, abort the HTTP stream promptly (interruption path).
            await stream.close()
