from typing import Any
from client.llm_client import LLMClient
from client.response import StreamEventType, TokenUsage
from context.manager import ContextManager
from context.mrl import MrlEncoder
from prompts.system import get_compression_prompt


class ChatCompactor:
    def __init__(self, client: LLMClient):
        self.client = client
        self._mrl_encoder = MrlEncoder()

    def _format_history_for_compaction(self, messages: list[dict[str, Any]]) -> str:
        mrl_snapshot = self._mrl_encoder.encode_and_render(messages)
        return (
            "Here is the MRL particle snapshot of the conversation to compress:\n\n"
            + mrl_snapshot
        )

    async def compress(
        self, context_manager: ContextManager
    ) -> tuple[str | None, TokenUsage | None]:
        messages = context_manager.get_messages()

        if len(messages) < 3:
            return None, None

        compression_messages = [
            {
                "role": "system",
                "content": get_compression_prompt(),
            },
            {
                "role": "user",
                "content": self._format_history_for_compaction(messages),
            },
        ]

        try:
            summary = ""
            usage = None
            async for event in self.client.chat_completion(
                compression_messages,
                stream=False,
            ):
                if event.type == StreamEventType.MESSAGE_COMPLETE:
                    usage = event.usage
                    summary += event.text_delta.content

            if not summary or not usage:
                return None, None

            return summary, usage
        except Exception:
            return None, None
