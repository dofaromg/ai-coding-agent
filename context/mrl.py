from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Any


class MrlParticleKind(str, Enum):
    GOAL = "GOAL"
    CTX = "CTX"
    ACT = "ACT"
    STATE = "STATE"
    CONSTRAINT = "CONSTRAINT"
    REMAIN = "REMAIN"
    NEXT = "NEXT"


@dataclass
class MrlParticle:
    kind: MrlParticleKind
    content: str
    tag: str | None = None

    def render(self) -> str:
        label = f"{self.kind.value}:{self.tag}" if self.tag else self.kind.value
        return f"[{label}] {self.content}"


class MrlEncoder:
    """Encodes conversation history into compact MRL particle language format.

    MRL (Message Representation Language) particles are atomic units that
    absorb, integrate, and compress the conversation context into a structured
    representation for efficient context restoration.
    """

    # Token-budget-aware truncation limits: keep each particle small enough
    # that a full conversation snapshot stays well within the LLM prompt budget
    # while retaining the most actionable information per role.
    MAX_CONTENT_LEN = 200
    MAX_TOOL_ARGS_LEN = 120
    MAX_TOOL_RESULT_LEN = 150

    def encode(self, messages: list[dict[str, Any]]) -> list[MrlParticle]:
        """Convert a list of conversation messages into MRL particles."""
        particles: list[MrlParticle] = []
        goal_set = False

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content") or ""

            if role == "system":
                continue

            if role == "user":
                truncated = self._truncate(content, self.MAX_CONTENT_LEN)
                if not goal_set and truncated:
                    particles.append(MrlParticle(MrlParticleKind.GOAL, truncated))
                    goal_set = True
                elif truncated:
                    particles.append(MrlParticle(MrlParticleKind.CTX, truncated))

            elif role == "assistant":
                if content:
                    truncated = self._truncate(content, self.MAX_CONTENT_LEN)
                    if truncated:
                        particles.append(
                            MrlParticle(MrlParticleKind.STATE, truncated)
                        )

                tool_calls = msg.get("tool_calls") or []
                for tc in tool_calls:
                    func = tc.get("function", {})
                    name = func.get("name", "unknown")
                    args = func.get("arguments", "")
                    args_summary = self._truncate(args, self.MAX_TOOL_ARGS_LEN)
                    particles.append(
                        MrlParticle(MrlParticleKind.ACT, args_summary, tag=name)
                    )

            elif role == "tool":
                result_summary = self._truncate(content, self.MAX_TOOL_RESULT_LEN)
                particles.append(
                    MrlParticle(MrlParticleKind.STATE, result_summary, tag="tool_result")
                )

        return particles

    def render(self, particles: list[MrlParticle]) -> str:
        """Render a list of MRL particles into a compact string representation."""
        return "\n".join(p.render() for p in particles)

    def encode_and_render(self, messages: list[dict[str, Any]]) -> str:
        """Encode messages into MRL particles and render them to a string."""
        particles = self.encode(messages)
        return self.render(particles)

    @staticmethod
    def _truncate(text: str, max_len: int) -> str:
        text = text.strip()
        if len(text) <= max_len:
            return text
        return text[:max_len] + "…"
