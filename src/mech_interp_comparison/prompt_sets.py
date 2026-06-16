"""Small prompt fixtures for smoke checks and method comparison planning."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptCase:
    """A minimal prompt record used before full benchmark integration."""

    case_id: str
    prompt: str
    expected_topic: str


MINI_PROMPTS: tuple[PromptCase, ...] = (
    PromptCase(
        case_id="factual_capital_france",
        prompt="The capital of France is",
        expected_topic="factual_completion",
    ),
    PromptCase(
        case_id="sentiment_simple",
        prompt="The movie was surprisingly thoughtful and moving. Sentiment:",
        expected_topic="sentiment",
    ),
)
