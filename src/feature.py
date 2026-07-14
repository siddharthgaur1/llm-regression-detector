"""Customer support email classifier — the LLM feature under test.

Loads a versioned prompt from ``prompts/*.yaml`` and calls the classifier
model with structured output, so every eval run is tied to a specific,
inspectable prompt version.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

Category = Literal["billing", "technical", "account", "general"]


class ClassificationResult(BaseModel):
    """Structured output produced by the classifier for a single email."""

    category: Category = Field(..., description="Predicted support category.")
    summary: str = Field(..., min_length=1, description="One-sentence summary of the customer's issue.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model's self-reported confidence, 0-1.")


@dataclass
class PromptConfig:
    """A versioned system prompt plus optional few-shot examples, loaded from YAML."""

    version_id: str
    timestamp: str
    system_prompt: str
    few_shot_examples: list[dict] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> "PromptConfig":
        """Load a prompt version from a YAML file.

        Raises:
            FileNotFoundError: if ``path`` doesn't exist.
            KeyError: if the YAML is missing ``version_id`` or ``system_prompt``.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        for required_key in ("version_id", "system_prompt"):
            if required_key not in data:
                raise KeyError(f"Prompt file {path} is missing required key '{required_key}'")
        return cls(
            version_id=data["version_id"],
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(path.stat().st_mtime)),
            system_prompt=data["system_prompt"],
            few_shot_examples=data.get("few_shot_examples") or [],
        )

    def build_messages(self, input_email: str) -> list[dict]:
        """Build the chat-completion message list for a given input email."""
        if not input_email or not input_email.strip():
            raise ValueError("input_email must be a non-empty string")

        messages = [{"role": "system", "content": self.system_prompt}]
        for ex in self.few_shot_examples:
            messages.append({"role": "user", "content": ex["input_email"]})
            messages.append(
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "category": ex["category"],
                            "summary": ex["summary"],
                            "confidence": ex["confidence"],
                        }
                    ),
                }
            )
        messages.append({"role": "user", "content": input_email})
        return messages


@dataclass
class FeatureResult:
    """Outcome of classifying one email: parsed result plus latency/token telemetry."""

    result: ClassificationResult
    latency_ms: float
    input_tokens: int
    output_tokens: int


async def classify_email(
    client: AsyncOpenAI, prompt_config: PromptConfig, input_email: str, model: str
) -> FeatureResult:
    """Classify a single email using the given prompt version and model.

    Args:
        client: An initialized AsyncOpenAI client.
        prompt_config: The prompt version to classify with.
        input_email: Raw email text. Must be non-empty.
        model: Chat-completions model name (e.g. "gpt-4o-mini").

    Returns:
        A FeatureResult with the parsed classification and timing/token usage.

    Raises:
        ValueError: if input_email is empty.
    """
    start = time.perf_counter()
    messages = prompt_config.build_messages(input_email)
    logger.debug("Classifying email (prompt=%s, model=%s)", prompt_config.version_id, model)
    response = await client.chat.completions.parse(
        model=model,
        messages=messages,
        response_format=ClassificationResult,
    )
    latency_ms = (time.perf_counter() - start) * 1000
    parsed = response.choices[0].message.parsed
    if parsed is None:
        raise ValueError(f"Model {model} returned no parsable classification for prompt {prompt_config.version_id}")
    usage = response.usage
    return FeatureResult(
        result=parsed,
        latency_ms=latency_ms,
        input_tokens=usage.prompt_tokens if usage else 0,
        output_tokens=usage.completion_tokens if usage else 0,
    )
