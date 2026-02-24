"""LLM-based biomedical entity extraction using Anthropic Claude."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import anthropic

from .config import VALID_DOMAINS, Settings, get_settings

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """\
Extract biomedical entities from the following text.

Text: {text}

Extract entities of these types: {domains}

Return a JSON array of entities with:
- text: the extracted phrase exactly as it appears in the input
- start_pos: character start position in the original text (0-indexed)
- end_pos: character end position in the original text (exclusive)
- domain: one of disease, chemical, gene, phenotype, anatomy, or organism
- confidence: 0.0 to 1.0

Only return the JSON array, no other text.\
"""


class ExtractionError(Exception):
    """Raised when entity extraction fails."""


class EntityExtractor:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        if not self._settings.anthropic_api_key:
            raise ExtractionError(
                "ANTHROPIC_API_KEY is not set; entity extraction is unavailable."
            )
        self._client = anthropic.AsyncAnthropic(api_key=self._settings.anthropic_api_key)

    async def extract(
        self,
        text: str,
        domains: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Extract biomedical entities from free text.

        Returns a list of entity dicts with keys:
            text, start_pos, end_pos, domain, extraction_confidence
        """
        effective_domains = domains or list(VALID_DOMAINS)
        # Validate
        invalid = [d for d in effective_domains if d not in VALID_DOMAINS]
        if invalid:
            raise ExtractionError(f"Invalid domains: {invalid}")

        prompt = EXTRACTION_PROMPT.format(
            text=text,
            domains=", ".join(effective_domains),
        )

        try:
            message = await self._client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
        except anthropic.APIError as exc:
            raise ExtractionError(f"Anthropic API error: {exc}") from exc

        content = message.content[0].text if message.content else ""
        return self._parse_response(content, text)

    def _parse_response(self, content: str, original_text: str) -> list[dict[str, Any]]:
        """Parse the LLM JSON response and validate/fix position offsets."""
        # Strip markdown code fences if present
        content = content.strip()
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
        content = content.strip()

        try:
            entities: list[dict[str, Any]] = json.loads(content)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse LLM entity extraction response: %s", exc)
            return []

        if not isinstance(entities, list):
            logger.warning("LLM returned non-list entity response")
            return []

        validated: list[dict[str, Any]] = []
        for ent in entities:
            if not isinstance(ent, dict):
                continue
            entity_text: str = ent.get("text") or ""
            domain: str = ent.get("domain") or ""
            confidence: float = float(ent.get("confidence") or 0.0)

            if not entity_text or domain not in VALID_DOMAINS:
                continue

            # Try to use model-provided positions; validate against original text
            start_pos: int | None = ent.get("start_pos")
            end_pos: int | None = ent.get("end_pos")

            # Re-derive positions if model's are wrong or missing
            if (
                start_pos is None
                or end_pos is None
                or not isinstance(start_pos, int)
                or not isinstance(end_pos, int)
                or original_text[start_pos:end_pos] != entity_text
            ):
                idx = original_text.lower().find(entity_text.lower())
                if idx == -1:
                    logger.debug(
                        "Discarding extracted entity (not found in text): %r [%s]",
                        entity_text,
                        domain,
                    )
                    continue
                start_pos = idx
                end_pos = idx + len(entity_text)

            validated.append(
                {
                    "text": entity_text,
                    "start_pos": start_pos,
                    "end_pos": end_pos,
                    "domain": domain,
                    "extraction_confidence": round(confidence, 4),
                }
            )

        return validated
