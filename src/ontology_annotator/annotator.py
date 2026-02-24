"""Core ontology annotation logic with multi-stage matching."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from .bioportal_client import BioPortalClient
from .config import VALID_DOMAINS, Settings, get_settings
from .ols_client import OLSClient

logger = logging.getLogger(__name__)

# Confidence scores for each match stage
CONFIDENCE_EXACT_LABEL = 0.98
CONFIDENCE_SYNONYM = 0.85
CONFIDENCE_OLS_SEARCH = 0.75
CONFIDENCE_BIOPORTAL = 0.70


class OntologyMatch:
    __slots__ = (
        "term_id",
        "label",
        "ontology",
        "match_type",
        "confidence",
        "definition",
        "synonyms",
        "cross_references",
    )

    def __init__(
        self,
        term_id: str,
        label: str,
        ontology: str,
        match_type: str,
        confidence: float,
        definition: str | None = None,
        synonyms: list[str] | None = None,
        cross_references: dict[str, str] | None = None,
    ) -> None:
        self.term_id = term_id
        self.label = label
        self.ontology = ontology
        self.match_type = match_type
        self.confidence = confidence
        self.definition = definition
        self.synonyms = synonyms or []
        self.cross_references = cross_references or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "term_id": self.term_id,
            "label": self.label,
            "ontology": self.ontology,
            "match_type": self.match_type,
            "confidence": self.confidence,
            "definition": self.definition,
            "synonyms": self.synonyms,
            "cross_references": self.cross_references,
        }


def _raw_to_match(raw: dict[str, Any], match_type: str, confidence: float) -> OntologyMatch:
    return OntologyMatch(
        term_id=raw.get("term_id") or "",
        label=raw.get("label") or "",
        ontology=raw.get("ontology") or "",
        match_type=match_type,
        confidence=confidence,
        definition=raw.get("definition"),
        synonyms=raw.get("synonyms") or [],
        cross_references=raw.get("cross_references") or {},
    )


def _deduplicate(matches: list[OntologyMatch]) -> list[OntologyMatch]:
    """Remove duplicates by term_id, keeping the first (highest confidence) occurrence."""
    seen: set[str] = set()
    unique: list[OntologyMatch] = []
    for m in matches:
        key = f"{m.ontology}:{m.term_id}" if m.term_id else f"{m.ontology}:{m.label}"
        if key not in seen:
            seen.add(key)
            unique.append(m)
    return unique


class OntologyAnnotator:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._ols = OLSClient(self._settings)
        self._bioportal: BioPortalClient | None = (
            BioPortalClient(self._settings) if self._settings.bioportal_enabled else None
        )

    async def close(self) -> None:
        await self._ols.close()
        if self._bioportal:
            await self._bioportal.close()

    async def __aenter__(self) -> OntologyAnnotator:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    def _resolve_ontologies(
        self, domain: str | None, preferred: list[str] | None
    ) -> list[str] | None:
        if preferred:
            return [o.lower() for o in preferred]
        if domain and domain in VALID_DOMAINS:
            return self._settings.ontologies_for_domain(domain)
        return None

    async def annotate(
        self,
        text: str,
        domain: str | None = None,
        preferred_ontologies: list[str] | None = None,
        use_bioportal_fallback: bool = True,
        min_confidence: float = 0.7,
    ) -> dict[str, Any]:
        """Annotate a single text term through the multi-stage pipeline.

        Pipeline:
          1. Exact label match via OLS
          2. Synonym match via OLS
          3. Fuzzy/search via OLS
          4. BioPortal fallback (if enabled and configured)
        """
        ontologies = self._resolve_ontologies(domain, preferred_ontologies)
        matches: list[OntologyMatch] = []

        # Stage 1: exact label
        exact = await self._ols.find_exact(text, ontologies)
        for raw in exact:
            matches.append(_raw_to_match(raw, "exact_label", CONFIDENCE_EXACT_LABEL))

        # Stage 2: synonym match (only if no exact label found)
        if not matches:
            synonyms = await self._ols.find_by_synonym(text, ontologies)
            for raw in synonyms:
                matches.append(_raw_to_match(raw, "synonym", CONFIDENCE_SYNONYM))

        # Stage 3: OLS fuzzy search (only if still no matches)
        if not matches:
            fuzzy = await self._ols.fuzzy_search(text, ontologies)
            for raw in fuzzy:
                matches.append(_raw_to_match(raw, "ols_search", CONFIDENCE_OLS_SEARCH))

        # Stage 4: BioPortal fallback
        if not matches and use_bioportal_fallback and self._bioportal:
            bp_results = await self._bioportal.find_exact(text, ontologies)
            if not bp_results:
                bp_results = await self._bioportal.fuzzy_search(text, ontologies)
            for raw in bp_results:
                matches.append(_raw_to_match(raw, "bioportal", CONFIDENCE_BIOPORTAL))

        # Deduplicate and filter by confidence
        matches = _deduplicate(matches)
        matches = [m for m in matches if m.confidence >= min_confidence]

        # Sort: highest confidence first
        matches.sort(key=lambda m: m.confidence, reverse=True)

        return {
            "input_text": text,
            "matches": [m.to_dict() for m in matches],
        }

    async def annotate_batch(
        self,
        texts: list[str],
        domain: str | None = None,
        preferred_ontologies: list[str] | None = None,
        use_bioportal_fallback: bool = True,
        min_confidence: float = 0.7,
    ) -> list[dict[str, Any]]:
        """Annotate multiple texts concurrently."""
        tasks = [
            self.annotate(
                text,
                domain=domain,
                preferred_ontologies=preferred_ontologies,
                use_bioportal_fallback=use_bioportal_fallback,
                min_confidence=min_confidence,
            )
            for text in texts
        ]
        return list(await asyncio.gather(*tasks))
