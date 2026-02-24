"""OLS4 (Ontology Lookup Service) API client."""

from __future__ import annotations

import logging
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .config import Settings, get_settings

logger = logging.getLogger(__name__)


class OLSError(Exception):
    """Raised when an OLS API call fails."""


class OLSClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._base_url = self._settings.ols_api_url.rstrip("/")
        self._client = httpx.AsyncClient(
            timeout=self._settings.ols_timeout,
            headers={"Accept": "application/json"},
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> OLSClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
        reraise=True,
    )
    async def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        try:
            response = await self._client.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            raise OLSError(f"OLS HTTP {exc.response.status_code} for {url}") from exc
        except (httpx.TimeoutException, httpx.ConnectError):
            raise
        except Exception as exc:
            raise OLSError(f"OLS request failed: {exc}") from exc

    async def search(
        self,
        query: str,
        ontologies: list[str] | None = None,
        exact: bool = False,
        rows: int | None = None,
    ) -> list[dict[str, Any]]:
        """Search OLS for terms matching `query`.

        Returns a list of raw term dicts from OLS response docs.
        """
        # OLS4 only returns synonym/description data when explicitly requested via fieldList
        params: dict[str, Any] = {
            "q": query,
            "rows": rows or self._settings.ols_max_results,
            "fieldList": (
                "id,iri,label,ontology_name,description,synonym,obo_xref,short_form,obo_id"
            ),
        }
        if ontologies:
            params["ontology"] = ",".join(ontologies)
        if exact:
            params["exact"] = "true"

        try:
            data = await self._get("/search", params)
        except OLSError:
            logger.warning("OLS search failed for query=%r", query)
            return []

        docs: list[dict[str, Any]] = (
            data.get("response", {}).get("docs", [])
        )
        return docs

    async def get_term(self, ontology: str, iri_or_id: str) -> dict[str, Any] | None:
        """Fetch a specific term by IRI or short-form ID."""
        # Try short-form lookup via search
        results = await self.search(iri_or_id, ontologies=[ontology], exact=True, rows=1)
        return results[0] if results else None

    def _parse_term(self, doc: dict[str, Any]) -> dict[str, Any]:
        """Normalise a raw OLS doc into a cleaner dict."""
        description = doc.get("description")
        if isinstance(description, list):
            description = description[0] if description else None

        synonyms: list[str] = doc.get("synonym", []) or []
        if isinstance(synonyms, str):
            synonyms = [synonyms]

        # Cross-references from obo_xref
        cross_refs: dict[str, str] = {}
        for xref in doc.get("obo_xref", []) or []:
            if isinstance(xref, dict):
                db = xref.get("database", "").lower()
                acc = xref.get("id", "")
                if db and acc:
                    cross_refs[db] = f"{db.upper()}:{acc}"

        obo_id: str = doc.get("obo_id") or doc.get("short_form") or ""

        return {
            "term_id": obo_id,
            "label": doc.get("label", ""),
            "ontology": doc.get("ontology_name", ""),
            "definition": description,
            "synonyms": synonyms,
            "iri": doc.get("iri", ""),
            "cross_references": cross_refs,
        }

    async def find_exact(
        self, query: str, ontologies: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Return terms where label exactly matches `query` (case-insensitive)."""
        docs = await self.search(query, ontologies=ontologies, exact=True)
        query_lower = query.lower()
        results = []
        for doc in docs:
            label = (doc.get("label") or "").lower()
            if label == query_lower:
                results.append(self._parse_term(doc))
        return results

    async def find_by_synonym(
        self, query: str, ontologies: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Return terms where a synonym exactly matches `query` (case-insensitive)."""
        docs = await self.search(query, ontologies=ontologies, exact=False)
        query_lower = query.lower()
        results = []
        for doc in docs:
            synonyms = doc.get("synonym", []) or []
            if isinstance(synonyms, str):
                synonyms = [synonyms]
            if any(s.lower() == query_lower for s in synonyms):
                # Ensure label doesn't already match (avoid duplicates)
                label = (doc.get("label") or "").lower()
                if label != query_lower:
                    results.append(self._parse_term(doc))
        return results

    async def fuzzy_search(
        self, query: str, ontologies: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Return top OLS search results for a free-text query."""
        docs = await self.search(query, ontologies=ontologies, exact=False)
        return [self._parse_term(d) for d in docs]
