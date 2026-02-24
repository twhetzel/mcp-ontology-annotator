"""BioPortal REST API client (optional fallback)."""

from __future__ import annotations

import logging
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .config import Settings, get_settings

logger = logging.getLogger(__name__)


class BioPortalError(Exception):
    """Raised when a BioPortal API call fails."""


# Map our generic ontology names to BioPortal acronyms
BIOPORTAL_ACRONYM_MAP: dict[str, str] = {
    "mondo": "MONDO",
    "doid": "DOID",
    "hp": "HP",
    "chebi": "CHEBI",
    "drugbank": "DRUGBANK",
    "hgnc": "HGNC",
    "ncbigene": "NCBIGENE",
    "mp": "MP",
    "uberon": "UBERON",
    "fma": "FMA",
    "ncbitaxon": "NCBITAXON",
}


class BioPortalClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._base_url = self._settings.bioportal_api_url.rstrip("/")
        self._api_key = self._settings.bioportal_api_key
        self._client = httpx.AsyncClient(
            timeout=self._settings.bioportal_timeout,
            headers={
                "Accept": "application/json",
                "Authorization": f"apikey token={self._api_key}",
            },
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> BioPortalClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
        reraise=True,
    )
    async def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        if not self._api_key:
            raise BioPortalError("BIOPORTAL_API_KEY is not configured")
        url = f"{self._base_url}{path}"
        try:
            response = await self._client.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            raise BioPortalError(
                f"BioPortal HTTP {exc.response.status_code} for {url}"
            ) from exc
        except (httpx.TimeoutException, httpx.ConnectError):
            raise
        except Exception as exc:
            raise BioPortalError(f"BioPortal request failed: {exc}") from exc

    def _ontology_acronyms(self, ontologies: list[str] | None) -> list[str]:
        if not ontologies:
            return []
        return [BIOPORTAL_ACRONYM_MAP.get(o.lower(), o.upper()) for o in ontologies]

    def _parse_result(self, item: dict[str, Any]) -> dict[str, Any]:
        prefLabel: str = item.get("prefLabel") or ""
        synonyms: list[str] = item.get("synonym", []) or []
        if isinstance(synonyms, str):
            synonyms = [synonyms]

        # Extract ontology acronym from @id or links
        ontology_acronym = ""
        links = item.get("links", {}) or {}
        onto_link: str = links.get("ontology", "") or ""
        if onto_link:
            ontology_acronym = onto_link.rstrip("/").split("/")[-1].lower()

        # Build a CURIe-style ID from @id if possible
        at_id: str = item.get("@id") or item.get("id") or ""
        # BioPortal uses full IRIs; try to make a short form
        term_id = at_id
        for sep in ("/", "#"):
            if sep in at_id:
                local = at_id.rsplit(sep, 1)[-1]
                if local:
                    term_id = local.replace("_", ":")

        definition: str | None = None
        definitions = item.get("definition", [])
        if isinstance(definitions, list) and definitions:
            definition = definitions[0]
        elif isinstance(definitions, str):
            definition = definitions

        return {
            "term_id": term_id,
            "label": prefLabel,
            "ontology": ontology_acronym,
            "definition": definition,
            "synonyms": synonyms,
            "iri": at_id,
            "cross_references": {},
        }

    async def search(
        self,
        query: str,
        ontologies: list[str] | None = None,
        exact: bool = False,
        rows: int | None = None,
    ) -> list[dict[str, Any]]:
        """Search BioPortal for terms matching `query`."""
        params: dict[str, Any] = {
            "q": query,
            "pagesize": rows or self._settings.bioportal_max_results,
            "display_links": "false",
            "display_context": "false",
        }
        acronyms = self._ontology_acronyms(ontologies)
        if acronyms:
            params["ontologies"] = ",".join(acronyms)
        if exact:
            params["require_exact_match"] = "true"

        try:
            data = await self._get("/search", params)
        except BioPortalError:
            logger.warning("BioPortal search failed for query=%r", query)
            return []

        if not isinstance(data, dict):
            return []

        items: list[dict[str, Any]] = data.get("collection", []) or []
        return [self._parse_result(item) for item in items]

    async def find_exact(
        self, query: str, ontologies: list[str] | None = None
    ) -> list[dict[str, Any]]:
        results = await self.search(query, ontologies=ontologies, exact=True)
        query_lower = query.lower()
        return [r for r in results if (r.get("label") or "").lower() == query_lower]

    async def fuzzy_search(
        self, query: str, ontologies: list[str] | None = None
    ) -> list[dict[str, Any]]:
        return await self.search(query, ontologies=ontologies, exact=False)
