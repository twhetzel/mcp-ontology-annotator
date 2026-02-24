"""Tests for the ontology annotator."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest
import pytest_httpx

from ontology_annotator.annotator import OntologyAnnotator, OntologyMatch, _deduplicate
from ontology_annotator.config import Settings
from ontology_annotator.extractor import EntityExtractor, ExtractionError
from ontology_annotator.ols_client import OLSClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def settings() -> Settings:
    return Settings(
        anthropic_api_key="test-key",
        ols_api_url="https://www.ebi.ac.uk/ols4/api",
        bioportal_api_key="",  # disabled
    )


@pytest.fixture
def mock_ols_exact_result() -> list[dict]:
    return [
        {
            "term_id": "MONDO:0005015",
            "label": "diabetes mellitus",
            "ontology": "mondo",
            "definition": "A metabolic disorder characterized by hyperglycaemia.",
            "synonyms": ["diabetes", "DM"],
            "cross_references": {"doid": "DOID:9351"},
        }
    ]


# ---------------------------------------------------------------------------
# Unit tests: _deduplicate
# ---------------------------------------------------------------------------


def test_deduplicate_removes_duplicates():
    m1 = OntologyMatch("MONDO:0005015", "diabetes mellitus", "mondo", "exact_label", 0.98)
    m2 = OntologyMatch("MONDO:0005015", "diabetes mellitus", "mondo", "synonym", 0.85)
    result = _deduplicate([m1, m2])
    assert len(result) == 1
    assert result[0].match_type == "exact_label"


def test_deduplicate_keeps_different_ontologies():
    m1 = OntologyMatch("MONDO:0005015", "diabetes mellitus", "mondo", "exact_label", 0.98)
    m2 = OntologyMatch("DOID:9351", "diabetes mellitus", "doid", "exact_label", 0.98)
    result = _deduplicate([m1, m2])
    assert len(result) == 2


# ---------------------------------------------------------------------------
# Unit tests: OntologyAnnotator
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_annotate_exact_match(settings, mock_ols_exact_result):
    with patch("ontology_annotator.annotator.OLSClient") as MockOLS:
        mock_ols = AsyncMock()
        mock_ols.find_exact = AsyncMock(return_value=mock_ols_exact_result)
        mock_ols.find_by_synonym = AsyncMock(return_value=[])
        mock_ols.fuzzy_search = AsyncMock(return_value=[])
        mock_ols.close = AsyncMock()
        MockOLS.return_value = mock_ols

        async with OntologyAnnotator(settings) as annotator:
            result = await annotator.annotate("diabetes mellitus", domain="disease")

    assert result["input_text"] == "diabetes mellitus"
    assert len(result["matches"]) == 1
    match = result["matches"][0]
    assert match["term_id"] == "MONDO:0005015"
    assert match["match_type"] == "exact_label"
    assert match["confidence"] == 0.98


@pytest.mark.asyncio
async def test_annotate_synonym_fallback(settings):
    synonym_result = [
        {
            "term_id": "MONDO:0005015",
            "label": "diabetes mellitus",
            "ontology": "mondo",
            "definition": None,
            "synonyms": ["diabetes", "DM"],
            "cross_references": {},
        }
    ]
    with patch("ontology_annotator.annotator.OLSClient") as MockOLS:
        mock_ols = AsyncMock()
        mock_ols.find_exact = AsyncMock(return_value=[])
        mock_ols.find_by_synonym = AsyncMock(return_value=synonym_result)
        mock_ols.fuzzy_search = AsyncMock(return_value=[])
        mock_ols.close = AsyncMock()
        MockOLS.return_value = mock_ols

        async with OntologyAnnotator(settings) as annotator:
            result = await annotator.annotate("diabetes", domain="disease")

    assert result["matches"][0]["match_type"] == "synonym"
    assert result["matches"][0]["confidence"] == 0.85


@pytest.mark.asyncio
async def test_annotate_min_confidence_filters(settings):
    low_conf_result = [
        {
            "term_id": "X:001",
            "label": "something",
            "ontology": "test",
            "definition": None,
            "synonyms": [],
            "cross_references": {},
        }
    ]
    with patch("ontology_annotator.annotator.OLSClient") as MockOLS:
        mock_ols = AsyncMock()
        mock_ols.find_exact = AsyncMock(return_value=[])
        mock_ols.find_by_synonym = AsyncMock(return_value=[])
        mock_ols.fuzzy_search = AsyncMock(return_value=low_conf_result)
        mock_ols.close = AsyncMock()
        MockOLS.return_value = mock_ols

        async with OntologyAnnotator(settings) as annotator:
            # OLS search returns confidence 0.75; min_confidence of 0.8 should filter it
            result = await annotator.annotate("something", min_confidence=0.8)

    assert result["matches"] == []


@pytest.mark.asyncio
async def test_annotate_batch_returns_list(settings, mock_ols_exact_result):
    with patch("ontology_annotator.annotator.OLSClient") as MockOLS:
        mock_ols = AsyncMock()
        mock_ols.find_exact = AsyncMock(return_value=mock_ols_exact_result)
        mock_ols.find_by_synonym = AsyncMock(return_value=[])
        mock_ols.fuzzy_search = AsyncMock(return_value=[])
        mock_ols.close = AsyncMock()
        MockOLS.return_value = mock_ols

        async with OntologyAnnotator(settings) as annotator:
            results = await annotator.annotate_batch(
                ["diabetes mellitus", "diabetes mellitus"], domain="disease"
            )

    assert len(results) == 2
    for r in results:
        assert "input_text" in r
        assert "matches" in r


# ---------------------------------------------------------------------------
# Unit tests: EntityExtractor
# ---------------------------------------------------------------------------


def test_extractor_raises_without_api_key():
    settings = Settings(anthropic_api_key="")
    with pytest.raises(ExtractionError, match="ANTHROPIC_API_KEY"):
        EntityExtractor(settings)


def test_extractor_parse_valid_json():
    settings = Settings(anthropic_api_key="test-key")
    with patch("anthropic.AsyncAnthropic"):
        extractor = EntityExtractor(settings)

    raw_json = """[
        {"text": "diabetes", "start_pos": 0, "end_pos": 8, "domain": "disease", "confidence": 0.95}
    ]"""
    original = "diabetes mellitus"
    result = extractor._parse_response(raw_json, original)
    assert len(result) == 1
    assert result[0]["text"] == "diabetes"
    assert result[0]["domain"] == "disease"
    assert result[0]["extraction_confidence"] == 0.95


def test_extractor_parse_markdown_fenced_json():
    settings = Settings(anthropic_api_key="test-key")
    with patch("anthropic.AsyncAnthropic"):
        extractor = EntityExtractor(settings)

    fenced = """```json
[{"text": "aspirin", "start_pos": 0, "end_pos": 7, "domain": "chemical", "confidence": 0.9}]
```"""
    result = extractor._parse_response(fenced, "aspirin")
    assert len(result) == 1
    assert result[0]["domain"] == "chemical"


def test_extractor_parse_invalid_json_returns_empty():
    settings = Settings(anthropic_api_key="test-key")
    with patch("anthropic.AsyncAnthropic"):
        extractor = EntityExtractor(settings)

    result = extractor._parse_response("not valid json", "some text")
    assert result == []


def test_extractor_skips_invalid_domain():
    settings = Settings(anthropic_api_key="test-key")
    with patch("anthropic.AsyncAnthropic"):
        extractor = EntityExtractor(settings)

    raw = (
        '[{"text": "x", "start_pos": 0, "end_pos": 1,'
        ' "domain": "invalid_domain", "confidence": 0.9}]'
    )
    result = extractor._parse_response(raw, "x")
    assert result == []


# ---------------------------------------------------------------------------
# Unit tests: OLSClient — fieldList and synonym matching
# ---------------------------------------------------------------------------

OLS_SEARCH_RESPONSE_WITH_SYNONYMS = {
    "response": {
        "docs": [
            {
                "obo_id": "CHEBI:15365",
                "label": "acetylsalicylic acid",
                "ontology_name": "chebi",
                "synonym": ["Aspirin", "ASA", "acetylsalicylate"],
                "description": None,
                "obo_xref": [],
                "iri": "http://purl.obolibrary.org/obo/CHEBI_15365",
            }
        ]
    }
}

OLS_SEARCH_RESPONSE_EMPTY = {"response": {"docs": []}}


@pytest.mark.asyncio
async def test_ols_search_always_sends_fieldlist(httpx_mock):
    """Every OLS search request must include fieldList so synonyms are returned."""
    httpx_mock.add_response(json=OLS_SEARCH_RESPONSE_WITH_SYNONYMS)

    settings = Settings(ols_api_url="https://www.ebi.ac.uk/ols4/api")
    async with OLSClient(settings) as client:
        await client.search("aspirin", ontologies=["chebi"], exact=False)

    request = httpx_mock.get_request()
    assert request is not None
    assert "fieldList" in request.url.params, (
        "fieldList must be present in every OLS search request"
    )
    assert "synonym" in request.url.params["fieldList"]


@pytest.mark.asyncio
async def test_ols_find_by_synonym_matches_when_synonym_in_response(httpx_mock):
    """find_by_synonym returns a match when the API response includes synonym data."""
    # Non-exact search (used by find_by_synonym) returns the term with synonyms
    httpx_mock.add_response(json=OLS_SEARCH_RESPONSE_WITH_SYNONYMS)

    settings = Settings(ols_api_url="https://www.ebi.ac.uk/ols4/api")
    async with OLSClient(settings) as client:
        results = await client.find_by_synonym("aspirin", ontologies=["chebi"])

    assert len(results) == 1
    assert results[0]["term_id"] == "CHEBI:15365"
    assert results[0]["label"] == "acetylsalicylic acid"
    assert "Aspirin" in results[0]["synonyms"]


@pytest.mark.asyncio
async def test_ols_find_by_synonym_misses_when_synonym_field_null(httpx_mock):
    """Regression: synonym=null (old OLS behaviour without fieldList) yields no match."""
    response_null_synonyms = {
        "response": {
            "docs": [
                {
                    "obo_id": "CHEBI:15365",
                    "label": "acetylsalicylic acid",
                    "ontology_name": "chebi",
                    "synonym": None,  # what OLS returned before the fix
                    "description": None,
                    "obo_xref": [],
                    "iri": "http://purl.obolibrary.org/obo/CHEBI_15365",
                }
            ]
        }
    }
    httpx_mock.add_response(json=response_null_synonyms)

    settings = Settings(ols_api_url="https://www.ebi.ac.uk/ols4/api")
    async with OLSClient(settings) as client:
        results = await client.find_by_synonym("aspirin", ontologies=["chebi"])

    assert results == [], "synonym=null must not produce a match"


def test_extractor_fixes_wrong_positions():
    settings = Settings(anthropic_api_key="test-key")
    with patch("anthropic.AsyncAnthropic"):
        extractor = EntityExtractor(settings)

    original = "Patient has diabetes and hypertension"
    # Model gives wrong positions
    raw = (
        '[{"text": "diabetes", "start_pos": 99, "end_pos": 107,'
        ' "domain": "disease", "confidence": 0.9}]'
    )
    result = extractor._parse_response(raw, original)
    assert len(result) == 1
    assert result[0]["start_pos"] == original.index("diabetes")
    assert result[0]["end_pos"] == original.index("diabetes") + len("diabetes")
