"""MCP server exposing ontology annotation tools."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from .annotator import OntologyAnnotator
from .config import VALID_DOMAINS, Settings, get_settings
from .extractor import EntityExtractor, ExtractionError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

ANNOTATE_TOOL = Tool(
    name="annotate_ontology_terms",
    description=(
        "Map known text terms to standardized ontology IDs. "
        "Use when you have specific terms to annotate."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "texts": {
                "oneOf": [
                    {"type": "string"},
                    {"type": "array", "items": {"type": "string"}},
                ],
                "description": "Term(s) to annotate",
            },
            "domain": {
                "type": "string",
                "enum": list(VALID_DOMAINS),
                "description": "Biomedical domain (optional; narrows ontology search)",
            },
            "preferred_ontologies": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Override default ontologies (e.g. ['mondo', 'doid'])",
            },
            "use_bioportal_fallback": {
                "type": "boolean",
                "default": True,
                "description": "Fall back to BioPortal when OLS finds nothing",
            },
            "min_confidence": {
                "type": "number",
                "default": 0.7,
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Minimum confidence threshold for returned matches",
            },
        },
        "required": ["texts"],
    },
)

EXTRACT_TOOL = Tool(
    name="extract_and_annotate",
    description=(
        "Extract biomedical entities from natural language and annotate them. "
        "Use for full sentences or queries."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Natural language text to process",
            },
            "domains": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": list(VALID_DOMAINS),
                },
                "description": "Entity types to extract (defaults to all domains)",
            },
            "preferred_ontologies": {
                "type": "object",
                "description": (
                    'Domain-specific ontology preferences, e.g. {"disease": ["mondo"]}'
                ),
            },
            "use_bioportal_fallback": {
                "type": "boolean",
                "default": True,
            },
            "min_confidence": {
                "type": "number",
                "default": 0.7,
                "minimum": 0.0,
                "maximum": 1.0,
            },
        },
        "required": ["text"],
    },
)


# ---------------------------------------------------------------------------
# Handler helpers
# ---------------------------------------------------------------------------


def _parse_texts(raw: Any) -> list[str]:
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [str(t) for t in raw]
    raise ValueError(f"'texts' must be a string or list of strings, got {type(raw).__name__}")


def _json_response(data: Any) -> list[TextContent]:
    return [TextContent(type="text", text=json.dumps(data, ensure_ascii=False, indent=2))]


def _error_response(message: str) -> list[TextContent]:
    return _json_response({"error": message})


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


def create_server() -> Server:
    settings = get_settings()
    app = Server("ontology-annotator")

    @app.list_tools()
    async def list_tools() -> list[Tool]:
        return [ANNOTATE_TOOL, EXTRACT_TOOL]

    @app.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        if name == "annotate_ontology_terms":
            return await _handle_annotate(arguments, settings)
        if name == "extract_and_annotate":
            return await _handle_extract(arguments, settings)
        return _error_response(f"Unknown tool: {name}")

    return app


async def _handle_annotate(
    args: dict[str, Any], settings: Settings
) -> list[TextContent]:
    try:
        texts = _parse_texts(args.get("texts"))
    except ValueError as exc:
        return _error_response(str(exc))

    domain: str | None = args.get("domain")
    preferred_ontologies: list[str] | None = args.get("preferred_ontologies")
    use_bioportal_fallback: bool = bool(args.get("use_bioportal_fallback", True))
    min_confidence: float = float(args.get("min_confidence", 0.7))

    if domain and domain not in VALID_DOMAINS:
        return _error_response(
            f"Invalid domain '{domain}'. Valid domains: {sorted(VALID_DOMAINS)}"
        )

    try:
        async with OntologyAnnotator(settings) as annotator:
            annotations = await annotator.annotate_batch(
                texts,
                domain=domain,
                preferred_ontologies=preferred_ontologies,
                use_bioportal_fallback=use_bioportal_fallback,
                min_confidence=min_confidence,
            )
    except Exception as exc:
        logger.exception("annotate_ontology_terms failed")
        return _error_response(f"Annotation failed: {exc}")

    return _json_response({"annotations": annotations})


async def _handle_extract(
    args: dict[str, Any], settings: Settings
) -> list[TextContent]:
    text: str = args.get("text", "")
    if not text:
        return _error_response("'text' is required and must not be empty")

    domains: list[str] | None = args.get("domains")
    preferred_ontologies: dict[str, list[str]] | None = args.get("preferred_ontologies")
    use_bioportal_fallback: bool = bool(args.get("use_bioportal_fallback", True))
    min_confidence: float = float(args.get("min_confidence", 0.7))

    # Step 1: extract entities with LLM
    try:
        extractor = EntityExtractor(settings)
    except ExtractionError as exc:
        return _error_response(str(exc))

    try:
        entities = await extractor.extract(text, domains=domains)
    except ExtractionError as exc:
        logger.exception("Entity extraction failed")
        return _error_response(f"Entity extraction failed: {exc}")

    if not entities:
        return _json_response({"extracted_entities": [], "original_text": text})

    # Step 2: annotate each entity
    try:
        async with OntologyAnnotator(settings) as annotator:
            annotated_entities = []
            for entity in entities:
                domain = entity["domain"]
                per_domain_onto = (
                    preferred_ontologies.get(domain) if preferred_ontologies else None
                )
                result = await annotator.annotate(
                    entity["text"],
                    domain=domain,
                    preferred_ontologies=per_domain_onto,
                    use_bioportal_fallback=use_bioportal_fallback,
                    min_confidence=min_confidence,
                )
                annotated_entities.append(
                    {
                        "text": entity["text"],
                        "start_pos": entity["start_pos"],
                        "end_pos": entity["end_pos"],
                        "domain": domain,
                        "extraction_confidence": entity["extraction_confidence"],
                        "matches": result["matches"],
                    }
                )
    except Exception as exc:
        logger.exception("Annotation step in extract_and_annotate failed")
        return _error_response(f"Annotation failed: {exc}")

    return _json_response({"extracted_entities": annotated_entities, "original_text": text})


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the MCP server over stdio."""
    app = create_server()

    async def _run() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await app.run(read_stream, write_stream, app.create_initialization_options())

    asyncio.run(_run())


if __name__ == "__main__":
    main()
