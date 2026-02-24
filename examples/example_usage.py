"""Example usage of the ontology annotator (run directly, not via MCP)."""

from __future__ import annotations

import asyncio
import json
import os

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from ontology_annotator.annotator import OntologyAnnotator
from ontology_annotator.config import get_settings
from ontology_annotator.extractor import EntityExtractor, ExtractionError


async def example_annotate_terms() -> None:
    print("=" * 60)
    print("Example 1: annotate_ontology_terms")
    print("=" * 60)

    terms = ["diabetes mellitus", "aspirin", "BRCA1", "hippocampus"]

    async with OntologyAnnotator() as annotator:
        results = await annotator.annotate_batch(terms)

    for annotation in results:
        print(f"\nTerm: {annotation['input_text']!r}")
        if annotation["matches"]:
            top = annotation["matches"][0]
            print(f"  Best match : {top['term_id']} ({top['ontology']})")
            print(f"  Label      : {top['label']}")
            print(f"  Match type : {top['match_type']}")
            print(f"  Confidence : {top['confidence']:.2f}")
        else:
            print("  No matches found.")


async def example_extract_and_annotate() -> None:
    print("\n" + "=" * 60)
    print("Example 2: extract_and_annotate")
    print("=" * 60)

    text = (
        "The patient was diagnosed with type 2 diabetes mellitus "
        "and prescribed metformin. Family history is positive for "
        "Alzheimer disease. Physical examination revealed hepatomegaly."
    )
    print(f"Text: {text}\n")

    settings = get_settings()
    if not settings.anthropic_api_key:
        print("ANTHROPIC_API_KEY not set — skipping extract_and_annotate example.")
        return

    try:
        extractor = EntityExtractor(settings)
    except ExtractionError as exc:
        print(f"Extractor unavailable: {exc}")
        return

    entities = await extractor.extract(text)
    print(f"Extracted {len(entities)} entities:\n")

    if not entities:
        print("  (none)")
        return

    async with OntologyAnnotator(settings) as annotator:
        for entity in entities:
            result = await annotator.annotate(entity["text"], domain=entity["domain"])
            print(f"  [{entity['domain']}] {entity['text']!r}  (conf={entity['extraction_confidence']:.2f})")
            for m in result["matches"][:2]:
                print(f"    -> {m['term_id']} ({m['ontology']}) [{m['match_type']}]")


async def main() -> None:
    await example_annotate_terms()
    await example_extract_and_annotate()


if __name__ == "__main__":
    asyncio.run(main())
