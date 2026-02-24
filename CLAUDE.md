# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`ontology-annotator-mcp` is a Python-based MCP (Model Context Protocol) server for ontology annotation. The project is in early setup; source code, configuration, and tooling have not yet been implemented.

## Architecture

This will be an MCP server exposing ontology annotation capabilities to AI assistants. MCP servers communicate with clients (e.g., Claude Desktop) via the Model Context Protocol, exposing tools and/or resources.

## Development Setup

```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install runtime + dev dependencies
uv pip install -e ".[dev]"
# or just runtime:
uv pip install -e .
```

Copy `.env.example` to `.env` and populate at minimum `ANTHROPIC_API_KEY`.

## Running the Server

```bash
ontology-annotator
# or
python -m ontology_annotator.server
```

## Running Tests

```bash
pytest
```

## Linting

```bash
ruff check src tests
ruff format src tests
```

## Key Implementation Notes

- `config.py` — All settings via `pydantic-settings`; reads from `.env` automatically.
- `ols_client.py` — OLS4 REST client with tenacity retry/backoff. Three search modes: `find_exact`, `find_by_synonym`, `fuzzy_search`.
- `bioportal_client.py` — Optional BioPortal client; only active when `BIOPORTAL_API_KEY` is set.
- `annotator.py` — Four-stage pipeline: exact label → synonym → OLS fuzzy → BioPortal. Confidence thresholding and deduplication.
- `extractor.py` — Uses `claude-haiku-4-5-20251001` for entity extraction; parses + validates JSON response, fixes wrong offsets.
- `server.py` — MCP server wired up via `mcp.server.Server`; two tools exposed.
