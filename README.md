# ontology-annotator-mcp

An MCP (Model Context Protocol) server for mapping biomedical text to standardized ontology terms.

## Features

- **`annotate_ontology_terms`** — Map specific terms to ontology IDs via a multi-stage pipeline:
  1. Exact label match (OLS)
  2. Synonym match (OLS)
  3. Fuzzy search (OLS)
  4. BioPortal fallback (optional, requires API key)
- **`extract_and_annotate`** — Extract biomedical entities from free text using Claude, then annotate each entity.
- Supports domains: `disease`, `chemical`, `gene`, `phenotype`, `anatomy`, `organism`
- Configurable default ontologies per domain
- Retry with exponential back-off on API failures

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) for dependency management

## Setup

```bash
# Clone the repo
git clone <repo-url>
cd ontology-annotator-mcp

# Create and activate virtual environment with uv
uv venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# Install the package and dependencies
uv pip install -e ".[dev]"    # includes dev/test dependencies
# or just the runtime:
uv pip install -e .
```

### Environment Variables

Copy `.env.example` to `.env` and fill in values:

```bash
cp .env.example .env
```

| Variable | Default | Required |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | For `extract_and_annotate` |
| `OLS_API_URL` | `https://www.ebi.ac.uk/ols4/api` | No |
| `BIOPORTAL_API_KEY` | — | For BioPortal fallback |
| `BIOPORTAL_API_URL` | `https://data.bioontology.org` | No |
| `DEFAULT_DISEASE_ONTOLOGIES` | `mondo,doid,hp` | No |
| `DEFAULT_CHEMICAL_ONTOLOGIES` | `chebi,drugbank` | No |
| `DEFAULT_GENE_ONTOLOGIES` | `hgnc,ncbigene` | No |
| `DEFAULT_PHENOTYPE_ONTOLOGIES` | `hp,mp` | No |
| `DEFAULT_ANATOMY_ONTOLOGIES` | `uberon,fma` | No |
| `DEFAULT_ORGANISM_ONTOLOGIES` | `ncbitaxon` | No |

## Running the MCP Server

```bash
# After installing with uv pip install -e .
ontology-annotator
# or
python -m ontology_annotator.server
```

The server communicates over stdio, as required by the MCP protocol.

### Claude Desktop Configuration

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS):

```json
{
  "mcpServers": {
    "ontology-annotator": {
      "command": "uv",
      "args": [
        "run",
        "--directory", "/path/to/ontology-annotator-mcp",
        "ontology-annotator"
      ],
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-...",
        "BIOPORTAL_API_KEY": ""
      }
    }
  }
}
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

## Example Usage

```bash
python examples/example_usage.py
```

## Tool Reference

### `annotate_ontology_terms`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `texts` | `string \| string[]` | required | Term(s) to annotate |
| `domain` | `string` | — | Biomedical domain (narrows ontology search) |
| `preferred_ontologies` | `string[]` | — | Override default ontologies |
| `use_bioportal_fallback` | `boolean` | `true` | Fall back to BioPortal |
| `min_confidence` | `number` | `0.7` | Minimum confidence (0–1) |

### `extract_and_annotate`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `text` | `string` | required | Natural language text |
| `domains` | `string[]` | all | Entity types to extract |
| `preferred_ontologies` | `object` | — | Per-domain ontology preferences |
| `use_bioportal_fallback` | `boolean` | `true` | Fall back to BioPortal |
| `min_confidence` | `number` | `0.7` | Minimum confidence |

## Project Structure

```
ontology-annotator-mcp/
├── src/
│   └── ontology_annotator/
│       ├── __init__.py
│       ├── server.py           # MCP server
│       ├── annotator.py        # Multi-stage annotation pipeline
│       ├── extractor.py        # LLM entity extraction
│       ├── ols_client.py       # OLS4 API client
│       ├── bioportal_client.py # BioPortal API client (optional)
│       └── config.py           # Settings via env vars
├── tests/
│   └── test_annotator.py
├── examples/
│   └── example_usage.py
├── pyproject.toml
├── .env.example
└── README.md
```
