"""Configuration via environment variables."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

DOMAIN_ONTOLOGY_DEFAULTS: dict[str, list[str]] = {
    "disease": ["mondo", "doid", "hp"],
    "chemical": ["chebi", "drugbank"],
    "gene": ["hgnc", "ncbigene"],
    "phenotype": ["hp", "mp"],
    "anatomy": ["uberon", "fma"],
    "organism": ["ncbitaxon"],
}

VALID_DOMAINS = frozenset(DOMAIN_ONTOLOGY_DEFAULTS.keys())


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Anthropic
    anthropic_api_key: str = ""

    # OLS
    ols_api_url: str = "https://www.ebi.ac.uk/ols4/api"
    ols_timeout: float = 30.0
    ols_max_results: int = 10

    # Bioportal
    bioportal_api_key: str = ""
    bioportal_api_url: str = "https://data.bioontology.org"
    bioportal_timeout: float = 30.0
    bioportal_max_results: int = 10

    # Default ontologies per domain (comma-separated)
    default_disease_ontologies: str = "mondo,doid,hp"
    default_chemical_ontologies: str = "chebi,drugbank"
    default_gene_ontologies: str = "hgnc,ncbigene"
    default_phenotype_ontologies: str = "hp,mp"
    default_anatomy_ontologies: str = "uberon,fma"
    default_organism_ontologies: str = "ncbitaxon"

    # Retry settings
    max_retries: int = 3
    retry_min_wait: float = 1.0
    retry_max_wait: float = 10.0

    def ontologies_for_domain(self, domain: str) -> list[str]:
        """Return the configured ontology list for a domain."""
        attr = f"default_{domain}_ontologies"
        raw = getattr(self, attr, None)
        if raw:
            return [o.strip().lower() for o in raw.split(",") if o.strip()]
        return DOMAIN_ONTOLOGY_DEFAULTS.get(domain, [])

    @property
    def bioportal_enabled(self) -> bool:
        return bool(self.bioportal_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
