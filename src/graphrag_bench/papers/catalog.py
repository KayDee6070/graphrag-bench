"""Small, explicit acquisition catalogs; no implicit network or license inference."""

from datetime import date
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from graphrag_bench.models import Identifier, Record, Sha256, Text, require_unique

MAX_PDF_BYTES = 16_000_000
MAX_CATALOG_BYTES = 32_000_000
ArxivVersion = Annotated[str, StringConstraints(pattern=r"^\d{4}\.\d{4,5}v[1-9]\d*$")]


class PaperError(ValueError):
    """Paper acquisition, parsing, or annotation validation failed."""


class Paper(Record):
    paper_id: Identifier
    arxiv_version: ArxivVersion
    title: Text
    authors: tuple[Text, ...] = Field(min_length=1)
    license_id: Literal["CC-BY-4.0", "CC-BY-NC-SA-4.0"]
    metadata_checked_on: date
    pdf_sha256: Sha256
    pdf_bytes: Annotated[int, Field(gt=0, le=MAX_PDF_BYTES, strict=True)]

    @property
    def pdf_url(self) -> str:
        return f"https://arxiv.org/pdf/{self.arxiv_version}"

    @property
    def metadata_url(self) -> str:
        return f"https://arxiv.org/abs/{self.arxiv_version}"

    @property
    def filename(self) -> str:
        return f"{self.arxiv_version}.pdf"

    @property
    def license_url(self) -> str:
        name = "by" if self.license_id == "CC-BY-4.0" else "by-nc-sa"
        return f"https://creativecommons.org/licenses/{name}/4.0/"


class PaperCatalog(Record):
    catalog_id: Identifier
    purpose: Literal["development-pilot"] = "development-pilot"
    papers: tuple[Paper, ...] = Field(min_length=1, max_length=10)

    @model_validator(mode="after")
    def check_unique_and_bounded(self) -> Self:
        require_unique(tuple(p.paper_id for p in self.papers), "paper IDs")
        require_unique(tuple(p.arxiv_version.split("v")[0] for p in self.papers), "arXiv works")
        if sum(p.pdf_bytes for p in self.papers) > MAX_CATALOG_BYTES:
            raise ValueError("pilot catalog exceeds 32 MB; plan a reviewed corpus expansion")
        return self


def load_catalog(path: Path) -> PaperCatalog:
    try:
        return PaperCatalog.model_validate_json(path.read_bytes())
    except (OSError, ValueError) as error:
        raise PaperError(f"cannot load paper catalog: {error}") from error
