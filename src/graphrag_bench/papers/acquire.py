"""Explicit, bounded downloads with verified cache reuse and no overwrite."""

import os
import tempfile
import time
from hashlib import sha256
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from graphrag_bench.papers.catalog import Paper, PaperCatalog, PaperError


def checked_pdf(path: Path, paper: Paper) -> bytes:
    """Verify an exact PDF snapshot before any parser sees it."""
    try:
        if path.is_symlink() or path.stat().st_size != paper.pdf_bytes:
            raise PaperError(f"PDF size mismatch or symlink: {path}")
        with path.open("rb") as stream:
            raw = stream.read(paper.pdf_bytes + 1)
        validate_pdf_bytes(raw, paper)
        return raw
    except OSError as error:
        raise PaperError(f"cannot read pinned PDF {path}: {error}") from error


def validate_pdf_bytes(raw: bytes, paper: Paper) -> None:
    if len(raw) != paper.pdf_bytes or sha256(raw).hexdigest() != paper.pdf_sha256:
        raise PaperError(f"PDF checksum/size mismatch: {paper.arxiv_version}")
    if not raw.startswith(b"%PDF-"):
        raise PaperError(f"not a PDF: {paper.arxiv_version}")


def fetch_papers(catalog: PaperCatalog, output: Path) -> dict[str, int]:
    """Download only missing PDFs; invalid cached bytes fail without replacement.

    Hashes are established when curating a catalog, never learned on the fly here.
    Versioned arXiv PDFs can be regenerated upstream; drift requires a new review.
    """
    downloaded = reused = 0
    try:
        if output.is_symlink():
            raise PaperError("raw output directory must not be a symlink")
        # Check all existing entries before any network request or write.
        for paper in catalog.papers:
            path = output / paper.filename
            if path.exists() or path.is_symlink():
                checked_pdf(path, paper)
        output.mkdir(parents=True, exist_ok=True)
        for paper in catalog.papers:
            path = output / paper.filename
            if path.exists():
                reused += 1
                continue
            if downloaded:
                time.sleep(3)  # Space serial arXiv requests; no parallel crawler.
            request = Request(paper.pdf_url, headers={"User-Agent": "GraphRAG-Bench/0.10"})
            with urlopen(request, timeout=60) as response:
                raw = response.read(paper.pdf_bytes + 1)
            validate_pdf_bytes(raw, paper)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=output, suffix=".part", delete=False) as f:
                    temporary = Path(f.name)
                    f.write(raw)
                # A concurrent writer cannot be overwritten by rename/replace.
                os.link(temporary, path)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            downloaded += 1
        return {"downloaded": downloaded, "reused": reused}
    except (OSError, URLError) as error:
        raise PaperError(f"cannot fetch pinned papers: {error}") from error
