"""Managed corpus workspaces for the interactive RRR product."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


class CorpusDiscoveryError(RuntimeError):
    """Raised when RRR cannot identify one PDF collection safely."""

    def __init__(self, message: str, candidates: Iterable[Path] = ()) -> None:
        super().__init__(message)
        self.candidates = tuple(Path(path) for path in candidates)


@dataclass(frozen=True)
class PreparedWorkspace:
    corpus_dir: Path
    workspace_dir: Path
    metadata_path: Path
    reused: bool
    pdf_count: int

    def as_dict(self) -> dict:
        return {
            "corpus_dir": str(self.corpus_dir),
            "workspace_dir": str(self.workspace_dir),
            "metadata_path": str(self.metadata_path),
            "reused": self.reused,
            "pdf_count": self.pdf_count,
        }


def _pdfs_in(directory: Path) -> list[Path]:
    return sorted(
        (
            path.resolve()
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() == ".pdf"
        ),
        key=lambda path: path.name.casefold(),
    )


def discover_corpus_dir(start: Path) -> Path:
    """Find one nearby directory containing PDFs without searching broadly."""
    start = Path(start).expanduser().resolve()
    if not start.is_dir():
        raise CorpusDiscoveryError(f"The selected location is not a folder: {start}")

    if _pdfs_in(start):
        return start

    standard = start / "corpus"
    if standard.is_dir() and _pdfs_in(standard):
        return standard.resolve()

    candidates: list[Path] = []
    for child in sorted(start.iterdir(), key=lambda path: path.name.casefold()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        resolved = child.resolve()
        if _pdfs_in(resolved):
            candidates.append(resolved)

    candidates = list(dict.fromkeys(candidates))
    if not candidates:
        raise CorpusDiscoveryError(
            "No PDF collection was found. Select or open the folder containing "
            "the PDFs you want RRR to analyse."
        )
    if len(candidates) > 1:
        raise CorpusDiscoveryError(
            "More than one PDF collection was found. Select the folder RRR "
            "should use.",
            candidates,
        )
    return candidates[0]


def rrr_data_dir() -> Path:
    override = os.environ.get("RRR_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "RRR"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "RRR"
    if os.environ.get("XDG_DATA_HOME"):
        return Path(os.environ["XDG_DATA_HOME"]) / "rrr"
    return Path.home() / ".local" / "share" / "rrr"


def workspace_for_corpus(corpus_dir: Path, data_dir: Optional[Path] = None) -> Path:
    corpus_dir = Path(corpus_dir).expanduser().resolve()
    root = Path(data_dir).expanduser().resolve() if data_dir else rrr_data_dir()
    slug = re.sub(r"[^A-Za-z0-9]+", "-", corpus_dir.name).strip("-").lower()
    slug = slug[:40] or "corpus"
    digest = hashlib.sha256(os.path.normcase(str(corpus_dir)).encode("utf-8")).hexdigest()[:12]
    return root / "corpora" / f"{slug}-{digest}"


def corpus_snapshot(corpus_dir: Path) -> list[dict]:
    return [
        {
            "path": str(path),
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in _pdfs_in(Path(corpus_dir))
    ]


def _manifest_path(workspace_dir: Path) -> Path:
    return workspace_dir / "workspace.json"


def _is_reusable(workspace_dir: Path, snapshot: list[dict]) -> bool:
    required = (
        workspace_dir / "metadata.csv",
        workspace_dir / "indices" / "bm25.pkl",
        workspace_dir / "indices" / "page_ids.npy",
        workspace_dir / "indices" / "docs.csv",
    )
    if not all(path.is_file() for path in required):
        return False
    page_text = workspace_dir / "data" / "page_text"
    if not page_text.is_dir() or not any(page_text.glob("*.txt")):
        return False
    try:
        manifest = json.loads(_manifest_path(workspace_dir).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    return manifest.get("corpus_snapshot") == snapshot


def _run(command: list[str], env: dict[str, str], quiet: bool = False) -> None:
    completed = subprocess.run(
        command,
        env=env,
        check=False,
        capture_output=quiet,
        text=quiet,
    )
    if quiet:
        if completed.stdout:
            sys.stderr.write(completed.stdout)
        if completed.stderr:
            sys.stderr.write(completed.stderr)
    if completed.returncode:
        raise SystemExit(completed.returncode)


def prepare_workspace(
    corpus_dir: Path,
    workspace_dir: Optional[Path] = None,
    bibliography: Optional[Path] = None,
    use_llm: bool = False,
    accept_low_confidence: bool = False,
    force: bool = False,
    quiet: bool = False,
) -> PreparedWorkspace:
    """Ingest, preprocess, and index a PDF folder in a managed workspace."""
    corpus_dir = Path(corpus_dir).expanduser().resolve()
    pdfs = _pdfs_in(corpus_dir)
    if not pdfs:
        raise CorpusDiscoveryError(f"No PDF files were found in {corpus_dir}")

    workspace_dir = (
        Path(workspace_dir).expanduser().resolve()
        if workspace_dir
        else workspace_for_corpus(corpus_dir)
    )
    snapshot = corpus_snapshot(corpus_dir)
    metadata_path = workspace_dir / "metadata.csv"
    if not force and _is_reusable(workspace_dir, snapshot):
        return PreparedWorkspace(
            corpus_dir=corpus_dir,
            workspace_dir=workspace_dir,
            metadata_path=metadata_path,
            reused=True,
            pdf_count=len(pdfs),
        )

    workspace_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["RRR_PROJECT_ROOT"] = str(workspace_dir)

    ingest = [
        sys.executable,
        "-m",
        "rrr.cli",
        "ingest",
        "--corpus",
        str(corpus_dir),
        "--output",
        str(metadata_path),
    ]
    if bibliography:
        ingest.extend(["--bib", str(Path(bibliography).expanduser().resolve())])
    if not use_llm:
        ingest.append("--no-llm")
    if accept_low_confidence:
        ingest.append("--accept-low-confidence")
    _run(ingest, env, quiet=quiet)
    _run(
        [sys.executable, "-m", "rrr.preprocess", "--metadata", str(metadata_path)],
        env,
        quiet=quiet,
    )
    _run(
        [sys.executable, "-m", "rrr.index", "--metadata", str(metadata_path)],
        env,
        quiet=quiet,
    )

    manifest = {
        "schema_version": 1,
        "corpus_dir": str(corpus_dir),
        "workspace_dir": str(workspace_dir),
        "metadata_path": str(metadata_path),
        "corpus_snapshot": snapshot,
    }
    _manifest_path(workspace_dir).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return PreparedWorkspace(
        corpus_dir=corpus_dir,
        workspace_dir=workspace_dir,
        metadata_path=metadata_path,
        reused=False,
        pdf_count=len(pdfs),
    )
