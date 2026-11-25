from __future__ import annotations

import datetime
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional


def _slugify(text: str, max_len: int = 60) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", (text or "").strip()).strip("_").lower()
    return slug[:max_len] or "topic"


def _utc_stamp() -> str:
    return datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def mint_run_id(topic: Optional[str] = None) -> str:
    stamp = _utc_stamp()
    return f"{stamp}_{_slugify(topic)}" if topic else stamp


def _stage_cache_root_for(workdir: Path) -> Path:
    override = os.environ.get("RRR_STAGE_CACHE_DIR")
    return Path(override).expanduser() if override else workdir / "cache"


def stage_cache_enabled() -> bool:
    return os.environ.get("RRR_STAGE_CACHE", "1") != "0"


@dataclass(frozen=True)
class Workspace:
    workdir: Path
    corpus_dir: Path
    metadata_path: Path
    bibliography_path: Path
    claim_cache_root: Path
    stage_cache_root: Path = None
    run_id: Optional[str] = None

    @classmethod
    def from_env(cls) -> "Workspace":
        override = os.environ.get("RRR_PROJECT_ROOT")
        if override:
            workdir = Path(override).expanduser().resolve()
        else:
            workdir = Path(__file__).resolve().parents[2]
        claim_override = os.environ.get("RRR_CLAIM_CACHE_DIR")
        claim_cache_root = (
            Path(claim_override).expanduser() if claim_override
            else workdir / "claim_cache"
        )
        return cls(
            workdir=workdir,
            corpus_dir=workdir / "corpus",
            metadata_path=workdir / "metadata.csv",
            bibliography_path=workdir / "bibliography.bib",
            claim_cache_root=claim_cache_root,
            stage_cache_root=_stage_cache_root_for(workdir),
            run_id=None,
        )

    @classmethod
    def from_corpus_dir(
        cls,
        corpus_dir: Path,
        workdir: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
        bibliography_path: Optional[Path] = None,
        run_id: Optional[str] = None,
    ) -> "Workspace":
        corpus_dir = Path(corpus_dir).expanduser().resolve()
        if workdir is None:
            workdir = corpus_dir.parent
        else:
            workdir = Path(workdir).expanduser().resolve()
        metadata_path = (
            Path(metadata_path).expanduser().resolve() if metadata_path
            else workdir / "metadata.csv"
        )
        bibliography_path = (
            Path(bibliography_path).expanduser().resolve() if bibliography_path
            else workdir / "bibliography.bib"
        )
        claim_override = os.environ.get("RRR_CLAIM_CACHE_DIR")
        claim_cache_root = (
            Path(claim_override).expanduser() if claim_override
            else workdir / "claim_cache"
        )
        return cls(
            workdir=workdir,
            corpus_dir=corpus_dir,
            metadata_path=metadata_path,
            bibliography_path=bibliography_path,
            claim_cache_root=claim_cache_root,
            stage_cache_root=_stage_cache_root_for(workdir),
            run_id=run_id,
        )

    def with_run_id(self, run_id: Optional[str]) -> "Workspace":
        return replace(self, run_id=run_id)

    @property
    def data_dir(self) -> Path:
        return self.workdir / "data"

    @property
    def page_text_dir(self) -> Path:
        return self.workdir / "data" / "page_text"

    @property
    def indices_dir(self) -> Path:
        return self.workdir / "indices"

    @property
    def runs_root(self) -> Path:
        return self.workdir / "runs"

    @property
    def runs_dir(self) -> Path:
        if self.run_id:
            return self.runs_root / self.run_id
        return self.runs_root

    @property
    def logs_dir(self) -> Path:
        return self.workdir / "logs"

    def data_path(self, *parts) -> Path:
        return self.data_dir.joinpath(*parts)

    def page_text_file(self, doc_id: str, page: int) -> Path:
        return self.page_text_dir / f"{doc_id}_page_{int(page)}.txt"

    def indices_path(self, *parts) -> Path:
        return self.indices_dir.joinpath(*parts)

    def runs_path(self, *parts) -> Path:
        return self.runs_dir.joinpath(*parts)

    def logs_path(self, *parts) -> Path:
        return self.logs_dir.joinpath(*parts)

    def claim_cache_path(self, *parts) -> Path:
        return self.claim_cache_root.joinpath(*parts)

    def stage_cache_path(self, *parts) -> Path:
        return self.stage_cache_root.joinpath(*parts)

    def repo_path(self, *parts) -> Path:
        return self.workdir.joinpath(*parts)


_DEFAULT_WORKSPACE: Workspace = Workspace.from_env()


def default() -> Workspace:
    return _DEFAULT_WORKSPACE


def set_default_run_id(run_id: Optional[str]) -> None:
    global _DEFAULT_WORKSPACE
    _DEFAULT_WORKSPACE = _DEFAULT_WORKSPACE.with_run_id(run_id)


def repo_path(*parts) -> Path:
    return default().repo_path(*parts)


def data_path(*parts) -> Path:
    return default().data_path(*parts)


def require_dir(path: Path, label: str) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def require_page_text_dir() -> Path:
    return require_dir(default().page_text_dir, "RRR page-text directory")


def require_indices_dir() -> Path:
    return require_dir(default().indices_dir, "RRR indices directory")


def page_text_path(doc_id: str, page: int) -> Path:
    return default().page_text_file(doc_id, page)


def indices_path(*parts) -> Path:
    return default().indices_path(*parts)


def runs_path(*parts) -> Path:
    return default().runs_path(*parts)


def logs_path(*parts) -> Path:
    return default().logs_path(*parts)


def claim_cache_path(*parts) -> Path:
    return default().claim_cache_path(*parts)


def stage_cache_path(*parts) -> Path:
    return default().stage_cache_path(*parts)
