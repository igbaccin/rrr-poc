"""Path resolution for RRR.

v15.9: introduces the Workspace dataclass — a single object that owns every
path the pipeline reads from or writes to. The module-level functions below
(project_root, runs_path, page_text_path, claim_cache_path, ...) are kept as
BACKWARD-COMPAT delegates that resolve against the module-level default
workspace (built from env at import time). Every existing caller keeps
working with zero source changes; new callers can construct a Workspace
explicitly (rrr.review(corpus_dir=..., topic=..., workdir=...)).

The one behavioural change: when a run_id is set on the default workspace
(via set_default_run_id), runs_path("foo.md") resolves to
`<workdir>/runs/<run_id>/foo.md` instead of `<workdir>/runs/foo.md`. This
makes concurrent runs on the same corpus not collide (improvement #4).
Setting no run_id preserves the flat layout — byte-identical to pre-v15.9.
"""
from __future__ import annotations

import datetime
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slugify(text: str, max_len: int = 60) -> str:
    """Lowercase-alnum slug with underscores. Mirrors the pattern
    scripts/run_small_validation.py:56-58 uses."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", (text or "").strip()).strip("_").lower()
    return slug[:max_len] or "topic"


def _utc_stamp() -> str:
    """UTC timestamp string, e.g. '20260702T115300Z'. Used as run_id prefix."""
    return datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def mint_run_id(topic: Optional[str] = None) -> str:
    """Compose `<utc_stamp>_<slug>` for a per-invocation run directory."""
    stamp = _utc_stamp()
    return f"{stamp}_{_slugify(topic)}" if topic else stamp


def _stage_cache_root_for(workdir: Path) -> Path:
    """v15.16: workspace-level root for outline and admission caches.
    Override with RRR_STAGE_CACHE_DIR (mirrors
    RRR_CLAIM_CACHE_DIR) to survive across sessions on an ephemeral pod."""
    override = os.environ.get("RRR_STAGE_CACHE_DIR")
    return Path(override).expanduser() if override else workdir / "cache"


def stage_cache_enabled() -> bool:
    """v15.16: RRR_STAGE_CACHE=0 disables stage-cache READS (writes still
    happen, for forensics). The battery sets 0 so repeat runs of the same
    topic measure whole-pipeline variance rather than replaying cached
    precheck/cluster/posture/order results; smokes keep the default 1 for
    fast iteration."""
    return os.environ.get("RRR_STAGE_CACHE", "1") != "0"


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Workspace:
    """Immutable resolver for every path the pipeline touches.

    Attributes:
        workdir: root under which data/, indices/, runs/, logs/ live.
                 Default: project_root() (repo root, or $RRR_PROJECT_ROOT).
        corpus_dir: folder of PDFs. Default: workdir/corpus/.
        metadata_path: metadata.csv location. Default: workdir/metadata.csv.
        bibliography_path: bibliography.bib. Default: workdir/bibliography.bib.
        claim_cache_root: cache root (usually outside workdir on the pod).
                          Default: $RRR_CLAIM_CACHE_DIR or workdir/claim_cache/.
        run_id: per-invocation subdir under runs/. None → flat layout
                (backward-compat with pre-v15.9 output).

    Everything else is a derived property or method.
    """
    workdir: Path
    corpus_dir: Path
    metadata_path: Path
    bibliography_path: Path
    claim_cache_root: Path
    # v15.16: outline and document-admission caches live at workspace level
    # like the claim cache.
    # They previously lived under runs_path("cache", ...), which the v15.9
    # per-run-id layout silently scoped PER RUN — replay was impossible and
    # every minted run recomputed all stages cold.
    stage_cache_root: Path = None  # type: ignore[assignment]
    run_id: Optional[str] = None

    # --- constructors ---

    @classmethod
    def from_env(cls) -> "Workspace":
        """Build the default workspace by reading env vars. Preserves pre-v15.9
        behaviour exactly: project_root() → workdir, all subdirs under it."""
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
        """Build a workspace anchored on an arbitrary corpus folder.
        The workdir defaults to `~/.rrr/corpora/<hash>` (per productisation
        plan) — but if the caller passes an explicit workdir we use that.
        For the initial v15.9, we default workdir to the corpus's parent
        so existing users keep D:\\RRR-style layouts working.
        """
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
        """Return a copy with run_id set. Preserves immutability."""
        return replace(self, run_id=run_id)

    # --- derived paths ---

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
        """Top-level runs/ dir (invariant across invocations)."""
        return self.workdir / "runs"

    @property
    def runs_dir(self) -> Path:
        """Per-invocation dir: runs_root/run_id if run_id is set, else the
        flat runs_root (byte-identical to pre-v15.9 layout)."""
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
        """Alias for workdir.joinpath — kept for backward compat with old
        callers that used `repo_path()` to mean 'the RRR project root'."""
        return self.workdir.joinpath(*parts)


# ---------------------------------------------------------------------------
# Module-level default + backward-compat delegates
# ---------------------------------------------------------------------------

_DEFAULT_WORKSPACE: Workspace = Workspace.from_env()


def default() -> Workspace:
    """Return the current module-level default workspace."""
    return _DEFAULT_WORKSPACE


def set_default_run_id(run_id: Optional[str]) -> None:
    """Convenience: set only the run_id on the module-level workspace."""
    global _DEFAULT_WORKSPACE
    _DEFAULT_WORKSPACE = _DEFAULT_WORKSPACE.with_run_id(run_id)


# The functions below are the pre-v15.9 API. They delegate to `default()` so
# every existing caller works unchanged. Do not add new callers — prefer
# `paths.default().<method>()` for new code.

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
    """Persistent, content-keyed claim cache root.

    Lives OUTSIDE runs/ so the smoke harness (which wipes runs/cache between
    topics) cannot evict it. Override with RRR_CLAIM_CACHE_DIR to point at a
    location that survives across sessions (e.g. /workspace/claim_cache on an
    ephemeral pod), making per-paper claim extraction a true one-time cost.
    """
    return default().claim_cache_path(*parts)


def stage_cache_path(*parts) -> Path:
    """v15.16: persistent stage-cache root for outline and document admission.
    It lives outside runs/, where the v15.9 per-run-id
    layout the old runs_path("cache", ...) location was silently scoped to
    each minted run, so caches could never be reused or replayed. Content
    keys (prompt version + model + inputs) make cross-run sharing safe;
    RRR_STAGE_CACHE=0 disables reads for cold-run measurement.
    """
    return default().stage_cache_path(*parts)
