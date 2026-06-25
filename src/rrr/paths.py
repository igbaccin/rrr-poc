from pathlib import Path
import os


def project_root() -> Path:
    override = os.environ.get("RRR_PROJECT_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def repo_path(*parts) -> Path:
    return project_root().joinpath(*parts)


def data_path(*parts) -> Path:
    return repo_path("data", *parts)


def require_dir(path: Path, label: str) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def require_data_dir() -> Path:
    return require_dir(data_path(), "RRR data directory")


def require_page_text_dir() -> Path:
    return require_dir(data_path("page_text"), "RRR page-text directory")


def require_indices_dir() -> Path:
    return require_dir(indices_path(), "RRR indices directory")


def page_text_path(doc_id: str, page: int) -> Path:
    return data_path("page_text", f"{doc_id}_page_{int(page)}.txt")


def indices_path(*parts) -> Path:
    return repo_path("indices", *parts)


def runs_path(*parts) -> Path:
    return repo_path("runs", *parts)


def logs_path(*parts) -> Path:
    return repo_path("logs", *parts)
