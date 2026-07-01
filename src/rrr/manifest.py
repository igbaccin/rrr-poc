import datetime as _dt
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

from rrr.paths import data_path, indices_path, repo_path, runs_path
from rrr.utils import ensure_dir

# v15.14: mirror of metrics._redact_env_value — the manifest's env snapshot
# ships in the same downloaded artifacts.
_SENSITIVE_ENV_RE = re.compile(r"KEY|TOKEN|SECRET|PASS|CREDENTIAL", re.IGNORECASE)


def _redact_env_value(key: str, value):
    if value and _SENSITIVE_ENV_RE.search(key or ""):
        return "***redacted***"
    return value


def _sha256_file(path: Path):
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_tree(root: Path, suffixes=(".txt", ".json", ".csv")):
    if not root.is_dir():
        return None
    h = hashlib.sha256()
    count = 0
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in suffixes):
        rel = path.relative_to(root).as_posix()
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(_sha256_file(path).encode("ascii"))
        h.update(b"\0")
        count += 1
    return {"sha256": h.hexdigest(), "files": count}


def _run_git(args):
    try:
        res = subprocess.run(
            ["git", *args],
            cwd=str(repo_path()),
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return None


def _git_info():
    status = _run_git(["status", "--short"]) or ""
    diff = _run_git(["diff", "--no-ext-diff", "--binary"]) or ""
    untracked = "\n".join(line for line in status.splitlines() if line.startswith("?? "))
    return {
        "commit": _run_git(["rev-parse", "HEAD"]),
        "branch": _run_git(["branch", "--show-current"]),
        "dirty": bool(status.strip()),
        "status_short": status.splitlines(),
        "tracked_diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
        "untracked_list_sha256": hashlib.sha256(untracked.encode("utf-8")).hexdigest(),
    }


def _hardware_info():
    info = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
    }
    try:
        res = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
        if res.returncode == 0:
            info["gpus"] = [line.strip() for line in res.stdout.splitlines() if line.strip()]
    except Exception:
        pass
    return info


def _env_snapshot():
    keep = {}
    for key in sorted(os.environ):
        if key.startswith("RRR_") or key in {
            "OLLAMA_HOST",
            "OLLAMA_NUM_PARALLEL",
            "OLLAMA_MAX_LOADED_MODELS",
            "PYTHONPATH",
        }:
            keep[key] = _redact_env_value(key, os.environ.get(key))
    return keep


def build_run_manifest(task: str, topic: str, meta_path, model: str, plan=None, extra=None):
    meta_path = Path(meta_path)
    index_files = {
        "bm25": indices_path("bm25.pkl"),
        "page_ids": indices_path("page_ids.npy"),
        "docs": indices_path("docs.csv"),
    }
    manifest = {
        "created_at": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "task": task,
        "topic": topic,
        "model": model,
        "repo_root": str(repo_path()),
        "git": _git_info(),
        "hardware": _hardware_info(),
        "env": _env_snapshot(),
        "artifacts": {
            "metadata": {
                "path": str(meta_path),
                "sha256": _sha256_file(meta_path),
            },
            "indices": {
                name: {"path": str(path), "sha256": _sha256_file(path)}
                for name, path in index_files.items()
            },
            "page_text": _hash_tree(data_path("page_text"), suffixes=(".txt",)),
        },
        "plan": plan or {},
        "extra": extra or {},
    }
    return manifest


def write_run_manifest(task: str, topic: str, meta_path, model: str, plan=None, extra=None):
    ensure_dir(str(runs_path()))
    manifest = build_run_manifest(task, topic, meta_path, model, plan=plan, extra=extra)
    # v15.14: atomic write — the manifest is the run's provenance record; a
    # crash mid-write must not leave a truncated one that parses as absent.
    path = str(runs_path("run_manifest.json"))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)
    return manifest
