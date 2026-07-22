#!/usr/bin/env python3
"""Install the RRR wheel bundled with the plugin into a private runtime."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import venv
from pathlib import Path


BOOTSTRAP_SCHEMA = 2


def _data_dir() -> Path:
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_bundle(plugin_root: Path) -> tuple[Path, dict]:
    runtime_dir = plugin_root / "runtime"
    manifest_path = runtime_dir / "runtime.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        wheel = runtime_dir / manifest["wheel"]
        expected_hash = manifest["sha256"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as exc:
        raise RuntimeError("The RRR plugin runtime manifest is missing or invalid.") from exc
    if not wheel.is_file():
        raise RuntimeError(f"The bundled RRR wheel is missing: {wheel}")
    actual_hash = _sha256(wheel)
    if actual_hash != expected_hash:
        raise RuntimeError("The bundled RRR wheel failed its integrity check.")
    return wheel, manifest


def _runtime_paths(install_root: Path) -> tuple[Path, Path, Path]:
    venv_dir = install_root / "venv"
    if os.name == "nt":
        return venv_dir, venv_dir / "Scripts" / "python.exe", venv_dir / "Scripts" / "rrr.exe"
    return venv_dir, venv_dir / "bin" / "python", venv_dir / "bin" / "rrr"


def bootstrap(
    plugin_root: Path,
    install_base: Path | None = None,
    install_api: bool = False,
) -> dict:
    wheel, bundle = _load_bundle(plugin_root)
    version = str(bundle["version"])
    install_base = install_base or (_data_dir() / "runtime")
    install_root = Path(install_base).expanduser().resolve() / version
    marker = install_root / "installed.json"
    venv_dir, python_exe, rrr_exe = _runtime_paths(install_root)

    requested_extras = {"api"} if install_api else set()
    reused = False
    installed_extras: set[str] = set()
    if marker.is_file() and python_exe.is_file() and rrr_exe.is_file():
        try:
            installed = json.loads(marker.read_text(encoding="utf-8"))
            installed_extras = set(installed.get("extras", []))
            reused = (
                installed.get("bootstrap_schema") == BOOTSTRAP_SCHEMA
                and installed.get("sha256") == bundle["sha256"]
                and requested_extras.issubset(installed_extras)
            )
        except (json.JSONDecodeError, OSError):
            reused = False

    if not reused:
        install_root.mkdir(parents=True, exist_ok=True)
        if not python_exe.is_file():
            venv.EnvBuilder(with_pip=True).create(venv_dir)
        wheel_spec = str(wheel) + ("[api]" if install_api else "")
        subprocess.run(
            [
                str(python_exe),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--upgrade",
                wheel_spec,
            ],
            check=True,
        )
        subprocess.run(
            [
                str(python_exe),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--force-reinstall",
                "--no-deps",
                str(wheel),
            ],
            check=True,
        )
        marker.write_text(
            json.dumps(
                {
                    "bootstrap_schema": BOOTSTRAP_SCHEMA,
                    "version": version,
                    "wheel": wheel.name,
                    "sha256": bundle["sha256"],
                    "extras": sorted(installed_extras | requested_extras),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    return {
        "version": version,
        "python": str(python_exe),
        "rrr": str(rrr_exe),
        "install_root": str(install_root),
        "extras": sorted(installed_extras | requested_extras),
        "reused": reused,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plugin-root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--install-base", default=None)
    ap.add_argument("--api", action="store_true", help="Install provider API dependencies")
    ap.add_argument("--json", action="store_true", dest="as_json")
    args = ap.parse_args()
    try:
        result = bootstrap(
            Path(args.plugin_root),
            Path(args.install_base) if args.install_base else None,
            install_api=args.api,
        )
    except (RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        sys.stderr.write(f"[RRR] installation failed: {exc}\n")
        raise SystemExit(2)
    if args.as_json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        action = "reused" if result["reused"] else "installed"
        print(f"[RRR] {action} RRR {result['version']}")
        print(f"[RRR] command: {result['rrr']}")


if __name__ == "__main__":
    main()
