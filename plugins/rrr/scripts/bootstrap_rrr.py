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

PIP_TIMEOUT_SECONDS = 30
PIP_RETRIES = 3


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


def _quiet_child() -> dict:
    if os.name == "nt":
        return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
    return {}


def runtime_command(python_exe: Path) -> list[str]:
    return [str(python_exe), "-m", "rrr"]


def runtime_starts(python_exe: Path) -> bool:
    try:
        completed = subprocess.run(
            [str(python_exe), "-c", "import rrr.cli"],
            capture_output=True,
            timeout=180,
            check=False,
            **_quiet_child(),
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def _say(message: str) -> None:
    sys.stderr.write(f"[RRR] {message}\n")
    sys.stderr.flush()


def resolve_wheelhouse(explicit: Path | None = None) -> Path | None:
    raw = str(explicit) if explicit else os.environ.get("RRR_WHEELHOUSE", "").strip()
    if not raw:
        return None
    folder = Path(raw).expanduser().resolve()
    if not folder.is_dir():
        raise RuntimeError(f"The wheelhouse folder does not exist: {folder}")
    return folder


def _pip_install(python_exe: Path, arguments: list[str], wheelhouse: Path | None) -> None:
    command = [str(python_exe), "-m", "pip", "install", "--disable-pip-version-check"]
    if "PIP_DEFAULT_TIMEOUT" not in os.environ:
        command += ["--timeout", str(PIP_TIMEOUT_SECONDS)]
    if "PIP_RETRIES" not in os.environ:
        command += ["--retries", str(PIP_RETRIES)]
    if wheelhouse is not None:
        command += ["--no-index", "--find-links", str(wheelhouse)]
    command += arguments
    try:
        sys.stderr.fileno()
        streams = {"stdout": sys.stderr, "stderr": sys.stderr}
    except (AttributeError, OSError, ValueError):
        streams = {}
    subprocess.run(command, check=True, **streams, **_quiet_child())


def _dependency_failure(exit_code: int, wheelhouse: Path | None, wheel: Path) -> str:
    if wheelhouse is not None:
        return (
            f"pip could not install RRR from the wheelhouse {wheelhouse} "
            f"(exit code {exit_code}). A package RRR needs is missing from "
            "that folder, or was downloaded for another Python version or "
            "another system. Fill the folder on a computer with the same "
            "Python version and system, with: python -m pip download --dest "
            f"<folder> \"{wheel}\""
        )
    return (
        f"pip could not download RRR's dependencies (exit code {exit_code}). "
        "The usual cause is that this computer cannot reach the Python "
        "package index (pypi.org) at the moment: a managed network may block "
        "it or let it through only at times. pip's own messages are above. "
        "What helps: (1) run the same step again in a few minutes; (2) if "
        "your organisation runs a package mirror, set PIP_INDEX_URL to its "
        "address; (3) behind a proxy, set HTTPS_PROXY; (4) install with no "
        "network at all from a folder of wheels, with --wheelhouse <folder> "
        "or RRR_WHEELHOUSE=<folder>."
    )


def bootstrap(
    plugin_root: Path,
    install_base: Path | None = None,
    install_api: bool = False,
    wheelhouse: Path | None = None,
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
    if marker.is_file() and python_exe.is_file():
        try:
            installed = json.loads(marker.read_text(encoding="utf-8"))
            installed_extras = set(installed.get("extras", []))
            reused = (
                installed.get("bootstrap_schema") == BOOTSTRAP_SCHEMA
                and installed.get("sha256") == bundle["sha256"]
                and requested_extras.issubset(installed_extras)
                and runtime_starts(python_exe)
            )
        except (json.JSONDecodeError, OSError):
            reused = False

    if not reused:
        wheelhouse = resolve_wheelhouse(wheelhouse)
        install_root.mkdir(parents=True, exist_ok=True)
        if not python_exe.is_file():
            _say(f"creating a private Python environment in {venv_dir}")
            venv.EnvBuilder(with_pip=True).create(venv_dir)
        wheel_spec = str(wheel) + ("[api]" if install_api else "")
        if wheelhouse is not None:
            _say(f"installing RRR's dependencies from the wheelhouse {wheelhouse} (no network)")
        else:
            _say(
                "downloading and installing RRR's dependencies; the first "
                "installation fetches a few hundred megabytes and can take "
                "several minutes on a slow or filtered connection"
            )
        try:
            _pip_install(python_exe, ["--upgrade", wheel_spec], wheelhouse)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                _dependency_failure(exc.returncode, wheelhouse, wheel)
            ) from exc
        _say(f"installing the bundled RRR {version} wheel")
        _pip_install(python_exe, ["--force-reinstall", "--no-deps", str(wheel)], wheelhouse)
        _say("checking that the runtime starts")
        if not runtime_starts(python_exe):
            raise RuntimeError(
                "RRR was installed, but its interpreter cannot import it: "
                f"{python_exe}"
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
        "command": runtime_command(python_exe),
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
    ap.add_argument(
        "--wheelhouse",
        default=None,
        metavar="DIR",
        help=(
            "Install from this folder of wheels and use no network "
            "(pip --no-index --find-links DIR). RRR_WHEELHOUSE does the same."
        ),
    )
    ap.add_argument("--json", action="store_true", dest="as_json")
    args = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    try:
        result = bootstrap(
            Path(args.plugin_root),
            Path(args.install_base) if args.install_base else None,
            install_api=args.api,
            wheelhouse=Path(args.wheelhouse) if args.wheelhouse else None,
        )
    except (RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        sys.stderr.write(f"[RRR] installation failed: {exc}\n")
        raise SystemExit(2)
    _say(
        f"RRR {result['version']} "
        + ("was already installed" if result["reused"] else "is installed")
        + ". Run it with: " + " ".join(result["command"])
    )
    if args.as_json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        action = "reused" if result["reused"] else "installed"
        print(f"[RRR] {action} RRR {result['version']}")
        print(f"[RRR] command: {' '.join(result['command'])}")


if __name__ == "__main__":
    main()
