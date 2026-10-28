#!/usr/bin/env python3
"""Build the public local-reference RRR wheel.

The hosted-agent plugin is a separate tested release. A local build leaves its
runtime untouched unless the caller explicitly requests synchronization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIST_DIR = ROOT / "dist"
PLUGIN_RUNTIME = ROOT / "plugins" / "rrr" / "runtime"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sync-local-wheel-to-plugin",
        action="store_true",
        help=(
            "Replace the hosted plugin runtime with the newly built local "
            "wheel. This is intentionally opt-in."
        ),
    )
    args = parser.parse_args()
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    def project_field(name: str) -> str:
        match = re.search(rf'(?m)^{re.escape(name)}\s*=\s*"([^"]+)"', pyproject)
        if not match:
            raise RuntimeError(f"Missing project field: {name}")
        return match.group(1)

    distribution = project_field("name")
    version = project_field("version")
    with tempfile.TemporaryDirectory(prefix="rrr-wheel-") as temp_dir:
        temp_path = Path(temp_dir)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                str(ROOT),
                "--no-deps",
                "--wheel-dir",
                str(temp_path),
            ],
            check=True,
        )
        wheels = list(temp_path.glob("*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(f"Expected one wheel, found {len(wheels)}")
        built_wheel = wheels[0]

        DIST_DIR.mkdir(parents=True, exist_ok=True)
        for stale_wheel in DIST_DIR.glob(f"{distribution.replace('-', '_')}-*.whl"):
            stale_wheel.unlink()
            stale_checksum = DIST_DIR / f"{stale_wheel.name}.sha256"
            if stale_checksum.exists():
                stale_checksum.unlink()
        dist_wheel = DIST_DIR / built_wheel.name
        shutil.copy2(built_wheel, dist_wheel)

    digest = sha256(dist_wheel)
    (DIST_DIR / f"{dist_wheel.name}.sha256").write_text(
        f"{digest}  {dist_wheel.name}\n",
        encoding="ascii",
    )
    if args.sync_local_wheel_to_plugin:
        PLUGIN_RUNTIME.mkdir(parents=True, exist_ok=True)
        for stale_wheel in PLUGIN_RUNTIME.glob(
            f"{distribution.replace('-', '_')}-*.whl"
        ):
            stale_wheel.unlink()
        plugin_wheel = PLUGIN_RUNTIME / dist_wheel.name
        shutil.copy2(dist_wheel, plugin_wheel)
        (PLUGIN_RUNTIME / "runtime.json").write_text(
            json.dumps(
                {
                    "distribution": distribution,
                    "version": version,
                    "wheel": plugin_wheel.name,
                    "sha256": digest,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[RRR] plugin runtime synchronized: {plugin_wheel}")
    else:
        print("[RRR] hosted plugin runtime left unchanged")
    print(f"[RRR] wheel: {dist_wheel}")
    print(f"[RRR] sha256: {digest}")


if __name__ == "__main__":
    main()
