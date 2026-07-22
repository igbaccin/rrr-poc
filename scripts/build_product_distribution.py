#!/usr/bin/env python3
"""Build the public RRR wheel and place a verified copy in the plugin."""
from __future__ import annotations

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
PORTABLE_SKILL_RUNTIME = ROOT / "skills" / "rrr" / "runtime"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
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
        PLUGIN_RUNTIME.mkdir(parents=True, exist_ok=True)
        PORTABLE_SKILL_RUNTIME.mkdir(parents=True, exist_ok=True)
        dist_wheel = DIST_DIR / built_wheel.name
        plugin_wheel = PLUGIN_RUNTIME / built_wheel.name
        portable_skill_wheel = PORTABLE_SKILL_RUNTIME / built_wheel.name
        shutil.copy2(built_wheel, dist_wheel)
        shutil.copy2(built_wheel, plugin_wheel)
        shutil.copy2(built_wheel, portable_skill_wheel)

    digest = sha256(dist_wheel)
    (DIST_DIR / f"{dist_wheel.name}.sha256").write_text(
        f"{digest}  {dist_wheel.name}\n",
        encoding="ascii",
    )
    runtime_manifest = (
        json.dumps(
            {
                "distribution": distribution,
                "version": version,
                "wheel": plugin_wheel.name,
                "sha256": digest,
            },
            indent=2,
        )
        + "\n"
    )
    for runtime_dir in (PLUGIN_RUNTIME, PORTABLE_SKILL_RUNTIME):
        (runtime_dir / "runtime.json").write_text(
            runtime_manifest,
            encoding="utf-8",
        )
    print(f"[RRR] wheel: {dist_wheel}")
    print(f"[RRR] sha256: {digest}")
    print(f"[RRR] plugin runtime: {plugin_wheel}")
    print(f"[RRR] portable skill runtime: {portable_skill_wheel}")


if __name__ == "__main__":
    main()
