#!/usr/bin/env python3
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
SOURCE_PACKAGE = ROOT / "src" / "rrr"
HOSTED_OVERLAY = ROOT / "profiles" / "hosted" / "rrr"
PLUGIN_RUNTIME = ROOT / "plugins" / "rrr" / "runtime"
HOSTED_TARGETS = {
    "cx": {
        "runtime_dir": PLUGIN_RUNTIME,
        "context": "Codex hosted agent",
    },
    "cl": {
        "runtime_dir": ROOT / "skills" / "rrr" / "runtime",
        "context": "Claude hosted agent",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_field(name: str, pyproject: str | None = None) -> str:
    text = pyproject if pyproject is not None else (
        ROOT / "pyproject.toml"
    ).read_text(encoding="utf-8")
    match = re.search(rf'(?m)^{re.escape(name)}\s*=\s*"([^"]+)"', text)
    if not match:
        raise RuntimeError(f"Missing project field: {name}")
    return match.group(1)


def hosted_overlay_modules() -> list[str]:
    return sorted(path.name for path in HOSTED_OVERLAY.glob("*.py"))


def assemble_hosted_source(dest: Path) -> Path:
    package = Path(dest) / "src" / "rrr"
    if package.exists():
        shutil.rmtree(package)
    package.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        SOURCE_PACKAGE,
        package,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for name in hosted_overlay_modules():
        if not (SOURCE_PACKAGE / name).is_file():
            raise RuntimeError(
                f"profiles/hosted/rrr/{name} replaces no module in src/rrr"
            )
        shutil.copyfile(HOSTED_OVERLAY / name, package / name)
    return package


def _pip_wheel(project: Path, wheel_dir: Path) -> Path:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            str(project),
            "--no-deps",
            "--wheel-dir",
            str(wheel_dir),
        ],
        check=True,
    )
    wheels = list(wheel_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"Expected one wheel, found {len(wheels)}")
    return wheels[0]


def build_local(sync_to_plugin: bool = False) -> Path:
    distribution = project_field("name")
    version = project_field("version")
    with tempfile.TemporaryDirectory(prefix="rrr-wheel-") as temp_dir:
        built_wheel = _pip_wheel(ROOT, Path(temp_dir))
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
    if sync_to_plugin:
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
    return dist_wheel


def hosted_pyproject(version: str) -> str:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    replaced, count = re.subn(
        r'(?m)^version\s*=\s*"[^"]+"',
        f'version = "{version}"',
        pyproject,
        count=1,
    )
    if count != 1:
        raise RuntimeError("pyproject.toml has no version field to replace")
    return replaced


def build_hosted(
    profiles: list[str],
    version: str,
    profile_version: str,
    out_dir: Path | None = None,
) -> Path:
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise RuntimeError(f"Hosted version must look like 15.19.4, got {version!r}")
    if not re.fullmatch(r"\d+\.\d+", profile_version):
        raise RuntimeError(
            f"Profile version must look like 2.4, got {profile_version!r}"
        )
    distribution = project_field("name")
    with tempfile.TemporaryDirectory(prefix="rrr-hosted-") as temp_dir:
        project = Path(temp_dir) / "project"
        assemble_hosted_source(project)
        (project / "pyproject.toml").write_text(
            hosted_pyproject(version), encoding="utf-8"
        )
        for name in ("README.md", "LICENSE.txt"):
            shutil.copyfile(ROOT / name, project / name)
        wheel_dir = Path(temp_dir) / "wheel"
        wheel_dir.mkdir()
        built_wheel = _pip_wheel(project, wheel_dir)
        digest = sha256(built_wheel)

        if out_dir is not None:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            rehearsal = out_dir / built_wheel.name
            shutil.copy2(built_wheel, rehearsal)
            print(f"[RRR] hosted wheel (rehearsal): {rehearsal}")
            print(f"[RRR] sha256: {digest}")
            return rehearsal

        installed = None
        for profile in profiles:
            target = HOSTED_TARGETS[profile]
            runtime_dir = target["runtime_dir"]
            runtime_dir.mkdir(parents=True, exist_ok=True)
            for stale_wheel in runtime_dir.glob(
                f"{distribution.replace('-', '_')}-*.whl"
            ):
                stale_wheel.unlink()
            installed = runtime_dir / built_wheel.name
            shutil.copy2(built_wheel, installed)
            (runtime_dir / "runtime.json").write_text(
                json.dumps(
                    {
                        "distribution": distribution,
                        "profile": profile,
                        "profile_version": f"v{profile}.{profile_version}",
                        "context": target["context"],
                        "version": version,
                        "wheel": installed.name,
                        "sha256": digest,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print(f"[RRR] {profile} runtime: {installed}")
        print(f"[RRR] sha256: {digest}")
        return installed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the RRR wheels (local by default)."
    )
    parser.add_argument(
        "--sync-local-wheel-to-plugin",
        action="store_true",
        help=(
            "Replace the hosted plugin runtime with the newly built local "
            "wheel. This is intentionally opt-in."
        ),
    )
    parser.add_argument(
        "--hosted",
        choices=["cx", "cl", "both"],
        default=None,
        help="Build the hosted wheel for Codex (cx), Claude (cl), or both.",
    )
    parser.add_argument(
        "--hosted-version",
        default=None,
        help="Package version of the hosted wheel, for example 15.19.4.",
    )
    parser.add_argument(
        "--profile-version",
        default=None,
        help="Profile version without its prefix, for example 2.4.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Write the hosted wheel here and leave the bundled runtimes alone.",
    )
    parser.add_argument(
        "--assemble-hosted",
        default=None,
        metavar="DIR",
        help="Write the assembled hosted source tree to DIR and stop.",
    )
    args = parser.parse_args()

    if args.assemble_hosted:
        package = assemble_hosted_source(Path(args.assemble_hosted))
        print(f"[RRR] hosted source assembled: {package}")
        return
    if args.hosted:
        if not args.hosted_version or not args.profile_version:
            parser.error("--hosted needs --hosted-version and --profile-version")
        profiles = ["cx", "cl"] if args.hosted == "both" else [args.hosted]
        build_hosted(
            profiles,
            args.hosted_version,
            args.profile_version,
            out_dir=Path(args.out_dir) if args.out_dir else None,
        )
        return
    build_local(sync_to_plugin=args.sync_local_wheel_to_plugin)


if __name__ == "__main__":
    main()
