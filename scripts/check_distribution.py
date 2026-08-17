#!/usr/bin/env python3
"""Validate ppmlx distribution files and write a release manifest."""

from __future__ import annotations

import argparse
import configparser
from email import policy
from email.parser import BytesParser
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
from typing import Any
import zipfile

from packaging.utils import canonicalize_name, parse_sdist_filename, parse_wheel_filename
from packaging.version import Version


PACKAGE_NAME = "ppmlx"
WHEEL_REQUIRED_FILES = {
    "ppmlx/__init__.py",
    "ppmlx/cli.py",
    "ppmlx/mcp_server.py",
    "ppmlx/registry_data.json",
    "ppmlx/server.py",
}
SDIST_REQUIRED_FILES = WHEEL_REQUIRED_FILES | {
    "LICENSE",
    "PKG-INFO",
    "README.md",
    "pyproject.toml",
}
REQUIRED_ENTRY_POINTS = {
    "ppmlx": "ppmlx.cli:main_entry",
    "ppmlx-memory-mcp": "ppmlx.mcp_server:main",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_metadata(raw: bytes, source: str) -> tuple[str, Version]:
    metadata = BytesParser(policy=policy.default).parsebytes(raw)
    name = metadata.get("Name")
    version_text = metadata.get("Version")
    if not name or not version_text:
        raise ValueError(f"{source} must contain Name and Version metadata.")
    if canonicalize_name(name) != PACKAGE_NAME:
        raise ValueError(f"{source} contains the incorrect package name: {name!r}.")
    return name, Version(version_text)


def validate_registry(raw: bytes, source: str) -> None:
    try:
        data: Any = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{source} does not contain valid JSON.") from exc
    if not isinstance(data, dict) or not isinstance(data.get("models"), dict):
        raise ValueError(f"{source} must contain a models object.")
    if not data["models"]:
        raise ValueError(f"{source} must contain at least one model.")


def validate_entry_points(raw: bytes, source: str) -> None:
    parser = configparser.ConfigParser(interpolation=None)
    try:
        parser.read_string(raw.decode("utf-8"))
    except (UnicodeDecodeError, configparser.Error) as exc:
        raise ValueError(f"{source} is not a valid entry point file.") from exc
    if not parser.has_section("console_scripts"):
        raise ValueError(f"{source} does not contain console_scripts.")
    actual = dict(parser.items("console_scripts"))
    missing = {
        name: value
        for name, value in REQUIRED_ENTRY_POINTS.items()
        if actual.get(name) != value
    }
    if missing:
        raise ValueError(f"{source} has incorrect entry points: {missing}.")


def validate_wheel(path: Path) -> Version:
    filename_name, filename_version, _, _ = parse_wheel_filename(path.name)
    if canonicalize_name(filename_name) != PACKAGE_NAME:
        raise ValueError(f"The wheel has the incorrect package name: {filename_name!r}.")

    with zipfile.ZipFile(path) as wheel:
        names = set(wheel.namelist())
        unsafe = [
            name
            for name in names
            if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
        ]
        if unsafe:
            raise ValueError(f"The wheel has unsafe paths: {sorted(unsafe)}.")
        missing = sorted(WHEEL_REQUIRED_FILES - names)
        if missing:
            raise ValueError(f"The wheel is missing required files: {missing}.")
        if any(name.startswith("tests/") for name in names):
            raise ValueError("The wheel must not contain the test suite.")

        metadata_files = sorted(name for name in names if name.endswith(".dist-info/METADATA"))
        entry_point_files = sorted(
            name for name in names if name.endswith(".dist-info/entry_points.txt")
        )
        license_files = sorted(
            name for name in names if name.endswith(".dist-info/licenses/LICENSE")
        )
        if len(metadata_files) != 1:
            raise ValueError("The wheel must contain one METADATA file.")
        if len(entry_point_files) != 1:
            raise ValueError("The wheel must contain one entry_points.txt file.")
        if len(license_files) != 1:
            raise ValueError("The wheel must contain one LICENSE file.")

        _, metadata_version = read_metadata(
            wheel.read(metadata_files[0]), f"{path.name}:{metadata_files[0]}"
        )
        validate_entry_points(
            wheel.read(entry_point_files[0]), f"{path.name}:{entry_point_files[0]}"
        )
        validate_registry(
            wheel.read("ppmlx/registry_data.json"), f"{path.name}:ppmlx/registry_data.json"
        )

    if metadata_version != filename_version:
        raise ValueError(
            f"The wheel version {filename_version} does not match METADATA version {metadata_version}."
        )
    return metadata_version


def validate_sdist(path: Path) -> Version:
    filename_name, filename_version = parse_sdist_filename(path.name)
    if canonicalize_name(filename_name) != PACKAGE_NAME:
        raise ValueError(f"The source archive has the incorrect package name: {filename_name!r}.")

    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        names = {member.name for member in members}
        unsafe = [
            name
            for name in names
            if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
        ]
        if unsafe:
            raise ValueError(f"The source archive has unsafe paths: {sorted(unsafe)}.")
        links = sorted(member.name for member in members if member.issym() or member.islnk())
        if links:
            raise ValueError(f"The source archive must not contain links: {links}.")

        roots = {PurePosixPath(name).parts[0] for name in names if PurePosixPath(name).parts}
        if len(roots) != 1:
            raise ValueError(f"The source archive must contain one root directory: {sorted(roots)}.")
        root = next(iter(roots))
        normalized_names = {
            str(PurePosixPath(*PurePosixPath(name).parts[1:]))
            for name in names
            if len(PurePosixPath(name).parts) > 1
        }
        missing = sorted(SDIST_REQUIRED_FILES - normalized_names)
        if missing:
            raise ValueError(f"The source archive is missing required files: {missing}.")

        metadata_member = archive.getmember(f"{root}/PKG-INFO")
        registry_member = archive.getmember(f"{root}/ppmlx/registry_data.json")
        if not metadata_member.isfile() or not registry_member.isfile():
            raise ValueError("The source archive has invalid required files.")
        metadata_source = archive.extractfile(metadata_member)
        registry_source = archive.extractfile(registry_member)
        if metadata_source is None or registry_source is None:
            raise ValueError("The source archive has unreadable required files.")
        _, metadata_version = read_metadata(metadata_source.read(), f"{path.name}:PKG-INFO")
        validate_registry(registry_source.read(), f"{path.name}:ppmlx/registry_data.json")

    expected_root = f"{PACKAGE_NAME}-{metadata_version}"
    if root != expected_root:
        raise ValueError(f"The source archive root {root!r} must be {expected_root!r}.")
    if metadata_version != filename_version:
        raise ValueError(
            "The source archive filename version does not match its PKG-INFO version."
        )
    return metadata_version


def append_github_output(path: Path, values: dict[str, str]) -> None:
    with path.open("a", encoding="utf-8") as output:
        for name, value in values.items():
            if "\n" in value or "\r" in value:
                raise ValueError(f"The output {name!r} contains a newline.")
            output.write(f"{name}={value}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dist_dir", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()

    wheels = sorted(args.dist_dir.glob("*.whl"))
    source_archives = sorted(args.dist_dir.glob("*.tar.gz"))
    if len(wheels) != 1 or len(source_archives) != 1:
        raise SystemExit("The distribution directory must contain one wheel and one source archive.")

    wheel = wheels[0]
    source_archive = source_archives[0]
    wheel_version = validate_wheel(wheel)
    source_version = validate_sdist(source_archive)
    if wheel_version != source_version:
        raise SystemExit(
            f"The wheel version {wheel_version} does not match the source version {source_version}."
        )

    artifacts = []
    for artifact_type, path in (("wheel", wheel), ("sdist", source_archive)):
        artifacts.append(
            {
                "filename": path.name,
                "sha256": sha256(path),
                "size": path.stat().st_size,
                "type": artifact_type,
            }
        )
    manifest = {
        "artifacts": artifacts,
        "name": PACKAGE_NAME,
        "version": str(wheel_version),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    outputs = {
        "manifest": str(args.manifest),
        "sdist": str(source_archive),
        "version": str(wheel_version),
        "wheel": str(wheel),
    }
    if args.github_output:
        append_github_output(args.github_output, outputs)
    print(
        f"Validated {PACKAGE_NAME} {wheel_version}: {wheel.name} and {source_archive.name}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
