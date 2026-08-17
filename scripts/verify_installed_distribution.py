#!/usr/bin/env python3
"""Verify an installed ppmlx wheel outside the source tree."""

from __future__ import annotations

import argparse
import importlib
from importlib import metadata, resources
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


PACKAGE_NAME = "ppmlx"
REQUIRED_ENTRY_POINTS = {
    "ppmlx": "ppmlx.cli:main_entry",
    "ppmlx-memory-mcp": "ppmlx.mcp_server:main",
}


def verify_server_health(expected_version: str) -> None:
    """Verify the health route from the installed ASGI application."""
    from fastapi.testclient import TestClient
    from ppmlx.server import app

    client = TestClient(app)
    try:
        response = client.get("/health")
    finally:
        client.close()
    if response.status_code != 200:
        raise SystemExit(
            f"The installed server health route returned HTTP {response.status_code}."
        )
    payload: Any = response.json()
    if not isinstance(payload, dict) or payload.get("status") != "ok":
        raise SystemExit("The installed server health response is invalid.")
    if payload.get("version") != expected_version:
        raise SystemExit(
            "The installed server health version "
            f"{payload.get('version')!r} does not match {expected_version!r}."
        )


def executable_path(name: str) -> Path:
    suffix = ".exe" if sys.platform == "win32" else ""
    return Path(sys.executable).parent / f"{name}{suffix}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-version", required=True)
    args = parser.parse_args()

    installed_version = metadata.version(PACKAGE_NAME)
    if installed_version != args.expected_version:
        raise SystemExit(
            f"Installed version {installed_version} does not match {args.expected_version}."
        )

    distribution = metadata.distribution(PACKAGE_NAME)
    actual_entry_points = {
        entry_point.name: entry_point.value
        for entry_point in distribution.entry_points
        if entry_point.group == "console_scripts"
    }
    missing_entry_points = {
        name: value
        for name, value in REQUIRED_ENTRY_POINTS.items()
        if actual_entry_points.get(name) != value
    }
    if missing_entry_points:
        raise SystemExit(f"Installed entry points are incorrect: {missing_entry_points}.")
    for name, value in REQUIRED_ENTRY_POINTS.items():
        path = executable_path(name)
        if not path.is_file():
            raise SystemExit(f"The installed entry point does not exist: {path}.")
        if os.name != "nt" and not os.access(path, os.X_OK):
            raise SystemExit(f"The installed entry point is not executable: {path}.")
        importlib.import_module(value.partition(":")[0])

    registry_path = resources.files(PACKAGE_NAME).joinpath("registry_data.json")
    try:
        data: Any = json.loads(registry_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit("The installed registry_data.json file is invalid.") from exc
    if not isinstance(data, dict) or not isinstance(data.get("models"), dict):
        raise SystemExit("The installed registry_data.json file must contain a models object.")
    if not data["models"]:
        raise SystemExit("The installed registry_data.json file must contain at least one model.")

    result = subprocess.run(
        [str(executable_path("ppmlx")), "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    expected_output = f"ppmlx {args.expected_version}"
    if result.stdout.strip() != expected_output:
        raise SystemExit(
            f"The ppmlx version output is {result.stdout.strip()!r}, not {expected_output!r}."
        )
    subprocess.run(
        [str(executable_path("ppmlx")), "--help"],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    verify_server_health(args.expected_version)

    print(
        f"Verified installed {PACKAGE_NAME} {installed_version}, its entry points, "
        "registry data, and server health route."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
