#!/usr/bin/env python3
"""Wait for a package index and verify its files against a local release manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen


DEFAULT_INDEX_BASE_URL = "https://test.pypi.org/pypi"
DEFAULT_APPROVED_FILE_HOST = "test-files.pythonhosted.org"
# Keep the old name for callers that imported the TestPyPI helper.
TESTPYPI_FILE_HOST = DEFAULT_APPROVED_FILE_HOST
USER_AGENT = "ppmlx-release-verifier/1"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class NotReadyError(RuntimeError):
    """The package index does not yet have the complete release."""


def append_github_output(path: Path, values: dict[str, str]) -> None:
    with path.open("a", encoding="utf-8") as output:
        for name, value in values.items():
            if "\n" in value or "\r" in value:
                raise ValueError(f"The output {name!r} contains a newline.")
            output.write(f"{name}={value}\n")


def load_manifest(path: Path) -> tuple[str, str, dict[str, dict[str, Any]]]:
    try:
        data: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"The release manifest is invalid: {path}.") from exc

    if not isinstance(data, dict) or data.get("name") != "ppmlx":
        raise ValueError("The release manifest has an incorrect package name.")
    version = data.get("version")
    artifacts = data.get("artifacts")
    if not isinstance(version, str) or not version:
        raise ValueError("The release manifest has no version.")
    if not isinstance(artifacts, list) or len(artifacts) != 2:
        raise ValueError("The release manifest must contain two artifacts.")

    records: dict[str, dict[str, Any]] = {}
    artifact_types: set[str] = set()
    for record in artifacts:
        if not isinstance(record, dict):
            raise ValueError("The release manifest contains an invalid artifact record.")
        filename = record.get("filename")
        digest = record.get("sha256")
        size = record.get("size")
        artifact_type = record.get("type")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError("The release manifest contains an invalid filename.")
        if not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None:
            raise ValueError(f"The release manifest has an invalid digest for {filename!r}.")
        if not isinstance(size, int) or size <= 0:
            raise ValueError(f"The release manifest has an invalid size for {filename!r}.")
        if artifact_type not in {"wheel", "sdist"}:
            raise ValueError(f"The release manifest has an invalid type for {filename!r}.")
        if filename in records:
            raise ValueError(f"The release manifest repeats {filename!r}.")
        records[filename] = record
        artifact_types.add(artifact_type)
    if artifact_types != {"wheel", "sdist"}:
        raise ValueError("The release manifest must contain one wheel and one source archive.")
    return data["name"], version, records


def get_release_data(url: str, timeout_seconds: float) -> dict[str, Any]:
    request = Request(url, headers={"Accept": "application/json", "User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
            raw = response.read()
    except HTTPError as exc:
        if exc.code == 404 or 500 <= exc.code < 600:
            raise NotReadyError(f"The package index returned HTTP {exc.code}.") from exc
        raise
    try:
        data: Any = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NotReadyError("The package index returned invalid JSON.") from exc
    if not isinstance(data, dict):
        raise NotReadyError("The package index returned an invalid release object.")
    return data


def match_remote_files(
    data: dict[str, Any], expected: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    urls = data.get("urls")
    if not isinstance(urls, list):
        raise NotReadyError("The package index does not list release files.")
    remote: dict[str, dict[str, Any]] = {}
    for record in urls:
        if isinstance(record, dict) and isinstance(record.get("filename"), str):
            remote[record["filename"]] = record

    if set(remote) != set(expected):
        missing = sorted(set(expected) - set(remote))
        extra = sorted(set(remote) - set(expected))
        if extra:
            raise ValueError(f"The package index contains unexpected release files: {extra}.")
        raise NotReadyError(f"The package index file set is incomplete. Missing: {missing}.")

    for filename, expected_record in expected.items():
        remote_record = remote[filename]
        digests = remote_record.get("digests")
        remote_digest = digests.get("sha256") if isinstance(digests, dict) else None
        if remote_digest != expected_record["sha256"]:
            raise ValueError(f"The package index has an incorrect SHA-256 digest for {filename}.")
        remote_size = remote_record.get("size")
        if remote_size != expected_record["size"]:
            raise ValueError(f"The package index has an incorrect file size for {filename}.")
    return remote


def download_and_verify(
    remote: dict[str, dict[str, Any]],
    expected: dict[str, dict[str, Any]],
    download_dir: Path,
    timeout_seconds: float,
    allow_http: bool,
    approved_file_host: str = DEFAULT_APPROVED_FILE_HOST,
) -> dict[str, Path]:
    download_dir.mkdir(parents=True, exist_ok=True)
    downloaded: dict[str, Path] = {}
    for filename, expected_record in expected.items():
        url = remote[filename].get("url")
        if not isinstance(url, str):
            raise NotReadyError(f"The package index does not give a URL for {filename}.")
        parsed = urlparse(url)
        if parsed.scheme != "https" and not (allow_http and parsed.scheme == "http"):
            raise ValueError(f"The package index gave an unsafe URL for {filename}.")
        if parsed.hostname != approved_file_host:
            raise ValueError(f"The package index gave an unapproved host for {filename}.")

        target = download_dir / filename
        temporary_target = download_dir / f"{filename}.part"
        digest = hashlib.sha256()
        size = 0
        request = Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
                with temporary_target.open("wb") as output:
                    while block := response.read(1024 * 1024):
                        output.write(block)
                        digest.update(block)
                        size += len(block)
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            raise NotReadyError(f"The download for {filename} is not ready: {exc}.") from exc

        if digest.hexdigest() != expected_record["sha256"]:
            raise ValueError(f"The downloaded SHA-256 digest is incorrect for {filename}.")
        if size != expected_record["size"]:
            raise ValueError(f"The downloaded file size is incorrect for {filename}.")
        temporary_target.replace(target)
        downloaded[expected_record["type"]] = target
    return downloaded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--download-dir", type=Path, required=True)
    parser.add_argument("--github-output", type=Path)
    parser.add_argument("--index-base-url", default=DEFAULT_INDEX_BASE_URL)
    parser.add_argument("--approved-file-host", default=DEFAULT_APPROVED_FILE_HOST)
    parser.add_argument("--attempts", type=int, default=12)
    parser.add_argument("--delay-seconds", type=float, default=10.0)
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument("--allow-http", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if not 1 <= args.attempts <= 20:
        raise SystemExit("--attempts must be from 1 through 20.")
    if not 0 <= args.delay_seconds <= 60:
        raise SystemExit("--delay-seconds must be from 0 through 60.")
    if not 1 <= args.timeout_seconds <= 60:
        raise SystemExit("--timeout-seconds must be from 1 through 60.")

    name, version, expected = load_manifest(args.manifest)
    base_url = args.index_base_url.rstrip("/")
    parsed_base_url = urlparse(base_url)
    if parsed_base_url.scheme != "https" and not (
        args.allow_http and parsed_base_url.scheme == "http"
    ):
        raise SystemExit("The index base URL must use HTTPS.")
    release_url = f"{base_url}/{quote(name)}/{quote(version)}/json"

    last_error = "The package index did not become ready."
    for attempt in range(1, args.attempts + 1):
        try:
            release_data = get_release_data(release_url, args.timeout_seconds)
            remote = match_remote_files(release_data, expected)
            downloaded = download_and_verify(
                remote,
                expected,
                args.download_dir,
                args.timeout_seconds,
                args.allow_http,
                args.approved_file_host,
            )
        except (NotReadyError, URLError, TimeoutError) as exc:
            last_error = str(exc)
            print(f"Attempt {attempt} of {args.attempts}: {last_error}", flush=True)
            if attempt < args.attempts:
                time.sleep(args.delay_seconds)
            continue

        outputs = {
            "sdist": str(downloaded["sdist"]),
            "version": version,
            "wheel": str(downloaded["wheel"]),
        }
        if args.github_output:
            append_github_output(args.github_output, outputs)
        print(f"Verified package index files for {name} {version}.")
        return 0

    raise SystemExit(last_error)


if __name__ == "__main__":
    raise SystemExit(main())
