from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_ppmlx_{name}", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not load {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check_distribution = load_script("check_distribution")
verify_installed_distribution = load_script("verify_installed_distribution")
verify_testpypi_artifact = load_script("verify_testpypi_artifact")


def metadata_bytes(version: str = "1.2.3", name: str = "ppmlx") -> bytes:
    return f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n\n".encode()


def entry_points_bytes() -> bytes:
    return (
        "[console_scripts]\n"
        "ppmlx = ppmlx.cli:main_entry\n"
        "ppmlx-memory-mcp = ppmlx.mcp_server:main\n"
    ).encode()


def registry_bytes() -> bytes:
    return json.dumps({"models": {"demo": {}}}).encode()


def write_wheel(
    directory: Path,
    *,
    filename: str = "ppmlx-1.2.3-py3-none-any.whl",
    metadata_version: str = "1.2.3",
    extra_names: tuple[str, ...] = (),
) -> Path:
    path = directory / filename
    dist_info = "ppmlx-1.2.3.dist-info"
    files = {
        "ppmlx/__init__.py": b"",
        "ppmlx/cli.py": b"",
        "ppmlx/mcp_server.py": b"",
        "ppmlx/registry_data.json": registry_bytes(),
        "ppmlx/server.py": b"",
        f"{dist_info}/METADATA": metadata_bytes(metadata_version),
        f"{dist_info}/entry_points.txt": entry_points_bytes(),
        f"{dist_info}/licenses/LICENSE": b"license",
    }
    files.update({name: b"extra" for name in extra_names})
    with zipfile.ZipFile(path, "w") as archive:
        for name, contents in files.items():
            archive.writestr(name, contents)
    return path


def write_sdist(
    directory: Path,
    *,
    filename: str = "ppmlx-1.2.3.tar.gz",
    root: str = "ppmlx-1.2.3",
    metadata_version: str = "1.2.3",
    extra_names: tuple[str, ...] = (),
) -> Path:
    path = directory / filename
    files = {
        "LICENSE": b"license",
        "PKG-INFO": metadata_bytes(metadata_version),
        "README.md": b"readme",
        "pyproject.toml": b"[project]\nname = 'ppmlx'\n",
        "ppmlx/__init__.py": b"",
        "ppmlx/cli.py": b"",
        "ppmlx/mcp_server.py": b"",
        "ppmlx/registry_data.json": registry_bytes(),
        "ppmlx/server.py": b"",
    }
    files.update({name: b"extra" for name in extra_names})
    import tarfile

    with tarfile.open(path, "w:gz") as archive:
        directory_info = tarfile.TarInfo(root)
        directory_info.type = tarfile.DIRTYPE
        archive.addfile(directory_info)
        for name, contents in files.items():
            info = tarfile.TarInfo(f"{root}/{name}")
            info.size = len(contents)
            archive.addfile(info, io.BytesIO(contents))
        for name in extra_names:
            if name.startswith("../"):
                info = tarfile.TarInfo(f"{root}/{name}")
                info.size = len(b"extra")
                archive.addfile(info, io.BytesIO(b"extra"))
    return path


def manifest_records() -> dict[str, dict[str, object]]:
    return {
        "ppmlx-1.2.3-py3-none-any.whl": {
            "filename": "ppmlx-1.2.3-py3-none-any.whl",
            "sha256": "a" * 64,
            "size": 10,
            "type": "wheel",
        },
        "ppmlx-1.2.3.tar.gz": {
            "filename": "ppmlx-1.2.3.tar.gz",
            "sha256": "b" * 64,
            "size": 20,
            "type": "sdist",
        },
    }


def test_check_distribution_accepts_valid_wheel_and_sdist(tmp_path: Path):
    wheel = write_wheel(tmp_path)
    sdist = write_sdist(tmp_path)

    assert check_distribution.validate_wheel(wheel) == check_distribution.Version("1.2.3")
    assert check_distribution.validate_sdist(sdist) == check_distribution.Version("1.2.3")


def test_check_distribution_rejects_unsafe_wheel_path(tmp_path: Path):
    wheel = write_wheel(tmp_path, extra_names=("../escape",))

    with pytest.raises(ValueError, match="unsafe paths"):
        check_distribution.validate_wheel(wheel)


def test_check_distribution_rejects_mismatched_wheel_metadata(tmp_path: Path):
    wheel = write_wheel(tmp_path, metadata_version="1.2.4")

    with pytest.raises(ValueError, match="version"):
        check_distribution.validate_wheel(wheel)


def test_load_manifest_accepts_valid_records(tmp_path: Path):
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps({"name": "ppmlx", "version": "1.2.3", "artifacts": list(manifest_records().values())}),
        encoding="utf-8",
    )

    name, version, records = verify_testpypi_artifact.load_manifest(path)

    assert name == "ppmlx"
    assert version == "1.2.3"
    assert records == manifest_records()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda records: records["ppmlx-1.2.3-py3-none-any.whl"].update(filename="../escape.whl"), "invalid filename"),
        (lambda records: records["ppmlx-1.2.3.tar.gz"].update(sha256="not-a-digest"), "invalid digest"),
    ],
)
def test_load_manifest_rejects_unsafe_or_mismatched_artifact_records(change, message, tmp_path: Path):
    records = manifest_records()
    change(records)
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps({"name": "ppmlx", "version": "1.2.3", "artifacts": list(records.values())}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        verify_testpypi_artifact.load_manifest(path)


def test_match_remote_files_rejects_missing_extra_and_bad_digest():
    expected = manifest_records()
    remote = {
        filename: {
            "filename": filename,
            "digests": {"sha256": record["sha256"]},
            "size": record["size"],
            "url": f"https://test-files.pythonhosted.org/{filename}",
        }
        for filename, record in expected.items()
    }

    assert verify_testpypi_artifact.match_remote_files({"urls": list(remote.values())}, expected) == remote

    missing = dict(remote)
    missing.pop(next(iter(missing)))
    with pytest.raises(verify_testpypi_artifact.NotReadyError, match="incomplete"):
        verify_testpypi_artifact.match_remote_files({"urls": list(missing.values())}, expected)

    extra = dict(remote)
    extra["unexpected.whl"] = {"filename": "unexpected.whl"}
    with pytest.raises(ValueError, match="unexpected release files"):
        verify_testpypi_artifact.match_remote_files({"urls": list(extra.values())}, expected)

    bad_digest = dict(remote)
    bad_digest[next(iter(bad_digest))] = {
        **bad_digest[next(iter(bad_digest))],
        "digests": {"sha256": "c" * 64},
    }
    with pytest.raises(ValueError, match="incorrect SHA-256 digest"):
        verify_testpypi_artifact.match_remote_files({"urls": list(bad_digest.values())}, expected)


class FakeResponse:
    def __init__(self, payload: bytes):
        self.payload = payload
        self.read_count = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self, _size: int = -1) -> bytes:
        if self.read_count:
            return b""
        self.read_count += 1
        return self.payload


@pytest.mark.parametrize(
    "approved_file_host",
    [
        verify_testpypi_artifact.DEFAULT_APPROVED_FILE_HOST,
        "files.pythonhosted.org",
    ],
)
def test_download_accepts_the_approved_file_host(
    tmp_path: Path, monkeypatch, approved_file_host: str
):
    payload = b"wheel-data"
    filename = "ppmlx-1.2.3-py3-none-any.whl"
    expected = {
        filename: {
            "filename": filename,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size": len(payload),
            "type": "wheel",
        }
    }
    remote = {
        filename: {"url": f"https://{approved_file_host}/{filename}"}
    }
    monkeypatch.setattr(verify_testpypi_artifact, "urlopen", lambda request, timeout: FakeResponse(payload))

    downloaded = verify_testpypi_artifact.download_and_verify(
        remote,
        expected,
        tmp_path / "downloads",
        timeout_seconds=1,
        allow_http=False,
        approved_file_host=approved_file_host,
    )

    assert downloaded["wheel"].read_bytes() == payload


@pytest.mark.parametrize(
    "url",
    [
        "https://evil.example/ppmlx.whl",
        "http://test-files.pythonhosted.org/ppmlx.whl",
    ],
)
def test_download_rejects_unsafe_testpypi_urls(tmp_path: Path, url: str):
    payload = b"wheel-data"
    filename = "ppmlx-1.2.3-py3-none-any.whl"
    expected = {
        filename: {
            "filename": filename,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size": len(payload),
            "type": "wheel",
        }
    }

    with pytest.raises(ValueError, match="unsafe URL|unapproved host"):
        verify_testpypi_artifact.download_and_verify(
            {filename: {"url": url}},
            expected,
            tmp_path / "downloads",
            timeout_seconds=1,
            allow_http=False,
        )


def test_verify_installed_distribution_accepts_matching_install(tmp_path: Path, monkeypatch):
    version = "1.2.3"
    entry_points = [
        SimpleNamespace(name=name, value=value, group="console_scripts")
        for name, value in verify_installed_distribution.REQUIRED_ENTRY_POINTS.items()
    ]
    executable_paths = {}
    for name in verify_installed_distribution.REQUIRED_ENTRY_POINTS:
        path = tmp_path / name
        path.write_text("#!/bin/sh\n", encoding="utf-8")
        path.chmod(0o755)
        executable_paths[name] = path

    class FakeResources:
        def joinpath(self, _name: str):
            return self

        def read_text(self, encoding: str):
            assert encoding == "utf-8"
            return json.dumps({"models": {"demo": {}}})

    distribution = SimpleNamespace(entry_points=entry_points)
    subprocess_calls = []
    imported_modules = []

    def fake_run(command, **kwargs):
        subprocess_calls.append((command, kwargs))
        return SimpleNamespace(stdout=f"ppmlx {version}\n")

    monkeypatch.setattr(verify_installed_distribution.metadata, "version", lambda _name: version)
    monkeypatch.setattr(verify_installed_distribution.metadata, "distribution", lambda _name: distribution)
    monkeypatch.setattr(verify_installed_distribution.resources, "files", lambda _name: FakeResources())
    monkeypatch.setattr(
        verify_installed_distribution.importlib,
        "import_module",
        imported_modules.append,
    )
    monkeypatch.setattr(verify_installed_distribution, "executable_path", executable_paths.__getitem__)
    monkeypatch.setattr(verify_installed_distribution.subprocess, "run", fake_run)
    health_versions = []
    monkeypatch.setattr(
        verify_installed_distribution,
        "verify_server_health",
        health_versions.append,
    )
    monkeypatch.setattr(verify_installed_distribution.sys, "argv", ["verify", "--expected-version", version])

    assert verify_installed_distribution.main() == 0
    assert [call[0][1:] for call in subprocess_calls] == [["--version"], ["--help"]]
    assert health_versions == [version]
    assert imported_modules == ["ppmlx.cli", "ppmlx.mcp_server"]


def test_verify_installed_distribution_rejects_version_mismatch(monkeypatch):
    monkeypatch.setattr(verify_installed_distribution.metadata, "version", lambda _name: "1.2.4")
    monkeypatch.setattr(
        verify_installed_distribution.sys,
        "argv",
        ["verify", "--expected-version", "1.2.3"],
    )

    with pytest.raises(SystemExit, match="does not match"):
        verify_installed_distribution.main()


def test_testpypi_main_retries_until_release_is_ready(tmp_path: Path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"name": "ppmlx", "version": "1.2.3", "artifacts": list(manifest_records().values())}),
        encoding="utf-8",
    )
    downloaded = {
        "wheel": tmp_path / "ppmlx-1.2.3-py3-none-any.whl",
        "sdist": tmp_path / "ppmlx-1.2.3.tar.gz",
    }
    attempts = []
    download_calls = []
    sleep_delays = []

    def fake_get_release_data(url: str, timeout_seconds: float):
        attempts.append((url, timeout_seconds))
        if len(attempts) < 3:
            raise verify_testpypi_artifact.NotReadyError("not ready")
        return {"urls": []}

    monkeypatch.setattr(verify_testpypi_artifact, "get_release_data", fake_get_release_data)
    monkeypatch.setattr(verify_testpypi_artifact, "match_remote_files", lambda data, expected: {})

    def fake_download_and_verify(
        remote,
        expected,
        download_dir,
        timeout_seconds,
        allow_http,
        approved_file_host,
    ):
        download_calls.append((download_dir, timeout_seconds, allow_http, approved_file_host))
        return downloaded

    monkeypatch.setattr(verify_testpypi_artifact, "download_and_verify", fake_download_and_verify)
    monkeypatch.setattr(verify_testpypi_artifact.time, "sleep", sleep_delays.append)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify",
            "--manifest",
            str(manifest),
            "--download-dir",
            str(tmp_path / "downloads"),
            "--index-base-url",
            "https://pypi.org/pypi",
            "--approved-file-host",
            "files.pythonhosted.org",
            "--attempts",
            "3",
            "--delay-seconds",
            "2",
            "--timeout-seconds",
            "4",
        ],
    )

    assert verify_testpypi_artifact.main() == 0
    assert len(attempts) == 3
    assert attempts[-1][0] == "https://pypi.org/pypi/ppmlx/1.2.3/json"
    assert download_calls[-1][3] == "files.pythonhosted.org"
    assert sleep_delays == [2.0, 2.0]
