"""Stubs out mlx/* at collection time so tests run on any platform."""
from __future__ import annotations
import sys
import types
import pytest


def _stub(name: str) -> types.ModuleType:
    m = types.ModuleType(name)
    sys.modules[name] = m
    return m


for _p in [
    "mlx", "mlx.core", "mlx.nn",
    "mlx_lm", "mlx_lm.utils",
    "mlx_vlm", "mlx_vlm.utils",
    "mlx_embeddings", "mlx_embeddings.utils",
]:
    if _p not in sys.modules:
        _stub(_p)

# Pre-import real ppmlx modules so that test_cli.py / test_server.py cannot
# replace them with MagicMock (both files guard injection with
# `if mod not in sys.modules`, so pre-importing here wins).
# This runs at collection time — before any test file is collected.
import ppmlx.config        # noqa: F401, E402
import ppmlx.schema        # noqa: F401, E402
import ppmlx.db            # noqa: F401, E402
import ppmlx.models        # noqa: F401, E402
import ppmlx.memory        # noqa: F401, E402
import ppmlx.quantize      # noqa: F401, E402
import ppmlx.templates     # noqa: F401, E402
import ppmlx.bench         # noqa: F401, E402
import ppmlx.engine        # noqa: F401, E402
import ppmlx.engine_embed  # noqa: F401, E402
import ppmlx.engine_vlm    # noqa: F401, E402
import ppmlx.bench         # noqa: F401, E402
import ppmlx.batch         # noqa: F401, E402
import ppmlx.rag           # noqa: F401, E402
import ppmlx.processor     # noqa: F401, E402
import ppmlx.agent         # noqa: F401, E402
import ppmlx.gateway       # noqa: F401, E402
import ppmlx.turboquant    # noqa: F401, E402

# Snapshot real module attributes HERE (module level) — before any test file
# is collected.  test_server.py's module-level code runs at collection time and
# pollutes ppmlx.engine.get_engine etc. with MagicMocks.  By snapshotting now
# we capture the clean state and can restore it before every test.
_CLEAN_MODULE_STATE: dict[str, dict] = {
    mod_name: {k: v for k, v in vars(mod).items() if not k.startswith("__")}
    for mod_name, mod in sys.modules.items()
    if mod_name.startswith("ppmlx.") and isinstance(mod, types.ModuleType)
}


@pytest.fixture(autouse=True)
def _restore_module_attrs():
    """Restore ppmlx.* modules to their pre-collection clean state before each test.

    Both test_cli.py and test_server.py monkey-patch real module attributes
    (some at collection time, some inside test functions).  This fixture
    restores the clean snapshot captured at conftest load time so every test
    starts from a known-good baseline.
    """
    # Restore BEFORE the test runs
    for mod_name, saved in _CLEAN_MODULE_STATE.items():
        mod = sys.modules.get(mod_name)
        if mod is None or not isinstance(mod, types.ModuleType):
            continue
        current = {k for k in vars(mod) if not k.startswith("__")}
        for key in current - set(saved):
            try:
                delattr(mod, key)
            except Exception:
                pass
        for key, val in saved.items():
            if vars(mod).get(key) is not val:
                try:
                    setattr(mod, key, val)
                except Exception:
                    pass
    yield


@pytest.fixture()
def tmp_home(tmp_path, monkeypatch):
    """Redirect ~/.ppmlx to a temp directory for isolation."""
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path
