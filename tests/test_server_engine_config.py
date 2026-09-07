from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_lifespan_passes_engine_limits_and_kv_cache(monkeypatch):
    import ppmlx.server as server

    kv_cache = SimpleNamespace(quantize="turboquant")
    cfg = SimpleNamespace(
        logging=SimpleNamespace(snapshot_interval_seconds=60),
        server=SimpleNamespace(max_loaded_models=3, ttl_seconds=60),
        defaults=SimpleNamespace(prompt_cache_limit=0),
        kv_cache=kv_cache,
    )
    engine = SimpleNamespace(load=lambda _repo: None)
    calls = []

    monkeypatch.setattr(server, "_preload_model", None)
    monkeypatch.setattr(server, "_preload_embed_model", None)
    monkeypatch.setattr(server, "_startup_overrides", {})
    monkeypatch.setattr("ppmlx.config.load_config", lambda: cfg)
    monkeypatch.setattr("ppmlx.db.get_db", lambda: SimpleNamespace(init=lambda: None))
    monkeypatch.setattr("ppmlx.engine.get_engine", lambda **kwargs: calls.append(kwargs) or engine)
    monkeypatch.setattr(server, "_route_policy_path", lambda: None)
    async def snapshot(_interval):
        return None
    monkeypatch.setattr(server, "_snapshot_loop", snapshot)

    class App:
        state = SimpleNamespace()

    async with server.lifespan(App()):
        pass

    assert calls == [{
        "max_loaded": 3,
        "ttl_seconds": 60,
        "prompt_cache_limit": 0,
        "kv_cache": kv_cache,
    }]


@pytest.mark.asyncio
async def test_lifespan_without_model_does_not_load(monkeypatch):
    import ppmlx.server as server

    cfg = SimpleNamespace(
        logging=SimpleNamespace(snapshot_interval_seconds=60),
        server=SimpleNamespace(max_loaded_models=2, ttl_seconds=0),
        defaults=SimpleNamespace(prompt_cache_limit=4),
        kv_cache=SimpleNamespace(quantize="off"),
    )
    def load(_repo):
        pytest.fail("startup loaded an unrequested model")
    engine = SimpleNamespace(load=load)
    monkeypatch.setattr(server, "_preload_model", None)
    monkeypatch.setattr(server, "_preload_embed_model", None)
    monkeypatch.setattr(server, "_startup_overrides", {})
    monkeypatch.setattr("ppmlx.config.load_config", lambda: cfg)
    monkeypatch.setattr("ppmlx.db.get_db", lambda: SimpleNamespace(init=lambda: None))
    monkeypatch.setattr("ppmlx.engine.get_engine", lambda **_kwargs: engine)
    monkeypatch.setattr(server, "_route_policy_path", lambda: None)
    async def snapshot(_interval):
        return None
    monkeypatch.setattr(server, "_snapshot_loop", snapshot)

    class App:
        state = SimpleNamespace()

    async with server.lifespan(App()):
        pass


def test_startup_override_reaches_cached_config(monkeypatch):
    import ppmlx.server as server
    from ppmlx.config import Config
    cfg = Config()
    seen = []
    def load_config(*, cli_overrides):
        seen.append(cli_overrides)
        cfg.kv_cache.quantize = cli_overrides["kv_quant"]
        return cfg
    monkeypatch.setattr("ppmlx.config.load_config", load_config)
    monkeypatch.setattr(server, "_startup_overrides", {})
    monkeypatch.setattr(server, "_cached_full_config", None)
    monkeypatch.setattr(server, "_full_config_loaded", False)
    monkeypatch.setattr(server, "_cached_server_config", None)
    monkeypatch.setattr(server, "_server_config_loaded", False)
    server.set_startup_overrides({"kv_quant": "turboquant"})
    assert server._get_config().kv_cache.quantize == "turboquant"
    assert seen == [{"kv_quant": "turboquant"}]
