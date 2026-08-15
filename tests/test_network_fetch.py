from __future__ import annotations

from utils import network_fetch


def _clear_proxy_environment(monkeypatch) -> None:
    for name in (
        "RWKV_ECRA_HTTP_PROXY",
        "RWKV_ECRA_HTTPS_PROXY",
        "RWKV_ECRA_AUTO_WINDOWS_PROXY",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        monkeypatch.delenv(name, raising=False)


def test_shell_loopback_proxy_is_not_rewritten(monkeypatch) -> None:
    _clear_proxy_environment(monkeypatch)
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:10808")
    monkeypatch.setenv("HTTPS_PROXY", "127.0.0.1:10808")

    assert network_fetch.get_network_proxies() == {
        "http": "http://127.0.0.1:10808",
        "https": "http://127.0.0.1:10808",
    }


def test_windows_registry_proxy_discovery_is_opt_in(monkeypatch) -> None:
    _clear_proxy_environment(monkeypatch)
    calls: list[bool] = []
    monkeypatch.setattr(
        network_fetch,
        "_windows_proxy_settings",
        lambda: calls.append(True) or {"https": "http://192.0.2.1:10808"},
    )

    assert network_fetch.get_network_proxies() == {}
    assert calls == []

    monkeypatch.setenv("RWKV_ECRA_AUTO_WINDOWS_PROXY", "1")
    assert network_fetch.get_network_proxies() == {"https": "http://192.0.2.1:10808"}
    assert calls == [True]


def test_explicit_project_proxy_takes_precedence(monkeypatch) -> None:
    _clear_proxy_environment(monkeypatch)
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:10808")
    monkeypatch.setenv("RWKV_ECRA_HTTP_PROXY", "proxy.internal:3128")

    assert network_fetch.get_network_proxies() == {
        "http": "http://proxy.internal:3128",
        "https": "http://proxy.internal:3128",
    }
