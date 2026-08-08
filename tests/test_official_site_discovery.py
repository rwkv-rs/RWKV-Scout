from unittest.mock import patch

from tools.official_site_discovery import _sitemap_urls, discover_official_urls
from utils.network_fetch import NetworkFetchError


def test_python_docs_adapter_finds_relevant_howto_without_a_search_engine():
    pages = {
        "https://docs.python.org/3.14/howto/": """
            <html><body>
              <a href="free-threading-python.html">Python support for free threading</a>
              <a href="free-threading-extensions.html">C API Extension support for free threading</a>
              <a href="../library/threading.html">threading library</a>
              <a href="https://example.com/wrong">third party</a>
            </body></html>
        """,
        "https://docs.python.org/3.14/": "<a href='howto/'>Python HOWTOs</a>",
        "https://docs.python.org/": "<a href='/3.14/'>Python 3.14 documentation</a>",
    }

    with patch(
        "tools.official_site_discovery.fetch_text",
        side_effect=lambda url, **_kwargs: pages.get(url, "<html></html>"),
    ):
        result = discover_official_urls(
            "python 3.14 free threading",
            ["docs.python.org"],
            answer_requirements=[{"type": "procedure"}],
        )

    urls = [row["url"] for row in result["results"]]
    assert "https://docs.python.org/3.14/howto/free-threading-python.html" in urls
    assert urls.index("https://docs.python.org/3.14/howto/free-threading-python.html") < urls.index(
        "https://docs.python.org/3.14/howto/free-threading-extensions.html"
    )
    assert all("example.com" not in url for url in urls)


def test_cisa_adapter_prioritizes_catalog_alert_links():
    pages = {
        "https://cisa.gov/news-events/cybersecurity-advisories": """
            <html><body>
              <a href="/news-events/alerts/2026/08/05/cisa-adds-one-known-exploited-vulnerability-catalog">
                CISA Adds One Known Exploited Vulnerability to Catalog
              </a>
            </body></html>
        """,
        "https://cisa.gov/known-exploited-vulnerabilities-catalog": """
            <a href="/sites/default/files/feeds/known_exploited_vulnerabilities.json">Download the KEV catalog</a>
        """,
        "https://cisa.gov/": (
            "<a href='/known-exploited-vulnerabilities-catalog'>Known Exploited Vulnerabilities Catalog</a>"
            "<a href='/news-events/cybersecurity-advisories'>Cybersecurity advisories</a>"
        ),
    }
    with patch(
        "tools.official_site_discovery.fetch_text",
        side_effect=lambda url, **_kwargs: pages.get(url, "<html></html>"),
    ):
        result = discover_official_urls(
            "CISA known exploited vulnerabilities latest additions CVE identifier",
            ["cisa.gov"],
            answer_requirements=[{"type": "cve_id"}],
            prefer_recent=True,
        )

    urls = [row["url"] for row in result["results"]]
    assert any("/2026/08/05/cisa-adds-one" in url for url in urls)
    assert any(url.endswith("known_exploited_vulnerabilities.json") for url in urls)


def test_kubernetes_sitemap_surfaces_the_exact_version_release_page():
    _sitemap_urls.cache_clear()
    pages = {
        "https://kubernetes.io/sitemap.xml": (
            "<sitemapindex><sitemap><loc>https://kubernetes.io/en/sitemap.xml</loc></sitemap></sitemapindex>"
        ),
        "https://kubernetes.io/en/sitemap.xml": (
            "<urlset>"
            "<url><loc>https://kubernetes.io/blog/2024/04/17/kubernetes-v1-30-release/</loc></url>"
            "<url><loc>https://kubernetes.io/blog/2023/12/13/kubernetes-v1-29-release/</loc></url>"
            "</urlset>"
        ),
    }
    with patch(
        "tools.official_site_discovery.fetch_text",
        side_effect=lambda url, **_kwargs: pages.get(url, "<html></html>"),
    ):
        result = discover_official_urls(
            "Kubernetes 1.30 release date",
            ["kubernetes.io"],
            answer_requirements=[{"type": "date"}],
        )

    urls = [row["url"] for row in result["results"]]
    assert "https://kubernetes.io/blog/2024/04/17/kubernetes-v1-30-release/" in urls
    assert "https://kubernetes.io/blog/2023/12/13/kubernetes-v1-29-release/" not in urls


def test_generic_seeds_do_not_embed_answer_specific_who_statement_url():
    _sitemap_urls.cache_clear()
    with patch(
        "tools.official_site_discovery.fetch_text",
        return_value="<html></html>",
    ):
        fastapi = discover_official_urls(
            "FastAPI latest version and release date",
            ["fastapi.tiangolo.com"],
            answer_requirements=[{"type": "version"}, {"type": "date"}],
        )
        who = discover_official_urls(
            "WHO COVID-19 no longer a public health emergency",
            ["who.int"],
            answer_requirements=[{"type": "person"}, {"type": "date"}],
        )

    assert fastapi["results"][0]["url"] == "https://fastapi.tiangolo.com/release-notes/"
    assert not any("05-05-2023-statement-on-the-fifteenth-meeting" in row["url"] for row in who["results"])


def test_unseen_official_site_uses_generic_sitemap():
    _sitemap_urls.cache_clear()
    pages = {
        "https://docs.djangoproject.com/sitemap.xml": (
            "<urlset>"
            "<url><loc>https://docs.djangoproject.com/en/5.2/releases/5.2/</loc></url>"
            "<url><loc>https://docs.djangoproject.com/en/5.1/releases/5.1/</loc></url>"
            "</urlset>"
        ),
    }
    with patch(
        "tools.official_site_discovery.fetch_text",
        side_effect=lambda url, **_kwargs: pages.get(url, "<html></html>"),
    ):
        result = discover_official_urls(
            "Django 5.2 official release date",
            ["docs.djangoproject.com"],
            answer_requirements=[{"type": "date"}],
        )

    urls = [row["url"] for row in result["results"]]
    assert urls[0] == "https://docs.djangoproject.com/en/5.2/releases/5.2/"
    assert "https://docs.djangoproject.com/en/5.1/releases/5.1/" not in urls


def test_official_sitemap_can_prove_a_request_scoped_canonical_hostname():
    _sitemap_urls.cache_clear()
    pages = {
        "https://project.example/sitemap.xml": (
            "<urlset>"
            "<url><loc>https://project-docs.example/news/2025/04/02/project-52-released/</loc></url>"
            "<url><loc>https://project-docs.example/download/</loc></url>"
            "</urlset>"
        ),
    }
    with patch(
        "tools.official_site_discovery.fetch_text",
        side_effect=lambda url, **_kwargs: pages.get(url, "<html></html>"),
    ):
        result = discover_official_urls(
            "Project 5.2 official release date",
            ["project.example"],
            answer_requirements=[{"type": "date"}],
        )

    assert result["results"][0]["url"].endswith("/project-52-released/")
    assert result["resolved_domains"] == ["project.example", "project-docs.example"]
    assert result["domain_aliases"] == [
        {
            "requested_domain": "project.example",
            "canonical_domain": "project-docs.example",
            "verification": "dominant_https_sitemap_host",
        }
    ]


def test_transient_sitemap_failure_is_not_cached_across_recovery_rounds():
    _sitemap_urls.cache_clear()
    calls = {"count": 0}

    def fetch(url, **_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise NetworkFetchError("temporary sitemap timeout")
        return (
            "<urlset>"
            "<url><loc>https://example.org/releases/5.2/</loc></url>"
            "</urlset>"
        )

    with patch("tools.official_site_discovery.fetch_text", side_effect=fetch):
        assert _sitemap_urls("example.org") == ()
        assert _sitemap_urls("example.org") == ("https://example.org/releases/5.2/",)

    assert calls["count"] == 2


def test_unseen_release_site_gets_generic_release_routes():
    _sitemap_urls.cache_clear()
    with patch("tools.official_site_discovery.fetch_text", return_value="<html></html>"):
        result = discover_official_urls(
            "Pydantic latest stable version and release date",
            ["docs.pydantic.dev"],
            answer_requirements=[{"type": "version"}, {"type": "date"}],
        )

    urls = [row["url"] for row in result["results"]]
    assert "https://docs.pydantic.dev/release-notes/" in urls


def test_exact_protocol_topic_beats_a_generic_guide_on_an_unseen_official_site():
    _sitemap_urls.cache_clear()
    pages = {
        "https://nginx.org/sitemap.xml": (
            "<urlset>"
            "<url><loc>https://nginx.org/en/docs/beginners_guide.html</loc></url>"
            "<url><loc>https://nginx.org/en/docs/http/websocket.html</loc></url>"
            "</urlset>"
        ),
    }
    with patch(
        "tools.official_site_discovery.fetch_text",
        side_effect=lambda url, **_kwargs: pages.get(url, "<html></html>"),
    ):
        result = discover_official_urls(
            "site:nginx.org nginx WebSocket proxy Upgrade Connection headers",
            ["nginx.org"],
            answer_requirements=[{"type": "procedure"}],
        )

    urls = [row["url"] for row in result["results"]]
    assert urls[0] == "https://nginx.org/en/docs/http/websocket.html"
    assert result["results"][0]["query_relevance"]["literal_satisfied"] is True


def test_model_invented_literal_does_not_exclude_user_requested_official_page():
    _sitemap_urls.cache_clear()
    pages = {
        "https://nginx.org/sitemap.xml": (
            "<urlset>"
            "<url><loc>https://nginx.org/en/docs/beginners_guide.html</loc></url>"
            "<url><loc>https://nginx.org/en/docs/http/websocket.html</loc></url>"
            "</urlset>"
        ),
    }
    with patch(
        "tools.official_site_discovery.fetch_text",
        side_effect=lambda url, **_kwargs: pages.get(url, "<html></html>"),
    ):
        result = discover_official_urls(
            "site:nginx.org nginx WebSocket Upgrade Sec-WebSocket-Key Sec-WebSocket-Version",
            ["nginx.org"],
            answer_requirements=[{"type": "procedure"}],
            constraint_query="按照 nginx 官方文档配置 WebSocket 反向代理",
        )

    assert result["results"][0]["url"] == "https://nginx.org/en/docs/http/websocket.html"
    assert "sec-websocket-key" not in result["results"][0]["query_relevance"]["literal_anchors"]


def test_version_slug_beats_old_release_schedule_without_product_rules():
    _sitemap_urls.cache_clear()
    pages = {
        "https://example.org/sitemap.xml": (
            "<urlset>"
            "<url><loc>https://example.org/blog/2025/apr/02/project-52-released/</loc></url>"
            "<url><loc>https://example.org/blog/2010/apr/07/project-1_2-release-schedule-update/</loc></url>"
            "</urlset>"
        ),
    }
    with patch(
        "tools.official_site_discovery.fetch_text",
        side_effect=lambda url, **_kwargs: pages.get(url, "<html></html>"),
    ):
        result = discover_official_urls(
            "Project 5.2 official release date",
            ["example.org"],
            answer_requirements=[{"type": "date"}],
        )

    assert result["results"][0]["url"].endswith("/project-52-released/")
    assert all("project-1_2" not in row["url"] for row in result["results"])
