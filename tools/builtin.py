"""Explicit loading of built-in tool registrations."""

from __future__ import annotations

from threading import Lock


_loaded = False
_lock = Lock()


def load_builtin_tools() -> None:
    """Import all registration modules exactly once."""
    global _loaded
    if _loaded:
        return
    with _lock:
        if _loaded:
            return
        import tools.chat  # noqa: F401
        import tools.crossref  # noqa: F401
        import tools.github_rest  # noqa: F401
        import tools.mediawiki  # noqa: F401
        import tools.paper_search  # noqa: F401
        import tools.static_ops  # noqa: F401
        import tools.weather  # noqa: F401
        import tools.web_search  # noqa: F401
        import tools.web_search_tavily  # noqa: F401
        import tools.web_search_keyless  # noqa: F401
        import tools.web_search_wigolo  # noqa: F401
        import workflows.map_reduce_flow  # noqa: F401
        import workflows.memory_query_flow  # noqa: F401
        import workflows.report_flow  # noqa: F401
        _loaded = True
