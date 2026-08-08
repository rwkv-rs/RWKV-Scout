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
        import tools.calculator  # noqa: F401
        import tools.connectors  # noqa: F401
        import tools.crossref  # noqa: F401
        import tools.date_calculator  # noqa: F401
        import tools.github_rest  # noqa: F401
        import tools.mediawiki  # noqa: F401
        import tools.paper_search  # noqa: F401
        import tools.static_ops  # noqa: F401
        import tools.time_tools  # noqa: F401
        import tools.weather  # noqa: F401
        # ``tools.web_search_generic`` is the active public web capability.
        # The former execute_web_search implementation is legacy and is no
        # longer registered by default.
        import tools.web_search_tavily  # noqa: F401
        import tools.web_search_keyless  # noqa: F401
        import tools.web_search_generic  # noqa: F401
        import tools.web_search_wigolo  # noqa: F401
        _loaded = True
