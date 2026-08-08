import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from config import _load_local_env_file, retire_search_api_key


def test_local_env_file_does_not_override_process_environment():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / ".env.local"
        path.write_text(
            "EXISTING_SETTING=from-file\nNEW_LOCAL_SETTING='local value'\n",
            encoding="utf-8",
        )
        with patch.dict(os.environ, {"EXISTING_SETTING": "from-process"}, clear=False):
            os.environ.pop("NEW_LOCAL_SETTING", None)
            _load_local_env_file(str(path))
            assert os.environ["EXISTING_SETTING"] == "from-process"
            assert os.environ["NEW_LOCAL_SETTING"] == "local value"


def test_retire_search_api_key_removes_only_target_from_runtime_and_file():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / ".env.local"
        path.write_text(
            'TAVILY_API_KEYS=["expired-key","healthy-key"]\n'
            "UNRELATED_SETTING=preserved\n",
            encoding="utf-8",
        )
        with patch.dict(
            os.environ,
            {"TAVILY_API_KEYS": '["expired-key","healthy-key"]'},
            clear=False,
        ):
            changed = retire_search_api_key(
                "tavily",
                "expired-key",
                local_env_file=str(path),
            )
            assert changed is True
            assert "expired-key" not in os.environ["TAVILY_API_KEYS"]
            assert "healthy-key" in os.environ["TAVILY_API_KEYS"]

        contents = path.read_text(encoding="utf-8")
        assert "expired-key" not in contents
        assert "healthy-key" in contents
        assert "UNRELATED_SETTING=preserved" in contents


def test_retire_search_api_key_deletes_empty_primary_line():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / ".env.local"
        path.write_text(
            "TAVILY_API_KEY=expired-key\nUNRELATED_SETTING=preserved\n",
            encoding="utf-8",
        )
        with patch.dict(os.environ, {"TAVILY_API_KEY": "expired-key"}, clear=False):
            changed = retire_search_api_key(
                "tavily",
                "expired-key",
                local_env_file=str(path),
            )
            assert changed is True
            assert "TAVILY_API_KEY" not in os.environ

        contents = path.read_text(encoding="utf-8")
        assert "TAVILY_API_KEY" not in contents
        assert "UNRELATED_SETTING=preserved" in contents
