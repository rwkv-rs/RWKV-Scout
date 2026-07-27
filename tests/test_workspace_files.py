from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

from app.services.workspace_files import (
    WorkspacePathError,
    list_workspace_files,
    read_workspace_file,
    resolve_workspace_path,
    save_uploaded_file,
)


class WorkspaceFileTests(unittest.TestCase):
    def test_paths_are_contained_and_normalized(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            resolved, relative = resolve_workspace_path(root, r"nested\note.md")
            self.assertEqual(relative, "nested/note.md")
            self.assertEqual(resolved, root / "nested" / "note.md")
            with self.assertRaises(WorkspacePathError):
                resolve_workspace_path(root, "../outside.txt")
            with self.assertRaises(WorkspacePathError):
                resolve_workspace_path(root, "C:/outside.txt")

    def test_upload_list_and_read_use_one_workspace_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            saved = save_uploaded_file(directory, "nested/note.md", io.BytesIO(b"hello"))
            self.assertEqual(saved, "nested/note.md")
            self.assertEqual(list_workspace_files(directory), ["nested/note.md"])
            loaded = read_workspace_file(directory, "nested/note.md")
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.text, "hello")


if __name__ == "__main__":
    unittest.main()
