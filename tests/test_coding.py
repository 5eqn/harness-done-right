"""
Tests for Coding tasks (PythonFileWritten, PythonWorkspaceBuilt).
Uses a fixed temp workspace created via tempfile.mkdtemp.
"""

import os
import subprocess
import tempfile

import pytest


class TestPythonFileWritten:
    """Tests for PythonFileWritten task class."""

    def test_py_file_not_exists(self):
        """Test PythonFileWritten fails when file does not exist"""
        from hdr.tasks.coding import PythonFileWritten

        with pytest.raises(AssertionError, match="does not exist"):
            PythonFileWritten(path="/nonexistent/path/12345.py")

    def test_py_file_must_end_with_py(self):
        """Test PythonFileWritten fails when path does not end with .py"""
        from hdr.tasks.coding import PythonFileWritten

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test.txt")
            with open(file_path, "w") as f:
                f.write("hello")
            with pytest.raises(AssertionError, match=r".*\.py.*"):
                PythonFileWritten(path=file_path)

    def test_valid_py_file(self):
        """Test PythonFileWritten passes for a clean Python file"""
        from hdr.tasks.coding import PythonFileWritten

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "hello.py")
            with open(file_path, "w") as f:
                f.write('name = "world"\nprint(f"Hello, {name}!")\n')
            f = PythonFileWritten(path=file_path)
            assert f.path == file_path
            assert 'name = "world"\nprint(f"Hello, {name}!")\n' in f.content

    def test_py_file_content_reflects_formatted_file(self):
        """Test PythonFileWritten content is read after ruff format mutates disk."""
        from hdr.tasks.coding import PythonFileWritten

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "formatted.py")
            with open(file_path, "w") as f:
                f.write("x=1\n")
            f = PythonFileWritten(path=file_path)
            assert f.content == "x = 1\n"


class TestMarkdownFileWritten:
    """Tests for MarkdownFileWritten task class."""

    def test_markdown_runs_non_fix_lint_after_format(self, monkeypatch):
        """Test MarkdownFileWritten runs a plain markdownlint pass after --fix."""
        from hdr.tasks.coding import MarkdownFileWritten

        calls: list[list[str]] = []

        def fake_run(
            command,
            capture_output=False,
            text=False,
            cwd=None,
        ):
            calls.append(command)
            return subprocess.CompletedProcess(command, 0, "", "")

        monkeypatch.setattr("shutil.which", lambda name: "/bin/markdownlint-cli2")
        monkeypatch.setattr("subprocess.run", fake_run)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test.md")
            with open(file_path, "w") as f:
                f.write("# Hello\n")
            MarkdownFileWritten(path=file_path)

        assert calls == [
            ["markdownlint-cli2", "--fix", file_path],
            ["markdownlint-cli2", file_path],
        ]


class TestPythonWorkspaceBuilt:
    """Tests for PythonWorkspaceBuilt task class."""

    def test_directory_exists(self):
        """Test PythonWorkspaceBuilt validation passes when directory exists"""
        from hdr.tasks.coding import PythonWorkspaceBuilt

        with tempfile.TemporaryDirectory() as tmpdir:
            d = PythonWorkspaceBuilt(path=tmpdir)
            assert d.path == tmpdir

    def test_directory_not_exists(self):
        """Test PythonWorkspaceBuilt validation fails when directory does not exist"""
        from hdr.tasks.coding import PythonWorkspaceBuilt

        with pytest.raises(AssertionError, match="does not exist"):
            PythonWorkspaceBuilt(path="/nonexistent/path/12345")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
