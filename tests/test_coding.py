"""
Tests for Coding contracts (PythonFile, PythonWorkspace).
Uses a fixed temp workspace created via tempfile.mkdtemp.
"""

import os
import subprocess
import tempfile

import pytest


class TestPythonFile:
    """Tests for PythonFile contract class."""

    def test_py_file_not_exists(self):
        """Test PythonFile fails when file does not exist"""
        from hdr.contracts.coding import PythonFile

        with pytest.raises(AssertionError, match="does not exist"):
            PythonFile(path="/nonexistent/path/12345.py")

    def test_py_file_must_end_with_py(self):
        """Test PythonFile fails when path does not end with .py"""
        from hdr.contracts.coding import PythonFile

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test.txt")
            with open(file_path, "w") as f:
                f.write("hello")
            with pytest.raises(AssertionError, match=r".*\.py.*"):
                PythonFile(path=file_path)

    def test_valid_py_file(self):
        """Test PythonFile passes for a clean Python file"""
        from hdr.contracts.coding import PythonFile

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "hello.py")
            with open(file_path, "w") as f:
                f.write('name = "world"\nprint(f"Hello, {name}!")\n')
            f = PythonFile(path=file_path)
            assert f.path == file_path
            assert 'name = "world"\nprint(f"Hello, {name}!")\n' in f.content

    def test_py_file_content_reflects_formatted_file(self):
        """Test PythonFile content is read after ruff format mutates disk."""
        from hdr.contracts.coding import PythonFile

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "formatted.py")
            with open(file_path, "w") as f:
                f.write("x=1\n")
            f = PythonFile(path=file_path)
            assert f.content == "x = 1\n"


class TestMarkdownFile:
    """Tests for MarkdownFile contract class."""

    def test_markdown_runs_non_fix_lint_after_format(self, monkeypatch):
        """Test MarkdownFile runs a plain markdownlint pass after --fix."""
        from hdr.contracts.coding import MarkdownFile

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
            MarkdownFile(path=file_path)

        assert calls == [
            ["markdownlint-cli2", "--fix", file_path],
            ["markdownlint-cli2", file_path],
        ]


class TestPythonWorkspace:
    """Tests for PythonWorkspace contract class."""

    def test_directory_exists(self):
        """Test PythonWorkspace validation passes when content is explicit."""
        from hdr.contracts.coding import PythonWorkspace

        with tempfile.TemporaryDirectory() as tmpdir:
            d = PythonWorkspace(path=tmpdir, content=[])
            assert d.path == tmpdir

    def test_directory_not_exists(self):
        """Test PythonWorkspace validation fails when directory does not exist"""
        from hdr.contracts.coding import PythonWorkspace

        with pytest.raises(AssertionError, match="does not exist"):
            PythonWorkspace(path="/nonexistent/path/12345", content=[])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
