"""
Tests for Directory contract class.
Uses a fixed temp workspace created via tempfile.mkdtemp.
"""

import tempfile
import pytest
from pydantic import ValidationError
from hdr.tasks.std import Directory, File
from hdr.tasks.coding import MarkdownFile


class TestFile:
    """Tests for File contract class."""

    def test_file_exists(self):
        """Test File validation passes when file exists"""
        with tempfile.TemporaryDirectory() as tmpdir:
            import os

            file_path = os.path.join(tmpdir, "test.txt")
            with open(file_path, "w") as f:
                f.write("hello")
            f = File(path=file_path)
            assert f.path == file_path
            assert f.content == "hello"

    def test_content_is_frozen_after_read(self):
        """Test File content cannot be manually assigned after init."""
        with tempfile.TemporaryDirectory() as tmpdir:
            import os

            file_path = os.path.join(tmpdir, "test.txt")
            with open(file_path, "w") as f:
                f.write("hello")
            f = File(path=file_path)
            with pytest.raises(ValidationError, match="Field is frozen"):
                f.content = "manual"

    def test_file_not_exists(self):
        """Test File validation fails when file does not exist"""
        with pytest.raises(AssertionError, match="does not exist"):
            File(path="/nonexistent/path/12345.txt")


class TestDirectory:
    """Tests for Directory contract class."""

    def test_directory_exists(self):
        """Test Directory validation passes when directory exists"""
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Directory(path=tmpdir)
            assert d.path == tmpdir

    def test_directory_not_exists(self):
        """Test Directory validation fails when directory does not exist"""
        with pytest.raises(AssertionError, match="does not exist"):
            Directory(path="/nonexistent/path/12345")

    def test_directory_with_content(self):
        """Test Directory accepts explicit content"""
        with tempfile.TemporaryDirectory() as tmpdir:
            import os

            # Create a file to use with explicit content
            file_path = os.path.join(tmpdir, "a.txt")
            with open(file_path, "w") as f:
                f.write("custom content")
            file = File(path=file_path)
            d = Directory(
                path=tmpdir,
                content=[file],
            )
            assert d.path == tmpdir
            assert len(d.content) == 1
            assert d.content[0].content == "custom content"

    def test_directory_auto_gathers_content(self):
        """Test Directory auto-fills content from files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            import os

            # Create some files
            with open(os.path.join(tmpdir, "a.txt"), "w") as f:
                f.write("content a")
            with open(os.path.join(tmpdir, "b.txt"), "w") as f:
                f.write("content b")
            d = Directory(path=tmpdir)
            assert len(d.content) == 2
            contents = [f.content for f in d.content]
            assert "content a" in contents
            assert "content b" in contents

    def test_directory_respects_nested_gitignore(self):
        """Test Directory applies .gitignore files in nested directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            import os

            nested = os.path.join(tmpdir, "nested")
            os.mkdir(nested)
            with open(os.path.join(nested, ".gitignore"), "w") as f:
                f.write("secret.txt\n")
            with open(os.path.join(nested, "secret.txt"), "w") as f:
                f.write("secret")
            with open(os.path.join(nested, "keep.txt"), "w") as f:
                f.write("keep")

            d = Directory(path=tmpdir)

            paths = {os.path.relpath(f.path, tmpdir) for f in d.content}
            assert "nested/keep.txt" in paths
            assert "nested/secret.txt" not in paths


class TestMarkdownFile:
    """Tests for MarkdownFile contract class."""

    def test_md_file_exists(self):
        """Test MarkdownFile validation passes when file exists"""
        with tempfile.TemporaryDirectory() as tmpdir:
            import os

            file_path = os.path.join(tmpdir, "test.md")
            with open(file_path, "w") as f:
                f.write("# Hello\n")
            f = MarkdownFile(path=file_path)
            assert f.path == file_path
            assert f.content == "# Hello\n"

    def test_md_file_must_end_with_md(self):
        """Test MarkdownFile fails when path does not end with .md"""
        with tempfile.TemporaryDirectory() as tmpdir:
            import os

            file_path = os.path.join(tmpdir, "test.txt")
            with open(file_path, "w") as f:
                f.write("hello")
            with pytest.raises(AssertionError, match=r".*\.md.*"):
                MarkdownFile(path=file_path)

    def test_md_file_not_exists(self):
        """Test MarkdownFile fails when file does not exist"""
        with pytest.raises(AssertionError, match="does not exist"):
            MarkdownFile(path="/nonexistent/path/12345.md")

    def test_md_file_invalid_syntax(self):
        """Test MarkdownFile passes when markdownlint finds no issues"""
        with tempfile.TemporaryDirectory() as tmpdir:
            import os

            file_path = os.path.join(tmpdir, "valid.md")
            with open(file_path, "w") as f:
                f.write("# Valid Markdown\n\nThis is fine.\n")
            f = MarkdownFile(path=file_path)
            assert f.path == file_path


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
