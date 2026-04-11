"""
Tests for Directory task class.
Uses a fixed temp workspace created via tempfile.mkdtemp.
"""

import tempfile
import pytest
from pydantic import ValidationError
from hdr.tasks.std import DirectoryCreated, FileWritten
from hdr.tasks.coding import MarkdownFileWritten


class TestFileWritten:
    """Tests for FileWritten task class."""

    def test_file_exists(self):
        """Test FileWritten validation passes when file exists"""
        with tempfile.TemporaryDirectory() as tmpdir:
            import os

            file_path = os.path.join(tmpdir, "test.txt")
            with open(file_path, "w") as f:
                f.write("hello")
            f = FileWritten(path=file_path)
            assert f.path == file_path
            assert f.content == "hello"

    def test_content_is_frozen_after_read(self):
        """Test FileWritten content cannot be manually assigned after init."""
        with tempfile.TemporaryDirectory() as tmpdir:
            import os

            file_path = os.path.join(tmpdir, "test.txt")
            with open(file_path, "w") as f:
                f.write("hello")
            f = FileWritten(path=file_path)
            with pytest.raises(ValidationError, match="Field is frozen"):
                f.content = "manual"

    def test_file_not_exists(self):
        """Test FileWritten validation fails when file does not exist"""
        with pytest.raises(AssertionError, match="does not exist"):
            FileWritten(path="/nonexistent/path/12345.txt")


class TestDirectoryCreated:
    """Tests for DirectoryCreated task class."""

    def test_directory_exists(self):
        """Test DirectoryCreated validation passes when directory exists"""
        with tempfile.TemporaryDirectory() as tmpdir:
            d = DirectoryCreated(path=tmpdir)
            assert d.path == tmpdir

    def test_directory_not_exists(self):
        """Test DirectoryCreated validation fails when directory does not exist"""
        with pytest.raises(AssertionError, match="does not exist"):
            DirectoryCreated(path="/nonexistent/path/12345")

    def test_directory_with_content(self):
        """Test DirectoryCreated accepts explicit content"""
        with tempfile.TemporaryDirectory() as tmpdir:
            import os

            # Create a file to use with explicit content
            file_path = os.path.join(tmpdir, "a.txt")
            with open(file_path, "w") as f:
                f.write("custom content")
            file = FileWritten(path=file_path)
            d = DirectoryCreated(
                path=tmpdir,
                content=[file],
            )
            assert d.path == tmpdir
            assert len(d.content) == 1
            assert d.content[0].content == "custom content"

    def test_directory_auto_gathers_content(self):
        """Test DirectoryCreated auto-fills content from files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            import os

            # Create some files
            with open(os.path.join(tmpdir, "a.txt"), "w") as f:
                f.write("content a")
            with open(os.path.join(tmpdir, "b.txt"), "w") as f:
                f.write("content b")
            d = DirectoryCreated(path=tmpdir)
            assert len(d.content) == 2
            contents = [f.content for f in d.content]
            assert "content a" in contents
            assert "content b" in contents

    def test_directory_respects_nested_gitignore(self):
        """Test DirectoryCreated applies .gitignore files in nested directories."""
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

            d = DirectoryCreated(path=tmpdir)

            paths = {os.path.relpath(f.path, tmpdir) for f in d.content}
            assert "nested/keep.txt" in paths
            assert "nested/secret.txt" not in paths


class TestMarkdownFileWritten:
    """Tests for MarkdownFileWritten task class."""

    def test_md_file_exists(self):
        """Test MarkdownFileWritten validation passes when file exists"""
        with tempfile.TemporaryDirectory() as tmpdir:
            import os

            file_path = os.path.join(tmpdir, "test.md")
            with open(file_path, "w") as f:
                f.write("# Hello\n")
            f = MarkdownFileWritten(path=file_path)
            assert f.path == file_path
            assert f.content == "# Hello\n"

    def test_md_file_must_end_with_md(self):
        """Test MarkdownFileWritten fails when path does not end with .md"""
        with tempfile.TemporaryDirectory() as tmpdir:
            import os

            file_path = os.path.join(tmpdir, "test.txt")
            with open(file_path, "w") as f:
                f.write("hello")
            with pytest.raises(AssertionError, match=r".*\.md.*"):
                MarkdownFileWritten(path=file_path)

    def test_md_file_not_exists(self):
        """Test MarkdownFileWritten fails when file does not exist"""
        with pytest.raises(AssertionError, match="does not exist"):
            MarkdownFileWritten(path="/nonexistent/path/12345.md")

    def test_md_file_invalid_syntax(self):
        """Test MarkdownFileWritten passes when markdownlint finds no issues"""
        with tempfile.TemporaryDirectory() as tmpdir:
            import os

            file_path = os.path.join(tmpdir, "valid.md")
            with open(file_path, "w") as f:
                f.write("# Valid Markdown\n\nThis is fine.\n")
            f = MarkdownFileWritten(path=file_path)
            assert f.path == file_path


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
