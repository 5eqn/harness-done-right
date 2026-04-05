"""
Tests for Directory task class.
Uses a fixed temp workspace created via tempfile.mkdtemp.
"""

import tempfile
import pytest
from hdr.tasks.std import DirectoryCreated, FileWritten


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

    def test_file_not_exists(self):
        """Test FileWritten validation fails when file does not exist"""
        with pytest.raises(AssertionError, match="does not exist"):
            FileWritten(path="/nonexistent/path/12345.txt")

    def test_file_with_content(self):
        """Test FileWritten accepts explicit content"""
        with tempfile.TemporaryDirectory() as tmpdir:
            import os

            file_path = os.path.join(tmpdir, "test.txt")
            with open(file_path, "w") as f:
                f.write("original")
            f = FileWritten(path=file_path, content="custom content")
            assert f.path == file_path
            assert f.content == "custom content"


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
            d = DirectoryCreated(
                path=tmpdir,
                content=[FileWritten(path=file_path, content="custom content")],
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
