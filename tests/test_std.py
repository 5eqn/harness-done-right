"""
Tests for Directory task class.
Uses a fixed temp workspace created via tempfile.mkdtemp.
"""

import os
import tempfile
import pytest
from hdr.tasks.std import Directory

# Enable mock mode for all tests
import hdr
hdr.set_mock_mode(True)


class TestDirectory:
    """Tests for Directory task class."""

    def test_directory_exists(self):
        """Test Directory validation passes when directory exists and exists=True"""
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Directory(path=tmpdir, exists=True)
            assert d.path == tmpdir
            assert d.exists is True

    def test_directory_not_exists(self):
        """Test Directory validation fails when directory does not exist and exists=True"""
        with pytest.raises(AssertionError, match="does not exist"):
            Directory(path="/nonexistent/path/12345", exists=True)

    def test_directory_exists_false_when_missing(self):
        """Test Directory validation passes when directory does not exist and exists=False"""
        d = Directory(path="/nonexistent/path/12345", exists=False)
        assert d.path == "/nonexistent/path/12345"
        assert d.exists is False

    def test_directory_exists_false_when_present(self):
        """Test Directory validation fails when directory exists but exists=False"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(AssertionError, match="should not exist"):
                Directory(path=tmpdir, exists=False)

    def test_directory_default_exists_true(self):
        """Test Directory defaults to exists=True"""
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Directory(path=tmpdir)
            assert d.exists is True

    def test_directory_with_files(self):
        """Test Directory validates that all listed files exist"""
        with tempfile.TemporaryDirectory() as tmpdir:
            subfile = os.path.join(tmpdir, "sample.py")
            with open(subfile, "w") as f:
                f.write("# sample")
            d = Directory(path=tmpdir, files=["sample.py"])
            assert d.path == tmpdir
            assert d.files == ["sample.py"]

    def test_directory_with_missing_file(self):
        """Test Directory raises when listed file does not exist"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(AssertionError, match="does not exist"):
                Directory(path=tmpdir, files=["nonexistent.py"])

    def test_directory_nested_files(self):
        """Test Directory validates nested files (relative path)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = os.path.join(tmpdir, "sub")
            os.makedirs(nested_dir)
            nested_file = os.path.join(nested_dir, "nested.py")
            with open(nested_file, "w") as f:
                f.write("# nested")
            d = Directory(path=tmpdir, files=["sub/nested.py"])
            assert d.files == ["sub/nested.py"]

    def test_directory_multiple_files(self):
        """Test Directory validates multiple files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_a = os.path.join(tmpdir, "a.py")
            file_b = os.path.join(tmpdir, "b.py")
            with open(file_a, "w") as f:
                f.write("# a")
            with open(file_b, "w") as f:
                f.write("# b")
            d = Directory(path=tmpdir, files=["a.py", "b.py"])
            assert d.files == ["a.py", "b.py"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])