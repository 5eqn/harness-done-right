"""
Tests for Directory task class.
Uses a fixed temp workspace created via tempfile.mkdtemp.
"""

import tempfile
import pytest
from hdr.tasks.std import Directory, File

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
        """Test Directory accepts File instances in files list"""
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Directory(path=tmpdir, files=[File(path="main.py", exists=False)])
            assert d.path == tmpdir
            assert len(d.files) == 1
            assert d.files[0].path == "main.py"

    def test_directory_nested_files(self):
        """Test Directory accepts File instances with nested paths"""
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Directory(path=tmpdir, files=[File(path="sub/nested.py", exists=False)])
            assert len(d.files) == 1
            assert d.files[0].path == "sub/nested.py"

    def test_directory_multiple_files(self):
        """Test Directory accepts multiple File instances"""
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Directory(path=tmpdir, files=[File(path="a.py", exists=False), File(path="b.py", exists=False)])
            assert len(d.files) == 2
            assert d.files[0].path == "a.py"
            assert d.files[1].path == "b.py"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])