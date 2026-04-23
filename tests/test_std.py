"""
Tests for Directory contract class.
Uses a fixed temp workspace created via tempfile.mkdtemp.
"""

import tempfile
import pytest
from pydantic import ValidationError
from hdr.contracts.std import Directory, File, Image
from hdr.contracts.coding import MarkdownFile


class TestFile:
    """Tests for File contract class."""

    def test_file_exists(self):
        """Test File validation passes when file exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            import os

            file_path = os.path.join(tmpdir, "test.txt")
            with open(file_path, "w") as f:
                f.write("hello")
            f = File(path=file_path)
            assert f.path == file_path
            assert f.content == "hello"

    def test_file_enforces_manual_content_when_provided(self):
        """Test File validates manual content against disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            import os

            file_path = os.path.join(tmpdir, "test.txt")
            with open(file_path, "w") as f:
                f.write("hello")

            f = File(path=file_path, content="hello")
            assert f.content == "hello"

    def test_file_rejects_mismatched_manual_content(self):
        """Test File rejects manual content that differs from disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            import os

            file_path = os.path.join(tmpdir, "test.txt")
            with open(file_path, "w") as f:
                f.write("hello")

            with pytest.raises(ValidationError, match="does not match"):
                File(path=file_path, content="goodbye")

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
        """Test Directory validation passes when explicit content matches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Directory(path=tmpdir, content=[])
            assert d.path == tmpdir

    def test_directory_not_exists(self):
        """Test Directory validation fails when directory does not exist"""
        with pytest.raises(AssertionError, match="does not exist"):
            Directory(path="/nonexistent/path/12345", content=[])

    def test_directory_requires_manual_content(self):
        """Test Directory rejects omitted content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(AssertionError, match="manually assigned"):
                Directory(path=tmpdir)

    def test_directory_with_content(self):
        """Test Directory accepts explicit content when it matches exactly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            import os

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

    def test_directory_rejects_missing_file_from_content(self):
        """Test Directory rejects content that omits a tracked file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            import os

            with open(os.path.join(tmpdir, "a.txt"), "w") as f:
                f.write("content a")
            with open(os.path.join(tmpdir, "b.txt"), "w") as f:
                f.write("content b")

            with pytest.raises(AssertionError, match="missing files: b.txt"):
                Directory(
                    path=tmpdir,
                    content=[File(path=os.path.join(tmpdir, "a.txt"))],
                )

    def test_directory_rejects_extra_file_in_content(self):
        """Test Directory rejects content that includes a file outside the directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            import os

            with open(os.path.join(tmpdir, "a.txt"), "w") as f:
                f.write("content a")

            with tempfile.NamedTemporaryFile(suffix=".txt") as external_file:
                external_file.write(b"external")
                external_file.flush()

                with pytest.raises(AssertionError, match="is not inside"):
                    Directory(
                        path=tmpdir,
                        content=[
                            File(path=os.path.join(tmpdir, "a.txt")),
                            File(path=external_file.name),
                        ],
                    )

    def test_directory_checks_only_immediate_files(self):
        """Test Directory ignores nested files because subdirectories validate themselves."""
        with tempfile.TemporaryDirectory() as tmpdir:
            import os

            nested = os.path.join(tmpdir, "nested")
            os.mkdir(nested)
            with open(os.path.join(nested, "keep.txt"), "w") as f:
                f.write("keep")

            d = Directory(
                path=tmpdir,
                content=[],
            )

            assert d.content == []

    def test_directory_respects_current_directory_gitignore(self):
        """Test Directory applies the current directory's .gitignore."""
        with tempfile.TemporaryDirectory() as tmpdir:
            import os

            with open(os.path.join(tmpdir, ".gitignore"), "w") as f:
                f.write("secret.txt\n")
            with open(os.path.join(tmpdir, "secret.txt"), "w") as f:
                f.write("secret")
            with open(os.path.join(tmpdir, "keep.txt"), "w") as f:
                f.write("keep")

            d = Directory(
                path=tmpdir,
                content=[
                    File(path=os.path.join(tmpdir, ".gitignore")),
                    File(path=os.path.join(tmpdir, "keep.txt")),
                ],
            )

            paths = {os.path.relpath(f.path, tmpdir) for f in d.content}
            assert ".gitignore" in paths
            assert "keep.txt" in paths
            assert "secret.txt" not in paths


class TestImage:
    """Tests for Image contract class."""

    def test_svg_image_reads_content_and_dimensions(self):
        """Test SVG images expose text content and dimensions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            import os

            image_path = os.path.join(tmpdir, "diagram.svg")
            with open(image_path, "w") as f:
                f.write(
                    '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="360">'
                    '<rect width="640" height="360" />'
                    "</svg>"
                )

            image = Image(path=image_path)

            assert image.path == image_path
            assert image.media_type == "image/svg+xml"
            assert image.is_vector is True
            assert image.width == 640
            assert image.height == 360
            assert image.content.startswith("<svg")
            assert image.size_bytes > 0

    def test_svg_image_uses_viewbox_dimensions(self):
        """Test SVG dimensions can be inferred from viewBox."""
        with tempfile.TemporaryDirectory() as tmpdir:
            import os

            image_path = os.path.join(tmpdir, "diagram.svg")
            with open(image_path, "w") as f:
                f.write(
                    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 450">'
                    "</svg>"
                )

            image = Image(path=image_path)

            assert image.width == 800
            assert image.height == 450

    def test_png_image_reads_metadata_without_text_content(self):
        """Test bitmap images expose metadata without reading binary as text."""
        with tempfile.TemporaryDirectory() as tmpdir:
            import os

            image_path = os.path.join(tmpdir, "diagram.png")
            png_header = (
                b"\x89PNG\r\n\x1a\n"
                b"\x00\x00\x00\r"
                b"IHDR"
                + (320).to_bytes(4, "big")
                + (180).to_bytes(4, "big")
                + b"\x08\x02\x00\x00\x00"
            )
            with open(image_path, "wb") as f:
                f.write(png_header)

            image = Image(path=image_path)

            assert image.media_type == "image/png"
            assert image.is_vector is False
            assert image.width == 320
            assert image.height == 180
            assert image.content == ""
            assert image.size_bytes == len(png_header)

    def test_image_rejects_unsupported_extension(self):
        """Test Image fails when extension is not an image type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            import os

            image_path = os.path.join(tmpdir, "diagram.txt")
            with open(image_path, "w") as f:
                f.write("not an image")

            with pytest.raises(AssertionError, match="Image path must end"):
                Image(path=image_path)

    def test_image_metadata_is_frozen(self):
        """Test Image auto-filled metadata cannot be manually assigned."""
        with tempfile.TemporaryDirectory() as tmpdir:
            import os

            image_path = os.path.join(tmpdir, "diagram.svg")
            with open(image_path, "w") as f:
                f.write('<svg xmlns="http://www.w3.org/2000/svg"></svg>')

            image = Image(path=image_path)

            with pytest.raises(ValidationError, match="Field is frozen"):
                image.media_type = "image/png"


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
