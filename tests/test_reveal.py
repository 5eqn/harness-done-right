import tempfile
from pathlib import Path

from hdr.contracts.coding import MarkdownFile
from hdr.contracts.reveal import Reveal


def _write_deck(tmpdir: str, content: str) -> Path:
    path = Path(tmpdir) / "deck.md"
    path.write_text(content, encoding="utf-8")
    return path


class TestReveal:
    def test_reveal_contains_markdown_file_only(self):
        assert list(Reveal.model_fields) == ["markdown"]

    def test_build_html_uses_frontmatter_and_body(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            markdown_path = _write_deck(
                tmpdir,
                "---\ntitle: Demo Deck\nauthor: Ada\n---\n\n## Hello & HDR\n",
            )

            reveal = Reveal(markdown=MarkdownFile(path=str(markdown_path)))
            html = reveal.build_html()

            assert "<title>Demo Deck</title>" in html
            assert '<meta name="author" content="Ada">' in html
            assert "## Hello &amp; HDR" in html
            assert "title: Demo Deck" not in html

    def test_write_html_defaults_to_markdown_sibling(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            markdown_path = _write_deck(tmpdir, "# Deck\n")

            reveal = Reveal(markdown=MarkdownFile(path=str(markdown_path)))
            output = reveal.write_html()

            assert output == markdown_path.with_suffix(".html")
            assert output.exists()
            assert "Reveal.initialize" in output.read_text(encoding="utf-8")

    def test_write_html_accepts_explicit_output_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            markdown_path = _write_deck(tmpdir, "# Deck\n")
            output_path = Path(tmpdir) / "public" / "slides.html"

            reveal = Reveal(markdown=MarkdownFile(path=str(markdown_path)))
            output = reveal.write_html(output_path)

            assert output == output_path
            assert output_path.exists()
