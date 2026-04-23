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
            html = reveal._build_html()

            assert "<title>Demo Deck</title>" in html
            assert '<meta name="author" content="Ada">' in html
            assert "## Hello &amp; HDR" in html
            assert "title: Demo Deck" not in html

    def test_request_handler_serves_html_from_memory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            markdown_path = _write_deck(tmpdir, "# Deck\n")

            reveal = Reveal(markdown=MarkdownFile(path=str(markdown_path)))
            handler = reveal._request_handler(reveal._build_html())

            assert handler.__name__ == "RevealRequestHandler"

    def test_reveal_does_not_expose_write_or_build_html_api(self):
        assert not hasattr(Reveal, "write_html")
        assert not hasattr(Reveal, "build_html")
