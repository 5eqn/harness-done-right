# Reveal Contracts

Reveal contract types in `hdr.contracts.reveal` for turning markdown slide decks
into reveal.js HTML and previewing them locally.

## Reveal

Wraps a single `MarkdownFile` and provides methods for rendering or hosting it
as a reveal.js deck.

**Fields:**
- `markdown: MarkdownFile` - Markdown source file for the deck.

**Key methods:**
- `build_html() -> str` - Returns the generated reveal.js HTML.
- `write_html(output_path: str | Path | None = None) -> Path` - Writes the HTML
  deck. Defaults to the markdown file's sibling `.html` path.
- `host(output_path: str | Path | None = None, *, bind: str = "127.0.0.1", port: int = 8000) -> None` - Writes the HTML and serves its directory with Python's `http.server` until interrupted.

**Example:**
```python
from hdr.contracts.coding import MarkdownFile
from hdr.contracts.reveal import Reveal

deck = Reveal(markdown=MarkdownFile(path="slides.md"))
deck.write_html()
deck.host(port=8000)
```
