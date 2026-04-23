# Reveal Contracts

Reveal contract types in `hdr.contracts.reveal` for hosting markdown slide decks
as reveal.js presentations.

## Reveal

Wraps a single `MarkdownFile` and hosts it as a reveal.js deck without writing a
generated HTML file to the project.

**Fields:**
- `markdown: MarkdownFile` - Markdown source file for the deck.

**Key methods:**
- `host(*, bind: str = "127.0.0.1", port: int = 8000) -> None` - Serves the
  generated reveal.js HTML from memory with Python's `http.server` until
  interrupted.

**Example:**
```python
from hdr.contracts.coding import MarkdownFile
from hdr.contracts.reveal import Reveal

deck = Reveal(markdown=MarkdownFile(path="slides.md"))
deck.host()
```
