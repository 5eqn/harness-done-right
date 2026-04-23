"""
Reveal.js contract for previewing markdown slide decks.
"""

from __future__ import annotations

import html
import socket
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from hdr.contracts.coding import MarkdownFile
from hdr.contracts.std import BaseContract


class Reveal(BaseContract):
    """
    A reveal.js deck backed by a single markdown file.
    """

    markdown: MarkdownFile

    def write_html(self, output_path: str | Path | None = None) -> Path:
        """
        Render the markdown deck to a standalone reveal.js HTML file.
        """
        output = self._resolve_output_path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(self.build_html(), encoding="utf-8")
        return output

    def build_html(self) -> str:
        """
        Build the reveal.js HTML document for this markdown deck.
        """
        metadata, body = self._split_frontmatter(self.markdown.content)
        title = metadata.get("title", Path(self.markdown.path).stem)
        author = metadata.get("author", "")
        escaped_markdown = html.escape(body, quote=False)

        return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{html.escape(title)}</title>
  <meta name="author" content="{html.escape(author)}">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/reveal.js@5/dist/reset.css">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/reveal.js@5/dist/reveal.css">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/reveal.js@5/dist/theme/white.css">
  <style>
    :root {{
      --r-main-font: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      --r-heading-font: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      --r-heading-text-transform: none;
      --r-main-color: #172026;
      --r-heading-color: #0e1a21;
      --r-link-color: #126b84;
      --r-selection-background-color: #d9edf2;
    }}
    .reveal {{
      font-size: 31px;
    }}
    .reveal h1,
    .reveal h2,
    .reveal h3 {{
      letter-spacing: 0;
      line-height: 1.12;
    }}
    .reveal h1 {{
      font-size: 2.05em;
    }}
    .reveal h2 {{
      font-size: 1.45em;
      margin-bottom: 0.65em;
    }}
    .reveal p,
    .reveal li {{
      line-height: 1.45;
    }}
    .reveal ul,
    .reveal ol {{
      margin-left: 1.25em;
    }}
    .reveal table {{
      font-size: 0.68em;
      width: 100%;
    }}
    .reveal th,
    .reveal td {{
      padding: 0.38em 0.5em;
      vertical-align: top;
    }}
    .reveal code {{
      color: #8a3b12;
      background: #f4f0eb;
      padding: 0.05em 0.22em;
      border-radius: 5px;
    }}
    .reveal pre code {{
      padding: 1em;
      border-radius: 8px;
    }}
    .reveal blockquote {{
      border-left: 6px solid #126b84;
      box-shadow: none;
      color: #26363d;
      font-size: 1em;
      padding: 0.25em 0 0.25em 0.8em;
      width: 86%;
    }}
  </style>
</head>
<body>
  <div class="reveal">
    <div class="slides">
      <section data-markdown data-separator="^---$" data-separator-vertical="^--$">
        <textarea data-template>
{escaped_markdown}
        </textarea>
      </section>
    </div>
  </div>
  <script src="https://cdn.jsdelivr.net/npm/reveal.js@5/dist/reveal.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/reveal.js@5/plugin/markdown/markdown.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/reveal.js@5/plugin/highlight/highlight.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/reveal.js@5/plugin/notes/notes.js"></script>
  <script>
    Reveal.initialize({{
      hash: true,
      slideNumber: true,
      transition: "slide",
      width: 1280,
      height: 720,
      margin: 0.08,
      plugins: [RevealMarkdown, RevealHighlight, RevealNotes]
    }});
  </script>
</body>
</html>
"""

    def host(
        self,
        output_path: str | Path | None = None,
        *,
        bind: str = "127.0.0.1",
        port: int = 8000,
    ) -> None:
        """
        Render the deck and serve it with Python's http.server until interrupted.
        """
        started_at = time.perf_counter()
        output = self.write_html(output_path)
        directory = output.parent
        handler = partial(SimpleHTTPRequestHandler, directory=str(directory))

        with ThreadingHTTPServer((bind, port), handler) as server:
            elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            local_url = self._local_url(bind, server.server_port, output.name)
            network_url = self._network_url(bind, server.server_port, output.name)
            self._print_host_banner(local_url, network_url, elapsed_ms)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                print("\nHDR Reveal server stopped.")

    def _resolve_output_path(self, output_path: str | Path | None) -> Path:
        if output_path is not None:
            return Path(output_path)
        return Path(self.markdown.path).with_suffix(".html")

    @staticmethod
    def _split_frontmatter(markdown: str) -> tuple[dict[str, str], str]:
        if not markdown.startswith("---\n"):
            return {}, markdown

        parts = markdown.split("---\n", 2)
        if len(parts) != 3:
            return {}, markdown

        metadata: dict[str, str] = {}
        for line in parts[1].splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip()
        return metadata, parts[2].lstrip()

    @staticmethod
    def _local_url(bind: str, port: int, filename: str) -> str:
        host = "localhost" if bind in {"", "0.0.0.0", "::"} else bind
        return f"http://{host}:{port}/{filename}"

    @staticmethod
    def _network_url(bind: str, port: int, filename: str) -> str | None:
        if bind not in {"", "0.0.0.0", "::"}:
            return None

        try:
            host = socket.gethostbyname(socket.gethostname())
        except OSError:
            return None
        if host.startswith("127."):
            return None
        return f"http://{host}:{port}/{filename}"

    @staticmethod
    def _print_host_banner(
        local_url: str, network_url: str | None, elapsed_ms: int
    ) -> None:
        print()
        print(f"  HDR Reveal ready in {elapsed_ms} ms")
        print()
        print(f"  -> Local:   {local_url}")
        if network_url is None:
            print("  -> Network: use --bind 0.0.0.0 to expose")
        else:
            print(f"  -> Network: {network_url}")
        print()
        print("  Press Ctrl+C to stop.")
