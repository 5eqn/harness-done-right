from __future__ import annotations

import argparse
import html
from pathlib import Path


DEFAULT_INPUT = Path(__file__).with_name("academic_report_reveal.md")
DEFAULT_OUTPUT = Path(__file__).with_name("academic_report_reveal.html")


def split_frontmatter(markdown: str) -> tuple[dict[str, str], str]:
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


def build_html(markdown_path: Path) -> str:
    metadata, body = split_frontmatter(markdown_path.read_text(encoding="utf-8"))
    title = metadata.get("title", markdown_path.stem)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a reveal.js HTML deck.")
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Markdown input path. Defaults to {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"HTML output path. Defaults to {DEFAULT_OUTPUT}",
    )
    args = parser.parse_args()

    args.output.write_text(build_html(args.input), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
