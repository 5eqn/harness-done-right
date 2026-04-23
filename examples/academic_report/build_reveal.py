from __future__ import annotations

import argparse
from pathlib import Path

from hdr.contracts.coding import MarkdownFile
from hdr.contracts.reveal import Reveal


DEFAULT_INPUT = Path(__file__).with_name("academic_report_reveal.md")
DEFAULT_OUTPUT = Path(__file__).with_name("academic_report_reveal.html")


def build_html(markdown_path: Path) -> str:
    return Reveal(markdown=MarkdownFile(path=str(markdown_path))).build_html()


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
    parser.add_argument(
        "--host",
        action="store_true",
        help="Serve the generated reveal.js deck with Python's http.server.",
    )
    parser.add_argument(
        "--bind",
        default="127.0.0.1",
        help="Interface to bind when hosting. Use 0.0.0.0 for network access.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind when hosting.",
    )
    args = parser.parse_args()

    reveal = Reveal(markdown=MarkdownFile(path=str(args.input)))
    if args.host:
        reveal.host(args.output, bind=args.bind, port=args.port)
    else:
        output = reveal.write_html(args.output)
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
