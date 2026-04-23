from __future__ import annotations

from pathlib import Path

from hdr.contracts.coding import MarkdownFile
from hdr.contracts.reveal import Reveal


def main() -> None:
    markdown = MarkdownFile(
        path=str(Path(__file__).with_name("academic_report_reveal.md"))
    )
    Reveal(markdown=markdown).host()


if __name__ == "__main__":
    main()
