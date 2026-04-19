# Academic Report Reveal Deck

This directory contains a reveal.js deck generated from
`academic_report_reveal.md`.

## Generate HTML

Run from this directory:

```bash
uv run python build_reveal.py
```

This writes `academic_report_reveal.html`.

## Preview

Run from this directory:

```bash
uv run python -m http.server 8000
```

Then open:

```text
http://localhost:8000/academic_report_reveal.html
```

The generated HTML loads reveal.js from the jsDelivr CDN, so the browser needs
network access while previewing.

