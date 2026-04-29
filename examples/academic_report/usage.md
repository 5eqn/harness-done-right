# Academic Report Reveal Deck

This directory contains a reveal.js deck hosted from `academic_report_reveal.md`.

## Preview

Run from this directory:

```bash
uv run python build_reveal.py
```

The `Reveal.host()` API prints local preview information, including the URL:

```text
http://127.0.0.1:8000/
```

The HTML is generated in memory and loads reveal.js from the jsDelivr CDN, so the
browser needs network access while previewing.
