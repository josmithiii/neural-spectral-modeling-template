# Presentation Assets (Archive)

This directory contains LaTeX sources and figures for long-form slide decks created during the early LHTE design phase. Treat it as a reference archive:

- `LHTE.tex`, `multihead.tex`, supporting `.sty` files – full conference-style deck.
- `NSMT-contents.tex`, `std...` files – legacy Neural Spectral Modeling Template material.
- `fig/`, `eps/`, `pdf/` – exported graphics used by the decks.

## When to Use
- Rebuilding or updating the original slide deck for conferences/meetups.
- Extracting vector figures for docs/blog posts.

## Quick Start
```bash
# Compile the main deck (requires latexmk, powerdot, etc.)
cd docs/presentation
make
```

> ⚠️ Dependencies: TeX Live + Powerdot packages. Builds are intentionally out of scope for everyday workflows.

## Prefer a Shorter Overview?
Consult [../vision_primer.md](../vision_primer.md) for a concise, Markdown-based briefing aimed at image-processing enthusiasts.
