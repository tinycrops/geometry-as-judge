# AGENTS.md

## Purpose

`/home/ath/writing` is the publication layer for **Geometry as Judge**.

This directory builds the static site published at:

- `tinycrops.github.io/geometry-as-judge`

The source research repo is `/home/ath/experiments`. This repo should reflect
that work accurately, not diverge from it.

## Source Of Truth

- Experimental logic, metrics, raw results, and primary findings live in
  `/home/ath/experiments`.
- Website structure, narrative framing, generated figures, and static HTML live
  here.
- If there is a conflict between prose here and artifacts in `experiments`,
  reconcile the discrepancy rather than papering over it.

## Generator-First Workflow

- `generate.py` is the main authoring surface.
- Prefer changing `generate.py` and regenerating the site instead of manually
  editing individual HTML pages.
- Navigation is defined in `generate.py` via `nav()`. If pages are added,
  removed, or renamed, update nav there and regenerate all affected pages.
- `generate.py` also copies selected figures from `/home/ath/experiments`, so
  keep cross-repo paths valid when changing filenames.

## Site Conventions

The current site style is intentional and should stay consistent unless the
task calls for a redesign:

- `system-ui` body font
- ~860px content width
- dark sticky nav
- callout boxes
- dark code blocks
- figure-heavy longform pages generated from Python

Preserve the existing tone: technical, explanatory, and research-forward.

## Editing Rules

- Do not hand-edit generated HTML unless there is a compelling one-off reason.
  In normal work, edit `generate.py`, rerun it, and let the HTML be derived.
- Keep filenames and URLs stable when possible because this repo is a published
  GitHub Pages site.
- When introducing a new result page, make sure it has:
  a nav entry, page shell integration, and any required copied/generated figure.
- Prefer reusing existing helper functions in `generate.py` such as page shell,
  callouts, and figure helpers rather than open-coding page markup repeatedly.

## Cross-Repo Publishing Checklist

- Confirm the underlying result exists in `/home/ath/experiments`.
- Copy or regenerate any needed figures into `/home/ath/writing/figures/`.
- Update `generate.py`.
- Regenerate the HTML pages.
- Spot-check at least the changed page and the nav/footer links before calling
  the site update complete.

