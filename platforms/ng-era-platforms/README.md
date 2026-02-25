# NG Era Platforms (Consolidated)

This subproject groups all NG-era platform artifacts into a single auditable, verifiable, and replayable bundle under `platforms/`. The original content is preserved to keep provenance and evidence chains intact.

## Layout
- `./ng-cross-era-platforms/meta/` — GL90-99 meta specifications and semantic core (cross-era)
- `./ng-era1-platforms/` — Era-1 code layer (enterprise architecture, governance, core/services/runtime)
- `./ng-era2-platforms/` — Era-2 microcode layer (runtime, data processing, monitoring, governance)
- `./ng-era3-platforms/` — Era-3 extensions layer (services, services-platform, integration-hub)

## Audit & Verification
- Run `python ecosystem/enforce.py` from the repository root to execute the standard governance enforcement suite.
- Each era directory retains its existing README/STATUS files for platform-specific validation steps.
- Evidence anchors and governance artifacts remain co-located with their respective platforms for traceability.

## Replayability
- All historical reports, TODOs, and deployment specs remain unchanged inside each era so execution flows can be replayed without reconstruction.
- Use git history to trace exact states when reproducing prior runs; no rewrites were performed during this consolidation.
