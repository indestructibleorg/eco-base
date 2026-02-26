# Platforms Refactor Retrieval Index

## Workflow Spec

- `platforms/PLATFORMS_REFACTOR_FORCED_RETRIEVAL_WORKFLOW.md`
- `scripts/platforms_refactor_retrieval.sh`
- `Makefile` target: `platforms-refactor-retrieval`

## Generated Artifacts

- `.tmp/refactor-retrieval/p0.launch.yaml`
- `.tmp/refactor-retrieval/facts.files.txt`
- `.tmp/refactor-retrieval/assumptions.raw.txt`
- `.tmp/refactor-retrieval/gaps.raw.txt`
- `.tmp/refactor-retrieval/security-redline.raw.txt`
- `.tmp/refactor-retrieval/summary.counts.txt`
- `.tmp/refactor-retrieval/decision-1.txt`
- `.tmp/refactor-retrieval/external.professional.sources.txt`
- `.tmp/refactor-retrieval/external.open.sources.txt`
- `.tmp/refactor-retrieval/verification-matrix.csv`
- `.tmp/refactor-retrieval/decision-2.txt`
- `.tmp/refactor-retrieval/next-questions.txt`
- `.tmp/refactor-retrieval/action-plan.md`
- `.tmp/refactor-retrieval/focused-q1-legacy-refs.txt`
- `.tmp/refactor-retrieval/focused-q2-supply-chain-coverage.csv`
- `.tmp/refactor-retrieval/focused-q2-supply-chain-review.txt`
- `.tmp/refactor-retrieval/focused-q3-mapping-delta.md`
- `.tmp/refactor-retrieval/p0.execution.md`
- `.tmp/refactor-retrieval/p1.execution.md`
- `.tmp/refactor-retrieval/emit-actions.md`
- `.tmp/refactor-retrieval/p2.execution.md`

## Dependency & Reference Mapping

- Internal retrieval scope: `platforms/**`, `docs/**`
- Security redline keywords: `secret`, `token`, `password`, `cosign`, `slsa`, `kyverno`, `policy`
- Governance dependency: `.github/workflows/supply-chain-gate.yml`
- Execution dependency: `bash`, `python3`, `grep`, `find`
- Architecture reference: `ARCHITECTURE.md`
- Platforms entry reference: `platforms/README.md`
