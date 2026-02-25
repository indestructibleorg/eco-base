# @ECO-governed
# @ECO-layer: GL90-99
# @ECO-semantic: documentation
# @ECO-audit-trail: ../../engine/governance/GL_SEMANTIC_ANCHOR.json
#
# GL Unified Architecture Governance Framework Activated
# Refactoring Validation Tools

Python validation tools for three-phase refactoring process.

## 📜 Available Validators

### Phase 1 Validator
Validates Deconstruction phase deliverables.

```bash
python3 tools/refactor/validate-phase1.py \
  --deliverables-path workspace/docs/refactor_playbooks/01_deconstruction \
  --output validation-report-phase1.json
```

**Validates:**
- Repository structure map (YAML)
- Dependency graph (JSON)
- Architecture violations report (Markdown)
- Legacy assets index (YAML)
- Priority matrix (YAML)

### Phase 2 Validator
Validates Integration phase deliverables.

```bash
python3 tools/refactor/validate-phase2.py \
  --deliverables-path workspace/docs/refactor_playbooks/02_integration \
  --output validation-report-phase2.json
```

**Validates:**
- Module boundary specifications (Markdown)
- API contract definitions (YAML)
- Integration strategy (Markdown)
- Migration roadmap (YAML)
- Integration test suites (existence check)

### Phase 3 Validator
Validates Refactor phase deliverables and final system state.

```bash
python3 tools/refactor/validate-phase3.py \
  --deliverables-path workspace/docs/refactor_playbooks/03_refactor \
  --output validation-report-phase3.json
```

**Validates:**
- P0/P1/P2 action plan (YAML)
- Validation reports directory
- Refactored codebase (existence)
- Architecture compliance configuration
- Test coverage configuration

## 🔧 Requirements

Install Python dependencies:

```bash
pip install pyyaml
```

## 📊 Output Format

All validators produce JSON reports with this structure:

```json
{
  "phase": 1,
  "deliverables_path": "/path/to/deliverables",
  "passed": ["file1.yaml", "file2.json"],
  "warnings": ["Warning message 1", "Warning message 2"],
  "errors": ["Error message 1"],
  "success": false,
  "recommendation": "address_errors"
}
```

## 🚀 Integration with Master Script

These validators are automatically called by `leader-refactor.sh`:

```bash
# Phase 1 validation
python3 tools/refactor/validate-phase1.py \
  --deliverables-path workspace/docs/refactor_playbooks/01_deconstruction

# Phase 2 validation
python3 tools/refactor/validate-phase2.py \
  --deliverables-path workspace/docs/refactor_playbooks/02_integration

# Phase 3 validation
python3 tools/refactor/validate-phase3.py \
  --deliverables-path workspace/docs/refactor_playbooks/03_refactor
```

## 📝 Exit Codes

- `0` - Validation passed (errors = 0)
- `1` - Validation failed (errors > 0)

## 🔗 Related Documentation

- Master Plan: `workspace/docs/THREE_PHASE_REFACTORING_EXECUTION_PLAN.md`
- Scripts: `scripts/refactor/README.md`
- Quick Reference: `workspace/docs/REFACTORING_QUICK_REFERENCE.md`
