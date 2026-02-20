# Contributing to IndestructibleEco

## Getting Started

```bash
git clone https://github.com/indestructibleorg/indestructibleeco.git
cd indestructibleeco
pip install pydantic fastapi httpx pytest pytest-asyncio jsonschema pyyaml numpy
PYTHONPATH=. pytest tests/ -v
```

## Development Workflow

1. Create a feature branch from `develop`
2. Make changes following the conventions below
3. Run all tests locally: `PYTHONPATH=. pytest tests/ -v`
4. Run CI validator: `python3 tools/ci-validator/validate.py`
5. Push and open a PR against `develop`

## Code Conventions

### Python (backend/ai, src/)
- Python 3.11+
- Pydantic v2 for all schemas
- All IDs use UUID v1
- All resources have `uri` + `urn` fields
- Environment variables use `ECO_*` prefix
- Docstrings with URI: `indestructibleeco://module/path`

### TypeScript (backend/api, packages/, platforms/)
- TypeScript strict mode
- Express for API service
- React + Vite for web frontend
- pnpm workspaces for package management

### .qyaml Governance
Every .qyaml manifest must contain 4 governance blocks:
1. `document_metadata` — unique_id (UUID v1), uri, urn, schema_version
2. `governance_info` — owner, compliance_tags, lifecycle_policy
3. `registry_binding` — service_endpoint, health_check_path
4. `vector_alignment_map` — alignment_model, coherence_vector_dim

### Naming Conventions
- Container names: `eco-*`
- Image registry: `ghcr.io/indestructibleorg/*`
- Namespace: `indestructibleeco`
- Config prefix: `ECO_*`

## Testing

```bash
# Core tests (Python)
PYTHONPATH=. pytest tests/ -v

# Skill tests
pytest tools/skill-creator/skills/ -v

# CI validator
python3 tools/ci-validator/validate.py

# YAML validation
node tools/yaml-toolkit/bin/cli.js validate k8s/base/*.qyaml
```

## CI/CD Pipeline

5-gate pipeline (all must pass):
1. **validate** — CI Validator Engine (8 validators)
2. **lint** — Python compile + JS syntax + YAML governance
3. **test** — Core tests + skill tests
4. **build** — Docker build + structure verification
5. **auto-fix** — Diagnostic mode on failure

## Commit Messages

Format: `type: description`

Types:
- `feat:` — New feature
- `fix:` — Bug fix
- `docs:` — Documentation
- `test:` — Tests
- `refactor:` — Code refactoring
- `ci:` — CI/CD changes
- `chore:` — Maintenance

## Security

- Never commit secrets or API keys
- Use `ECO_*` environment variables for all configuration
- Report vulnerabilities via GitHub Security Advisories
- See [SECURITY.md](SECURITY.md) for details

## License

Apache-2.0. By contributing, you agree that your contributions will be licensed under the same license.
