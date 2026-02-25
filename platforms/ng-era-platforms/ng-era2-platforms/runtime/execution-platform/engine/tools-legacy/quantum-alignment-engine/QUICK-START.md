# @ECO-layer: GQS-L0
<!-- @ECO-governed -->
<!-- @ECO-layer: GL90-99 -->
<!-- @ECO-semantic: governed-documentation -->
<!-- @ECO-audit-trail: engine/governance/GL_SEMANTIC_ANCHOR.json -->

# 🚀 Quantum Code Alignment Engine - Quick Start

## One-Command Transformation

### Step 1: Navigate to Tool

```bash
cd /workspace/mno-repository-understanding-system/workspace/tools/quantum-alignment-engine
```

### Step 2: Run Transformation

```bash
python -m src.core.transformer /path/to/external/xxx --policy axiom-naming-v9 --output ./transformed_xxx
```

### Step 3: Get Result

```
✅ Transformation complete: ./transformed_xxx/transformed_code.py
```

## What Just Happened?

The engine automatically:

1. ✅ Scanned all Python files in `/path/to/external/xxx`
2. ✅ Extracted functions, classes, and dependencies
3. ✅ Encoded into 8192-dimensional quantum states
4. ✅ Mapped to MachineNativeOps namespaces
5. ✅ Applied error correction (decoherence calibration)
6. ✅ Generated aligned Python code

## Key Files

- **Entry Point:** `src/core/transformer.py`
- **Main Class:** `QuantumCodeTransformer`
- **Core Method:** `transform(source_path, policy, output_path)`

## Architecture Overview

```
External Code
    ↓
Phase 1: Quantum Encoding (Hadamard gates)
    ↓
Phase 2: Semantic Parsing (AST analysis)
    ↓
Phase 3: Entanglement Mapping (Grover's algorithm)
    ↓
Phase 4: QIR Generation (Quantum circuits)
    ↓
Phase 5: Decoherence Calibration (Error correction)
    ↓
Phase 6: Classical Collapse (Code generation)
    ↓
Aligned Code
```

## Essential Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `QuantumCodeTransformer` | `src/core/transformer.py` | Main orchestrator |
| `SemanticLattice` | `src/core/transformer.py` | 8192D quantum space |
| `EntanglementMapper` | `src/core/transformer.py` | Namespace mapping |
| `NamespaceRegistry` | `src/core/transformer.py` | Policy management |
| `DecoherenceCalibrator` | `src/core/transformer.py` | Error correction |

## Scanning Process

### How It Scans

1. **Recursive Directory Traversal**: `Path.rglob("*.py")`
2. **AST Parsing**: `ast.parse()` for each file
3. **Element Extraction**: Functions, classes, imports
4. **Dependency Analysis**: Import statements
5. **Semantic Encoding**: Hash → 8192D vector

### What It Finds

- ✅ Functions (`ast.FunctionDef`)
- ✅ Classes (`ast.ClassDef`)
- ✅ Methods (`ast.FunctionDef` in classes)
- ✅ Imports (`ast.Import`, `ast.ImportFrom`)
- ✅ Dependencies (import analysis)

## Transformation Rules

### Namespace Mapping

External namespace → MachineNativeOps namespace

Example:
```
external.cli.tool → workspace.tools.integrated-cli
external.utils.helpers → workspace.src.governance.utils
```

### Naming Convention

Follows `axiom-naming-v9` policy:

- `snake_case` for functions/variables
- `PascalCase` for classes
- Lowercase for modules

### Dependency Resolution

Updates imports automatically:

```python
# Before
import external_module

# After
from workspace.src.governance import external_module
```

## Verification

### Check Output

```bash
# View structure
tree ./transformed_xxx/

# Read generated code
cat ./transformed_xxx/transformed_code.py
```

### Validate

```python
# Import test
python -c "import transformed_code"
```

## Error Handling

| Error | Cause | Solution |
|-------|-------|----------|
| `Found 0 code elements` | No Python files | Check source path |
| `SemanticDecoherenceError` | Low coherence | Manual review |
| `ModuleNotFoundError` | Missing deps | Install requirements |

## Advanced Options

### Custom Policy

```bash
python -m src.core.transformer /path/to/code --policy custom-policy
```

### Custom Output

```bash
python -m src.core.transformer /path/to/code --output /custom/path
```

### Python API

```python
from src.core.transformer import QuantumCodeTransformer

transformer = QuantumCodeTransformer()
result = transformer.transform("/path/to/xxx", output_path="./output")
```

## Next Steps

After successful transformation:

1. ✅ Review generated code
2. ✅ Run tests
3. ✅ Update imports
4. ✅ Integrate with CI/CD
5. ✅ Update documentation

---

**Ready to transform any external codebase in one command! 🎉**
</content>