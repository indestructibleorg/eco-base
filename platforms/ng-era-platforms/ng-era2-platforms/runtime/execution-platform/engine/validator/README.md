<!-- @ECO-governed -->
<!-- @ECO-layer: GL90-99 -->
<!-- @ECO-semantic: governed-documentation -->
<!-- @ECO-audit-trail: engine/governance/GL_SEMANTIC_ANCHOR.json -->

<!--
@gl-layer ECO-90-META
@gl-module engine/validator/docs
@gl-semantic-anchor ECO-90-META-DOC
@gl-evidence-required false
-->

# Validator Stage - Stage 4

## Overview

The Validator Stage validates configuration data against schemas, module manifests, and naming conventions to ensure quality and compliance.

## Components

### SchemaValidator
JSON Schema validator using AJv.

**Features:**
- AJv-based validation
- Comprehensive error reporting
- Schema registration
- Custom error formatting

**Usage:**
```typescript
import { SchemaValidator } from './validator/schema_validator';

const validator = new SchemaValidator();
const result = await validator.validate(data, schema);
```

### ModuleValidator
Module configuration validator.

**Features:**
- Manifest validation
- Structure validation
- Naming convention checks
- Dependency validation

**Usage:**
```typescript
import { ModuleValidator } from './validator/module_validator';

const validator = new ModuleValidator();
const result = await validator.validate(moduleConfig, manifest);
```

### ErrorReporter
Structured error reporter with grouping and formatting.

**Features:**
- Error grouping by category
- Severity classification
- Formatted output (text, JSON)
- Filtering and sorting

**Usage:**
```typescript
import { ErrorReporter } from './validator/error_reporter';

const reporter = new ErrorReporter();
const report = reporter.generateReport(errors, warnings);
```

## Evidence Records

All validator operations generate evidence records with:
- Validation results
- Error counts and details
- Schema references
- Performance metrics

## Output

**ValidationResult:**
- `valid`: boolean - Overall validation status
- `errors`: string[] - Validation errors
- `warnings`: string[] - Validation warnings
- `duration`: number - Validation time in ms
- `evidence`: EvidenceRecord[] - Complete evidence chain
