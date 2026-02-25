# @ECO-governed
# @ECO-layer: GL90-99
# @ECO-semantic: documentation
# @ECO-audit-trail: ../../engine/governance/GL_SEMANTIC_ANCHOR.json
#
# GL Unified Architecture Governance Framework Activated
# @machine-native-ops/aep-engine

> Architecture Execution Pipeline (AEP) Engine - Machine-Native Declarative Architecture Engine

[![npm version]([EXTERNAL_URL_REMOVED])]([EXTERNAL_URL_REMOVED])
[![License: MIT]([EXTERNAL_URL_REMOVED])]([EXTERNAL_URL_REMOVED])
[![GL Charter]([EXTERNAL_URL_REMOVED])]([EXTERNAL_URL_REMOVED])
<!--
@gl-layer ECO-90-META
@gl-module engine/docs
@gl-semantic-anchor ECO-90-META-DOC
@gl-evidence-required false
-->

# AEP Engine - Architecture Execution Pipeline

## Overview

The AEP Engine is a governance-driven infrastructure automation pipeline that transforms declarative configurations into executable artifacts with full audit trail and compliance verification.

## Features

- **Multi-Stage Pipeline**: Loader → Parser → Normalizer → Validator → Renderer → Executor
- **Governance Integration**: Built-in GL (Governance Layer) compliance checking
- **Evidence Chain**: Complete audit trail for all operations
- **Type Safety**: Full TypeScript support with comprehensive type definitions
- **Extensible**: Plugin architecture for custom stages and validators

## Installation

```bash
npm install @machine-native-ops/aep-engine
```

## Quick Start

```typescript
import {
  FsLoader,
  YamlParser,
  SchemaValidator,
  GovernanceEngine,
  TemplateEngine,
  LocalExecutor
} from '@machine-native-ops/aep-engine';

// Initialize components
const loader = new FsLoader('./config');
const parser = new YamlParser();
const validator = new SchemaValidator();
const governance = new GovernanceEngine();
const renderer = new TemplateEngine();
const executor = new LocalExecutor();

// Execute pipeline
async function runPipeline() {
  // Load configuration files
  const loadResult = await loader.load();

  // Process each loaded file through the pipeline
  for (const { content, path: filePath } of loadResult.files.values()) {
    // Parse YAML content
    const parseResult = await parser.parse(content, filePath);
    const data = parseResult.data as Record<string, unknown>;

    // Validate against schema
    const schema = { type: 'object' };
    const validationResult = await validator.validate(data, schema);
    if (!validationResult.valid) {
      continue;
    }

    // Enforce governance policies
    const governanceResult = await governance.enforce(data, 'production');
    if (!governanceResult.passed) {
      continue;
    }

    // Render templates
    const renderResult = await renderer.render('./templates/example.njk', data);

    // Execute artifacts
    const executionResult = await executor.execute(
      {
        id: `artifact-${filePath}`,
        name: 'rendered-template',
        type: 'command',
        command: 'echo',
        args: [renderResult.content]
      },
      'production'
    );
  }
}
```

## Architecture

### Pipeline Stages

| Stage | Component | GL Layer | Description |
|-------|-----------|----------|-------------|
| Loader | FsLoader, GitLoader | GL30-49 | Load configuration files |
| Parser | YamlParser, JsonPassthrough | GL30-49 | Parse file content |
| Normalizer | EnvMerger, DefaultsApplier | GL30-49 | Normalize configuration |
| Validator | SchemaValidator, ModuleValidator | GL30-49 | Validate against schemas |
| Renderer | TemplateEngine, ArtifactWriter | GL30-49 | Render templates |
| Executor | LocalExecutor, RemoteExecutor | GL30-49 | Execute artifacts |
| Governance | GovernanceEngine, RuleEvaluator | GL00-99 | Enforce policies |
| Artifacts | ArtifactManager, EvidenceChain | GL90-99 | Manage artifacts |

### GL Layer Mapping

- **GL00-09**: Foundation Layer - Core definitions and schemas
- **GL10-29**: Operational Layer - Process and resource management
- **GL30-49**: Execution Layer - Pipeline stages
- **GL50-69**: Integration Layer - External system integration
- **GL70-89**: Presentation Layer - UI and reporting
- **GL90-99**: Meta Layer - Audit, evidence, and compliance

## API Reference

### Interfaces

#### EvidenceRecord

Records audit trail for all pipeline operations.

```typescript
interface EvidenceRecord {
  timestamp: string;      // ISO 8601 timestamp
  stage: string;          // Pipeline stage
  app.kubernetes.io/component: string;      // Component name
  action: string;         // Action performed
  status: 'success' | 'error' | 'warning';
  input: Record<string, unknown>;
  output: Record<string, unknown>;
  metrics: Record<string, unknown>;
}
```

#### GLEvent

Represents governance events for audit and compliance.

```typescript
interface GLEvent {
  id: string;             // UUID v4
  timestamp: string;      // ISO 8601 timestamp
  type: string;           // Event type
  stage: string;          // Pipeline stage
  app.kubernetes.io/component: string;      // Component name
  data: Record<string, unknown>;
  metadata: Record<string, unknown>;
}
```

### Components

#### FsLoader

Loads configuration files from the file system.

```typescript
const loader = new FsLoader('./config', {
  ignore: ['*.test.yaml', 'node_modules']
});
const result = await loader.load();
```

#### YamlParser

Parses YAML content with comprehensive error handling.

```typescript
const parser = new YamlParser();
const result = await parser.parse(content, filePath);
```

#### SchemaValidator

Validates configuration against JSON Schema.

```typescript
const validator = new SchemaValidator({
  allErrors: true,
  strict: true
});
const result = await validator.validate(data, schema);
```

#### GovernanceEngine

Enforces governance policies on configuration.

```typescript
const governance = new GovernanceEngine();
const result = await governance.enforce(config, 'production', context);
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AEP_LOG_LEVEL` | Logging level | `info` |
| `AEP_DRY_RUN` | Enable dry run mode | `false` |
| `AEP_EVIDENCE_DIR` | Evidence output directory | `./evidence` |
| `AEP_ARTIFACTS_DIR` | Artifacts output directory | `./artifacts` |

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for contribution guidelines.

## License

MIT License - see [LICENSE](../LICENSE) for details.

## GL Metadata

```json
{
  "charter-activated": true,
  "layer": "GL30-49",
  "component": "Execution Layer",
  "version": "1.0.0",
  "audit-compliant": true
}
```
