/**
 * @ECO-governed
 * @ECO-layer: core
 * @ECO-semantic: engine-index
 * @ECO-audit-trail: ../governance/GL_SEMANTIC_ANCHOR.json
 * 
 * GL Unified Charter Activated
 */

/**
 * @module index
 * @description AEP Engine main entry point
 * @gl-governed
 * GL Unified Charter Activated
 * @gl-layer ECO-30-EXECUTION
 * @gl-module engine/core
 * @gl-semantic-anchor ECO-30-EXEC-TS
 * @gl-evidence-required true
 */

/**
 * Architecture Execution Pipeline (AEP) - Main Entry Point
 * 
 * @description Core pipeline orchestrator for the AEP engine
 * @version 1.0.0
 * @author MachineNativeOps Team
 * @license MIT
 * @module engine
 * @gl-layer GL30-49 Execution Layer - Core Pipeline Orchestrator
 * @gl-charter-activated true
 * 
 * @example
 * ```typescript
 * import { FsLoader, YamlParser, GovernanceEngine } from '@machine-native-ops/aep-engine';
 * 
 * const loader = new FsLoader('./config');
 * const parser = new YamlParser();
 * const governance = new GovernanceEngine();
 * ```
 */

// Export all interfaces
export * from './interfaces.d';

// Export loader components
export { FsLoader } from './loader/fs_loader';
export { GitLoader } from './loader/git_loader';
export { MergeIndex } from './loader/merge_index';

// Export parser components
export { YamlParser } from './parser/yaml_parser';
export { AnchorResolver as ParserAnchorResolver } from './parser/anchor_resolver';
export { JsonPassthroughParser } from './parser/json_passthrough';

// Export normalizer components
export { EnvMerger } from './normalizer/env_merger';
export { DefaultsApplier } from './normalizer/defaults_applier';
export { ModuleDefaults } from './normalizer/module_defaults';

// Export validator components
export { SchemaValidator } from './validator/schema_validator';
export { ModuleValidator } from './validator/module_validator';
export { ErrorReporter } from './validator/error_reporter';

// Export governance components
export { GovernanceEngine } from './governance/gl_engine';
export { RuleEvaluator } from './governance/rule_evaluator';
export { AnchorResolver as GovernanceAnchorResolver } from './governance/anchor_resolver';
export { EventsWriter } from './governance/events_writer';

// Export renderer components
export { TemplateEngine } from './renderer/template_engine';
export { ModuleMapper } from './renderer/module_mapper';
export { ArtifactWriter } from './renderer/artifact_writer';

// Export executor components
export { LocalExecutor } from './executor/local_executor';
export { RemoteExecutor } from './executor/remote_executor';
export { RollbackManager } from './executor/rollback';

// Export artifacts components
export { ArtifactManager } from './artifacts/artifact_manager';
export { EvidenceChain } from './artifacts/evidence_chain';
export { ManifestGenerator } from './artifacts/manifest_generator';

/**
 * Engine version
 * @constant
 */
export const ENGINE_VERSION = '1.0.0';

/**
 * GL Charter activation status
 * @constant
 */
export const GL_CHARTER_ACTIVATED = true;
