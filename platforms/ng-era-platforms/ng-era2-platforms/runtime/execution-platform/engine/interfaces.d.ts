/**
 * @ECO-governed
 * @ECO-layer: core
 * @ECO-semantic: engine-interfaces.d
 * @ECO-audit-trail: ../governance/GL_SEMANTIC_ANCHOR.json
 * 
 * GL Unified Charter Activated
 */

/**
 * Evidence Record Interface
 * 
 * @description Records audit trail for all pipeline operations
 * @since 1.0.0
 * @module engine/interfaces
 * @gl-layer GL90-99 Meta Layer
 */
 * @module interfaces
 * @description Core interface definitions for AEP Engine
 * @gl-governed
 * GL Unified Charter Activated
 */

import type {
  ConfigObject,
  DataPayload,
  MetadataObject,
  ExecutableArtifact
} from './types';

 * @gl-layer ECO-30-EXECUTION
 * @gl-module engine/types
 * @gl-semantic-anchor ECO-30-EXEC-TYPEDEF
 * @gl-evidence-required true
 */

export interface EvidenceRecord {
  /** ISO 8601 timestamp of the operation */
  timestamp: string;
  /** Pipeline stage (loader, parser, normalizer, validator, renderer, executor) */
  stage: string;
  /** Component name within the stage */
  app.kubernetes.io/component: string;
  /** Action performed */
  action: string;
  /** Operation status */
  status: 'success' | 'error' | 'warning';
  /** Input parameters */
  input: Record<string, unknown>;
  /** Output results */
  output: Record<string, unknown>;
  /** Performance metrics */
  input: Record<string, unknown>;
  output: Record<string, unknown>;
  metrics: Record<string, unknown>;
}

/**
 * Governance Layer Event Interface
 * 
 * @description Represents governance events for audit and compliance
 * @since 1.0.0
 * @module engine/interfaces
 * @gl-layer GL00-99 Unified Governance Framework
 */
export interface GLEvent {
  /** Unique event identifier (UUID v4) */
  id: string;
  /** ISO 8601 timestamp of the event */
  timestamp: string;
  /** Event type (policy_check, rule_evaluation, anchor_resolution) */
  type: string;
  /** Pipeline stage where event occurred */
  stage: string;
  /** Component that generated the event */
  app.kubernetes.io/component: string;
  /** Event payload data */
  data: Record<string, unknown>;
  /** Additional metadata */
  data: Record<string, unknown>;
  metadata: Record<string, unknown>;
}

/**
 * Load Result Interface
 * 
 * @description Result of file loading operations
 * @since 1.0.0
 * @module engine/interfaces
 * @gl-layer GL30-49 Execution Layer - Loader Stage
 */
export interface LoadResult {
  /** Operation status */
  status: 'success' | 'error' | 'warning';
  /** Map of file paths to parsed content */
  files: Map<string, unknown>;
  /** Error messages if any */
  files: Map<string, ConfigObject>;
  errors: string[];
  /** Audit trail records */
  evidence: EvidenceRecord[];
}

/**
 * Loader Interface
 * 
 * @description Contract for file loading implementations
 * @since 1.0.0
 * @module engine/interfaces
 * @gl-layer GL30-49 Execution Layer - Loader Stage
 */
export interface LoaderInterface {
  /** Load files from source */
  load(): Promise<LoadResult>;
  /** Retrieve audit trail */
  getEvidence(): EvidenceRecord[];
}

/**
 * Parse Result Interface
 * 
 * @description Result of parsing operations
 * @since 1.0.0
 * @module engine/interfaces
 * @gl-layer GL30-49 Execution Layer - Parser Stage
 */
export interface ParseResult {
  /** Operation status */
  status: 'success' | 'error' | 'warning';
  /** Parsed data structure */
  data: unknown;
  /** Error messages if any */
  data: ConfigObject;
  errors: string[];
  /** Audit trail records */
  evidence: EvidenceRecord[];
}

/**
 * Parser Interface
 * 
 * @description Contract for content parsing implementations
 * @since 1.0.0
 * @module engine/interfaces
 * @gl-layer GL30-49 Execution Layer - Parser Stage
 */
export interface ParserInterface {
  /** Parse content from file */
  parse(content: string, filePath: string): Promise<ParseResult>;
  /** Retrieve audit trail */
  getEvidence(): EvidenceRecord[];
}

/**
 * Normalizer Result Interface
 * 
 * @description Result of normalization operations
 * @since 1.0.0
 * @module engine/interfaces
 * @gl-layer GL30-49 Execution Layer - Normalizer Stage
 */
export interface NormalizerResult {
  /** Operation status */
  status: 'success' | 'error' | 'warning';
  /** Normalized data structure */
  data: unknown;
  /** Error messages if any */
  data: ConfigObject;
  errors: string[];
  /** Audit trail records */
  evidence: EvidenceRecord[];
}

/**
 * Normalizer Interface
 * 
 * @description Contract for data normalization implementations
 * @since 1.0.0
 * @module engine/interfaces
 * @gl-layer GL30-49 Execution Layer - Normalizer Stage
 */
export interface NormalizerInterface {
  /** Normalize data with optional context */
  normalize(data: unknown, context?: unknown): Promise<NormalizerResult>;
  /** Retrieve audit trail */
  normalize(data: ConfigObject, context?: MetadataObject): Promise<NormalizerResult>;
  getEvidence(): EvidenceRecord[];
}

/**
 * Validation Result Interface
 * 
 * @description Result of validation operations
 * @since 1.0.0
 * @module engine/interfaces
 * @gl-layer GL30-49 Execution Layer - Validator Stage
 */
export interface ValidationResult {
  /** Whether validation passed */
  valid: boolean;
  /** Validation error messages */
  errors: string[];
  /** Validation warning messages */
  warnings: string[];
  /** Validation duration in milliseconds */
  duration: number;
  /** Audit trail records */
  evidence: EvidenceRecord[];
}

/**
 * Validator Interface
 * 
 * @description Contract for validation implementations
 * @since 1.0.0
 * @module engine/interfaces
 * @gl-layer GL30-49 Execution Layer - Validator Stage
 */
export interface ValidatorInterface {
  /** Validate data against optional schema */
  validate(data: unknown, schema?: unknown): Promise<ValidationResult>;
  /** Retrieve audit trail */
  validate(data: ConfigObject, schema?: ConfigObject): Promise<ValidationResult>;
  getEvidence(): EvidenceRecord[];
}

/**
 * Render Result Interface
 * 
 * @description Result of rendering operations
 * @since 1.0.0
 * @module engine/interfaces
 * @gl-layer GL30-49 Execution Layer - Renderer Stage
 */
export interface RenderResult {
  /** Operation status */
  status: 'success' | 'error' | 'warning';
  /** Rendered content */
  content: string;
  /** Output file path if written */
  outputPath?: string;
  /** Error messages if any */
  errors: string[];
  /** Audit trail records */
  evidence: EvidenceRecord[];
}

/**
 * Renderer Interface
 * 
 * @description Contract for template rendering implementations
 * @since 1.0.0
 * @module engine/interfaces
 * @gl-layer GL30-49 Execution Layer - Renderer Stage
 */
export interface RendererInterface {
  /** Render template with data */
  render(template: string, data: unknown, outputPath?: string): Promise<RenderResult>;
  /** Retrieve audit trail */
  render(template: string, data: DataPayload, outputPath?: string): Promise<RenderResult>;
  getEvidence(): EvidenceRecord[];
}

/**
 * Execution Result Interface
 * 
 * @description Result of execution operations
 * @since 1.0.0
 * @module engine/interfaces
 * @gl-layer GL30-49 Execution Layer - Executor Stage
 */
export interface ExecutionResult {
  /** Operation status */
  status: 'success' | 'error' | 'warning';
  /** Execution output */
  output: string;
  /** Error messages if any */
  errors: string[];
  /** Execution duration in milliseconds */
  duration: number;
  /** Audit trail records */
  evidence: EvidenceRecord[];
}

/**
 * Executor Interface
 * 
 * @description Contract for artifact execution implementations
 * @since 1.0.0
 * @module engine/interfaces
 * @gl-layer GL30-49 Execution Layer - Executor Stage
 */
export interface ExecutorInterface {
  /** Execute artifact in specified environment */
  execute(artifact: unknown, env: string): Promise<ExecutionResult>;
  /** Retrieve audit trail */
  execute(artifact: ExecutableArtifact, env: string): Promise<ExecutionResult>;
  getEvidence(): EvidenceRecord[];
}

/**
 * Governance Violation Interface
 * 
 * @description Represents a governance policy violation
 * @since 1.0.0
 * @module engine/interfaces
 * @gl-layer GL00-99 Unified Governance Framework
 */
export interface GovernanceViolation {
  /** JSON path to violation location */
  path: string;
  /** Violation description */
  message: string;
  /** Severity level (critical, high, medium, low) */
  severity: string;
}

/**
 * Governance Enforcement Result Interface
 * 
 * @description Result of governance enforcement operations
 * @since 1.0.0
 * @module engine/interfaces
 * @gl-layer GL00-99 Unified Governance Framework
 */
export interface GovernanceEnforcementResult {
  /** Generated governance events */
  events: GLEvent[];
  /** Policy violations found */
  violations: GovernanceViolation[];
  /** Whether all policies passed */
  passed: boolean;
}

/**
 * Governance Engine Interface
 * 
 * @description Contract for governance enforcement implementations
 * @since 1.0.0
 * @module engine/interfaces
 * @gl-layer GL00-99 Unified Governance Framework
 */
export interface GovernanceEngineInterface {
  /** Enforce governance policies on configuration */
  enforce(config: unknown, env: string, context?: unknown): Promise<GovernanceEnforcementResult>;
  /** Retrieve audit trail */
  enforce(config: ConfigObject, env: string, context?: MetadataObject): Promise<{
    events: GLEvent[];
    violations: Array<{ path: string; message: string; severity: string }>;
    passed: boolean;
  }>;
  getEvidence(): EvidenceRecord[];
}

/**
 * Merge Strategy Type
 * 
 * @description Strategy for resolving merge conflicts
 * @since 1.0.0
 * @module engine/interfaces
 * @gl-layer GL30-49 Execution Layer - Loader Stage
 */
export type MergeStrategy = 'error' | 'first' | 'last' | 'newest' | 'custom';

/**
 * Pipeline Stage Type
 * 
 * @description Valid pipeline stages
 * @since 1.0.0
 * @module engine/interfaces
 * @gl-layer GL30-49 Execution Layer
 */
export type PipelineStage = 'loader' | 'parser' | 'normalizer' | 'validator' | 'renderer' | 'executor';

/**
 * Operation Status Type
 * 
 * @description Valid operation statuses
 * @since 1.0.0
 * @module engine/interfaces
 * @gl-layer GL90-99 Meta Layer
 */
export type OperationStatus = 'success' | 'error' | 'warning';

/**
 * Severity Level Type
 * 
 * @description Valid severity levels for violations
 * @since 1.0.0
 * @module engine/interfaces
 * @gl-layer GL00-99 Unified Governance Framework
 */
export type SeverityLevel = 'critical' | 'high' | 'medium' | 'low';
