/**
 * @ECO-governed
 * @ECO-layer: types
 * @ECO-semantic: types-index
 * @ECO-audit-trail: ../governance/GL_SEMANTIC_ANCHOR.json
 * 
 * GL Unified Charter Activated
 */

/**
 * Engine Type Definitions
 * 
 * @description Centralized type definitions for the AEP engine
 * @version 1.0.0
 * @module engine/types
 * @gl-layer GL90-99 Meta Layer - Type System
 * @gl-charter-activated true
 */

// Re-export all interface types
export * from '../interfaces.d';

/**
 * Configuration object type
 * @description Generic configuration structure
 */
export interface ConfigObject {
  [key: string]: unknown;
}

/**
 * File metadata type
 * @description Metadata for loaded files
 */
export interface FileMetadata {
  /** File path relative to base directory */
  path: string;
  /** File size in bytes */
  size: number;
  /** Last modified timestamp */
  modified: string;
  /** File hash (SHA-256) */
  hash: string;
  /** File type (yaml, json, etc.) */
  type: string;
}

/**
 * Pipeline context type
 * @description Context passed through pipeline stages
 */
export interface PipelineContext {
  /** Current environment (dev, staging, prod) */
  env: string;
  /** Base directory for operations */
  baseDir: string;
  /** Dry run mode flag */
  dryRun: boolean;
  /** Verbose logging flag */
  verbose: boolean;
  /** Additional context data */
  metadata: Record<string, unknown>;
}

/**
 * Artifact descriptor type
 * @description Describes a generated artifact
 */
export interface ArtifactDescriptor {
  /** Artifact unique identifier */
  id: string;
  /** Artifact name */
  name: string;
  /** Artifact type */
  type: string;
  /** Output path */
  path: string;
  /** Content hash */
  hash: string;
  /** Creation timestamp */
  created: string;
  /** Source files */
  sources: string[];
}

/**
 * Rule definition type
 * @description Governance rule definition
 */
export interface RuleDefinition {
  /** Rule unique identifier */
  id: string;
  /** Rule name */
  name: string;
  /** Rule description */
  description: string;
  /** Rule severity */
  severity: 'critical' | 'high' | 'medium' | 'low';
  /** Rule category */
  category: string;
  /** Rule condition expression */
  condition: string;
  /** Rule enabled flag */
  enabled: boolean;
}

/**
 * Policy definition type
 * @description Governance policy definition
 */
export interface PolicyDefinition {
  /** Policy unique identifier */
  id: string;
  /** Policy name */
  name: string;
  /** Policy description */
  description: string;
  /** Policy version */
  version: string;
  /** Associated rules */
  rules: string[];
  /** Policy enabled flag */
  enabled: boolean;
}

/**
 * Anchor definition type
 * @description Semantic anchor definition
 */
export interface AnchorDefinition {
  /** Anchor name */
  name: string;
  /** Anchor value */
  value: unknown;
  /** Source file path */
  source: string;
  /** Anchor type */
  type: string;
}

/**
 * Merge conflict type
 * @description Describes a merge conflict
 */
export interface MergeConflict {
  /** Conflict path */
  path: string;
  /** Source A value */
  sourceA: unknown;
  /** Source B value */
  sourceB: unknown;
  /** Resolution strategy used */
  resolution: string;
  /** Resolved value */
  resolved: unknown;
}

/**
 * Execution plan type
 * @description Describes an execution plan
 */
export interface ExecutionPlan {
  /** Plan unique identifier */
  id: string;
  /** Plan name */
  name: string;
  /** Target environment */
  env: string;
  /** Artifacts to execute */
  artifacts: ArtifactDescriptor[];
  /** Execution order */
  order: string[];
  /** Dependencies map */
  dependencies: Record<string, string[]>;
  /** Rollback plan */
  rollback: RollbackPlan;
}

/**
 * Rollback plan type
 * @description Describes a rollback plan
 */
export interface RollbackPlan {
  /** Rollback enabled flag */
  enabled: boolean;
  /** Backup location */
  backupPath: string;
  /** Rollback steps */
  steps: RollbackStep[];
}

/**
 * Rollback step type
 * @description Single rollback step
 */
export interface RollbackStep {
  /** Step order */
  order: number;
  /** Step action */
  action: string;
  /** Step target */
  target: string;
  /** Step parameters */
  params: Record<string, unknown>;
}

/**
 * Health check result type
 * @description Result of health check operations
 */
export interface HealthCheckResult {
  /** Overall health status */
  healthy: boolean;
  /** Individual check results */
  checks: HealthCheck[];
  /** Check timestamp */
  timestamp: string;
}

/**
 * Health check type
 * @description Single health check
 */
export interface HealthCheck {
  /** Check name */
  name: string;
  /** Check status */
  status: 'pass' | 'fail' | 'warn';
  /** Check message */
  message: string;
  /** Check duration in ms */
  duration: number;
}
 * @module types
 * @description Core type definitions for AEP Engine
 * @gl-governed
 * GL Unified Charter Activated
 */

// ============================================================
// Base Types
// ============================================================

/**
 * Generic configuration object type
 */
export type ConfigObject = Record<string, unknown>;

/**
 * Generic data payload type
 */
export type DataPayload = Record<string, unknown>;

/**
 * Generic metadata type
 */
export type MetadataObject = Record<string, unknown>;

// ============================================================
// Artifact Types
// ============================================================

/**
 * Artifact content type
 */
export interface ArtifactContent {
  type: string;
  name: string;
  version?: string;
  data: DataPayload;
  metadata?: MetadataObject;
}

/**
 * Artifact metadata type
 */
export interface ArtifactMetadata {
  id?: string;
  type?: string;
  tags?: string[];
  createdAt?: string;
  updatedAt?: string;
  checksum?: string;
  [key: string]: unknown;
}

/**
 * Artifact list item
 */
export interface ArtifactListItem {
  id: string;
  type: string;
  name: string;
  tags: string[];
  createdAt: string;
  checksum: string;
}

// ============================================================
// Manifest Types
// ============================================================

/**
 * Manifest artifact entry
 */
export interface ManifestArtifact {
  id: string;
  type: string;
  name: string;
  path?: string;
  checksum?: string;
  dependencies?: string[];
  metadata?: MetadataObject;
}

/**
 * Manifest generation input
 */
export interface ManifestInput {
  name: string;
  version: string;
  artifacts: ManifestArtifact[];
  metadata?: MetadataObject;
}

/**
 * Generated manifest structure
 */
export interface GeneratedManifest {
  name: string;
  version: string;
  generatedAt: string;
  artifacts: ManifestArtifact[];
  dependencies: string[];
  verification: {
    checksum: string;
    algorithm: string;
  };
  metadata?: MetadataObject;
}

/**
 * Manifest generation result
 */
export interface ManifestResult {
  status: 'success' | 'error' | 'warning';
  manifest: GeneratedManifest;
  errors: string[];
  warnings: string[];
}

// ============================================================
// Executor Types
// ============================================================

/**
 * Executable artifact type
 */
export interface ExecutableArtifact {
  id: string;
  type: string;
  name: string;
  command?: string;
  script?: string;
  args?: string[];
  env?: Record<string, string>;
  workingDir?: string;
  timeout?: number;
  metadata?: MetadataObject;
}

/**
 * Remote execution target
 */
export interface RemoteTarget {
  host: string;
  port?: number;
  user?: string;
  keyPath?: string;
  password?: string;
  timeout?: number;
}

/**
 * Backup entry type
 */
export interface BackupEntry {
  id: string;
  artifactId: string;
  timestamp: string;
  data: DataPayload;
  metadata?: MetadataObject;
}

// ============================================================
// Governance Types
// ============================================================

/**
 * Anchor definition
 */
export interface AnchorDefinition {
  name: string;
  path: string;
  value: unknown;
}

/**
 * Anchor usage reference
 */
export interface AnchorUsage {
  name: string;
  path: string;
}

/**
 * Anchor resolution result
 */
export interface AnchorResolutionResult {
  status: 'success' | 'error' | 'warning';
  config: ConfigObject;
  resolved: ConfigObject;
  anchors: AnchorDefinition[];
  usages: AnchorUsage[];
  errors: string[];
}

/**
 * Policy definition
 */
export interface PolicyDefinition {
  name: string;
  description?: string;
  environments?: string[];
  rules: RuleDefinition[];
  severity?: 'error' | 'warning' | 'info';
}

/**
 * Rule definition
 */
export interface RuleDefinition {
  name: string;
  type: 'required' | 'forbidden' | 'pattern' | 'range' | 'enum' | 'custom';
  path?: string;
  pattern?: string;
  min?: number;
  max?: number;
  values?: unknown[];
  message?: string;
  severity?: 'error' | 'warning' | 'info';
}

/**
 * Rule evaluation result
 */
export interface RuleEvaluationResult {
  passed: boolean;
  violations: RuleViolation[];
  warnings: string[];
  duration: number;
}

/**
 * Rule violation
 */
export interface RuleViolation {
  rule: string;
  path: string;
  message: string;
  severity: 'error' | 'warning' | 'info';
  expected?: unknown;
  actual?: unknown;
}

/**
 * Custom rule evaluator function type
 */
export type CustomRuleEvaluator = (config: ConfigObject, rule: RuleDefinition) => boolean;

// ============================================================
// Loader Types
// ============================================================

/**
 * File entry for merge operations
 */
export interface FileEntry {
  path: string;
  content: string;
  parsed?: ConfigObject;
  timestamp?: string;
  priority?: number;
}

/**
 * Merge result
 */
export interface MergeResult {
  file?: ConfigObject;
  error?: string;
}

// ============================================================
// Normalizer Types
// ============================================================

/**
 * Defaults application result
 */
export interface DefaultsApplicationResult {
  status: 'success' | 'error' | 'warning';
  applied: ConfigObject;
  changes: string[];
  errors: string[];
}

/**
 * Environment merge result
 */
export interface EnvMergeResult {
  status: 'success' | 'error' | 'warning';
  merged: ConfigObject;
  changes: string[];
  errors: string[];
}

/**
 * Module defaults application result
 */
export interface ModuleDefaultsResult {
  status: 'success' | 'error' | 'warning';
  applied: ConfigObject;
  modulesApplied: string[];
  errors: string[];
}

// ============================================================
// Parser Types
// ============================================================

/**
 * Anchor extraction result
 */
export interface ExtractedAnchor {
  name: string;
  line: number;
  value: unknown;
}

/**
 * Anchor definition storage
 */
export interface StoredAnchorDefinition {
  filePath: string;
  value: unknown;
}

// ============================================================
// Renderer Types
// ============================================================

/**
 * Template render input
 */
export interface TemplateRenderInput {
  path: string;
  data: DataPayload;
  output?: string;
}

/**
 * Artifact write options
 */
export interface ArtifactWriteOptions {
  artifactId: string;
  content: string;
  outputPath: string;
  metadata?: ArtifactMetadata;
}

/**
 * Module mapping result
 */
export interface ModuleMappingResult {
  status: 'success' | 'error' | 'warning';
  artifacts: MappedArtifact[];
  errors: string[];
}

/**
 * Mapped artifact
 */
export interface MappedArtifact {
  moduleId: string;
  artifactId: string;
  type: string;
  data: DataPayload;
  metadata?: MetadataObject;
}

/**
 * Module manifest
 */
export interface ModuleManifest {
  name: string;
  version?: string;
  artifacts?: string[];
  dependencies?: string[];
  config?: ConfigObject;
}

/**
 * Template filter function type
 */
export type TemplateFilter = (...args: unknown[]) => unknown;

/**
 * Template engine options
 */
export interface TemplateEngineOptions {
  templatesDir?: string;
  outputDir?: string;
  filters?: Map<string, TemplateFilter>;
}

// ============================================================
// Validator Types
// ============================================================

/**
 * Validation error details
 */
export interface ValidationErrorDetails {
  path?: string;
  keyword?: string;
  params?: Record<string, unknown>;
  message?: string;
  [key: string]: unknown;
}

/**
 * Error report entry
 */
export interface ErrorReportEntry {
  path: string;
  message: string;
  severity: 'error' | 'warning' | 'info';
  code?: string;
  details?: ValidationErrorDetails;
}

/**
 * Module validation input
 */
export interface ModuleValidationInput {
  config: ConfigObject;
  manifest?: ModuleManifest;
}

/**
 * Schema validation input
 */
export interface SchemaValidationInput {
  config: ConfigObject;
  schema: ConfigObject;
}

/**
 * AJV error type
 */
export interface AjvError {
  instancePath: string;
  keyword: string;
  params: Record<string, unknown>;
  message?: string;
  schemaPath: string;
}

// ============================================================
// Utility Types
// ============================================================

/**
 * Deep partial type
 */
export type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

/**
 * Nullable type
 */
export type Nullable<T> = T | null;

/**
 * Optional type
 */
export type Optional<T> = T | undefined;
