/**
 * @ECO-governed
 * @ECO-layer: core
 * @ECO-semantic: engine-interfaces-fix.d
 * @ECO-audit-trail: ../governance/GL_SEMANTIC_ANCHOR.json
 * 
 * GL Unified Charter Activated
 */

/**
 * @module interfaces-fix
 * @description Fixed interface definitions for AEP Engine
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
  timestamp: string;
  stage: string;
  app.kubernetes.io/component: string;
  action: string;
  status: 'success' | 'error' | 'warning';
  input: Record<string, unknown>;
  output: Record<string, unknown>;
  metrics: Record<string, unknown>;
}

export interface GLEvent {
  id: string;
  timestamp: string;
  type: string;
  stage: string;
  app.kubernetes.io/component: string;
  data: Record<string, unknown>;
  metadata: Record<string, unknown>;
}

export interface LoadResult {
  status: 'success' | 'error' | 'warning';
  files: Map<string, ConfigObject>;
  errors: string[];
  evidence: EvidenceRecord[];
}

export interface LoaderInterface {
  load(): Promise<LoadResult>;
  getEvidence(): EvidenceRecord[];
}

export interface ParseResult {
  status: 'success' | 'error' | 'warning';
  data: ConfigObject;
  errors: string[];
  evidence: EvidenceRecord[];
}

export interface ParserInterface {
  parse(content: string, filePath: string): Promise<ParseResult>;
  getEvidence(): EvidenceRecord[];
}

export interface NormalizerResult {
  status: 'success' | 'error' | 'warning';
  data: ConfigObject;
  errors: string[];
  evidence: EvidenceRecord[];
}

export interface NormalizerInterface {
  normalize(data: ConfigObject, context?: MetadataObject): Promise<NormalizerResult>;
  getEvidence(): EvidenceRecord[];
}

export interface ValidationResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
  duration: number;
  evidence: EvidenceRecord[];
}

export interface ValidatorInterface {
  validate(data: ConfigObject, schema?: ConfigObject): Promise<ValidationResult>;
  getEvidence(): EvidenceRecord[];
}

export interface RenderResult {
  status: 'success' | 'error' | 'warning';
  content: string;
  outputPath?: string;
  errors: string[];
  evidence: EvidenceRecord[];
}

export interface RendererInterface {
  render(template: string, data: DataPayload, outputPath?: string): Promise<RenderResult>;
  getEvidence(): EvidenceRecord[];
}

export interface ExecutionResult {
  status: 'success' | 'error' | 'warning';
  output: string;
  errors: string[];
  duration: number;
  evidence: EvidenceRecord[];
}

export interface ExecutorInterface {
  execute(artifact: ExecutableArtifact, env: string): Promise<ExecutionResult>;
  getEvidence(): EvidenceRecord[];
}

export interface GovernanceEngineInterface {
  enforce(config: ConfigObject, env: string, context?: MetadataObject): Promise<{
    events: GLEvent[];
    violations: Array<{ path: string; message: string; severity: string }>;
    passed: boolean;
  }>;
  getEvidence(): EvidenceRecord[];
}

export type MergeStrategy = 'error' | 'first' | 'last' | 'newest' | 'custom';