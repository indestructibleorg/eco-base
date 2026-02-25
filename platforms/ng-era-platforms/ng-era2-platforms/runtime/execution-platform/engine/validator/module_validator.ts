/**
 * @ECO-governed
 * @ECO-layer: validator
 * @ECO-semantic: validator-module_validator
 * @ECO-audit-trail: ../governance/GL_SEMANTIC_ANCHOR.json
 * 
 * GL Unified Charter Activated
 */

/**
 * @module module_validator
 * @description Validate module manifests and configurations
 * @gl-governed
 * GL Unified Charter Activated
 * @gl-layer ECO-30-EXECUTION
 * @gl-module engine/validator
 * @gl-semantic-anchor ECO-30-EXEC-TS
 * @gl-evidence-required true
 * @version 1.0.0
 * @since 2026-01-24
 * @author MachineNativeOps Team
 */

import { ValidatorInterface, ValidationResult, EvidenceRecord } from '../interfaces.d';
import type { ConfigObject, ModuleManifest, ModuleValidationInput } from '../types';

/**
 * Module Validator
 * 
 * GL30-49: Execution Layer - Validator Stage
 * 
 * Validates module manifests, schemas, and templates
 * against governance policies and module conventions.
 */
export class ModuleValidator implements ValidatorInterface {
  private evidence: EvidenceRecord[] = [];
  private readonly moduleRegistry: Map<string, ModuleManifest> = new Map();

  constructor() {
    // Initialize with default module requirements
  }

  /**
   * Validate module configuration
   */
  async validate(
    moduleConfig: ConfigObject,
    moduleName: string,
    moduleManifest?: ModuleManifest
  ): Promise<ValidationResult> {
    const startTime = Date.now();
    const errors: string[] = [];
    const warnings: string[] = [];

    try {
      // Validate module exists
      if (!moduleConfig || typeof moduleConfig !== 'object') {
        errors.push(`Module ${moduleName} configuration must be an object`);
        return this.createResult(false, errors, warnings, startTime);
      }

      // Validate module manifest if provided
      if (moduleManifest) {
        const manifestErrors = this.validateManifest(moduleManifest, moduleName);
        errors.push(...manifestErrors.errors);
        warnings.push(...manifestErrors.warnings);
      }

      // Validate module structure
      const structureErrors = this.validateStructure(moduleConfig, moduleName);
      errors.push(...structureErrors.errors);
      warnings.push(...structureErrors.warnings);

      // Validate module naming
      const namingErrors = this.validateNaming(moduleName);
      errors.push(...namingErrors.errors);
      warnings.push(...namingErrors.warnings);

      // Validate required fields
      const requiredErrors = this.validateRequiredFields(moduleConfig, moduleName);
      errors.push(...requiredErrors.errors);
      warnings.push(...requiredErrors.warnings);

      const valid = errors.length === 0;

      // Generate evidence record
      this.evidence.push({
        timestamp: new Date().toISOString(),
        stage: 'validator',
        component: 'module_validator',
        action: 'validate',
        status: valid ? 'success' : (errors.length > 0 ? 'error' : 'warning'),
        input: { moduleName, hasManifest: !!moduleManifest },
        output: {
          valid,
          errorCount: errors.length,
          warningCount: warnings.length
        },
        metrics: { duration: Date.now() - startTime }
      });

      return this.createResult(valid, errors, warnings, startTime);
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : String(error);
      
      this.evidence.push({
        timestamp: new Date().toISOString(),
        stage: 'validator',
        component: 'module_validator',
        action: 'validate',
        status: 'error',
        input: { moduleName },
        output: { error: errorMsg },
        metrics: { duration: Date.now() - startTime }
      });

      return this.createResult(false, [errorMsg], [], startTime);
    }
  }

  /**
   * Validate module manifest
   */
  private validateManifest(
    manifest: ModuleManifest,
    moduleName: string
  ): { errors: string[]; warnings: string[] } {
    const errors: string[] = [];
    const warnings: string[] = [];

    // Required manifest fields
    if (!manifest.name) {
      errors.push(`Module ${moduleName}: Required manifest field missing: name`);
    }

    // Validate version format
    if (manifest.version && !/^\d+\.\d+\.\d+$/.test(manifest.version)) {
      warnings.push(`Module ${moduleName}: Version ${manifest.version} does not follow semver format`);
    }

    // Validate config exists
    if (manifest.config && typeof manifest.config !== 'object') {
      errors.push(`Module ${moduleName}: Config must be an object`);
    }

    // Validate artifacts exist
    if (manifest.artifacts && !Array.isArray(manifest.artifacts)) {
      errors.push(`Module ${moduleName}: Artifacts must be an array`);
    }

    return { errors, warnings };
  }

  /**
   * Validate module structure
   */
  private validateStructure(
    config: ConfigObject,
    moduleName: string
  ): { errors: string[]; warnings: string[] } {
    const errors: string[] = [];
    const warnings: string[] = [];

    // Check for unexpected top-level keys
    const allowedKeys = ['enabled', 'config', 'version', 'metadata'];
    for (const key of Object.keys(config)) {
      if (!allowedKeys.includes(key)) {
        warnings.push(`Module ${moduleName}: Unexpected top-level key: ${key}`);
      }
    }

    // Validate config structure
    if (config.config && typeof config.config !== 'object') {
      errors.push(`Module ${moduleName}: Config must be an object`);
    }

    // Validate enabled is boolean
    if (config.enabled !== undefined && typeof config.enabled !== 'boolean') {
      errors.push(`Module ${moduleName}: Enabled must be a boolean`);
    }

    return { errors, warnings };
  }

  /**
   * Validate module naming conventions
   */
  private validateNaming(moduleName: string): { errors: string[]; warnings: string[] } {
    const errors: string[] = [];
    const warnings: string[] = [];

    // Check kebab-case naming
    if (!/^[a-z][a-z0-9-]*$/.test(moduleName)) {
      errors.push(`Module name "${moduleName}" must be in kebab-case (lowercase, hyphens only)`);
    }

    // Check name length
    if (moduleName.length < 3) {
      warnings.push(`Module name "${moduleName}" is shorter than recommended minimum of 3 characters`);
    }

    if (moduleName.length > 50) {
      warnings.push(`Module name "${moduleName}" is longer than recommended maximum of 50 characters`);
    }

    return { errors, warnings };
  }

  /**
   * Validate required module fields
   */
  private validateRequiredFields(
    config: ConfigObject,
    moduleName: string
  ): { errors: string[]; warnings: string[] } {
    const errors: string[] = [];
    const warnings: string[] = [];

    // Check if module has any configuration
    const configObj = config.config as ConfigObject | undefined;
    if (!configObj || Object.keys(configObj).length === 0) {
      warnings.push(`Module ${moduleName}: No configuration provided`);
    }

    return { errors, warnings };
  }

  /**
   * Validate multiple modules
   */
  async validateBatch(
    modules: Map<string, ModuleValidationInput>
  ): Promise<Map<string, ValidationResult>> {
    const results: Map<string, ValidationResult> = new Map();

    for (const [moduleName, module] of modules.entries()) {
      const result = await this.validate(module.config, moduleName, module.manifest);
      results.set(moduleName, result);
    }

    return results;
  }

  /**
   * Register module manifest
   */
  registerModule(name: string, manifest: ModuleManifest): void {
    this.moduleRegistry.set(name, manifest);
  }

  /**
   * Get registered module
   */
  getModule(name: string): ModuleManifest | undefined {
    return this.moduleRegistry.get(name);
  }

  /**
   * Create validation result
   */
  private createResult(
    valid: boolean,
    errors: string[],
    warnings: string[],
    startTime: number
  ): ValidationResult {
    return {
      valid,
      errors,
      warnings,
      duration: Date.now() - startTime,
      evidence: this.evidence
    };
  }

  /**
   * Get evidence records
   */
  getEvidence(): EvidenceRecord[] {
    return this.evidence;
  }
}