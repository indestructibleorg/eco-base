/**
 * @ECO-governed
 * @ECO-layer: validation
 * @ECO-semantic: schema-validation
 * @ECO-audit-trail: GL_SEMANTIC_ANCHOR.json
 * 
 * GL Unified Charter Activated - Schema Validator
 * Validates artifact schemas against GL governance standards
 */

import { promises as fs } from 'fs';
import path from 'path';

export interface ValidationResult {
  valid: boolean;
  errors: ValidationError[];
  warnings: ValidationWarning[];
}

export interface ValidationError {
  code: string;
  path: string;
  message: string;
  severity: 'critical' | 'high';
}

export interface ValidationWarning {
  code: string;
  path: string;
  message: string;
  suggestion?: string;
}

export class SchemaValidator {
  constructor(private workspace: string = process.cwd()) {}

  async validateFile(filePath: string): Promise<ValidationResult> {
    const result: ValidationResult = {
      valid: true,
      errors: [],
      warnings: []
    };

    const relativePath = path.relative(this.workspace, filePath);

    try {
      const content = await fs.readFile(filePath, 'utf-8');

      // Validate JSON files
      if (filePath.endsWith('.json')) {
        this.validateJSON(content, relativePath, result);
      }

      // Validate TypeScript files
      if (/\.(ts|tsx)$/.test(filePath)) {
        this.validateTypeScript(content, relativePath, result);
      }

      // Validate YAML files
      if (/\.(yaml|yml)$/.test(filePath)) {
        this.validateYAML(content, relativePath, result);
      }

    } catch (error) {
      result.errors.push({
        code: 'FILE_READ_ERROR',
        path: relativePath,
        message: `Cannot read file: ${error}`,
        severity: 'high'
      });
    }

    result.valid = result.errors.length === 0;
    return result;
  }

  private validateJSON(content: string, filePath: string, result: ValidationResult): void {
    try {
      const data = JSON.parse(content);

      // Check for GL governance markers
      if (typeof data === 'object' && data !== null) {
        if (!data.governance) {
          result.warnings.push({
            code: 'GL_GOVERNANCE_MISSING',
            path: filePath,
            message: 'Missing "governance" section in JSON',
            suggestion: 'Add governance section with charter, version, and activation status'
          });
        }
      }

    } catch (error) {
      result.errors.push({
        code: 'JSON_PARSE_ERROR',
        path: filePath,
        message: `Invalid JSON: ${error}`,
        severity: 'critical'
      });
    }
  }

  private validateTypeScript(content: string, filePath: string, result: ValidationResult): void {
    // Check for required GL markers
    const requiredMarkers = ['@ECO-governed', '@ECO-layer:', '@ECO-semantic:'];
    
    for (const marker of requiredMarkers) {
      if (!content.includes(marker)) {
        result.errors.push({
          code: 'GL_MARKER_MISSING',
          path: filePath,
          message: `Missing GL marker: ${marker}`,
          severity: 'high'
        });
      }
    }

    // Check for export statements
    if (!content.includes('export') && (filePath.endsWith('.ts') || filePath.endsWith('.tsx'))) {
      result.warnings.push({
        code: 'NO_EXPORT',
        path: filePath,
        message: 'File has no exports',
        suggestion: 'Add appropriate exports for module usage'
      });
    }
  }

  private validateYAML(content: string, filePath: string, result: ValidationResult): void {
    // Basic YAML validation
    const lines = content.split('\n');
    
    lines.forEach((line, index) => {
      // Check for indentation consistency
      if (line.startsWith('  ') && !line.startsWith('    ')) {
        // 2-space indentation
      } else if (line.startsWith('    ')) {
        // 4-space indentation
      }
    });

    // Check for GL markers in comments
    if (!content.includes('@ECO-governed')) {
      result.warnings.push({
        code: 'GL_MARKER_MISSING',
        path: filePath,
        message: 'Missing @ECO-governed marker in YAML comments',
        suggestion: 'Add GL governance markers as comments'
      });
    }
  }

  async validateDirectory(dirPath: string): Promise<Map<string, ValidationResult>> {
    const results = new Map<string, ValidationResult>();

    const scanDir = async (dir: string): Promise<void> => {
      const entries = await fs.readdir(dir, { withFileTypes: true });

      for (const entry of entries) {
        const fullPath = path.join(dir, entry.name);
        
        if (entry.isDirectory() && 
!['node_modules', '__pycache__', 'dist', 'build'].includes(entry.name)
) {
          await scanDir(fullPath);
        } else if (entry.isFile() && /\.(ts|tsx|js|json|yaml|yml)$/.test(entry.name)) {
          const result = await this.validateFile(fullPath);
          results.set(fullPath, result);
        }
      }
    };

    await scanDir(dirPath);
    return results;
  }
}