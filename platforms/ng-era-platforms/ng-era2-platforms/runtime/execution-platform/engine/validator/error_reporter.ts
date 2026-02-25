/**
 * @ECO-governed
 * @ECO-layer: validator
 * @ECO-semantic: validator-error_reporter
 * @ECO-audit-trail: ../governance/GL_SEMANTIC_ANCHOR.json
 * 
 * GL Unified Charter Activated
 */

/**
 * @module error_reporter
 * @description Format and report validation errors
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

import { EvidenceRecord } from '../interfaces.d';
import type { ErrorReportEntry, ValidationErrorDetails } from '../types';

interface ReportedError {
  timestamp: string;
  path: string;
  message: string;
  severity: 'error' | 'warning' | 'info';
  code?: string;
  details?: ValidationErrorDetails;
}

interface ErrorSummary {
  total: number;
  errors: number;
  warnings: number;
  info: number;
}

interface ErrorReport {
  summary: ErrorSummary;
  groups: Map<string, ReportedError[]>;
  formatted: string;
}

interface JsonReport {
  summary: ErrorSummary;
  groups: Record<string, ReportedError[]>;
  timestamp: string;
}

/**
 * Error Reporter
 * 
 * GL30-49: Execution Layer - Validator Stage
 * 
 * Formats and reports validation errors with structured
 * output, grouping, and severity classification.
 */
export class ErrorReporter {
  private evidence: EvidenceRecord[] = [];
  private readonly errorGroups: Map<string, ReportedError[]> = new Map();

  /**
   * Report validation error
   */
  report(error: ErrorReportEntry): void {
    const timestamp = new Date().toISOString();

    // Add to evidence
    this.evidence.push({
      timestamp,
      stage: 'validator',
      component: 'error_reporter',
      action: 'report',
      status: error.severity === 'error' ? 'error' : 'warning',
      input: { path: error.path, severity: error.severity },
      output: { message: error.message, code: error.code },
      metrics: {}
    });

    // Group by path
    const groupKey = error.path || 'root';
    if (!this.errorGroups.has(groupKey)) {
      this.errorGroups.set(groupKey, []);
    }
    this.errorGroups.get(groupKey)!.push({
      timestamp,
      ...error
    });
  }

  /**
   * Report multiple errors
   */
  reportBatch(errors: ErrorReportEntry[]): void {
    for (const error of errors) {
      this.report(error);
    }
  }

  /**
   * Get formatted error report
   */
  getReport(): ErrorReport {
    // Calculate summary
    let total = 0;
    let errors = 0;
    let warnings = 0;
    let info = 0;

    for (const group of this.errorGroups.values()) {
      for (const err of group) {
        total++;
        if (err.severity === 'error') errors++;
        else if (err.severity === 'warning') warnings++;
        else info++;
      }
    }

    // Generate formatted report
    const formatted = this.generateFormattedReport(total, errors, warnings, info);

    return {
      summary: { total, errors, warnings, info },
      groups: this.errorGroups,
      formatted
    };
  }

  /**
   * Generate formatted text report
   */
  private generateFormattedReport(
    total: number,
    errors: number,
    warnings: number,
    info: number
  ): string {
    const lines: string[] = [];

    lines.push('═'.repeat(60));
    lines.push('VALIDATION ERROR REPORT');
    lines.push('═'.repeat(60));
    lines.push('');
    lines.push(`Summary: ${total} issues (${errors} errors, ${warnings} warnings, ${info} info)`);
    lines.push('');

    // Group errors by path
    const sortedPaths = Array.from(this.errorGroups.keys()).sort();

    for (const path of sortedPaths) {
      const group = this.errorGroups.get(path)!;
      
      lines.push(`─`.repeat(60));
      lines.push(`Path: ${path}`);
      lines.push(`─`.repeat(60));

      for (const err of group) {
        const icon = err.severity === 'error' ? '❌' : (err.severity === 'warning' ? '⚠️' : 'ℹ️');
        lines.push(`${icon} ${err.message}`);
        
        if (err.code) {
          lines.push(`   Code: ${err.code}`);
        }
        
        if (err.details) {
          lines.push(`   Details: ${JSON.stringify(err.details, null, 2)}`);
        }
        
        lines.push('');
      }
    }

    lines.push('═'.repeat(60));

    return lines.join('\n');
  }

  /**
   * Get JSON report
   */
  getJsonReport(): JsonReport {
    const report = this.getReport();

    return {
      summary: report.summary,
      groups: Object.fromEntries(report.groups),
      timestamp: new Date().toISOString()
    };
  }

  /**
   * Get errors by severity
   */
  getBySeverity(severity: 'error' | 'warning' | 'info'): ReportedError[] {
    const results: ReportedError[] = [];

    for (const group of this.errorGroups.values()) {
      for (const err of group) {
        if (err.severity === severity) {
          results.push(err);
        }
      }
    }

    return results;
  }

  /**
   * Get errors by path
   */
  getByPath(path: string): ReportedError[] {
    return this.errorGroups.get(path) || [];
  }

  /**
   * Get errors by code
   */
  getByCode(code: string): ReportedError[] {
    const results: ReportedError[] = [];

    for (const group of this.errorGroups.values()) {
      for (const err of group) {
        if (err.code === code) {
          results.push(err);
        }
      }
    }

    return results;
  }

  /**
   * Clear all reported errors
   */
  clear(): void {
    this.errorGroups.clear();
  }

  /**
   * Check if there are any errors
   */
  hasErrors(): boolean {
    return this.getBySeverity('error').length > 0;
  }

  /**
   * Check if there are any warnings
   */
  hasWarnings(): boolean {
    return this.getBySeverity('warning').length > 0;
  }

  /**
   * Get evidence records
   */
  getEvidence(): EvidenceRecord[] {
    return this.evidence;
  }
}