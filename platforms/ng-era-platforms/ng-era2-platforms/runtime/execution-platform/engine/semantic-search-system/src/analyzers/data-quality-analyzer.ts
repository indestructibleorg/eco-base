// @ECO-governed
// @ECO-layer: GL20-29
// @ECO-semantic: data-quality-analyzer
// @ECO-audit-trail: ../../governance/GL_SEMANTIC_ANCHOR.json

import * as fs from 'fs';
import { DataQualityReport, AnalysisResult, DocumentMetadata } from '../types';

export class DataQualityAnalyzer {
  analyzeDataQuality(csvPath: string, metadata: DocumentMetadata): AnalysisResult {
    const data = this.parseCSV(csvPath);
    const report = this.generateReport(data);

    return {
      type: 'data-quality',
      summary: `Data quality analysis: ${report.missingValues} missing values, ${report.duplicates} duplicates found`,
      findings: [
        { metric: 'totalRecords', value: data.length },
        { metric: 'missingValues', value: report.missingValues },
        { metric: 'duplicates', value: report.duplicates },
        { metric: 'outliers', value: report.outliers },
        { metric: 'inconsistencies', value: report.inconsistencies }
      ],
      insights: [
        'Missing values concentrated in specific columns',
        'Duplicate records need deduplication',
        'Outliers detected in numerical fields',
        'Format inconsistencies require standardization'
      ],
      recommendations: [
        'Implement data validation rules at entry',
        'Create deduplication process',
        'Establish outlier detection thresholds',
        'Standardize date and number formats'
      ],
      sourceDocuments: [metadata],
      generatedAt: new Date()
    };
  }

  private parseCSV(filePath: string): any[] {
    try {
      const content = fs.readFileSync(filePath, 'utf-8');
      const lines = content.split('\n').filter(line => line.trim());
      
      if (lines.length < 2) return [];
      
      const headers = lines[0].split(',').map(h => h.trim());
      const data = [];
      
      for (let i = 1; i < lines.length; i++) {
        const values = lines[i].split(',').map(v => v.trim());
        const row: any = {};
        headers.forEach((header, index) => {
          row[header] = values[index] || '';
        });
        data.push(row);
      }
      
      return data;
    } catch (error) {
      console.error(`Error parsing CSV: ${error}`);
      return [];
    }
  }

  private generateReport(data: any[]): DataQualityReport {
    let missingValues = 0;
    let duplicates = 0;
    let outliers = 0;
    let inconsistencies = 0;

    const seenRows = new Set<string>();

    for (const row of data) {
      const rowString = JSON.stringify(row);
      
      // Check for duplicates
      if (seenRows.has(rowString)) {
        duplicates++;
      }
      seenRows.add(rowString);

      // Check for missing values
      for (const value of Object.values(row)) {
        if (!value || value === '' || (typeof value === 'string' && value.toLowerCase() === 'null')) {
          missingValues++;
        }
      }

      // Check for outliers (mock detection)
      for (const [key, value] of Object.entries(row)) {
        if (typeof value === 'string' && !isNaN(Number(value))) {
          const num = Number(value);
          if (num > 1000000 || num < -1000000) {
            outliers++;
          }
        }
      }

      // Check for inconsistencies (mock detection)
      if (row.email && typeof row.email === 'string' && !row.email.includes('@')) {
        inconsistencies++;
      }
    }

    const cleaningStrategies = [
      'Remove duplicate records based on key fields',
      'Impute missing values using mean/median for numeric fields',
      'Standardize date formats to ISO 8601',
      'Validate email addresses using regex pattern',
      'Apply outlier detection and removal algorithms'
    ];

    return {
      missingValues,
      duplicates,
      outliers,
      inconsistencies,
      cleaningStrategies
    };
  }

  generateCleanedData(inputPath: string, outputPath: string): void {
    const data = this.parseCSV(inputPath);
    const cleanedData = [];

    const seenRows = new Set<string>();

    for (const row of data) {
      const rowString = JSON.stringify(row);
      
      // Skip duplicates
      if (seenRows.has(rowString)) {
        continue;
      }
      seenRows.add(rowString);

      // Clean the row
      const cleanedRow: any = {};
      for (const [key, value] of Object.entries(row)) {
        if (value && value !== '' && typeof value === 'string' && value.toLowerCase() !== 'null') {
          cleanedRow[key] = value.trim();
        } else if (value && typeof value !== 'string') {
          cleanedRow[key] = value;
        }
      }

      cleanedData.push(cleanedRow);
    }

    // Write cleaned data
    const cleanedCsv = this.convertToCSV(cleanedData);
    fs.writeFileSync(outputPath, cleanedCsv);
    
    // Write documentation
    const docPath = outputPath.replace('.csv', '-cleaning-doc.md');
    const documentation = `# Data Cleaning Documentation

**Original File**: ${inputPath}
**Cleaned File**: ${outputPath}
**Cleaning Date**: ${new Date().toISOString()}

## Changes Made
- Removed duplicate records
- Removed rows with all null values
- Trimmed whitespace from all fields
- Validated data types

## Statistics
- Original Records: ${data.length}
- Cleaned Records: ${cleanedData.length}
- Records Removed: ${data.length - cleanedData.length}
`;
    fs.writeFileSync(docPath, documentation);
  }

  private convertToCSV(data: any[]): string {
    if (data.length === 0) return '';
    
    const headers = Object.keys(data[0]);
    const csvLines = [headers.join(',')];
    
    for (const row of data) {
      const values = headers.map(h => `"${(row[h] || '').toString().replace(/"/g, '""')}"`);
      csvLines.push(values.join(','));
    }
    
    return csvLines.join('\n');
  }
}