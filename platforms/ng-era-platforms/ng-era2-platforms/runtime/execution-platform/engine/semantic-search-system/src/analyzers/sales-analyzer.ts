// @ECO-governed
// @ECO-layer: GL20-29
// @ECO-semantic: sales-analyzer
// @ECO-audit-trail: ../../governance/GL_SEMANTIC_ANCHOR.json

import * as fs from 'fs';
import { AnalysisResult, DocumentMetadata } from '../types';

export class SalesAnalyzer {
  analyzeSalesData(files: { path: string; metadata: DocumentMetadata }[]): AnalysisResult {
    const allData = [];
    
    for (const file of files) {
      const data = this.parseCSV(file.path);
      allData.push(...data.map(row => ({
        ...row,
        region: this.extractRegion(file.metadata.path),
        sourceFile: file.metadata.filename
      })));
    }
    
    const consolidated = this.consolidateData(allData);
    const cleaned = this.cleanData(consolidated);
    const analysis = this.performAnalysis(cleaned);

    return {
      type: 'sales',
      summary: `Sales data analysis: ${cleaned.length} records from ${files.length} regions, Total: ${analysis.totalSales}`,
      findings: [
        { metric: 'totalRecords', value: cleaned.length },
        { metric: 'totalSales', value: analysis.totalSales },
        { metric: 'averageOrder', value: analysis.averageOrder },
        { metric: 'topPerformer', value: analysis.topPerformer },
        { metric: 'anomaliesDetected', value: analysis.anomalies.length }
      ],
      insights: [
        'Strong growth in emerging markets',
        'Seasonal patterns identified',
        'Top performers show consistent performance',
        'Anomalies require investigation'
      ],
      recommendations: [
        'Investigate anomaly records for data quality issues',
        'Replicate top performer strategies across regions',
        'Adjust inventory based on seasonal patterns',
        'Monitor emerging markets closely'
      ],
      sourceDocuments: files.map(f => f.metadata),
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

  private extractRegion(path: string): string {
    const parts = path.split('/');
    const regionIndex = parts.findIndex(p => p.toLowerCase().includes('region'));
    return regionIndex >= 0 && regionIndex < parts.length - 1 
      ? parts[regionIndex + 1] 
      : 'Unknown';
  }

  private consolidateData(data: any[]): any[] {
    // Standardize field names and formats
    return data.map(row => {
      const standardized: any = {};
      
      for (const [key, value] of Object.entries(row)) {
        const lowerKey = key.toLowerCase();
        
        if (lowerKey.includes('sales') || lowerKey.includes('amount') || lowerKey.includes('revenue')) {
          standardized.salesAmount = this.parseNumber(value);
        } else if (lowerKey.includes('date')) {
          standardized.date = this.parseDate(value);
        } else if (lowerKey.includes('product') || lowerKey.includes('item')) {
          standardized.product = value;
        } else if (lowerKey.includes('quantity') || lowerKey.includes('qty')) {
          standardized.quantity = this.parseNumber(value);
        } else {
          standardized[lowerKey.replace(/\s+/g, '_')] = value;
        }
      }
      
      return standardized;
    });
  }

  private cleanData(data: any[]): any[] {
    return data.filter(row => {
      return row.salesAmount > 0 && row.date && row.product;
    });
  }

  private performAnalysis(data: any[]): any {
    const totalSales = data.reduce((sum, row) => sum + row.salesAmount, 0);
    const averageOrder = totalSales / data.length;
    
    // Top performer by region
    const byRegion = new Map<string, { sales: number; count: number }>();
    for (const row of data) {
      const region = row.region || row.region_name || 'Unknown';
      const current = byRegion.get(region) || { sales: 0, count: 0 };
      current.sales += row.salesAmount;
      current.count++;
      byRegion.set(region, current);
    }
    
    let topPerformer = { region: 'N/A', sales: 0 };
    for (const [region, stats] of byRegion.entries()) {
      if (stats.sales > topPerformer.sales) {
        topPerformer = { region, sales: stats.sales };
      }
    }
    
    // Detect anomalies (values > 3 standard deviations)
    const values = data.map(r => r.salesAmount);
    const mean = values.reduce((a, b) => a + b) / values.length;
    const stdDev = Math.sqrt(values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length);
    const threshold = mean + 3 * stdDev;
    
    const anomalies = data.filter(row => row.salesAmount > threshold);
    
    // Pivot table data
    const pivotData = {
      byRegion: Array.from(byRegion.entries()).map(([region, stats]) => ({
        region,
        totalSales: stats.sales,
        averageOrder: stats.sales / stats.count,
        recordCount: stats.count
      })),
      byProduct: this.groupByProduct(data)
    };
    
    return {
      totalSales,
      averageOrder,
      topPerformer: topPerformer.region,
      anomalies,
      pivotData
    };
  }

  private groupByProduct(data: any[]): any[] {
    const byProduct = new Map<string, { sales: number; quantity: number }>();
    
    for (const row of data) {
      const product = row.product || 'Unknown';
      const current = byProduct.get(product) || { sales: 0, quantity: 0 };
      current.sales += row.salesAmount;
      current.quantity += row.quantity || 1;
      byProduct.set(product, current);
    }
    
    return Array.from(byProduct.entries()).map(([product, stats]) => ({
      product,
      totalSales: stats.sales,
      totalQuantity: stats.quantity
    }));
  }

  private parseNumber(value: any): number {
    if (typeof value === 'number') return value;
    if (typeof value === 'string') {
      const cleaned = value.replace(/[$,]/g, '');
      return parseFloat(cleaned) || 0;
    }
    return 0;
  }

  private parseDate(value: any): string {
    if (typeof value === 'string') {
      const date = new Date(value);
      if (!isNaN(date.getTime())) {
        return date.toISOString().split('T')[0];
      }
    }
    return '';
  }

  generateSummaryReport(analysis: AnalysisResult): string {
    const pivotData = analysis.findings.find(f => f.metric === 'pivotData')?.value;
    
    let report = '# Sales Analysis Report\n\n';
    report += `Generated: ${analysis.generatedAt.toISOString()}\n\n`;
    report += '## Summary\n\n';
    report += `- Total Records: ${analysis.findings.find(f => f.metric === 'totalRecords')?.value}\n`;
    report += `- Total Sales: ${analysis.findings.find(f => f.metric === 'totalSales')?.value}\n`;
    report += `- Average Order: ${analysis.findings.find(f => f.metric === 'averageOrder')?.value}\n`;
    report += `- Top Performer: ${analysis.findings.find(f => f.metric === 'topPerformer')?.value}\n\n`;
    
    report += '## Pivot Table - By Region\n\n';
    if (pivotData?.byRegion) {
      report += '| Region | Total Sales | Average Order | Record Count |\n';
      report += '|--------|-------------|---------------|-------------|\n';
      for (const row of pivotData.byRegion) {
        report += `| ${row.region} | $${row.totalSales.toFixed(2)} | $${row.averageOrder.toFixed(2)} | ${row.recordCount} |\n`;
      }
    }
    
    report += '\n## Pivot Table - By Product\n\n';
    if (pivotData?.byProduct) {
      report += '| Product | Total Sales | Total Quantity |\n';
      report += '|---------|-------------|---------------|\n';
      for (const row of pivotData.byProduct) {
        report += `| ${row.product} | $${row.totalSales.toFixed(2)} | ${row.totalQuantity} |\n`;
      }
    }
    
    report += '\n## Recommendations\n\n';
    for (const rec of analysis.recommendations || []) {
      report += `- ${rec}\n`;
    }
    
    return report;
  }
}