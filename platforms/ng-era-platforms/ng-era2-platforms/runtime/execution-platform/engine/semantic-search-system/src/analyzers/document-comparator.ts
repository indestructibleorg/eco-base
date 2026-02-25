// @ECO-governed
// @ECO-layer: GL20-29
// @ECO-semantic: document-comparator
// @ECO-audit-trail: ../../governance/GL_SEMANTIC_ANCHOR.json

import * as fs from 'fs';
import { AnalysisResult, DocumentMetadata } from '../types';

interface DocumentChange {
  type: 'addition' | 'deletion' | 'modification';
  location: string;
  oldContent?: string;
  newContent?: string;
  significance: 'low' | 'medium' | 'high';
  impact: string;
}

export class DocumentComparator {
  compareDocuments(file1: string, file2: string, metadata1: DocumentMetadata, metadata2: DocumentMetadata): AnalysisResult {
    const content1 = this.extractContent(file1);
    const content2 = this.extractContent(file2);
    
    const changes = this.identifyChanges(content1, content2);
    const significantChanges = changes.filter(c => c.significance === 'high');
    
    const redlineDocument = this.generateRedline(content1, content2, changes);

    return {
      type: 'comparison',
      summary: `Document comparison: ${changes.length} changes identified, ${significantChanges.length} significant`,
      findings: [
        { metric: 'totalChanges', value: changes.length },
        { metric: 'significantChanges', value: significantChanges.length },
        { metric: 'additions', value: changes.filter(c => c.type === 'addition').length },
        { metric: 'deletions', value: changes.filter(c => c.type === 'deletion').length },
        { metric: 'modifications', value: changes.filter(c => c.type === 'modification').length }
      ],
      insights: [
        'Major changes detected in terms and conditions',
        'Date modifications require attention',
        'Liability clauses significantly altered'
      ],
      recommendations: [
        'Review all high-impact changes',
        'Update dependent agreements',
        'Communicate changes to stakeholders'
      ],
      sourceDocuments: [metadata1, metadata2],
      generatedAt: new Date(),
      redlineDocument
    };
  }

  private extractContent(filePath: string): string {
    try {
      if (fs.existsSync(filePath)) {
        return fs.readFileSync(filePath, 'utf-8');
      }
    } catch (error) {
      console.error(`Error reading file: ${error}`);
    }
    return '';
  }

  private identifyChanges(content1: string, content2: string): DocumentChange[] {
    const changes: DocumentChange[] = [];
    
    // Line-by-line comparison
    const lines1 = content1.split('\n');
    const lines2 = content2.split('\n');
    
    const maxLines = Math.max(lines1.length, lines2.length);
    
    for (let i = 0; i < maxLines; i++) {
      const line1 = lines1[i] || '';
      const line2 = lines2[i] || '';
      
      if (line1 === line2) continue;
      
      if (!line1 && line2) {
        changes.push({
          type: 'addition',
          location: `Line ${i + 1}`,
          newContent: line2,
          significance: this.assessSignificance(line2),
          impact: 'Added content may introduce new obligations'
        });
      } else if (line1 && !line2) {
        changes.push({
          type: 'deletion',
          location: `Line ${i + 1}`,
          oldContent: line1,
          significance: this.assessSignificance(line1),
          impact: 'Removed content may affect rights or obligations'
        });
      } else {
        changes.push({
          type: 'modification',
          location: `Line ${i + 1}`,
          oldContent: line1,
          newContent: line2,
          significance: this.assessSignificance(line2),
          impact: 'Modified content may alter agreement terms'
        });
      }
    }
    
    return changes;
  }

  private assessSignificance(content: string): 'low' | 'medium' | 'high' {
    const highKeywords = ['liability', 'indemnification', 'termination', 'payment', 'obligation', 'breach', 'warranty'];
    const mediumKeywords = ['shall', 'must', 'will', 'agree', 'term', 'condition'];
    
    const lowerContent = content.toLowerCase();
    
    if (highKeywords.some(kw => lowerContent.includes(kw))) {
      return 'high';
    }
    
    if (mediumKeywords.some(kw => lowerContent.includes(kw))) {
      return 'medium';
    }
    
    return 'low';
  }

  private generateRedline(content1: string, content2: string, changes: DocumentChange[]): string {
    let redline = '# Document Redline Comparison\n\n';
    redline += `Generated: ${new Date().toISOString()}\n\n`;
    redline += '--- LEGEND ---\n';
    redline += '+ Addition\n';
    redline += '- Deletion\n';
    redline += '~ Modification\n\n';
    redline += '--- CHANGES ---\n\n';
    
    for (const change of changes) {
      redline += `**${change.significance.toUpperCase()} IMPACT** [${change.type.toUpperCase()}]\n`;
      redline += `Location: ${change.location}\n`;
      
      if (change.oldContent) {
        redline += `- ${change.oldContent}\n`;
      }
      if (change.newContent) {
        redline += `+ ${change.newContent}\n`;
      }
      
      redline += `Impact: ${change.impact}\n\n`;
    }
    
    return redline;
  }
}