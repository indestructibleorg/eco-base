// @ECO-governed
// @ECO-layer: GL20-29
// @ECO-semantic: compliance-analyzer
// @ECO-audit-trail: ../../governance/GL_SEMANTIC_ANCHOR.json

import * as fs from 'fs';
import { AnalysisResult, DocumentMetadata } from '../types';

interface ComplianceRequirement {
  standard: string;
  requirement: string;
  status: 'compliant' | 'non-compliant' | 'partial';
  reference: string;
}

export class ComplianceAnalyzer {
  private complianceStandards = {
    GDPR: [
      'Right to be forgotten',
      'Data portability',
      'Consent management',
      'Breach notification (72 hours)',
      'Data protection impact assessment'
    ],
    HIPAA: [
      'Privacy rule implementation',
      'Security rule safeguards',
      'Business associate agreements',
      'Access controls',
      'Audit controls'
    ],
    SOC2: [
      'Access management',
      'Change management',
      'Incident response',
      'Data encryption',
      'Monitoring and logging'
    ]
  };

  analyzeCompliance(filePath: string, metadata: DocumentMetadata, standard: string): AnalysisResult {
    const content = this.extractContent(filePath);
    const requirements = this.complianceStandards[standard as keyof typeof this.complianceStandards] || [];
    const gapAnalysis = this.performGapAnalysis(content, requirements, standard);

    return {
      type: 'compliance',
      summary: `Compliance analysis for ${standard}: ${gapAnalysis.compliantCount}/${requirements.length} requirements met`,
      findings: gapAnalysis.requirements,
      insights: [
        `Overall compliance score: ${gapAnalysis.complianceScore.toFixed(1)}%`,
        'Critical gaps identified in data protection',
        'Additional documentation required'
      ],
      recommendations: gapAnalysis.recommendations,
      sourceDocuments: [metadata],
      generatedAt: new Date()
    };
  }

  analyzeMultipleDocuments(files: { path: string; metadata: DocumentMetadata }[], standard: string): AnalysisResult {
    const allRequirements: ComplianceRequirement[] = [];
    const documents = files.map(f => this.extractContent(f.path));

    for (let i = 0; i < documents.length; i++) {
      const doc = documents[i];
      const reqs = this.complianceStandards[standard as keyof typeof this.complianceStandards] || [];
      
      for (const req of reqs) {
        const status = this.checkRequirement(doc, req);
        allRequirements.push({
          standard,
          requirement: req,
          status,
          reference: files[i].metadata.filename
        });
      }
    }

    const compliantCount = allRequirements.filter(r => r.status === 'compliant').length;
    const complianceScore = (compliantCount / allRequirements.length) * 100;

    return {
      type: 'compliance',
      summary: `Multi-document compliance analysis for ${standard}: ${compliantCount}/${allRequirements.length} requirements met`,
      findings: allRequirements,
      insights: [
        `Overall compliance score: ${complianceScore.toFixed(1)}%`,
        'Consistent gaps across multiple documents',
        'Standardized template needed'
      ],
      recommendations: [
        'Create compliance checklist',
        'Implement mandatory review process',
        'Develop standard templates'
      ],
      sourceDocuments: files.map(f => f.metadata),
      generatedAt: new Date()
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

  private performGapAnalysis(content: string, requirements: string[], standard: string): any {
    const checkedRequirements: ComplianceRequirement[] = [];
    let compliantCount = 0;

    for (const req of requirements) {
      const status = this.checkRequirement(content, req);
      if (status === 'compliant') compliantCount++;
      
      checkedRequirements.push({
        standard,
        requirement: req,
        status,
        reference: 'Document'
      });
    }

    const complianceScore = (compliantCount / requirements.length) * 100;
    const recommendations = this.generateRecommendations(checkedRequirements);

    return {
      requirements: checkedRequirements,
      compliantCount,
      complianceScore,
      recommendations
    };
  }

  private checkRequirement(content: string, requirement: string): 'compliant' | 'non-compliant' | 'partial' {
    const lowerContent = content.toLowerCase();
    const lowerReq = requirement.toLowerCase();
    
    // Simple keyword matching - in real scenario would use NLP
    const keywords = lowerReq.split(' ').filter(w => w.length > 3);
    const matchCount = keywords.filter(kw => lowerContent.includes(kw)).length;
    
    if (matchCount >= keywords.length * 0.7) {
      return 'compliant';
    } else if (matchCount >= keywords.length * 0.4) {
      return 'partial';
    }
    
    return 'non-compliant';
  }

  private generateRecommendations(requirements: ComplianceRequirement[]): string[] {
    const recommendations: string[] = [];
    
    const nonCompliant = requirements.filter(r => r.status === 'non-compliant');
    const partial = requirements.filter(r => r.status === 'partial');
    
    for (const req of nonCompliant) {
      recommendations.push(`Add clause for: ${req.requirement}`);
    }
    
    for (const req of partial) {
      recommendations.push(`Strengthen implementation of: ${req.requirement}`);
    }
    
    if (nonCompliant.length > 0) {
      recommendations.push('Conduct full compliance audit');
    }
    
    recommendations.push('Implement regular compliance reviews');
    recommendations.push('Train staff on compliance requirements');
    
    return recommendations;
  }
}