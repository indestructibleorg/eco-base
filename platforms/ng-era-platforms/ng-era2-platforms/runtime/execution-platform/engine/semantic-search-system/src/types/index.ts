// @ECO-governed
// @ECO-layer: GL20-29
// @ECO-semantic: semantic-search-types
// @ECO-audit-trail: ../../governance/GL_SEMANTIC_ANCHOR.json

export interface DocumentMetadata {
  id: string;
  filename: string;
  path: string;
  fileType: string;
  size: number;
  lastModified: Date;
  module: string;
  glLayer?: string;
}

export interface SearchQuery {
  query: string;
  limit?: number;
  threshold?: number;
  modules?: string[];
  fileTypes?: string[];
}

export interface SearchResult {
  document: DocumentMetadata;
  score: number;
  context: string;
  pageNumber?: number;
  highlights: string[];
}

export interface AnalysisResult {
  type: 'search' | 'survey' | 'research' | 'data-quality' | 'contract' | 'compliance' | 'comparison' | 'sales';
  summary: string;
  findings: any[];
  insights: string[];
  recommendations?: string[];
  sourceDocuments: DocumentMetadata[];
  generatedAt: Date;
  redlineDocument?: string;
}

export interface SurveyAnalysis {
  responseRate: number;
  totalResponses: number;
  trends: any[];
  segments: any[];
  visualizations?: any[];
}

export interface DataQualityReport {
  missingValues: number;
  duplicates: number;
  outliers: number;
  inconsistencies: number;
  cleaningStrategies: string[];
}