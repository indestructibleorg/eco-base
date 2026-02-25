// @ECO-governed
// @ECO-layer: GL20-29
// @ECO-semantic: semantic-search-main
// @ECO-audit-trail: ../../governance/GL_SEMANTIC_ANCHOR.json

import * as path from 'path';
import { DocumentIndexer } from './indexer';
import { SemanticSearcher } from './searcher';
import * as analyzers from './analyzers';
import { AnalysisResult, SearchQuery } from './types';

export class SemanticSearchSystem {
  private indexer: DocumentIndexer;
  private searcher: SemanticSearcher;

  constructor(rootPath: string) {
    this.indexer = new DocumentIndexer(rootPath);
    this.searcher = new SemanticSearcher(this.indexer);
  }

  async initialize(): Promise<void> {
    console.log('Initializing Semantic Search System...');
    const rootPath = process.cwd();
    await this.indexer.indexDirectory(rootPath);
    console.log('System initialized successfully');
  }

  search(query: SearchQuery) {
    return this.searcher.search(query);
  }

  searchByModule(module: string, query: string = '') {
    return this.searcher.searchByModule(module, query);
  }

  searchByGLLayer(glLayer: string, query: string = '') {
    return this.searcher.searchByGLLayer(glLayer, query);
  }

  searchByFileType(fileType: string, query: string = '') {
    return this.searcher.searchByFileType(fileType, query);
  }

  getStatistics() {
    return this.searcher.getStatistics();
  }

  analyzeSurvey(csvPath: string, metadata: any): AnalysisResult {
    const analyzer = new analyzers.SurveyAnalyzer();
    return analyzer.analyzeSurveyData(csvPath, metadata);
  }

  analyzeResearch(pdfPath: string, metadata: any): AnalysisResult {
    const analyzer = new analyzers.ResearchAnalyzer();
    return analyzer.analyzeResearchPaper(pdfPath, metadata);
  }

  analyzeDataQuality(csvPath: string, metadata: any): AnalysisResult {
    const analyzer = new analyzers.DataQualityAnalyzer();
    return analyzer.analyzeDataQuality(csvPath, metadata);
  }

  analyzeContract(filePath: string, metadata: any): AnalysisResult {
    const analyzer = new analyzers.ContractAnalyzer();
    return analyzer.analyzeContract(filePath, metadata);
  }

  analyzeCompliance(filePath: string, metadata: any, standard: string): AnalysisResult {
    const analyzer = new analyzers.ComplianceAnalyzer();
    return analyzer.analyzeCompliance(filePath, metadata, standard);
  }

  compareDocuments(file1: string, file2: string, metadata1: any, metadata2: any): AnalysisResult {
    const analyzer = new analyzers.DocumentComparator();
    return analyzer.compareDocuments(file1, file2, metadata1, metadata2);
  }

  analyzeSales(files: { path: string; metadata: any }[]): AnalysisResult {
    const analyzer = new analyzers.SalesAnalyzer();
    return analyzer.analyzeSalesData(files);
  }

  exportIndex(outputPath: string): void {
    this.indexer.saveIndex(outputPath);
  }
}

// Export types
export * from './types';
export { DocumentIndexer } from './indexer';
export { SemanticSearcher } from './searcher';
export * from './analyzers';