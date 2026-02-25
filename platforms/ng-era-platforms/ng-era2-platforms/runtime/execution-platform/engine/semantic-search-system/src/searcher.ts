// @ECO-governed
// @ECO-layer: GL20-29
// @ECO-semantic: semantic-searcher
// @ECO-audit-trail: ../../governance/GL_SEMANTIC_ANCHOR.json

import * as fs from 'fs';
import { DocumentIndexer } from './indexer';
import { SearchQuery, SearchResult, DocumentMetadata } from './types';

export class SemanticSearcher {
  constructor(private indexer: DocumentIndexer) {}

  search(query: SearchQuery): SearchResult[] {
    const terms = query.query.toLowerCase().split(/\s+/);
    const results: Map<string, SearchResult> = new Map();

    for (const term of terms) {
      if (term.length < 2) continue;

      // Find documents containing this term in their filename
      const documents = this.indexer.getAllDocuments().filter(doc => {
        const filename = doc.filename.toLowerCase();
        const content = this.extractContent(doc.path);
        return filename.includes(term) || content.toLowerCase().includes(term);
      });

      for (const doc of documents) {
        const score = this.calculateScore(doc, term);
        
        if (!results.has(doc.id)) {
          results.set(doc.id, {
            document: doc,
            score: 0,
            context: '',
            highlights: []
          });
        }
        
        const result = results.get(doc.id)!;
        result.score += score;
        result.highlights.push(`Found "${term}" in ${doc.filename}`);
      }
    }

    // Filter by modules if specified
    let filteredResults = Array.from(results.values());
    if (query.modules && query.modules.length > 0) {
      filteredResults = filteredResults.filter(r => 
        query.modules!.includes(r.document.module)
      );
    }

    // Filter by file types if specified
    if (query.fileTypes && query.fileTypes.length > 0) {
      filteredResults = filteredResults.filter(r =>
        query.fileTypes!.includes(r.document.fileType)
      );
    }

    // Filter by threshold
    const threshold = query.threshold || 0.1;
    filteredResults = filteredResults.filter(r => r.score >= threshold);

    // Sort by score and limit
    filteredResults.sort((a, b) => b.score - a.score);
    return filteredResults.slice(0, query.limit || 50);
  }

  private extractContent(filePath: string): string {
    try {
      if (fs.existsSync(filePath)) {
        const content = fs.readFileSync(filePath, 'utf-8');
        return content.substring(0, 1000); // First 1000 chars for context
      }
    } catch (error) {
      // Ignore read errors
    }
    return '';
  }

  private calculateScore(doc: DocumentMetadata, term: string): number {
    let score = 0;
    
    // Filename match gets higher score
    if (doc.filename.toLowerCase().includes(term)) {
      score += 1.0;
    }
    
    // Path match gets medium score
    if (doc.path.toLowerCase().includes(term)) {
      score += 0.5;
    }
    
    // GL layer match
    if (doc.glLayer && doc.glLayer.toLowerCase().includes(term)) {
      score += 0.3;
    }
    
    return score;
  }

  searchByModule(module: string, query: string = ''): SearchResult[] {
    const documents = this.indexer.getAllDocuments().filter(doc => 
      doc.module === module
    );

    if (!query) {
      return documents.map(doc => ({
        document: doc,
        score: 1.0,
        context: '',
        highlights: ['All documents in module']
      }));
    }

    return this.search({ query, modules: [module] });
  }

  searchByGLLayer(glLayer: string, query: string = ''): SearchResult[] {
    const documents = this.indexer.getAllDocuments().filter(doc =>
      doc.glLayer === glLayer
    );

    if (!query) {
      return documents.map(doc => ({
        document: doc,
        score: 1.0,
        context: '',
        highlights: [`GL Layer: ${glLayer}`]
      }));
    }

    return this.search({ query });
  }

  searchByFileType(fileType: string, query: string = ''): SearchResult[] {
    const documents = this.indexer.getAllDocuments().filter(doc =>
      doc.fileType === fileType
    );

    if (!query) {
      return documents.map(doc => ({
        document: doc,
        score: 1.0,
        context: '',
        highlights: [`File type: ${fileType}`]
      }));
    }

    return this.search({ query, fileTypes: [fileType] });
  }

  getStatistics(): any {
    const documents = this.indexer.getAllDocuments();
    
    const byModule = new Map<string, number>();
    const byFileType = new Map<string, number>();
    const byGLLayer = new Map<string, number>();

    for (const doc of documents) {
      byModule.set(doc.module, (byModule.get(doc.module) || 0) + 1);
      byFileType.set(doc.fileType, (byFileType.get(doc.fileType) || 0) + 1);
      if (doc.glLayer) {
        byGLLayer.set(doc.glLayer, (byGLLayer.get(doc.glLayer) || 0) + 1);
      }
    }

    return {
      totalDocuments: documents.length,
      byModule: Object.fromEntries(byModule),
      byFileType: Object.fromEntries(byFileType),
      byGLLayer: Object.fromEntries(byGLLayer)
    };
  }
}