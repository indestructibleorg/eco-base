// @ECO-governed
// @ECO-layer: GL20-29
// @ECO-semantic: document-indexer
// @ECO-audit-trail: ../../governance/GL_SEMANTIC_ANCHOR.json

import * as fs from 'fs';
import * as path from 'path';
import { DocumentMetadata } from './types';

export class DocumentIndexer {
  private indexedDocuments: Map<string, DocumentMetadata> = new Map();
  private searchIndex: Map<string, string[]> = new Map(); // word -> document IDs

  constructor(private rootPath: string) {}

  async indexDirectory(directory: string): Promise<void> {
    const files = this.walkDirectory(directory);
    console.log(`Found ${files.length} files to index`);
    
    for (const file of files) {
      await this.indexFile(file);
    }
    
    console.log(`Indexed ${this.indexedDocuments.size} documents`);
  }

  private walkDirectory(dir: string): string[] {
    const files: string[] = [];
    
    const entries = fs.readdirSync(dir, { withFileTypes: true });
    
    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);
      
      if (entry.isDirectory()) {
        // Skip node_modules, .git, outputs
        if (!['node_modules', '.git', 'outputs', 'dist', 'build'].includes(entry.name)) {
          files.push(...this.walkDirectory(fullPath));
        }
      } else if (entry.isFile()) {
        const ext = path.extname(entry.name).toLowerCase();
        const supportedExts = ['.pdf', '.docx', '.xlsx', '.pptx', '.csv', '.json', '.yaml', '.yml', '.txt', '.md', '.ts', '.js', '.py'];
        
        if (supportedExts.includes(ext)) {
          files.push(fullPath);
        }
      }
    }
    
    return files;
  }

  private async indexFile(filePath: string): Promise<void> {
    const stats = fs.statSync(filePath);
    const relativePath = path.relative(this.rootPath, filePath);
    
    // Extract module from path
    const parts = relativePath.split(path.sep);
    const module = parts[0] || 'root';
    
    // Extract GL layer from path if applicable
    let glLayer: string | undefined;
    if (parts.some(p => p.includes('GL') || p.includes('governance'))) {
      glLayer = this.extractGLLayer(relativePath);
    }
    
    const metadata: DocumentMetadata = {
      id: this.generateId(relativePath),
      filename: path.basename(filePath),
      path: relativePath,
      fileType: path.extname(filePath).replace('.', ''),
      size: stats.size,
      lastModified: stats.mtime,
      module,
      glLayer
    };
    
    this.indexedDocuments.set(metadata.id, metadata);
    
    // Simple word-based indexing (placeholder for semantic search)
    const words = path.basename(filePath, path.extname(filePath)).toLowerCase().split(/[\s_-]+/);
    for (const word of words) {
      if (word.length > 2) {
        if (!this.searchIndex.has(word)) {
          this.searchIndex.set(word, []);
        }
        this.searchIndex.get(word)!.push(metadata.id);
      }
    }
  }

  private generateId(path: string): string {
    return Buffer.from(path).toString('base64').replace(/[/+=]/g, '');
  }

  private extractGLLayer(filePath: string): string | undefined {
    if (filePath.includes('GL00') || filePath.includes('strategic')) return 'GL00-09';
    if (filePath.includes('GL10') || filePath.includes('operational')) return 'GL10-29';
    if (filePath.includes('GL20') || filePath.includes('data')) return 'GL20-29';
    if (filePath.includes('GL30') || filePath.includes('execution')) return 'GL30-49';
    if (filePath.includes('GL50') || filePath.includes('observability')) return 'GL50-59';
    if (filePath.includes('GL60') || filePath.includes('feedback')) return 'GL60-80';
    if (filePath.includes('GL90') || filePath.includes('meta')) return 'GL90-99';
    return undefined;
  }

  getDocument(id: string): DocumentMetadata | undefined {
    return this.indexedDocuments.get(id);
  }

  getAllDocuments(): DocumentMetadata[] {
    return Array.from(this.indexedDocuments.values());
  }

  getDocumentCount(): number {
    return this.indexedDocuments.size;
  }

  exportIndex(): any {
    return {
      metadata: Array.from(this.indexedDocuments.values()),
      stats: {
        totalDocuments: this.indexedDocuments.size,
        totalWords: this.searchIndex.size,
        indexedAt: new Date().toISOString()
      }
    };
  }

  saveIndex(outputPath: string): void {
    const indexData = this.exportIndex();
    fs.writeFileSync(outputPath, JSON.stringify(indexData, null, 2));
    console.log(`Index saved to ${outputPath}`);
  }
}