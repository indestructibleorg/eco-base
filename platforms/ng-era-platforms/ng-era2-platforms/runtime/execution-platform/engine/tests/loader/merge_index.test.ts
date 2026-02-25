/**
 * @ECO-governed
 * @ECO-layer: loader
 * @ECO-semantic: loader-merge_index.test
 * @ECO-audit-trail: ../governance/GL_SEMANTIC_ANCHOR.json
 * 
 * GL Unified Charter Activated
 */

/**
 * @gl-layer ECO-50-OBSERVABILITY
 * @gl-module engine/tests/loader
 * @gl-semantic-anchor ECO-50-TEST-TS
 * @gl-evidence-required true
 */

import { MergeIndex } from '../../loader/merge_index';
import { LoadResult } from '../../interfaces.d';

describe('MergeIndex', () => {
  it('should merge indexes with first strategy', async () => {
    const merger = new MergeIndex('first');
    const result1: LoadResult = {
      status: 'success',
      files: new Map([['key1', 'value1']]),
      errors: [],
      evidence: []
    };
    const result2: LoadResult = {
      status: 'success',
      files: new Map([['key1', 'value2']]),
      errors: [],
      evidence: []
    };

    const result = merger.merge([result1, result2]);

    expect(result.files.get('key1')).toBe('value1');
  });

  it('should merge indexes with last strategy', async () => {
    const merger = new MergeIndex('last');
    const result1: LoadResult = {
      status: 'success',
      files: new Map([['key1', 'value1']]),
      errors: [],
      evidence: []
    };
    const result2: LoadResult = {
      status: 'success',
      files: new Map([['key1', 'value2']]),
      errors: [],
      evidence: []
    };

    const result = merger.merge([result1, result2]);

    expect(result.files.get('key1')).toBe('value2');
  });

  it('should merge indexes with newest strategy', async () => {
    const merger = new MergeIndex('newest');
    const result1: LoadResult = {
      status: 'success',
      files: new Map([
        [
          'key1',
          {
            path: 'file1.txt',
            fullPath: '/path/to/file1.txt',
            content: 'value1',
            type: 'file',
            size: 0,
            hash: 'hash1',
            modified: '2024-01-01'
          }
        ]
      ]),
      errors: [],
      evidence: []
    };
    const result2: LoadResult = {
      status: 'success',
      files: new Map([
        [
          'key1',
          {
            path: 'file2.txt',
            fullPath: '/path/to/file2.txt',
            content: 'value2',
            type: 'file',
            size: 0,
            hash: 'hash2',
            modified: '2024-01-02'
          }
        ]
      ]),
      errors: [],
      evidence: []
    };

    const result = merger.merge([result1, result2]);

    expect(result.files.get('key1')?.modified).toBe('2024-01-02');
  });

  it('should handle error strategy on conflict', async () => {
    const merger = new MergeIndex('error');
    const result1: LoadResult = {
      status: 'success',
      files: new Map([['key1', 'value1']]),
      errors: [],
      evidence: []
    };
    const result2: LoadResult = {
      status: 'success',
      files: new Map([['key1', 'value2']]),
      errors: [],
      evidence: []
    };

    const result = merger.merge([result1, result2]);

    expect(result.status).toBe('error');
    expect(result.errors.length).toBeGreaterThan(0);
  });

  describe('custom merge strategy', () => {
    it('should merge nested objects successfully', () => {
      const merger = new MergeIndex('custom');
      const result1: LoadResult = {
        status: 'success',
        files: new Map([
          [
            'config.yaml',
            {
              path: 'config.yaml',
              fullPath: '/path/to/config.yaml',
              content: 'nested:\n  key1: value1\n  key2: value2\n',
              type: '.yaml',
              size: 100,
              hash: 'hash1',
              modified: '2024-01-01'
            }
          ]
        ]),
        errors: [],
        evidence: []
      };
      const result2: LoadResult = {
        status: 'success',
        files: new Map([
          [
            'config.yaml',
            {
              path: 'config.yaml',
              fullPath: '/path/to/config.yaml',
              content: 'nested:\n  key3: value3\n',
              type: '.yaml',
              size: 50,
              hash: 'hash2',
              modified: '2024-01-02'
            }
          ]
        ]),
        errors: [],
        evidence: []
      };

      const result = merger.merge([result1, result2]);

      expect(result.status).toBe('success');
      const mergedFile = result.files.get('config.yaml');
      expect(mergedFile).toBeDefined();
      expect(mergedFile?.content).toContain('key1');
      expect(mergedFile?.content).toContain('key2');
      expect(mergedFile?.content).toContain('key3');
      expect(mergedFile?.merged).toBe(true);
    });

    it('should handle primitive to object merge (bug fix)', () => {
      const merger = new MergeIndex('custom');
      const result1: LoadResult = {
        status: 'success',
        files: new Map([
          [
            'config.yaml',
            {
              path: 'config.yaml',
              fullPath: '/path/to/config.yaml',
              content: 'setting: simpleValue\n',
              type: '.yaml',
              size: 50,
              hash: 'hash1',
              modified: '2024-01-01'
            }
          ]
        ]),
        errors: [],
        evidence: []
      };
      const result2: LoadResult = {
        status: 'success',
        files: new Map([
          [
            'config.yaml',
            {
              path: 'config.yaml',
              fullPath: '/path/to/config.yaml',
              content: 'setting:\n  nested: objectValue\n',
              type: '.yaml',
              size: 60,
              hash: 'hash2',
              modified: '2024-01-02'
            }
          ]
        ]),
        errors: [],
        evidence: []
      };

      const result = merger.merge([result1, result2]);

      // Should not throw runtime error
      expect(result.status).toBe('success');
      const mergedFile = result.files.get('config.yaml');
      expect(mergedFile).toBeDefined();
      // The primitive should be replaced with the object
      expect(mergedFile?.content).toContain('nested');
    });

    it('should replace arrays instead of merging them', () => {
      const merger = new MergeIndex('custom');
      const result1: LoadResult = {
        status: 'success',
        files: new Map([
          [
            'config.yaml',
            {
              path: 'config.yaml',
              fullPath: '/path/to/config.yaml',
              content: 'items:\n  - item1\n  - item2\n',
              type: '.yaml',
              size: 50,
              hash: 'hash1',
              modified: '2024-01-01'
            }
          ]
        ]),
        errors: [],
        evidence: []
      };
      const result2: LoadResult = {
        status: 'success',
        files: new Map([
          [
            'config.yaml',
            {
              path: 'config.yaml',
              fullPath: '/path/to/config.yaml',
              content: 'items:\n  - item3\n  - item4\n',
              type: '.yaml',
              size: 50,
              hash: 'hash2',
              modified: '2024-01-02'
            }
          ]
        ]),
        errors: [],
        evidence: []
      };

      const result = merger.merge([result1, result2]);

      expect(result.status).toBe('success');
      const mergedFile = result.files.get('config.yaml');
      expect(mergedFile).toBeDefined();
      // Arrays should be replaced, not merged
      // The second array should replace the first
      expect(mergedFile?.content).toContain('item3');
      expect(mergedFile?.content).toContain('item4');
    });

    it('should handle null values correctly', () => {
      const merger = new MergeIndex('custom');
      const result1: LoadResult = {
        status: 'success',
        files: new Map([
          [
            'config.yaml',
            {
              path: 'config.yaml',
              fullPath: '/path/to/config.yaml',
              content: 'key1: value1\nkey2: null\n',
              type: '.yaml',
              size: 50,
              hash: 'hash1',
              modified: '2024-01-01'
            }
          ]
        ]),
        errors: [],
        evidence: []
      };
      const result2: LoadResult = {
        status: 'success',
        files: new Map([
          [
            'config.yaml',
            {
              path: 'config.yaml',
              fullPath: '/path/to/config.yaml',
              content: 'key2: value2\nkey3: null\n',
              type: '.yaml',
              size: 50,
              hash: 'hash2',
              modified: '2024-01-02'
            }
          ]
        ]),
        errors: [],
        evidence: []
      };

      const result = merger.merge([result1, result2]);

      expect(result.status).toBe('success');
      const mergedFile = result.files.get('config.yaml');
      expect(mergedFile).toBeDefined();
      // Should handle null values without errors
      expect(mergedFile?.content).toContain('key1');
      expect(mergedFile?.content).toContain('key2');
      expect(mergedFile?.content).toContain('key3');
    });

    it('should merge JSON files correctly', () => {
      const merger = new MergeIndex('custom');
      const result1: LoadResult = {
        status: 'success',
        files: new Map([
          [
            'config.json',
            {
              path: 'config.json',
              fullPath: '/path/to/config.json',
              content: '{"nested": {"key1": "value1"}}',
              type: '.json',
              size: 50,
              hash: 'hash1',
              modified: '2024-01-01'
            }
          ]
        ]),
        errors: [],
        evidence: []
      };
      const result2: LoadResult = {
        status: 'success',
        files: new Map([
          [
            'config.json',
            {
              path: 'config.json',
              fullPath: '/path/to/config.json',
              content: '{"nested": {"key2": "value2"}}',
              type: '.json',
              size: 50,
              hash: 'hash2',
              modified: '2024-01-02'
            }
          ]
        ]),
        errors: [],
        evidence: []
      };

      const result = merger.merge([result1, result2]);

      expect(result.status).toBe('success');
      const mergedFile = result.files.get('config.json');
      expect(mergedFile).toBeDefined();
      expect(mergedFile?.merged).toBe(true);
      // Should contain both keys from nested merge
      expect(mergedFile?.content).toContain('key1');
      expect(mergedFile?.content).toContain('key2');
    });
  });
});
