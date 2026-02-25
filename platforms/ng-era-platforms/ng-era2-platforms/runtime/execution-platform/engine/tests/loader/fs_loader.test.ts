/**
 * @ECO-governed
 * @ECO-layer: loader
 * @ECO-semantic: loader-fs_loader.test
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

import { FsLoader } from '../../loader/fs_loader';
import * as fs from 'fs';
import * as path from 'path';

describe('FsLoader', () => {
  const testDir = path.join(__dirname, '../fixtures/test-loader');
  
  beforeEach(() => {
    if (!fs.existsSync(testDir)) {
      fs.mkdirSync(testDir, { recursive: true });
      fs.writeFileSync(path.join(testDir, 'test.yaml'), 'name: test\nvalue: 123');
      fs.writeFileSync(path.join(testDir, 'test.json'), '{"name":"test","value":123}');
    }
  });

  afterEach(() => {
    if (fs.existsSync(testDir)) {
      fs.rmSync(testDir, { recursive: true, force: true });
    }
  });

  it('should load YAML and JSON files', async () => {
    const loader = new FsLoader(testDir);
    const result = await loader.load();

    expect(result.status).toBe('success');
    expect(result.files.size).toBeGreaterThan(0);
    expect(result.errors).toHaveLength(0);
  });

  it('should generate evidence records', async () => {
    const loader = new FsLoader(testDir);
    await loader.load();
    const evidence = loader.getEvidence();

    expect(evidence.length).toBeGreaterThan(0);
    expect(evidence[0]).toHaveProperty('timestamp');
    expect(evidence[0]).toHaveProperty('stage');
    expect(evidence[0]).toHaveProperty('component');
  });

  it('should handle non-existent directory', async () => {
    const loader = new FsLoader('/nonexistent/path');
    const result = await loader.load();

    expect(result.status).toBe('error');
    expect(result.errors.length).toBeGreaterThan(0);
  });

  it('should ignore specified patterns', async () => {
    const loader = new FsLoader(testDir, { ignore: ['*.json'] });
    const result = await loader.load();

    expect(result.status).toBe('success');
    const files = Array.from(result.files.keys()).filter(f => f.endsWith('.json'));
    expect(files.length).toBe(0);
  });
});
