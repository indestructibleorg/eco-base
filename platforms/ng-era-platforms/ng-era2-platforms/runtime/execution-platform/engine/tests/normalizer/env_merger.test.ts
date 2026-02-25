/**
 * @ECO-governed
 * @ECO-layer: normalizer
 * @ECO-semantic: normalizer-env_merger.test
 * @ECO-audit-trail: ../governance/GL_SEMANTIC_ANCHOR.json
 * 
 * GL Unified Charter Activated
 */

/**
 * @gl-layer ECO-50-OBSERVABILITY
 * @gl-module engine/tests/normalizer
 * @gl-semantic-anchor ECO-50-TEST-TS
 * @gl-evidence-required true
 */

import { EnvMerger } from '../../normalizer/env_merger';

describe('EnvMerger', () => {
  it('should merge environment config with base config', async () => {
    const merger = new EnvMerger();
    const baseConfig = { name: 'app', version: '1.0', env: 'dev' };
    const envConfig = { env: 'prod', debug: false };

    const result = await merger.merge(baseConfig, envConfig, 'production');

    expect(result.merged.name).toBe('app');
    expect(result.merged.env).toBe('prod');
    expect(result.merged.debug).toBe(false);
  });

  it('should perform deep merge', async () => {
    const merger = new EnvMerger();
    const baseConfig = { 
      config: { 
        database: { host: 'localhost', port: 5432 },
        cache: { ttl: 300 }
      }
    };
    const envConfig = {
      config: {
        database: { host: 'prod-db', port: 5433 },
        cache: { enabled: true }
      }
    };

    const result = await merger.merge(baseConfig, envConfig, 'production');

    expect(result.merged.config.database.host).toBe('prod-db');
    expect(result.merged.config.database.port).toBe(5433);
    expect(result.merged.config.cache.ttl).toBe(300);
    expect(result.merged.config.cache.enabled).toBe(true);
  });

  it('should track override changes', async () => {
    const merger = new EnvMerger();
    const baseConfig = { name: 'app', env: 'dev' };
    const envConfig = { env: 'prod' };

    const result = await merger.merge(baseConfig, envConfig, 'production');

    expect(result.overrides).toBeDefined();
    expect(result.overrides.length).toBeGreaterThan(0);
  });
});
