/**
 * @ECO-governed
 * @ECO-layer: artifacts
 * @ECO-semantic: artifacts-evidence_chain.test
 * @ECO-audit-trail: ../governance/GL_SEMANTIC_ANCHOR.json
 * 
 * GL Unified Charter Activated
 */

/**
 * @gl-layer ECO-50-OBSERVABILITY
 * @gl-module engine/tests/artifacts
 * @gl-semantic-anchor ECO-50-TEST-TS
 * @gl-evidence-required true
 */

import { EvidenceChain } from '../../artifacts/evidence_chain';

describe('EvidenceChain', () => {
  it('should add evidence records', () => {
    const chain = new EvidenceChain();
    const record = {
      timestamp: new Date().toISOString(),
      stage: 'test',
      component: 'test-component',
      action: 'test-action',
      status: 'success' as const,
      input: {},
      output: {},
      metrics: {}
    };

    chain.add(record);
    const result = chain.generate();

    expect(result.evidence).toHaveLength(1);
    expect(result.evidence[0]).toEqual(record);
  });

  it('should add batch evidence records', () => {
    const chain = new EvidenceChain();
    const records = [
      {
        timestamp: new Date().toISOString(),
        stage: 'test1',
        component: 'test-component',
        action: 'test-action',
        status: 'success' as const,
        input: {},
        output: {},
        metrics: {}
      },
      {
        timestamp: new Date().toISOString(),
        stage: 'test2',
        component: 'test-component',
        action: 'test-action',
        status: 'success' as const,
        input: {},
        output: {},
        metrics: {}
      }
    ];

    chain.addBatch(records);
    const result = chain.generate();

    expect(result.evidence).toHaveLength(2);
  });

  it('should group evidence by stage', () => {
    const chain = new EvidenceChain();
    chain.add({
      timestamp: new Date().toISOString(),
      stage: 'loader',
      component: 'test',
      action: 'test',
      status: 'success',
      input: {},
      output: {},
      metrics: {}
    });
    chain.add({
      timestamp: new Date().toISOString(),
      stage: 'loader',
      component: 'test',
      action: 'test',
      status: 'success',
      input: {},
      output: {},
      metrics: {}
    });
    chain.add({
      timestamp: new Date().toISOString(),
      stage: 'validator',
      component: 'test',
      action: 'test',
      status: 'success',
      input: {},
      output: {},
      metrics: {}
    });

    const result = chain.generate();

    expect(result.byStage.get('loader')).toHaveLength(2);
    expect(result.byStage.get('validator')).toHaveLength(1);
  });

  it('should generate hash for evidence chain', () => {
    const chain = new EvidenceChain();
    chain.add({
      timestamp: new Date().toISOString(),
      stage: 'test',
      component: 'test',
      action: 'test',
      status: 'success',
      input: {},
      output: {},
      metrics: {}
    });

    const result = chain.generate();

    expect(result.hash).toBeDefined();
    expect(result.hash).toMatch(/^[a-f0-9]{64}$/);
  });

  it('should generate unique chain ID', () => {
    const chain1 = new EvidenceChain();
    const chain2 = new EvidenceChain();

    const id1 = chain1.generate().chainId;
    const id2 = chain2.generate().chainId;

    expect(id1).toBeDefined();
    expect(id2).toBeDefined();
    expect(id1).not.toBe(id2);
  });
});
