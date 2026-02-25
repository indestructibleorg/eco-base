/**
 * @ECO-governed
 * @ECO-layer: tests
 * @ECO-semantic: governance-rule_evaluator.test
 * @ECO-audit-trail: ../governance/GL_SEMANTIC_ANCHOR.json
 * 
 * GL Unified Charter Activated
 */

/**
 * @gl-layer ECO-50-OBSERVABILITY
 * @gl-module engine/tests/governance
 * @gl-semantic-anchor ECO-50-TEST-TS
 * @gl-evidence-required true
 */

import { RuleEvaluator } from '../../governance/rule_evaluator';

describe('RuleEvaluator', () => {
  it('should evaluate required rules', async () => {
    const evaluator = new RuleEvaluator();
    evaluator.registerRule('test-required', {
      type: 'required',
      path: 'customField',
      environments: ['production']
    });

    const result = await evaluator.evaluate({ 
      name: 'test-name', 
      metadata: {}, 
      customField: 'value' 
    }, 'production');

    expect(result.violations).toHaveLength(0);
  });

  it('should detect missing required fields', async () => {
    const evaluator = new RuleEvaluator();
    evaluator.registerRule('test-required', {
      type: 'required',
      path: 'customField',
      environments: ['production']
    });

    const result = await evaluator.evaluate({ 
      name: 'test-name', 
      metadata: {} 
    }, 'production');

    expect(result.violations.length).toBeGreaterThan(0);
    expect(result.violations.some(v => v.path === 'customField')).toBe(true);
  });

  it('should evaluate forbidden rules', async () => {
    const evaluator = new RuleEvaluator();
    evaluator.registerRule('test-forbidden', {
      type: 'forbidden',
      path: 'password',
      environments: ['production']
    });

    const result = await evaluator.evaluate({ password: 'secret' }, 'production');

    expect(result.violations.some(v => v.path === 'password')).toBe(true);
  });

  it('should evaluate pattern rules', async () => {
    const evaluator = new RuleEvaluator();
    evaluator.registerRule('test-pattern', {
      type: 'pattern',
      path: 'email',
      pattern: '^[^@]+@[^@]+\\.[^@]+$',
      environments: ['production']
    });

    const validResult = await evaluator.evaluate({ 
      name: 'test-name', 
      metadata: {}, 
      email: 'test@example.com' 
    }, 'production');
    expect(validResult.violations).toHaveLength(0);

    const invalidResult = await evaluator.evaluate({ 
      name: 'test-name', 
      metadata: {}, 
      email: 'invalid' 
    }, 'production');
    expect(invalidResult.violations.some(v => v.path === 'email')).toBe(true);
  });

  it('should evaluate range rules', async () => {
    const evaluator = new RuleEvaluator();
    evaluator.registerRule('test-range', {
      type: 'range',
      path: 'age',
      min: 0,
      max: 120,
      environments: ['production']
    });

    const validResult = await evaluator.evaluate({ 
      name: 'test-name', 
      metadata: {}, 
      age: 30 
    }, 'production');
    expect(validResult.violations).toHaveLength(0);

    const invalidResult = await evaluator.evaluate({ 
      name: 'test-name', 
      metadata: {}, 
      age: 150 
    }, 'production');
    expect(invalidResult.violations.some(v => v.path === 'age')).toBe(true);
  });

  it('should evaluate enum rules', async () => {
    const evaluator = new RuleEvaluator();
    evaluator.registerRule('test-enum', {
      type: 'enum',
      path: 'status',
      values: ['active', 'inactive', 'pending'],
      environments: ['production']
    });

    const validResult = await evaluator.evaluate({ 
      name: 'test-name', 
      metadata: {}, 
      status: 'active' 
    }, 'production');
    expect(validResult.violations).toHaveLength(0);

    const invalidResult = await evaluator.evaluate({ 
      name: 'test-name', 
      metadata: {}, 
      status: 'unknown' 
    }, 'production');
    expect(invalidResult.violations.some(v => v.path === 'status')).toBe(true);
  });

  it('should generate governance events', async () => {
    const evaluator = new RuleEvaluator();
    evaluator.registerRule('test-required', {
      type: 'required',
      path: 'customField',
      environments: ['production']
    });

    const result = await evaluator.evaluate({ 
      name: 'test-name', 
      metadata: {}, 
      customField: 'value' 
    }, 'production');

    expect(result.events.length).toBeGreaterThan(0);
    expect(result.events[0]).toHaveProperty('type');
    expect(result.events[0]).toHaveProperty('timestamp');
  });
});
