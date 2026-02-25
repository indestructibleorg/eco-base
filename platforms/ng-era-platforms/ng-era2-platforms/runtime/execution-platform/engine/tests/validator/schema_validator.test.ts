/**
 * @ECO-governed
 * @ECO-layer: validator
 * @ECO-semantic: validator-schema_validator.test
 * @ECO-audit-trail: ../governance/GL_SEMANTIC_ANCHOR.json
 * 
 * GL Unified Charter Activated
 */

/**
 * @gl-layer ECO-50-OBSERVABILITY
 * @gl-module engine/tests/validator
 * @gl-semantic-anchor ECO-50-TEST-TS
 * @gl-evidence-required true
 */

import { SchemaValidator } from '../../validator/schema_validator';

describe('SchemaValidator', () => {
  const schema = {
    type: 'object',
    properties: {
      name: { type: 'string' },
      age: { type: 'number', minimum: 0 },
      email: { type: 'string', format: 'email' }
    },
    required: ['name', 'email']
  };

  it('should validate valid data against schema', async () => {
    const validator = new SchemaValidator();
    const validData = {
      name: 'John Doe',
      age: 30,
      email: 'john@example.com'
    };

    const result = await validator.validate(validData, schema);

    expect(result.valid).toBe(true);
    expect(result.errors).toHaveLength(0);
  });

  it('should reject invalid data against schema', async () => {
    const validator = new SchemaValidator();
    const invalidData = {
      name: 'John',
      age: -5,
      email: 'invalid-email'
    };

    const result = await validator.validate(invalidData, schema);

    expect(result.valid).toBe(false);
    expect(result.errors.length).toBeGreaterThan(0);
  });

  it('should generate evidence records', async () => {
    const validator = new SchemaValidator();
    await validator.validate({ name: 'test', email: 'test@test.com' }, schema);
    const evidence = validator.getEvidence();

    expect(evidence.length).toBeGreaterThan(0);
    expect(evidence[0]).toHaveProperty('timestamp');
    expect(evidence[0]).toHaveProperty('stage', 'validator');
  });

  it('should handle missing required fields', async () => {
    const validator = new SchemaValidator();
    const incompleteData = { name: 'John' };

    const result = await validator.validate(incompleteData, schema);

    expect(result.valid).toBe(false);
    expect(result.errors.some(e => e.includes('email'))).toBe(true);
  });
});
