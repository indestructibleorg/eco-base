//# @GL-governed
//# @GL-layer: GL90-99
//# @GL-semantic: archive-tools
//# @GL-audit-trail: ../../engine/governance/GL_SEMANTIC_ANCHOR.json
//#
//# GL Unified Charter Activated
// @GL-status: active

/**
 * SynergyMesh Configuration Validator
 * 
 * Validates all YAML config files against their JSON schemas
 * and checks cross-references between configs.
 */

const fs = require('fs');
const path = require('path');
const yaml = require('js-yaml');
const Ajv = require('ajv');
const addFormats = require('ajv-formats');

const ajv = new Ajv({ allErrors: true, strict: false });
addFormats(ajv);

// Config files and their schemas
const CONFIG_FILES = [
  { config: 'config/environment.yaml', schema: 'schemas/environment.schema.json' },
  { config: 'config/dependencies.yaml', schema: 'schemas/dependencies.schema.json' },
  { config: 'config/ai-constitution.yaml', schema: 'schemas/ai-constitution.schema.json' },
  { config: 'config/agents/team/virtual-experts.yaml', schema: 'config/agents/schemas/virtual-experts.schema.json' },
  { config: 'config/safety-mechanisms.yaml', schema: 'schemas/safety-mechanisms.schema.json' }
];

// Colors for console output
const colors = {
  reset: '\x1b[0m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m'
};

function log(message, color = 'reset') {
  console.log(`${colors[color]}${message}${colors.reset}`);
}

function loadYaml(filePath) {
  try {
    const content = fs.readFileSync(filePath, 'utf8');
    return yaml.load(content);
  } catch (error) {
    return null;
  }
}

function loadJson(filePath) {
  try {
    const content = fs.readFileSync(filePath, 'utf8');
    return JSON.parse(content);
  } catch (error) {
    return null;
  }
}

function validateSchema(configPath, schemaPath) {
  const config = loadYaml(configPath);
  const schema = loadJson(schemaPath);
  
  if (!config) {
    return { valid: false, errors: [`Failed to load config: ${configPath}`] };
  }
  
  if (!schema) {
    return { valid: false, errors: [`Failed to load schema: ${schemaPath}`] };
  }
  
  const validate = ajv.compile(schema);
  const valid = validate(config);
  
  return {
    valid,
    errors: valid ? [] : validate.errors.map(e => `${e.instancePath} ${e.message}`)
  };
}

function checkCrossReferences(configs) {
  const errors = [];
  
  // Get all valid IDs from each config
  const envIds = new Set((configs.environment?.environments || []).map(e => e.id));
  const toolIds = new Set((configs.dependencies?.tools || []).map(t => t.id));
  const modelIds = new Set((configs.dependencies?.models || []).map(m => m.id));
  const principleIds = new Set((configs.constitution?.principles || []).map(p => p.id));
  const expertIds = new Set((configs.experts?.experts || []).map(e => e.id));
  
  // Check virtual-experts references
  for (const expert of (configs.experts?.experts || [])) {
    // Check governed_by references
    for (const principleId of (expert.governed_by || [])) {
      if (!principleIds.has(principleId)) {
        errors.push(`Expert '${expert.id}' references unknown principle: ${principleId}`);
      }
    }
    
    // Check allowed_tools references
    for (const toolId of (expert.allowed_tools || [])) {
      if (!toolIds.has(toolId)) {
        errors.push(`Expert '${expert.id}' references unknown tool: ${toolId}`);
      }
    }
    
    // Check environments references
    for (const envId of (expert.environments || [])) {
      if (!envIds.has(envId)) {
        errors.push(`Expert '${expert.id}' references unknown environment: ${envId}`);
      }
    }
  }
  
  // Check safety-mechanisms references
  for (const mechanism of (configs.safety?.mechanisms || [])) {
    // Check constitution_refs
    for (const principleId of (mechanism.constitution_refs || [])) {
      if (!principleIds.has(principleId)) {
        errors.push(`Mechanism '${mechanism.id}' references unknown principle: ${principleId}`);
      }
    }
    
    // Check reviewers
    for (const expertId of (mechanism.reviewers || [])) {
      if (!expertIds.has(expertId)) {
        errors.push(`Mechanism '${mechanism.id}' references unknown expert: ${expertId}`);
      }
    }
    
    // Check environments
    for (const envId of (mechanism.environments || [])) {
      if (!envIds.has(envId)) {
        errors.push(`Mechanism '${mechanism.id}' references unknown environment: ${envId}`);
      }
    }
  }
  
  return errors;
}

async function main() {
  log('\n🔍 SynergyMesh Configuration Validator\n', 'blue');
  
  let hasErrors = false;
  const configs = {};
  
  // Validate each config against its schema
  log('📋 Schema Validation:', 'blue');
  for (const { config, schema } of CONFIG_FILES) {
    const configPath = path.join(process.cwd(), config);
    const schemaPath = path.join(process.cwd(), schema);
    
    if (!fs.existsSync(configPath)) {
      log(`  ⚠️  ${config} - File not found`, 'yellow');
      continue;
    }
    
    if (!fs.existsSync(schemaPath)) {
      log(`  ⚠️  ${schema} - Schema not found`, 'yellow');
      continue;
    }
    
    const result = validateSchema(configPath, schemaPath);
    
    if (result.valid) {
      log(`  ✅ ${config}`, 'green');
      // Store config for cross-reference checking
      const configName = path.basename(config, '.yaml').replace(/-/g, '_');
      configs[configName === 'ai_constitution' ? 'constitution' : 
             configName === 'virtual_experts' ? 'experts' :
             configName === 'safety_mechanisms' ? 'safety' : configName] = loadYaml(configPath);
    } else {
      log(`  ❌ ${config}`, 'red');
      result.errors.forEach(err => log(`     ${err}`, 'red'));
      hasErrors = true;
    }
  }
  
  // Check cross-references
  log('\n🔗 Cross-Reference Validation:', 'blue');
  const crossRefErrors = checkCrossReferences(configs);
  
  if (crossRefErrors.length === 0) {
    log('  ✅ All cross-references valid', 'green');
  } else {
    crossRefErrors.forEach(err => log(`  ❌ ${err}`, 'red'));
    hasErrors = true;
  }
  
  // Summary
  log('\n' + '='.repeat(50), 'blue');
  if (hasErrors) {
    log('❌ Validation failed with errors', 'red');
    process.exit(1);
  } else {
    log('✅ All validations passed', 'green');
    process.exit(0);
  }
}

main().catch(err => {
  log(`Error: ${err.message}`, 'red');
  process.exit(1);
});
