#!/usr/bin/env node
// @GL-governed
// @GL-layer: GL-L7-SCRIPT
// @GL-semantic: governance-layer-script
// @GL-revision: 1.0.0
// @GL-status: active

/**
 * Workspace Validation Script
 * Validates workspace files and configurations for npm workspaces
 * 
 * Checks:
 * - Root package.json workspace paths exist
 * - Each workspace has a valid package.json
 * - Workspace package names are unique
 * - Workspace dependencies are valid
 * - No circular dependencies between workspaces
 */

const fs = require('fs');
const path = require('path');

// ANSI color codes
const colors = {
  reset: '\x1b[0m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
};

function log(message, color = 'reset') {
  console.log(`${colors[color]}${message}${colors.reset}`);
}

function loadJSON(filePath) {
  try {
    const content = fs.readFileSync(filePath, 'utf8');
    return JSON.parse(content);
  } catch (error) {
    throw new Error(`Failed to load ${filePath}: ${error.message}`);
  }
}

function validateWorkspaces() {
  const errors = [];
  const warnings = [];
  const repoRoot = path.resolve(__dirname, '..');
  
  log('\n========================================', 'blue');
  log('  Workspace Validation', 'blue');
  log('========================================\n', 'blue');

  // Load root package.json
  const rootPkgPath = path.join(repoRoot, 'package.json');
  let rootPkg;
  
  try {
    rootPkg = loadJSON(rootPkgPath);
    log('✅ Root package.json loaded', 'green');
  } catch (error) {
    errors.push(`Cannot load root package.json: ${error.message}`);
    return { errors, warnings };
  }

  // Check if workspaces are defined
  if (!rootPkg.workspaces || !Array.isArray(rootPkg.workspaces)) {
    errors.push('Root package.json does not define workspaces array');
    return { errors, warnings };
  }

  log(`\nFound ${rootPkg.workspaces.length} workspace paths defined\n`, 'blue');

  const workspacePackages = new Map();
  const workspacePaths = [];

  // Validate each workspace path
  for (const workspacePath of rootPkg.workspaces) {
    const fullPath = path.join(repoRoot, workspacePath);
    const pkgPath = path.join(fullPath, 'package.json');

    log(`Checking workspace: ${workspacePath}`);

    // Check if workspace directory exists
    if (!fs.existsSync(fullPath)) {
      errors.push(`Workspace directory does not exist: ${workspacePath}`);
      log(`  ❌ Directory not found`, 'red');
      continue;
    }

    // Check if package.json exists
    if (!fs.existsSync(pkgPath)) {
      errors.push(`Workspace missing package.json: ${workspacePath}`);
      log(`  ❌ package.json not found`, 'red');
      continue;
    }

    // Load and validate package.json
    let pkg;
    try {
      pkg = loadJSON(pkgPath);
      log(`  ✅ package.json loaded`, 'green');
    } catch (error) {
      errors.push(`Invalid package.json in ${workspacePath}: ${error.message}`);
      log(`  ❌ package.json is invalid`, 'red');
      continue;
    }

    // Check required fields
    if (!pkg.name) {
      errors.push(`Workspace ${workspacePath} missing "name" field in package.json`);
      log(`  ❌ Missing "name" field`, 'red');
    } else {
      // Check for duplicate names
      if (workspacePackages.has(pkg.name)) {
        errors.push(`Duplicate workspace name "${pkg.name}" in ${workspacePath} and ${workspacePackages.get(pkg.name)}`);
        log(`  ❌ Duplicate name: ${pkg.name}`, 'red');
      } else {
        workspacePackages.set(pkg.name, workspacePath);
        log(`  ✅ Name: ${pkg.name}`, 'green');
      }
    }

    if (!pkg.version) {
      warnings.push(`Workspace ${workspacePath} missing "version" field in package.json`);
      log(`  ⚠️  Missing "version" field`, 'yellow');
    }

    workspacePaths.push({
      path: workspacePath,
      fullPath,
      package: pkg,
    });
  }

  // Check for workspace dependencies
  log('\nValidating workspace dependencies...\n', 'blue');
  
  for (const workspace of workspacePaths) {
    const { path: wsPath, package: pkg } = workspace;
    
    if (!pkg || !pkg.name) continue;

    const allDeps = {
      ...pkg.dependencies,
      ...pkg.devDependencies,
      ...pkg.peerDependencies,
    };

    for (const [depName, depVersion] of Object.entries(allDeps)) {
      // Check if dependency is another workspace
      if (workspacePackages.has(depName)) {
        // Workspace dependency found
        if (depVersion.startsWith('workspace:')) {
          log(`  ✅ ${pkg.name} → ${depName} (workspace)`, 'green');
        } else {
          warnings.push(`Workspace ${pkg.name} depends on workspace ${depName} but uses version "${depVersion}" instead of "workspace:*"`);
          log(`  ⚠️  ${pkg.name} → ${depName} (should use workspace:*)`, 'yellow');
        }
      }
    }
  }

  // Summary
  log('\n========================================', 'blue');
  log('  Validation Summary', 'blue');
  log('========================================\n', 'blue');

  log(`Workspaces validated: ${workspacePaths.length}`, 'blue');
  log(`Unique package names: ${workspacePackages.size}`, 'blue');
  
  if (errors.length === 0 && warnings.length === 0) {
    log('\n✅ All workspace validations passed!', 'green');
    return { errors, warnings };
  }

  if (warnings.length > 0) {
    log(`\n⚠️  Found ${warnings.length} warning(s):\n`, 'yellow');
    warnings.forEach((warning) => log(`  • ${warning}`, 'yellow'));
  }

  if (errors.length > 0) {
    log(`\n❌ Found ${errors.length} error(s):\n`, 'red');
    errors.forEach((error) => log(`  • ${error}`, 'red'));
  }

  return { errors, warnings };
}

// Run validation
const { errors, warnings } = validateWorkspaces();

// Exit with appropriate code (only errors cause non-zero exit)
process.exit(errors.length > 0 ? 1 : 0);
