// @GL-governed
// @GL-layer: GL30-49
// @GL-semantic: script-execution
// @GL-audit-trail: ../../engine/governance/GL_SEMANTIC_ANCHOR.json
//
// GL Unified Charter Activated - Add GL Markers to package.json files

const fs = require('fs');
const path = require('path');

const workspace = process.cwd();
const excludedPaths = ['node_modules', '.next', 'dist', 'build', '.git', 'coverage', 'gl-audit-reports'];

function shouldExclude(filePath) {
  return excludedPaths.some(excluded => filePath.includes(excluded));
}

function addGLMarkerToJsonFile(filePath) {
  try {
    const content = fs.readFileSync(filePath, 'utf8');
    
    // Check if already has GL marker
    if (content.includes('@GL-governed')) {
      return { status: 'skipped', reason: 'already_has_marker' };
    }
    
    // Determine layer based on path
    let layer = 'GL90-99';
    if (filePath.includes('/engine/')) layer = 'GL10-29';
    else if (filePath.includes('/file-organizer-system/')) layer = 'GL10-29';
    else if (filePath.includes('/instant/')) layer = 'GL10-29';
    else if (filePath.includes('/elasticsearch-search-system/')) layer = 'GL10-29';
    else if (filePath.includes('/infrastructure/')) layer = 'GL30-49';
    else if (filePath.includes('/esync-platform/')) layer = 'GL10-29';
    else if (filePath.includes('/gl-gate/')) layer = 'GL30-49';
    else if (filePath.includes('/.github/')) layer = 'GL90-99';
    
    // Add GL marker as a comment (before JSON)
    // Note: JSON doesn't support comments, so we add as a property
    const json = JSON.parse(content);
    json._gl = {
      governed: true,
      layer: layer,
      semantic: 'package-configuration',
      charter_version: '2.0.0'
    };
    
    fs.writeFileSync(filePath, JSON.stringify(json, null, 2));
    return { status: 'success', layer };
  } catch (error) {
    return { status: 'error', error: error.message };
  }
}

function processDirectory(dir) {
  let results = [];
  
  const files = fs.readdirSync(dir);
  
  for (const file of files) {
    const fullPath = path.join(dir, file);
    const stat = fs.statSync(fullPath);
    
    if (shouldExclude(fullPath)) {
      continue;
    }
    
    if (stat.isDirectory()) {
      results = results.concat(processDirectory(fullPath));
    } else if (file === 'package.json') {
      const result = addGLMarkerToJsonFile(fullPath);
      results.push({ file: fullPath, ...result });
    }
  }
  
  return results;
}

console.log('🚀 Starting GL Marker Addition for package.json files');
console.log('📂 Workspace:', workspace);

const results = processDirectory(workspace);

const success = results.filter(r => r.status === 'success');
const skipped = results.filter(r => r.status === 'skipped');
const errors = results.filter(r => r.status === 'error');

console.log(`\n✅ Added GL markers to ${success.length} package.json files`);
console.log(`⏭️  Skipped ${skipped.length} files (already have markers)`);
console.log(`❌ Errors: ${errors.length}`);

if (errors.length > 0) {
  console.log('\nError details:');
  errors.forEach(e => {
    console.log(`  - ${e.file}: ${e.error}`);
  });
}

console.log(`\n📊 Summary:`);
console.log(`   - Total package.json files: ${results.length}`);
console.log(`   - Processed successfully: ${success.length}`);
console.log(`   - Already marked: ${skipped.length}`);