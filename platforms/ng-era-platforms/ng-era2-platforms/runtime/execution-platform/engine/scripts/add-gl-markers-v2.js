const fs = require('fs');
const path = require('path');

// Configuration
const WORKSPACE = '/workspace/machine-native-ops';
const EXCLUDED_PATHS = [
  'node_modules',
  '.next',
  'dist',
  'build',
  '.git',
  'coverage',
  'gl-audit-reports'
];

// GL Governance Layer Definitions
const LAYER_DEFINITIONS = {
  'engine/': {
    layer: 'GL-L1-CORE',
    semantic: 'governance-layer-core'
  },
  'engine/governance/': {
    layer: 'GL-L1-CORE',
    semantic: 'governance-layer-core'
  },
  'engine/gl-gate/': {
    layer: 'GL-L2-GATE',
    semantic: 'governance-layer-gate'
  },
  'engine/tests/': {
    layer: 'GL-L3-TEST',
    semantic: 'governance-layer-test'
  },
  
  // File Organizer
  'file-organizer-system/': {
    layer: 'GL-L4-APP',
    semantic: 'governance-layer-application'
  },
  
  // Config
  'config/': {
    layer: 'GL-L5-CONFIG',
    semantic: 'governance-layer-configuration'
  },
  
  // Archive (legacy files)
  'archive/': {
    layer: 'GL-L9-ARCHIVE',
    semantic: 'governance-layer-archive'
  },
  
  // Namespace MCP
  'ns-root/': {
    layer: 'GL-L6-NAMESPACE',
    semantic: 'governance-layer-namespace'
  },
  
  // Scripts
  'scripts/': {
    layer: 'GL-L7-SCRIPT',
    semantic: 'governance-layer-script'
  },
  
  // Docs
  'docs/': {
    layer: 'GL-L8-DOC',
    semantic: 'governance-layer-documentation'
  },
  
  // Dashboard
  'dashboard/': {
    layer: 'GL-L9-DASHBOARD',
    semantic: 'governance-layer-dashboard'
  },
  
  // Workspace
  'workspace/': {
    layer: 'GL-L10-WORKSPACE',
    semantic: 'governance-layer-workspace'
  },
  
  // Root config files (outside of main directories)
  '': {
    layer: 'GL-L11-ROOT',
    semantic: 'governance-layer-root'
  }
};

function isExcluded(filePath) {
  return EXCLUDED_PATHS.some(excludedPath => 
    filePath.includes(excludedPath)
  );
}

function getLayerInfo(filePath) {
  // Sort paths by length (longest first) to match most specific path
  const sortedPaths = Object.entries(LAYER_DEFINITIONS)
    .sort((a, b) => b[0].length - a[0].length);
  
  for (const [basePath, info] of sortedPaths) {
    if (basePath === '' || filePath.startsWith(basePath)) {
      return info;
    }
  }
  
  // Default for unclassified files
  return {
    layer: 'GL-L0-UNCLASSIFIED',
    semantic: 'governance-layer-unclassified'
  };
}

function hasGLMarkers(content) {
  return content.includes('@GL-governed');
}

function addGLMarkers(content, filePath) {
  const { layer, semantic } = getLayerInfo(filePath);
  const extension = path.extname(filePath);
  
  // Determine comment style
  let commentStart = '//';
  let commentEnd = '';
  
  if (extension === '.ts' || extension === '.tsx') {
    commentStart = '//';
  } else if (extension === '.js' || extension === '.jsx') {
    commentStart = '//';
  }
  
  const header = `${commentStart} @GL-governed
${commentStart} @GL-layer: ${layer}
${commentStart} @GL-semantic: ${semantic}
${commentStart} @GL-revision: 1.0.0
${commentStart} @GL-status: active

`;
  
  // Add markers after shebang if present
  if (content.startsWith('#!')) {
    const lines = content.split('\n');
    const shebang = lines[0];
    const rest = lines.slice(1).join('\n');
    return shebang + '\n' + header + rest;
  }
  
  return header + content;
}

function processDirectory(dir, processed = []) {
  console.log(`📂 Scanning directory: ${dir}`);
  const files = fs.readdirSync(dir, { withFileTypes: true });
  
  for (const file of files) {
    const fullPath = path.join(dir, file.name);
    const relativePath = path.relative(WORKSPACE, fullPath);
    
    if (isExcluded(relativePath)) {
      console.log(`  ⏭️  Skipping excluded: ${relativePath}`);
      continue;
    }
    
    if (file.isDirectory()) {
      processDirectory(fullPath, processed);
    } else if (file.isFile()) {
      const ext = path.extname(file.name);
      if (['.ts', '.tsx', '.js', '.jsx'].includes(ext)) {
        const content = fs.readFileSync(fullPath, 'utf-8');
        
        if (!hasGLMarkers(content)) {
          console.log(`  ➕ Adding markers to: ${relativePath}`);
          const newContent = addGLMarkers(content, relativePath);
          fs.writeFileSync(fullPath, newContent, 'utf-8');
          processed.push(relativePath);
        }
      }
    }
  }
  
  return processed;
}

function main() {
  console.log('🚀 Starting GL Marker Addition Process V2\n');
  console.log(`📁 Workspace: ${WORKSPACE}`);
  console.log(`🚫 Excluded Paths: ${EXCLUDED_PATHS.join(', ')}`);
  console.log('\n📝 Processing files...\n');
  
  try {
    const processed = processDirectory(WORKSPACE);
    
    console.log(`\n✅ Added GL markers to ${processed.length} files`);
    console.log(`\n📊 Summary:`);
    console.log(`   - Total files processed: ${processed.length}`);
    console.log(`   - Files with existing markers: skipped`);
    
    if (processed.length === 0) {
      console.log(`   - All files already have GL markers! 🎉`);
    }
  } catch (error) {
    console.error(`❌ Error: ${error.message}`);
    console.error(error.stack);
    process.exit(1);
  }
}

main();