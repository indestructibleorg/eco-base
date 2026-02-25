#!/usr/bin/env python3
"""
GL Marker Addition Script for Python Files
GL Unified Architecture Governance Framework Activated
"""
import os
import sys
from pathlib import Path

GL_HEADER = '''/**
 * @ECO-governed
 * @ECO-layer: {}
 * @ECO-semantic: {}
 * @ECO-audit-trail: {}
 *
 * GL Unified Architecture Governance Framework Activated
 */

'''

def add_gl_marker_to_file(file_path, layer, semantic, audit_trail):
    """Add GL marker to a Python file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if already has GL marker
        if '@ECO-governed' in content:
            return False
        
        # Add GL marker at the beginning
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(GL_HEADER.format(layer, semantic, audit_trail))
            f.write(content)
        
        return True
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def get_layer_from_path(file_path, workspace):
    """Determine GL layer from file path"""
    rel_path = Path(file_path).relative_to(workspace)
    parts = rel_path.parts
    
    if parts[0] == 'engine':
        return 'gl-platform.gl-platform.governance'
    elif parts[0] == 'file-organizer-system':
        return 'application'
    elif parts[0] == 'instant':
        return 'data'
    elif parts[0] == 'elasticsearch-search-system':
        return 'search'
    elif parts[0] == 'infrastructure':
        return 'infrastructure'
    elif parts[0] == 'esync-platform':
        return 'platform'
    elif parts[0] == '.github':
        return 'github'
    else:
        return 'common'

def get_semantic_from_path(file_path):
    """Determine semantic from file path"""
    path = Path(file_path)
    return path.stem

def get_audit_trail(file_path):
    """Determine audit trail reference"""
    workspace = Path.cwd().parent
    return f"../../engine/gl-platform.gl-platform.governance/GL_SEMANTIC_ANCHOR.json"

def process_directory(workspace):
    """Process all Python files in workspace"""
    workspace_path = Path(workspace)
    excluded = ['node_modules', '.next', 'dist', 'build', '.git', 'coverage', 'gl-audit-reports', 'summarized_conversations', '.github/gl-platform.gl-platform.governance-legacy']
    
    files_processed = 0
    files_modified = 0
    
    for py_file in workspace_path.rglob('*.py'):
        # Skip excluded directories
        if any(excluded in str(py_file) for excluded in excluded):
            continue
        
        files_processed += 1
        
        layer = get_layer_from_path(py_file, workspace_path)
        semantic = get_semantic_from_path(py_file)
        audit_trail = get_audit_trail(py_file)
        
        if add_gl_marker_to_file(py_file, layer, semantic, audit_trail):
            print(f"✓ Added GL marker to {py_file}")
            files_modified += 1
        else:
            print(f"⊘ Skipped {py_file} (already has GL marker)")
    
    print(f"\n📊 Summary:")
    print(f"   - Total files processed: {files_processed}")
    print(f"   - Files modified: {files_modified}")
    print(f"   - Files skipped: {files_processed - files_modified}")

if __name__ == '__main__':
    workspace = Path.cwd()
    print("🚀 Starting GL Marker Addition Process (Python)\n")
    print(f"📁 Workspace: {workspace}")
    process_directory(workspace)
    print("\n✅ GL Marker Addition Complete!")