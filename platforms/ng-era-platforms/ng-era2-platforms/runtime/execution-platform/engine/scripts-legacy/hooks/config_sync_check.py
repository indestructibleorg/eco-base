#
# @ECO-governed
# @ECO-layer: gl-platform.gl-platform.governance
# @ECO-semantic: config-sync-check
# @ECO-audit-trail: ../../engine/gl-platform.gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
Config Sync Check - Auto-Fix Bot 配置文件同步驗證與自動複製
GL Layer: GL30-49 Execution Layer
Purpose: Ensure auto-fix-bot configuration files stay in sync across the repository
This hook:
1. Detects when the source config file is modified
2. Automatically syncs to target locations
3. Validates all copies are identical
"""
import hashlib
import os
import shutil
import sys
# Configuration: Source and target paths for config sync
CONFIG_SYNC_MAPPINGS = [
    {
        "source": "config/.auto-fix-bot.yml",
        "targets": [
            "workspace/config/.auto-fix-bot.yml",
        ],
        "description": "Auto-Fix Bot configuration sync",
    },
]
def get_file_hash(filepath: str) -> str:
    """Calculate SHA-256 hash of a file."""
    try:
        with open(filepath, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except FileNotFoundError:
        return ""
def sync_config_files(workspace_root: str) -> tuple[bool, list[str]]:
    """
    Sync configuration files from source to targets.
    Returns:
        Tuple of (success, list of messages)
    """
    messages = []
    all_synced = True
    for mapping in CONFIG_SYNC_MAPPINGS:
        source_path = os.path.join(workspace_root, mapping["source"])
        if not os.path.exists(source_path):
            messages.append(f"⚠️  Source not found: {mapping['source']}")
            continue
        source_hash = get_file_hash(source_path)
        for target in mapping["targets"]:
            target_path = os.path.join(workspace_root, target)
            # Ensure target directory exists
            target_dir = os.path.dirname(target_path)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir, exist_ok=True)
                messages.append(f"📁 Created directory: {target_dir}")
            target_hash = get_file_hash(target_path)
            if source_hash != target_hash:
                # Copy source to target
                try:
                    shutil.copy2(source_path, target_path)
                    messages.append(f"✅ Synced: {mapping['source']} → {target}")
                except OSError as e:
                    messages.append(f"❌ Failed to sync: {mapping['source']} → {target}: {e}")
                    all_synced = False
            else:
                messages.append(f"✓  Already in sync: {target}")
    return all_synced, messages
def verify_config_sync(workspace_root: str) -> tuple[bool, list[str]]:
    """
    Verify all configuration files are in sync.
    Returns:
        Tuple of (all_in_sync, list of messages)
    """
    messages = []
    all_in_sync = True
    for mapping in CONFIG_SYNC_MAPPINGS:
        source_path = os.path.join(workspace_root, mapping["source"])
        if not os.path.exists(source_path):
            messages.append(f"⚠️  Source not found: {mapping['source']}")
            continue
        source_hash = get_file_hash(source_path)
        for target in mapping["targets"]:
            target_path = os.path.join(workspace_root, target)
            target_hash = get_file_hash(target_path)
            if source_hash != target_hash:
                messages.append(f"❌ Out of sync: {mapping['source']} ≠ {target}")
                all_in_sync = False
            else:
                messages.append(f"✅ In sync: {target}")
    return all_in_sync, messages
def main():
    """Main entry point for pre-commit hook."""
    # Determine workspace root
    workspace_root = os.environ.get("WORKSPACE_ROOT", os.getcwd())
    # Check if running in pre-commit context
    args = sys.argv[1:]
    # Check if any relevant config files are being modified
    relevant_files = []
    for mapping in CONFIG_SYNC_MAPPINGS:
        relevant_files.append(mapping["source"])
        relevant_files.extend(mapping["targets"])
    files_modified = [f for f in args if f in relevant_files]
    if not files_modified and args:
        # No relevant files in this commit
        print("ℹ️  No config files to sync in this commit")
        sys.exit(0)
    print("🔄 Auto-Fix Bot Config Sync Check")
    print("=" * 40)
    # First, sync files
    sync_success, sync_messages = sync_config_files(workspace_root)
    for msg in sync_messages:
        print(msg)
    # Then verify
    print("\n📋 Verification:")
    verify_success, verify_messages = verify_config_sync(workspace_root)
    for msg in verify_messages:
        print(msg)
    if sync_success and verify_success:
        print("\n✅ Config sync check passed")
        sys.exit(0)
    else:
        print("\n❌ Config sync check failed - please review the messages above")
        sys.exit(1)
if __name__ == "__main__":
    main()
