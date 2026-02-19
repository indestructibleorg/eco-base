#!/usr/bin/env python3
"""
Core GitHub Actions Policy Validation Logic

Shared validation functions used by both the standalone validator and CI engine integration.
This module eliminates code duplication and ensures consistent validation behavior.
"""

import re
from pathlib import Path
from typing import List, Dict, Optional

# Compile SHA pattern once at module level for performance
SHA_PATTERN = re.compile(r'^[a-f0-9]{40}$', re.IGNORECASE)


def extract_actions_from_workflow(workflow_file: Path, repo_root: Path) -> List[Dict]:
    """Extract all 'uses:' statements from a workflow file"""
    try:
        with open(workflow_file, 'r') as f:
            content = f.read()
        
        # Find all 'uses:' statements
        # Pattern: uses: owner/repo@ref or uses: owner/repo/path@ref
        pattern = r'uses:\s*([^\s#]+)'
        actions = []
        
        for line_num, line in enumerate(content.split('\n'), 1):
            match = re.search(pattern, line)
            if match:
                action_ref = match.group(1).strip()
                actions.append({
                    'action': action_ref,
                    'line': line_num,
                    'file': str(workflow_file.relative_to(repo_root))
                })
        
        return actions
    except Exception as e:
        print(f"Error parsing workflow {workflow_file}: {e}")
        return []


def is_local_action(action_ref: str) -> bool:
    """Check if action is a local action reference (./path/to/action)"""
    return action_ref.startswith('./')


def is_docker_action(action_ref: str) -> bool:
    """Check if action is a docker:// action reference"""
    return action_ref.startswith('docker://')


def validate_action_reference(
    action_ref: str,
    policy: Dict,
    enforcement_level: Optional[str] = None
) -> List[str]:
    """
    Validate a single action reference against policy
    
    Args:
        action_ref: The action reference (e.g., 'owner/repo@ref')
        policy: The policy configuration dictionary
        enforcement_level: Override enforcement level ('error' or 'warning')
    
    Returns:
        List of violation messages (empty if valid)
    """
    violations = []
    
    # Allow local actions and docker actions
    if is_local_action(action_ref) or is_docker_action(action_ref):
        return violations
    
    # Parse action reference: owner/repo@ref or owner/repo/path@ref
    if '@' not in action_ref:
        violations.append(
            f"Action '{action_ref}' missing version/ref specifier (expected owner/repo@ref)"
        )
        return violations
    
    action_path, ref = action_ref.rsplit('@', 1)
    parts = action_path.split('/')
    
    if len(parts) < 2:
        violations.append(
            f"Invalid action format '{action_ref}' (expected owner/repo@ref)"
        )
        return violations
    
    owner = parts[0]
    repository_name = parts[1]
    action_base = f"{owner}/{repository_name}"
    
    # Check if action is explicitly blocked
    blocked_actions = policy.get('blocked_actions', [])
    for blocked in blocked_actions:
        if action_path.startswith(blocked):
            violations.append(
                f"Action '{action_base}' is explicitly blocked by policy. "
                f"Use manual commands instead."
            )
            break  # Only report one blocklist violation per action
    
    # Check organization ownership requirement
    policy_config = policy.get('policy', {})
    if policy_config.get('require_org_ownership', False):
        allowed_orgs = policy_config.get('allowed_organizations', [])
        if owner not in allowed_orgs:
            violations.append(
                f"Action from '{owner}' is not allowed. "
                f"Only actions from {', '.join(allowed_orgs)} are permitted."
            )
    
    # Check SHA pinning requirement
    if policy_config.get('require_sha_pinning', False):
        if not SHA_PATTERN.match(ref):
            violations.append(
                f"Action '{action_base}@{ref}' must be pinned to a full-length commit SHA "
                f"(40 hex characters); '{ref}' is not a full-length commit SHA."
            )
    
    return violations


def get_default_policy() -> Dict:
    """Get the default policy configuration"""
    return {
        'policy': {
            'require_org_ownership': True,
            'allowed_organizations': ['indestructibleorg'],
            'require_sha_pinning': True,
            'block_tag_references': True
        },
        'blocked_actions': [],
        'approved_actions': []
    }


def load_policy_file(policy_file: Path) -> Optional[Dict]:
    """
    Load policy from YAML file with graceful fallback
    
    Returns:
        Policy dictionary if successful, None if PyYAML not available or file doesn't exist
    """
    if not policy_file.exists():
        return None
    
    try:
        import yaml
    except ImportError:
        # PyYAML not available - return None to use default policy
        return None
    
    try:
        with open(policy_file, 'r') as f:
            loaded = yaml.safe_load(f)
        
        if loaded is None:
            # Empty file - return None to use default policy
            return None
        
        return loaded
    except Exception:
        # Parse error - return None to use default policy
        return None
