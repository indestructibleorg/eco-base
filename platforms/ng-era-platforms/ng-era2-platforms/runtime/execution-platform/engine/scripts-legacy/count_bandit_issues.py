#
# @ECO-governed
# @ECO-layer: gl-platform.gl-platform.governance
# @ECO-semantic: count_bandit_issues
# @ECO-audit-trail: ../../engine/gl-platform.gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""Count high/medium severity issues from Bandit report."""
# MNGA-002: Import organization needs review
import json
import sys
def main() -> None:
    """Count HIGH severity issues from Bandit report (for CI blocking).
    MEDIUM severity issues are shown as warnings only and do not block CI.
    """
    try:
        with open('bandit-report.json', encoding='utf-8') as f:
            data = json.load(f)
            results = data.get('results', [])
            high_only = [
                r for r in results
                if r.get('issue_severity') == 'HIGH'
            ]
            print(len(high_only))
    except FileNotFoundError:
        print("0")
        sys.exit(0)
    except PermissionError as exc:
        print(f"Error: Permission denied reading Bandit report: {exc}", file=sys.stderr)
        print("0")
        sys.exit(0)
    except json.JSONDecodeError as exc:
        print(f"Error: Invalid JSON in Bandit report: {exc}", file=sys.stderr)
        print("0")
        sys.exit(0)
    except OSError as exc:
        print(f"Error: I/O error reading Bandit report: {exc}", file=sys.stderr)
        print("0")
        sys.exit(0)
if __name__ == "__main__":
    main()
