#
# @ECO-governed
# @ECO-layer: gl-platform.gl-platform.governance
# @ECO-semantic: convert-snyk-to-sarif
# @ECO-audit-trail: ../../engine/gl-platform.gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
Convert Snyk JSON output to SARIF format for GitHub Security.
This script converts Snyk vulnerability scan results from JSON format
to SARIF (Static Analysis Results Interchange Format) for upload to
GitHub Security.
Usage:
    python3 convert-snyk-to-sarif.py <input_json> <output_sarif>
"""
# MNGA-002: Import organization needs review
import json
import sys
def _map_severity_to_level(severity: str) -> str:
    """
    Map Snyk severity levels to SARIF levels.
    Args:
        severity: Snyk severity level (e.g., 'high', 'medium', 'low', 'critical')
    Returns:
        SARIF level ('error', 'warning', or 'note')
    """
    severity_lower = severity.lower() if severity else ''
    if severity_lower in ('critical', 'high'):
        return 'error'
    elif severity_lower == 'medium':
        return 'warning'
    else:  # low or unknown
        return 'note'
def convert_snyk_to_sarif(input_file: str, output_file: str) -> bool:
    """
    Convert Snyk JSON results to SARIF format.
    Args:
        input_file: Path to Snyk JSON results file
        output_file: Path to output SARIF file
    Returns:
        True if conversion was successful, False otherwise
    """
    try:
        # Read Snyk JSON results
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Create SARIF structure
        sarif = {
            'version': '2.1.0',
            '$schema': 'https://json.schemastore.org/sarif-2.1.0.json',
            'runs': [{
                'tool': {
                    'driver': {
                        'name': 'Snyk',
                        'version': '1.0.0',
                        'informationUri': 'https://snyk.io'
                    }
                },
                'results': []
            }]
        }
        # Convert vulnerabilities to SARIF results
        if 'vulnerabilities' in data:
            for vuln in data['vulnerabilities']:
                severity = vuln.get('severity', 'medium')
                sarif['runs'][0]['results'].append({
                    'ruleId': vuln.get('id', 'unknown'),
                    'level': _map_severity_to_level(severity),
                    'message': {
                        'text': vuln.get('title', 'Unknown vulnerability')
                    },
                    'locations': [{
                        'physicalLocation': {
                            'artifactLocation': {
                                'uri': vuln.get('package', 'unknown')
                            }
                        }
                    }]
                })
        # Write SARIF output
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sarif, f, indent=2)
        print(f'Successfully converted {input_file} to {output_file}')
        print(f'Found {len(sarif["runs"][0]["results"])} vulnerabilities')
        return True
    except FileNotFoundError:
        print(f'Error: Input file {input_file} not found', file=sys.stderr)
        return False
    except json.JSONDecodeError as e:
        print(f'Error: Invalid JSON in {input_file}: {e}', file=sys.stderr)
        return False
    except Exception as e:
        print(f'Error converting Snyk results to SARIF: {e}', file=sys.stderr)
        return False
def create_empty_sarif(output_file: str) -> None:
    """Create an empty SARIF file when conversion fails."""
    empty_sarif = {
        'version': '2.1.0',
        '$schema': 'https://json.schemastore.org/sarif-2.1.0.json',
        'runs': []
    }
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(empty_sarif, f, indent=2)
    print(f'Created empty SARIF file: {output_file}')
def main():
    """Main entry point for the script."""
    if len(sys.argv) != 3:
        print('Usage: convert-snyk-to-sarif.py <input_json> <output_sarif>', file=sys.stderr)
        print('Example: convert-snyk-to-sarif.py snyk-results.json snyk.sarif', file=sys.stderr)
        sys.exit(1)
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    # Try conversion, create empty SARIF on failure
    if not convert_snyk_to_sarif(input_file, output_file):
        create_empty_sarif(output_file)
        sys.exit(1)
if __name__ == '__main__':
    main()