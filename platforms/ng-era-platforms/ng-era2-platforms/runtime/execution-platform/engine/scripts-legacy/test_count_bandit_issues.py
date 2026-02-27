#
# @ECO-governed
# @ECO-layer: gl-platform.gl-platform.governance
# @ECO-semantic: test-count-bandit-issues
# @ECO-audit-trail: ../../engine/gl-platform.gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""Test Count Bandit Issues Script
Validates that the count_bandit_issues script correctly counts high/medium severity
issues from Bandit JSON reports and handles edge cases gracefully.
"""
# MNGA-002: Import organization needs review
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Tuple
# Constants
SCRIPT_TIMEOUT = 10  # seconds
BANDIT_REPORT_FILENAME = "bandit-report.json"
SCRIPT_NAME = "count_bandit_issues.py"
# Test Bandit report samples
TEST_REPORTS = {
    "empty_results": {
        "results": [],
        "metrics": {
            "_totals": {
                "SEVERITY.HIGH": 0,
                "SEVERITY.MEDIUM": 0,
                "SEVERITY.LOW": 0
            }
        }
    },
    "high_severity_only": {
        "results": [
            {
                "code": "import pickle",
                "filename": "test.py",
                "issue_confidence": "HIGH",
                "issue_severity": "HIGH",
                "issue_text": "Consider possible security implications"
            },
            {
                # SECURITY WARNING: exec() usage - ensure input is trusted
                "code": "exec(user_input)",
                "filename": "test2.py",
                "issue_confidence": "HIGH",
                "issue_severity": "HIGH",
                "issue_text": "Use of exec detected"
            }
        ],
        "metrics": {
            "_totals": {
                "SEVERITY.HIGH": 2,
                "SEVERITY.MEDIUM": 0,
                "SEVERITY.LOW": 0
            }
        }
    },
    "medium_severity_only": {
        "results": [
            {
                "code": "hashlib.md5()",
                "filename": "test.py",
                "issue_confidence": "HIGH",
                "issue_severity": "MEDIUM",
                "issue_text": "Use of insecure MD5 hash function"
            }
        ],
        "metrics": {
            "_totals": {
                "SEVERITY.HIGH": 0,
                "SEVERITY.MEDIUM": 1,
                "SEVERITY.LOW": 0
            }
        }
    },
    "mixed_severity": {
        "results": [
            {
                "code": "import pickle",
                "filename": "test.py",
                "issue_confidence": "HIGH",
                "issue_severity": "HIGH",
                "issue_text": "Security issue"
            },
            {
                "code": "hashlib.md5()",
                "filename": "test2.py",
                "issue_confidence": "HIGH",
                "issue_severity": "MEDIUM",
                "issue_text": "Use of insecure hash"
            },
            {
                "code": "assert True",
                "filename": "test3.py",
                "issue_confidence": "LOW",
                "issue_severity": "LOW",
                "issue_text": "Use of assert detected"
            }
        ],
        "metrics": {
            "_totals": {
                "SEVERITY.HIGH": 1,
                "SEVERITY.MEDIUM": 1,
                "SEVERITY.LOW": 1
            }
        }
    },
    "low_severity_only": {
        "results": [
            {
                "code": "assert True",
                "filename": "test.py",
                "issue_confidence": "LOW",
                "issue_severity": "LOW",
                "issue_text": "Use of assert detected"
            }
        ],
        "metrics": {
            "_totals": {
                "SEVERITY.HIGH": 0,
                "SEVERITY.MEDIUM": 0,
                "SEVERITY.LOW": 1
            }
        }
    },
    "malformed_missing_severity": {
        "results": [
            {
                "code": "some code",
                "filename": "test.py",
                "issue_confidence": "HIGH"
                # Missing issue_severity field
            }
        ],
        "metrics": {}
    }
}
def run_count_script_in_temp_dir(report_data) -> Tuple[str, int]:
    """Helper function to run count_bandit_issues.py with a test report.
    Args:
        report_data: Dictionary or string to write to bandit-report.json
    Returns:
        Tuple[str, int]: (stdout, exit_code)
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        try:
            # Create bandit-report.json
            with open(BANDIT_REPORT_FILENAME, "w", encoding='utf-8') as f:
                if isinstance(report_data, dict):
                    json.dump(report_data, f)
                else:
                    f.write(report_data)
            # Run the script
            script_path = Path(__file__).parent / SCRIPT_NAME
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=SCRIPT_TIMEOUT,
            )
            return result.stdout.strip(), result.returncode
        finally:
            os.chdir(original_dir)
def test_script_import():
    """Test that the count_bandit_issues script can be imported and has expected functionality"""
    print("Testing import...")
    try:
        spec = importlib.util.spec_from_file_location(
            "count_bandit_issues",
            Path(__file__).parent / SCRIPT_NAME,
        )
        count_bandit = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(count_bandit)
        # Verify the module has the expected main function
        if not hasattr(count_bandit, 'main'):
            print("❌ Module is missing expected 'main' function")
            return False
        print("✅ Successfully imported count_bandit_issues module with expected functions")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False
def test_empty_results():
    """Test counting with empty results"""
    print("\nTesting empty results...")
    try:
        output, exit_code = run_count_script_in_temp_dir(TEST_REPORTS["empty_results"])
        expected = "0"
        if output == expected:
            print(f"✅ Empty results: got {output} as expected")
            return True
        else:
            print(f"❌ Expected {expected}, got {output}")
            return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
def test_high_severity_count():
    """Test counting high severity issues"""
    print("\nTesting high severity count...")
    try:
        output, exit_code = run_count_script_in_temp_dir(TEST_REPORTS["high_severity_only"])
        expected = "2"  # 2 HIGH severity issues
        if output == expected:
            print(f"✅ High severity count: got {output} as expected")
            return True
        else:
            print(f"❌ Expected {expected}, got {output}")
            return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
def test_medium_severity_count():
    """Test counting medium severity issues"""
    print("\nTesting medium severity count...")
    try:
        output, exit_code = run_count_script_in_temp_dir(TEST_REPORTS["medium_severity_only"])
        expected = "1"  # 1 MEDIUM severity issue
        if output == expected:
            print(f"✅ Medium severity count: got {output} as expected")
            return True
        else:
            print(f"❌ Expected {expected}, got {output}")
            return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
def test_mixed_severity_count():
    """Test counting mixed severity issues (should count HIGH + MEDIUM only)"""
    print("\nTesting mixed severity count...")
    try:
        output, exit_code = run_count_script_in_temp_dir(TEST_REPORTS["mixed_severity"])
        expected = "2"  # 1 HIGH + 1 MEDIUM, LOW should be excluded
        if output == expected:
            print(f"✅ Mixed severity count: got {output} (HIGH+MEDIUM only)")
            return True
        else:
            print(f"❌ Expected {expected}, got {output}")
            return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
def test_low_severity_excluded():
    """Test that LOW severity issues are not counted"""
    print("\nTesting LOW severity exclusion...")
    try:
        output, exit_code = run_count_script_in_temp_dir(TEST_REPORTS["low_severity_only"])
        expected = "0"  # LOW severity should not be counted
        if output == expected:
            print(f"✅ LOW severity excluded: got {output} as expected")
            return True
        else:
            print(f"❌ Expected {expected}, got {output}")
            return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
def test_missing_file():
    """Test handling of missing bandit-report.json file"""
    print("\nTesting missing file handling...")
    try:
        # Create a temporary directory without bandit-report.json
        with tempfile.TemporaryDirectory() as temp_dir:
            original_dir = os.getcwd()
            os.chdir(temp_dir)
            try:
                script_path = Path(__file__).parent / SCRIPT_NAME
                result = subprocess.run(
                    [sys.executable, str(script_path)],
                    capture_output=True,
                    text=True,
                    timeout=SCRIPT_TIMEOUT,
                )
                output = result.stdout.strip()
                expected = "0"
                exit_code = result.returncode
                if output == expected and exit_code == 0:
                    print(f"✅ Missing file handled gracefully: got {output}, exit code {exit_code}")
                    return True
                else:
                    print(f"❌ Expected output {expected} and exit code 0, got {output} and {exit_code}")
                    return False
            finally:
                os.chdir(original_dir)
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
def test_malformed_json():
    """Test handling of malformed JSON"""
    print("\nTesting malformed JSON handling...")
    try:
        # Use clearly invalid JSON
        malformed_json = '{"key": unclosed'
        output, exit_code = run_count_script_in_temp_dir(malformed_json)
        expected = "0"
        if output == expected and exit_code == 0:
            print(f"✅ Malformed JSON handled gracefully: got {output}, exit code {exit_code}")
            return True
        else:
            print(f"❌ Expected output {expected} and exit code 0, got {output} and {exit_code}")
            return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
def test_malformed_missing_severity_field():
    """Test handling of report with missing severity fields"""
    print("\nTesting missing severity field handling...")
    try:
        output, exit_code = run_count_script_in_temp_dir(TEST_REPORTS["malformed_missing_severity"])
        # Should handle gracefully and return 0 since issue has no severity
        expected = "0"
        if output == expected:
            print(f"✅ Missing severity field handled: got {output}")
            return True
        else:
            print(f"❌ Expected {expected}, got {output}")
            return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
def main():
    """Run all tests"""
    print("=" * 60)
    print("count_bandit_issues.py Test Suite")
    print("=" * 60)
    tests = [
        ("Import Test", test_script_import),
        ("Empty Results", test_empty_results),
        ("High Severity Count", test_high_severity_count),
        ("Medium Severity Count", test_medium_severity_count),
        ("Mixed Severity Count", test_mixed_severity_count),
        ("LOW Severity Excluded", test_low_severity_excluded),
        ("Missing File Handling", test_missing_file),
        ("Malformed JSON Handling", test_malformed_json),
        ("Missing Severity Field Handling", test_malformed_missing_severity_field),
    ]
    results = []
    for test_name, test_func in tests:
        print(f"\n{'=' * 60}")
        print(f"Running: {test_name}")
        print(f"{'=' * 60}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    # Summary
    print(f"\n{'=' * 60}")
    print("Test Summary")
    print(f"{'=' * 60}")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    print(f"\nResults: {passed}/{total} tests passed")
    if passed == total:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        sys.exit(1)
if __name__ == "__main__":
    main()
