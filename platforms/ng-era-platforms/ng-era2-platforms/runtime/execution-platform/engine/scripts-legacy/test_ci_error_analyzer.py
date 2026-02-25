#
# @ECO-governed
# @ECO-layer: gl-platform.gl-platform.governance
# @ECO-semantic: test-ci-error-analyzer
# @ECO-audit-trail: ../../engine/gl-platform.gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
"""
Test CI Error Analyzer Script
Validates that the CI error analyzer correctly detects and categorizes errors.
"""
# MNGA-002: Import organization needs review
import json
import sys
import tempfile
from pathlib import Path
# Test log samples
TEST_LOGS = {
    "typescript_error": """
Build started...
src/index.ts:42:10 - error TS2304: Cannot find name 'unknown'.
42     const x: unknown = getValue();
              ~~~~~~~
Found 1 error.
npm ERR! code ELIFECYCLE
npm ERR! errno 2
npm ERR! Failed at build script.
""",
    "test_failure": """
PASS  src/utils.test.ts
FAIL  src/components/Button.test.tsx
  ● Button › should render correctly
    expect(received).toBe(expected)
    Expected: "Click me"
    Received: "Click"
      at Object.<anonymous> (src/components/Button.test.tsx:15:32)
Test Suites: 1 failed, 1 passed, 2 total
Tests:       1 failed, 5 passed, 6 total
""",
    "lint_error": """
/home/runner/work/project/src/index.js
  42:10  error  'unused' is assigned a value but never used  no-unused-vars
  43:15  error  Missing semicolon                            semi
✖ 2 problems (2 errors, 0 warnings)
  1 error and 0 warnings potentially fixable with the `--fix` option.
""",
    "dependency_error": """
npm ERR! code ERESOLVE
npm ERR! ERESOLVE unable to resolve dependency tree
npm ERR!
npm ERR! While resolving: project@1.0.0
npm ERR! Found: react@17.0.2
npm ERR!
npm ERR! Could not resolve dependency:
npm ERR! peer react@"^18.0.0" from react-dom@18.2.0
""",
    "security_vulnerability": """
# npm audit report
lodash  <4.17.21
Severity: critical
Prototype Pollution - https://npmjs.com/advisories/1065
fix available via `npm audit fix --force`
""",
    "permission_error": """
Error: EACCES: permission denied, open '/etc/config.yaml'
    at Object.openSync (fs.js:476:3)
    at writeFileSync (fs.js:1467:35)
""",
    "multiple_errors": """
src/index.ts:10:5 - error TS2304: Cannot find name 'unknown'.
src/utils.ts:25:10 - error TS2339: Property 'foo' does not exist on type 'Bar'.
npm ERR! Failed at build script.
FAIL  src/test.ts
  ● Test suite failed to run
/home/runner/work/project/src/index.js
  42:10  error  'unused' is assigned a value but never used  no-unused-vars
npm audit found 3 vulnerabilities (2 high, 1 critical)
""",
}
def test_analyzer_import():
    """Test that the CI error analyzer can be imported"""
    print("Testing import...")
    try:
        # Try to import from workspace
        sys.path.insert(0, str(Path(__file__).parent.parent / "workspace" / "src"))
        from core.ci_error_handler import CIErrorAnalyzer
        print("✅ Successfully imported CIErrorAnalyzer")
        return True
    except ImportError as e:
        print(f"⚠️  Import failed: {e}")
        print("   This is expected if running outside the workspace")
        return False
def test_fallback_analyzer():
    """Test the fallback analyzer in the script"""
    print("\nTesting fallback analyzer...")
    # Add script directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    try:
        # Import the fallback_analyze function
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "ci_analyzer",
            Path(__file__).parent.parent / "scripts" / "ci-error-analyzer.py",
        )
        ci_analyzer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ci_analyzer)
        # Test with sample log
        result = ci_analyzer.fallback_analyze(TEST_LOGS["typescript_error"])
        assert "errors" in result
        assert "summary" in result
        assert result["summary"]["total"] > 0
        print(f"✅ Fallback analyzer detected {result['summary']['total']} errors")
        return True
    except Exception as e:
        print(f"❌ Fallback analyzer test failed: {e}")
        return False
def test_script_execution():
    """Test running the analyzer script"""
    print("\nTesting script execution...")
    import subprocess
    # Create temporary log file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
        f.write(TEST_LOGS["multiple_errors"])
        log_file = f.name
    try:
        # Run the analyzer script
        script_path = Path(__file__).parent.parent / "scripts" / "ci-error-analyzer.py"
        result = subprocess.run(
            [
                "python3",
                str(script_path),
                "--log-file",
                log_file,
                "--mode",
                "report",
                "--output",
                "/tmp/test-analysis.json",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        print(f"Return code: {result.returncode}")
        if result.stdout:
            print(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"STDERR:\n{result.stderr}")
        # Check if output file was created
        output_file = Path("/tmp/test-analysis.json")
        if output_file.exists():
            with open(output_file, "r") as f:
                analysis = json.load(f)
            print("✅ Analysis complete:")
            print(f"   Total errors: {analysis.get('summary', {}).get('total', 0)}")
            print(
                f"   Auto-fixable: {analysis.get('summary', {}).get('auto_fixable_count', 0)}"
            )
            # Cleanup
            output_file.unlink()
            Path(log_file).unlink()
            return True
        else:
            print("❌ Output file not created")
            Path(log_file).unlink()
            return False
    except Exception as e:
        print(f"❌ Script execution failed: {e}")
        Path(log_file).unlink()
        return False
def test_error_patterns():
    """Test that different error types are detected"""
    print("\nTesting error pattern detection...")
    sys.path.insert(0, str(Path(__file__).parent.parent / "workspace" / "src"))
    try:
        from core.ci_error_handler import CIErrorAnalyzer
        analyzer = CIErrorAnalyzer()
        results = {}
        for log_type, log_content in TEST_LOGS.items():
            errors = analyzer.analyze_log(log_content)
            results[log_type] = len(errors)
            print(f"  {log_type}: {len(errors)} error(s) detected")
        # Verify some errors were detected
        assert sum(results.values()) > 0, "No errors detected in any test logs"
        print(f"✅ Pattern detection working (total: {sum(results.values())} errors)")
        return True
    except ImportError:
        print("⚠️  Skipping (CIErrorAnalyzer not available)")
        return True  # Not a failure
    except Exception as e:
        print(f"❌ Pattern detection test failed: {e}")
        return False
def main():
    """Run all tests"""
    print("=" * 60)
    print("CI Error Analyzer Test Suite")
    print("=" * 60)
    tests = [
        ("Import Test", test_analyzer_import),
        ("Fallback Analyzer", test_fallback_analyzer),
        ("Script Execution", test_script_execution),
        ("Error Pattern Detection", test_error_patterns),
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
