#
# @ECO-governed
# @ECO-layer: gl-platform.gl-platform.governance
# @ECO-semantic: test_gl_aep_engine_auditor
# @ECO-audit-trail: ../../engine/gl-platform.gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
Unit tests for GL AEP Engine Auditor - TypeScript any/@ts-ignore detection.
"""
import re
import sys
from pathlib import Path
import pytest
# Add the tools/gl-platform.gl-platform.governance-audit directory to the path
tools_path = Path(__file__).parent.parent.parent / "tools" / "gl-platform.gl-platform.governance-audit"
sys.path.insert(0, str(tools_path))
from gl_aep_engine_auditor import ETLPipeline, IssueType
def extract_any_count_from_message(message: str) -> int:
    """Helper function to extract the count of 'any' usages from error message."""
    match = re.search(r'Found (\d+) uses', message)
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not extract count from message: {message}")
class TestTypeScriptAnyDetection:
    """Tests for TypeScript 'any' type detection with @ts-ignore scoping."""
    @pytest.fixture
    def pipeline(self, tmp_path):
        """Create ETL pipeline instance."""
        return ETLPipeline(str(tmp_path))
    def test_any_without_ignore_is_detected(self, pipeline, tmp_path):
        """Test that 'any' usage without @ts-ignore is detected."""
        test_file = tmp_path / "test.ts"
        test_file.write_text("function test(x: any) { return x; }")
        extracted = pipeline.extract("test.ts")
        transformed = pipeline.transform(extracted)
        # Find type error issues
        type_errors = [i for i in transformed["issues"] if i["type"] == IssueType.TYPE_ERROR.value]
        assert len(type_errors) == 1
        assert "any" in type_errors[0]["message"].lower()
        # Extract the count from the message and verify it's exactly 1
        assert extract_any_count_from_message(type_errors[0]["message"]) == 1
    def test_any_with_ignore_on_previous_line_is_exempted(self, pipeline, tmp_path):
        """Test that 'any' with @ts-ignore on previous line is exempted."""
        test_file = tmp_path / "test.ts"
        test_file.write_text("// @ts-ignore\nfunction test(x: any) { return x; }")
        extracted = pipeline.extract("test.ts")
        transformed = pipeline.transform(extracted)
        # Should not find type errors
        type_errors = [i for i in transformed["issues"] if i["type"] == IssueType.TYPE_ERROR.value]
        assert len(type_errors) == 0
    def test_any_with_ignore_on_same_line_is_not_exempted(self, pipeline, tmp_path):
        """Test that @ts-ignore on same line does NOT exempt the 'any' usage."""
        # In TypeScript, @ts-ignore must be on the preceding line to take effect
        test_file = tmp_path / "test.ts"
        test_file.write_text("function test(x: any) { // @ts-ignore\n return x; }")
        extracted = pipeline.extract("test.ts")
        transformed = pipeline.transform(extracted)
        # Should still find the type error
        type_errors = [i for i in transformed["issues"] if i["type"] == IssueType.TYPE_ERROR.value]
        assert len(type_errors) == 1
    def test_ignore_without_space_is_recognized(self, pipeline, tmp_path):
        """Test that //@ts-ignore (without space) is recognized."""
        test_file = tmp_path / "test.ts"
        test_file.write_text("//@ts-ignore\nfunction test(x: any) { return x; }")
        extracted = pipeline.extract("test.ts")
        transformed = pipeline.transform(extracted)
        # Should not find type errors
        type_errors = [i for i in transformed["issues"] if i["type"] == IssueType.TYPE_ERROR.value]
        assert len(type_errors) == 0
    def test_ignore_with_extra_whitespace_is_recognized(self, pipeline, tmp_path):
        """Test that //  @ts-ignore (extra spaces) is recognized."""
        test_file = tmp_path / "test.ts"
        test_file.write_text("//  @ts-ignore\nfunction test(x: any) { return x; }")
        extracted = pipeline.extract("test.ts")
        transformed = pipeline.transform(extracted)
        # Should not find type errors
        type_errors = [i for i in transformed["issues"] if i["type"] == IssueType.TYPE_ERROR.value]
        assert len(type_errors) == 0
    def test_multiple_any_with_selective_ignores(self, pipeline, tmp_path):
        """Test multiple 'any' usages where only some have @ts-ignore."""
        test_file = tmp_path / "test.ts"
        test_content = """function test1(x: any) { }
// @ts-ignore
function test2(y: any) { }
function test3(z: any) { }"""
        test_file.write_text(test_content)
        extracted = pipeline.extract("test.ts")
        transformed = pipeline.transform(extracted)
        # Should find 1 type_error issue aggregating 2 'any' occurrences (test1 and test3, but not test2)
        type_errors = [i for i in transformed["issues"] if i["type"] == IssueType.TYPE_ERROR.value]
        assert len(type_errors) == 1
        # Extract the count from the message and verify it's exactly 2
        assert extract_any_count_from_message(type_errors[0]["message"]) == 2
    def test_no_any_usage_no_error(self, pipeline, tmp_path):
        """Test that files without 'any' usage don't trigger errors."""
        test_file = tmp_path / "test.ts"
        test_file.write_text("function test(x: string): string { return x; }")
        extracted = pipeline.extract("test.ts")
        transformed = pipeline.transform(extracted)
        # Should not find type errors
        type_errors = [i for i in transformed["issues"] if i["type"] == IssueType.TYPE_ERROR.value]
        assert len(type_errors) == 0
    def test_any_in_comment_is_not_counted(self, pipeline, tmp_path):
        """Test that 'any' in comments is not counted."""
        test_file = tmp_path / "test.ts"
        test_file.write_text("// This could be any value\nfunction test(x: string) { return x; }")
        extracted = pipeline.extract("test.ts")
        transformed = pipeline.transform(extracted)
        # Should not find type errors (the 'any' is in a comment, not a type annotation)
        type_errors = [i for i in transformed["issues"] if i["type"] == IssueType.TYPE_ERROR.value]
        assert len(type_errors) == 0
    def test_type_annotation_in_line_comment_not_counted(self, pipeline, tmp_path):
        """Test that type annotations like ': any' in line comments are not counted."""
        test_file = tmp_path / "test.ts"
        test_file.write_text("// foo: any - this is just a comment\nfunction test(x: string) { return x; }")
        extracted = pipeline.extract("test.ts")
        transformed = pipeline.transform(extracted)
        # Should not find type errors (': any' is in a comment, not actual code)
        type_errors = [i for i in transformed["issues"] if i["type"] == IssueType.TYPE_ERROR.value]
        assert len(type_errors) == 0
    def test_type_annotation_in_block_comment_not_counted(self, pipeline, tmp_path):
        """Test that type annotations in block comments are not counted."""
        test_file = tmp_path / "test.ts"
        test_file.write_text("/* foo: any - block comment */\nfunction test(x: string) { return x; }")
        extracted = pipeline.extract("test.ts")
        transformed = pipeline.transform(extracted)
        # Should not find type errors (': any' is in a block comment, not actual code)
        type_errors = [i for i in transformed["issues"] if i["type"] == IssueType.TYPE_ERROR.value]
        assert len(type_errors) == 0
    def test_multiline_block_comment_with_any_not_counted(self, pipeline, tmp_path):
        """Test that multiline block comments with ': any' are not counted."""
        test_file = tmp_path / "test.ts"
        test_content = """/*
# This function used to accept: any
# but now it's typed properly
function test(x: string) { return x; }"""
        test_file.write_text(test_content)
        extracted = pipeline.extract("test.ts")
        transformed = pipeline.transform(extracted)
        # Should not find type errors (': any' is in a multiline comment)
        type_errors = [i for i in transformed["issues"] if i["type"] == IssueType.TYPE_ERROR.value]
        assert len(type_errors) == 0
    def test_single_line_block_comment_not_counted(self, pipeline, tmp_path):
        """Test that single-line block comments with ': any' are not counted."""
        test_file = tmp_path / "test.ts"
        test_file.write_text("/* old signature: any */ function test(x: string) { return x; }")
        extracted = pipeline.extract("test.ts")
        transformed = pipeline.transform(extracted)
        # Should not find type errors (': any' is in a single-line block comment)
        type_errors = [i for i in transformed["issues"] if i["type"] == IssueType.TYPE_ERROR.value]
        assert len(type_errors) == 0
    def test_multiline_comment_end_with_code_after(self, pipeline, tmp_path):
        """Test that code after multiline block comment end is properly scanned."""
        test_file = tmp_path / "test.ts"
        test_content = """/*
# Some comment
#/ const x: any = 5;"""
        test_file.write_text(test_content)
        extracted = pipeline.extract("test.ts")
        transformed = pipeline.transform(extracted)
        # Should find the type error (code after */ should be scanned)
        type_errors = [i for i in transformed["issues"] if i["type"] == IssueType.TYPE_ERROR.value]
        assert len(type_errors) == 1
        assert extract_any_count_from_message(type_errors[0]["message"]) == 1
    def test_spacing_variations(self, pipeline, tmp_path):
        """Test detection of :any and : any spacing variations."""
        test_file = tmp_path / "test.ts"
        test_content = """function test1(x: any) { }
function test2(y:any) { }"""
        test_file.write_text(test_content)
        extracted = pipeline.extract("test.ts")
        transformed = pipeline.transform(extracted)
        # Should find both spacing variations
        type_errors = [i for i in transformed["issues"] if i["type"] == IssueType.TYPE_ERROR.value]
        assert len(type_errors) == 1
        # Extract the count from the message and verify it's exactly 2
        assert extract_any_count_from_message(type_errors[0]["message"]) == 2
    def test_ignore_must_start_with_double_slash(self, pipeline, tmp_path):
        """Test that @ts-ignore must be in a // comment to be recognized."""
        test_file = tmp_path / "test.ts"
        test_file.write_text("/* @ts-ignore */\nfunction test(x: any) { return x; }")
        extracted = pipeline.extract("test.ts")
        transformed = pipeline.transform(extracted)
        # Should still find the error (block comments don't count)
        type_errors = [i for i in transformed["issues"] if i["type"] == IssueType.TYPE_ERROR.value]
        assert len(type_errors) == 1
    def test_first_line_any_without_previous_line(self, pipeline, tmp_path):
        """Test 'any' on first line (no previous line to check)."""
        test_file = tmp_path / "test.ts"
        test_file.write_text("const x: any = 5;")
        extracted = pipeline.extract("test.ts")
        transformed = pipeline.transform(extracted)
        # Should find the type error
        type_errors = [i for i in transformed["issues"] if i["type"] == IssueType.TYPE_ERROR.value]
        assert len(type_errors) == 1
    def test_ignore_in_middle_of_comment_not_recognized(self, pipeline, tmp_path):
        """Test that @ts-ignore in middle of comment is NOT recognized."""
        test_file = tmp_path / "test.ts"
        test_file.write_text("// This is not @ts-ignore but mentions it\nfunction test(x: any) { return x; }")
        extracted = pipeline.extract("test.ts")
        transformed = pipeline.transform(extracted)
        # Should still find the error (the @ts-ignore is not at the start)
        type_errors = [i for i in transformed["issues"] if i["type"] == IssueType.TYPE_ERROR.value]
        assert len(type_errors) == 1
    def test_fast_path_no_any_keyword(self, pipeline, tmp_path):
        """Test fast-path optimization: files without ':any' or ': any' skip detailed processing."""
        test_file = tmp_path / "test.ts"
        # File with 'any' in other contexts (comments, identifiers) but no type annotations
        test_file.write_text("// any comment\nfunction test(x: string) { const anything = 5; return x; }")
        extracted = pipeline.extract("test.ts")
        transformed = pipeline.transform(extracted)
        # Should not find type errors (fast-path should skip processing)
        type_errors = [i for i in transformed["issues"] if i["type"] == IssueType.TYPE_ERROR.value]
        assert len(type_errors) == 0
    def test_any_with_multiple_spaces(self, pipeline, tmp_path):
        """Test detection of 'any' with multiple spaces between ':' and 'any'."""
        test_file = tmp_path / "test.ts"
        test_file.write_text("function test(x:  any) { return x; }")
        extracted = pipeline.extract("test.ts")
        transformed = pipeline.transform(extracted)
        # Should find the type error (multiple spaces should still be detected)
        type_errors = [i for i in transformed["issues"] if i["type"] == IssueType.TYPE_ERROR.value]
        assert len(type_errors) == 1
        assert extract_any_count_from_message(type_errors[0]["message"]) == 1
    def test_any_in_identifier_not_counted(self, pipeline, tmp_path):
        """Test that 'any' in identifiers like anyThing or anyType is NOT counted."""
        test_file = tmp_path / "test.ts"
        test_file.write_text("function test(x: anyThing) { return x; }\nconst y: anyType = 5;")
        extracted = pipeline.extract("test.ts")
        transformed = pipeline.transform(extracted)
        # Should not find type errors (anyThing and anyType are not 'any')
        type_errors = [i for i in transformed["issues"] if i["type"] == IssueType.TYPE_ERROR.value]
        assert len(type_errors) == 0
    def test_ts_ignore_next_line_not_recognized(self, pipeline, tmp_path):
        """Test that @ts-ignore-next-line is NOT recognized as @ts-ignore."""
        test_file = tmp_path / "test.ts"
        test_file.write_text("// @ts-ignore-next-line\nfunction test(x: any) { return x; }")
        extracted = pipeline.extract("test.ts")
        transformed = pipeline.transform(extracted)
        # Should still find the error (@ts-ignore-next-line is not @ts-ignore)
        type_errors = [i for i in transformed["issues"] if i["type"] == IssueType.TYPE_ERROR.value]
        assert len(type_errors) == 1
    def test_ts_ignore_with_reason_is_recognized(self, pipeline, tmp_path):
        """Test that @ts-ignore with a reason comment is still recognized."""
        test_file = tmp_path / "test.ts"
        test_file.write_text("// @ts-ignore legacy code\nfunction test(x: any) { return x; }")
        extracted = pipeline.extract("test.ts")
        transformed = pipeline.transform(extracted)
        # Should not find type errors (@ts-ignore followed by space and reason is valid)
        type_errors = [i for i in transformed["issues"] if i["type"] == IssueType.TYPE_ERROR.value]
        assert len(type_errors) == 0
class TestETLPipelineBasics:
    """Basic tests for ETLPipeline functionality."""
    def test_pipeline_initialization(self, tmp_path):
        """Test that ETLPipeline initializes correctly."""
        pipeline = ETLPipeline(str(tmp_path))
        assert pipeline.base_path == Path(tmp_path)
    def test_pipeline_extract_reads_file(self, tmp_path):
        """Test that extract phase reads file content."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        pipeline = ETLPipeline(str(tmp_path))
        extracted = pipeline.extract("test.txt")
        assert "content" in extracted
        assert extracted["content"] == "test content"
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
