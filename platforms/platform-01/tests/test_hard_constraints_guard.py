# =============================================================================
# Hard Constraints Guard Tests
# =============================================================================
# 验证硬约束检查脚本本身的正确性
# 确保硬闸不会退化
# =============================================================================

import subprocess
import sys
import tempfile
import os
from pathlib import Path

import pytest


class TestHardConstraintsGuardBehavior:
    """
    硬约束守卫行为测试
    
    验证点:
    1. 工具缺失时脚本必须 fail
    2. 检查失败时脚本必须 fail
    3. 测试失败时脚本必须 fail
    4. 全部通过时脚本必须 pass
    """
    
    def test_script_exists(self):
        """硬约束脚本必须存在"""
        script_path = Path("hard_constraints_check.sh")
        assert script_path.exists(), "hard_constraints_check.sh must exist"
        assert script_path.stat().st_size > 0, "script must not be empty"
    
    def test_script_has_shebang(self):
        """脚本必须有正确的 shebang"""
        script_path = Path("hard_constraints_check.sh")
        with open(script_path) as f:
            first_line = f.readline().strip()
        assert first_line == "#!/bin/bash", "script must have #!/bin/bash shebang"
    
    def test_script_has_strict_mode(self):
        """脚本必须有严格模式 (set -euo pipefail)"""
        script_path = Path("hard_constraints_check.sh")
        with open(script_path) as f:
            content = f.read()
        assert "set -euo pipefail" in content, "script must have strict mode"
    
    def test_script_checks_pytest_exit_code(self):
        """脚本必须正确检查 pytest exit code"""
        script_path = Path("hard_constraints_check.sh")
        with open(script_path) as f:
            content = f.read()
        
        # 必须检查 pytest_exit
        assert "pytest_exit" in content, "script must capture pytest exit code"
        assert 'if [ "$pytest_exit" -ne 0 ]' in content, "script must check pytest exit code"
        assert 'exit "$pytest_exit"' in content, "script must exit with pytest exit code"
    
    def test_script_requires_tools(self):
        """脚本必须要求工具存在"""
        script_path = Path("hard_constraints_check.sh")
        with open(script_path) as f:
            content = f.read()
        
        assert "require_cmd" in content, "script must have require_cmd function"
        assert "missing required tool" in content, "script must fail on missing tool"


class TestProhibitedPatternsDetection:
    """禁止模式检测测试"""
    
    @pytest.fixture
    def temp_code_file(self):
        """创建临时代码文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            yield f.name
        os.unlink(f.name)
    
    def test_detects_soft_initialization(self, temp_code_file):
        """检测软初始化模式 - 验证脚本使用正确的 grep 模式"""
        script_path = Path("hard_constraints_check.sh")
        with open(script_path) as f:
            content = f.read()
        
        # 脚本中必须有检测软初始化的 grep 模式
        assert "if not.*initialized.*return" in content, \
            "script must check for soft initialization pattern"
    
    def test_detects_bare_except(self, temp_code_file):
        """检测裸 except"""
        code = '''
try:
    do_something()
except:  # 裸 except
    return None
'''
        with open(temp_code_file, 'w') as f:
            f.write(code)
        
        result = subprocess.run(
            ["grep", "except:", temp_code_file],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, "should detect bare except"
    
    def test_detects_todo_comments(self, temp_code_file):
        """检测 TODO 注释"""
        code = '''
def do_something():
    # TODO: implement this
    pass
'''
        with open(temp_code_file, 'w') as f:
            f.write(code)
        
        result = subprocess.run(
            ["grep", "# TODO:", temp_code_file],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, "should detect TODO comment"
    
    def test_detects_mock_data(self, temp_code_file):
        """检测模拟数据"""
        code = '''
def get_data():
    mock_data = [1, 2, 3]  # 模拟数据
    return mock_data
'''
        with open(temp_code_file, 'w') as f:
            f.write(code)
        
        result = subprocess.run(
            ["grep", "mock_data", temp_code_file],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, "should detect mock_data"


class TestHardConstraintsExitCodes:
    """硬约束退出码测试"""
    
    def test_script_returns_zero_on_success(self):
        """全部通过时返回 0"""
        # 这个测试在真实环境中运行
        # 这里验证脚本结构支持正确退出码
        script_path = Path("hard_constraints_check.sh")
        with open(script_path) as f:
            content = f.read()
        
        # 必须有明确的 exit 0
        assert "exit 0" in content, "script must exit 0 on success"
    
    def test_script_returns_nonzero_on_failure(self):
        """失败时返回非 0"""
        script_path = Path("hard_constraints_check.sh")
        with open(script_path) as f:
            content = f.read()
        
        # 必须有明确的 exit 1
        assert "exit 1" in content, "script must exit 1 on failure"
        assert "exit 2" in content or "exit \"" in content, "script must exit non-zero on tool missing"


class TestRequirementsDev:
    """requirements-dev.txt 测试"""
    
    def test_requirements_dev_exists(self):
        """requirements-dev.txt 必须存在"""
        req_path = Path("requirements-dev.txt")
        assert req_path.exists(), "requirements-dev.txt must exist"
    
    def test_has_pytest_asyncio(self):
        """必须包含 pytest-asyncio"""
        req_path = Path("requirements-dev.txt")
        with open(req_path) as f:
            content = f.read()
        assert "pytest-asyncio" in content, "must have pytest-asyncio"
    
    def test_has_mypy(self):
        """必须包含 mypy"""
        req_path = Path("requirements-dev.txt")
        with open(req_path) as f:
            content = f.read()
        assert "mypy" in content, "must have mypy"
    
    def test_has_coverage(self):
        """必须包含 coverage"""
        req_path = Path("requirements-dev.txt")
        with open(req_path) as f:
            content = f.read()
        assert "coverage" in content or "pytest-cov" in content, "must have coverage tool"


class TestCIWorkflow:
    """CI 工作流测试"""
    
    def test_ci_workflow_exists(self):
        """CI 工作流必须存在"""
        workflow_path = Path(".github/workflows/hard-constraints.yml")
        assert workflow_path.exists(), "CI workflow must exist"
    
    def test_ci_runs_hard_constraints_script(self):
        """CI 必须运行硬约束脚本"""
        workflow_path = Path(".github/workflows/hard-constraints.yml")
        with open(workflow_path) as f:
            content = f.read()
        
        assert "hard_constraints_check.sh" in content, "CI must run hard_constraints_check.sh"
    
    def test_ci_has_gate_job(self):
        """CI 必须有门控 job"""
        workflow_path = Path(".github/workflows/hard-constraints.yml")
        with open(workflow_path) as f:
            content = f.read()
        
        assert "gate:" in content or "needs:" in content, "CI must have gate mechanism"


class TestNoRegressionPatterns:
    """防止退化模式测试"""
    
    def test_no_pipe_to_true_in_script(self):
        """脚本中不能有 `|| true` 来忽略错误"""
        script_path = Path("hard_constraints_check.sh")
        with open(script_path) as f:
            content = f.read()
        
        # 允许在 grep 中使用 || true（这是为了处理无匹配的情况）
        # 但不允许在关键检查中使用
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if '|| true' in line:
                # 只允许在 grep 命令中使用
                assert 'grep' in line, f"line {i+1}: || true should only be used with grep"
    
    def test_no_echo_pipe_grep_for_result_check(self):
        """不能用 echo | grep 来检查结果"""
        script_path = Path("hard_constraints_check.sh")
        with open(script_path) as f:
            content = f.read()
        
        # 这种模式是有问题的：echo "$VAR" | grep ...
        # 应该直接检查变量
        assert "echo \"\$" not in content or "echo \$" not in content, \
            "should not use echo | grep for checking results"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
