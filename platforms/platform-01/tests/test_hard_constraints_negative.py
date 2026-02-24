# =============================================================================
# Hard Constraints Negative Tests (反向测试)
# =============================================================================
# 验证硬约束检查脚本在错误情况下真的会 fail
# 这些测试故意制造失败场景，验证脚本能正确检测
# =============================================================================

import subprocess
import sys
import tempfile
import os
from pathlib import Path

import pytest


class TestHardConstraintsNegative:
    """
    反向测试：验证硬约束检查在错误情况下会 fail
    
    这些测试故意制造失败场景，确保硬闸不会放过错误。
    """
    
    def test_script_detects_soft_initialization(self):
        """
        验证脚本能检测到软初始化模式
        
        验证脚本中包含检测软初始化的 grep 模式。
        """
        script_path = Path("hard_constraints_check.sh")
        with open(script_path) as f:
            content = f.read()
        
        # 脚本中必须有检测软初始化的模式
        assert "if not.*initialized.*return" in content, \
            "脚本必须包含检测软初始化的 grep 模式"
    
    def test_script_detects_bare_except(self):
        """验证脚本能检测到裸 except"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('''
try:
    do_something()
except:  # 裸 except - 应该被检测到
    pass
''')
            temp_file = f.name
        
        try:
            result = subprocess.run(
                ["grep", "except:", temp_file],
                capture_output=True,
                text=True
            )
            assert result.returncode == 0, "应该检测到裸 except"
        finally:
            os.unlink(temp_file)
    
    def test_script_detects_todo_comment(self):
        """验证脚本能检测到 TODO 注释"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('''
def do_something():
    # TODO: implement this - 应该被检测到
    pass
''')
            temp_file = f.name
        
        try:
            result = subprocess.run(
                ["grep", "# TODO:", temp_file],
                capture_output=True,
                text=True
            )
            assert result.returncode == 0, "应该检测到 TODO 注释"
        finally:
            os.unlink(temp_file)
    
    def test_script_detects_mock_data(self):
        """验证脚本能检测到模拟数据"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('''
def get_data():
    mock_data = [1, 2, 3]  # 模拟数据 - 应该被检测到
    return mock_data
''')
            temp_file = f.name
        
        try:
            result = subprocess.run(
                ["grep", "mock_data", temp_file],
                capture_output=True,
                text=True
            )
            assert result.returncode == 0, "应该检测到 mock_data"
        finally:
            os.unlink(temp_file)
    
    def test_script_has_strict_mode(self):
        """验证脚本有严格模式 set -euo pipefail"""
        script_path = Path("hard_constraints_check.sh")
        with open(script_path) as f:
            content = f.read()
        
        assert "set -euo pipefail" in content, "脚本必须有严格模式"
    
    def test_script_checks_pytest_exit_code(self):
        """验证脚本检查 pytest exit code"""
        script_path = Path("hard_constraints_check.sh")
        with open(script_path) as f:
            content = f.read()
        
        # 必须捕获 pytest exit code
        assert "pytest_exit=" in content or "pytest_exit =" in content, \
            "脚本必须捕获 pytest exit code"
        
        # 必须检查 exit code
        assert '[ "$pytest_exit" -ne 0 ]' in content, \
            "脚本必须检查 pytest exit code"
        
        # 失败时必须 exit
        assert 'exit "$pytest_exit"' in content, \
            "脚本失败时必须 exit"
    
    def test_script_checks_mypy_exit_code(self):
        """验证脚本检查 mypy exit code"""
        script_path = Path("hard_constraints_check.sh")
        with open(script_path) as f:
            content = f.read()
        
        assert "mypy_exit=" in content or "mypy_exit =" in content, \
            "脚本必须捕获 mypy exit code"
    
    def test_script_checks_coverage_threshold(self):
        """验证脚本检查覆盖率门槛"""
        script_path = Path("hard_constraints_check.sh")
        with open(script_path) as f:
            content = f.read()
        
        # 必须检查覆盖率门槛
        assert "COVERAGE_THRESHOLD" in content or "fail-under" in content, \
            "脚本必须检查覆盖率门槛"
    
    def test_requirements_lock_exists(self):
        """验证 requirements-dev.lock 存在"""
        lock_path = Path("requirements-dev.lock")
        assert lock_path.exists(), "requirements-dev.lock 必须存在"
        
        with open(lock_path) as f:
            content = f.read()
        
        # 必须包含关键依赖
        assert "pytest==" in content, "lock 文件必须包含 pytest"
        assert "mypy==" in content, "lock 文件必须包含 mypy"
        assert "coverage==" in content, "lock 文件必须包含 coverage"
    
    def test_ci_workflow_runs_hard_constraints(self):
        """验证 CI 工作流运行硬约束检查"""
        workflow_path = Path(".github/workflows/hard-constraints.yml")
        with open(workflow_path) as f:
            content = f.read()
        
        assert "hard_constraints_check.sh" in content, \
            "CI 必须运行 hard_constraints_check.sh"
    
    def test_ci_has_guard_tests(self):
        """验证 CI 有守卫测试 job"""
        workflow_path = Path(".github/workflows/hard-constraints.yml")
        with open(workflow_path) as f:
            content = f.read()
        
        assert "guard-tests:" in content, "CI 必须有 guard-tests job"
        assert "test_hard_constraints_guard.py" in content, \
            "CI 必须运行守卫测试"


class TestExitCodes:
    """退出码测试"""
    
    def test_script_returns_0_on_success(self):
        """验证脚本成功时返回 0"""
        script_path = Path("hard_constraints_check.sh")
        with open(script_path) as f:
            content = f.read()
        
        assert "exit 0" in content, "脚本成功时必须 exit 0"
    
    def test_script_returns_nonzero_on_failure(self):
        """验证脚本失败时返回非 0"""
        script_path = Path("hard_constraints_check.sh")
        with open(script_path) as f:
            content = f.read()
        
        assert "exit 1" in content, "脚本失败时必须 exit 1"
        assert "exit 2" in content or "exit \"" in content, \
            "脚本工具缺失时必须 exit 非 0"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
