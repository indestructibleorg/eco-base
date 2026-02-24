# 硬约束规范 (Hard Constraints Specification)

> **核心原则**: 任何代码如果不能通过以下验收门槛，CI 必须阻断，不能合并。

---

## 1. 接口契约约束 (Interface Contract Constraints)

### 1.1 每个公共函数必须满足

```python
# ❌ 禁止模式：软失败
def do_something():
    if not self._initialized:
        return None  # 或 return False, return []
    try:
        ...
    except Exception:
        return None  # 吞掉异常

# ✅ 强制模式：硬契约
def do_something() -> ResultType:
    """
    前置条件: self._initialized == True, 否则 raise RuntimeError
    后置条件: 返回值必须符合 ResultType 契约
    异常: 所有异常必须向上传播，禁止吞掉
    """
    if not self._initialized:
        raise RuntimeError("Service not initialized. Call initialize() first.")
    ...  # 异常自然传播，不捕获
```

### 1.2 契约验证清单

| 检查项 | 验收测试 | CI 阻断 |
|--------|----------|---------|
| 所有公共方法有类型注解 | `mypy --strict` | ✅ |
| 所有公共方法有 docstring | `pydocstyle` | ✅ |
| 无 bare except | `flake8 --select=E722` | ✅ |
| 无隐式 None 返回 | `mypy --warn-return-any` | ✅ |

---

## 2. 初始化硬约束 (Initialization Hard Constraints)

### 2.1 服务初始化契约

```python
class MyService:
    def __init__(self):
        self._initialized = False
    
    async def initialize(self) -> None:
        """
        硬约束:
        - 初始化失败必须 raise 异常，不能返回 False
        - 初始化成功后 _initialized = True
        - 所有依赖必须验证可用性
        """
        # 验证依赖可用性
        await self._validate_dependencies()  # 失败则 raise
        
        self._initialized = True
    
    def _ensure_initialized(self) -> None:
        """运行时检查，失败直接抛异常"""
        if not self._initialized:
            raise RuntimeError(
                f"{self.__class__.__name__} not initialized. "
                "Call initialize() during application startup."
            )
```

### 2.2 应用启动契约

```python
@app.on_event("startup")
async def startup():
    """
    硬约束: 任何初始化失败都导致应用启动失败
    """
    # 顺序初始化，任何一步失败都抛异常
    await db.initialize()      # 失败 -> 启动失败
    await cache.initialize()   # 失败 -> 启动失败
    await service.initialize() # 失败 -> 启动失败
```

---

## 3. 测试硬约束 (Testing Hard Constraints)

### 3.1 测试必须全部通过

```python
# ❌ 禁止模式
@pytest.mark.skip("Not implemented yet")
def test_feature():
    ...

# ❌ 禁止模式
def test_feature():
    if not SERVICE_AVAILABLE:
        pytest.skip("Service not available")
    ...

# ✅ 强制模式
def test_feature():
    """
    硬约束: 此测试必须在 CI 中通过，不能 skip
    如果依赖不可用，测试失败而非 skip
    """
    result = service.do_something()  # 失败则测试失败
    assert result.success  # 硬断言
```

### 3.2 测试覆盖率门槛

| 模块 | 行覆盖率 | 分支覆盖率 | CI 阻断 |
|------|----------|------------|---------|
| core/ | >= 90% | >= 80% | ✅ |
| services/ | >= 85% | >= 75% | ✅ |
| api/ | >= 80% | >= 70% | ✅ |

---

## 4. 代码审查硬约束 (Code Review Hard Constraints)

### 4.1 禁止模式清单

| 模式 | 示例 | 阻断原因 |
|------|------|----------|
| 软初始化 | `if not initialized: return None` | 隐藏依赖问题 |
| 裸 except | `except:` 或 `except Exception:` | 吞掉错误 |
| 教学注释 | `# 实际实现需要...` | 占位符 |
| 模拟数据 | `mock_data = [...]` | 非真实实现 |
| 条件跳过 | `if not available: return` | 绕过核心逻辑 |
| 隐式返回 | 函数末尾无 return | 可能返回 None |

### 4.2 自动化检查脚本

```bash
#!/bin/bash
# hard_constraints_check.sh
# 任何检查失败都返回非 0，阻断 CI

set -e

echo "=== 硬约束检查 ==="

# 1. 类型检查
echo "1. 类型检查 (mypy)..."
mypy --strict --warn-return-any eco-backend/app/

# 2. 禁止模式检查
echo "2. 禁止模式检查..."
# 检查软初始化
if grep -rn "if not.*initialized.*return" --include="*.py" eco-backend/app/; then
    echo "❌ 发现软初始化模式"
    exit 1
fi

# 检查裸 except
if grep -rn "except:" --include="*.py" eco-backend/app/ | grep -v "except Exception as"; then
    echo "❌ 发现裸 except"
    exit 1
fi

# 检查教学注释
if grep -rn "# 实际\|# 生產\|# 生产\|# 需要实现\|# TODO:" --include="*.py" eco-backend/app/; then
    echo "❌ 发现教学注释/TODO"
    exit 1
fi

# 3. 测试检查
echo "3. 测试检查..."
# 检查是否有 skip
if grep -rn "@pytest.mark.skip\|pytest.skip" --include="*.py" eco-backend/tests/; then
    echo "❌ 发现测试 skip"
    exit 1
fi

# 4. 测试运行
echo "4. 运行测试..."
pytest eco-backend/tests/ -v --tb=short

echo "✅ 所有硬约束检查通过"
```

---

## 5. 重构 `platform_integration_service.py` 示例

### 5.1 硬约束版本

```python
class PlatformIntegrationService:
    """
    硬约束: 此服务必须在应用启动时初始化成功
    任何初始化失败都导致应用启动失败
    """
    
    def __init__(self):
        self._service: Optional[EcoPlatformService] = None
        self._initialized = False
    
    async def initialize(self, config: Dict[str, Any]) -> None:
        """
        硬约束: 初始化失败 raise 异常，不返回 False
        """
        if not PLATFORM_INTEGRATIONS_AVAILABLE:
            raise RuntimeError(
                "eco-platform-integrations not available. "
                "Install: pip install -e ./eco-platform-integrations"
            )
        
        self._service = eco_service
        
        # 注册适配器 - 失败则抛异常
        register_all_adapters()  # 内部失败会抛异常
        
        # 配置提供商 - 失败则抛异常
        await self._configure_providers(config)  # 内部失败会抛异常
        
        # 验证所有必需提供商可用
        await self._validate_providers()
        
        self._initialized = True
        logger.info("platform_integration_service_initialized")
    
    async def _validate_providers(self) -> None:
        """验证所有配置的提供商可用"""
        health = await self.health_check()
        if health["status"] != "healthy":
            raise RuntimeError(f"Providers unhealthy: {health}")
    
    def _ensure_initialized(self) -> EcoPlatformService:
        """
        硬约束: 返回服务实例，未初始化直接抛异常
        """
        if not self._initialized or self._service is None:
            raise RuntimeError(
                "PlatformIntegrationService not initialized. "
                "Call initialize() during application startup."
            )
        return self._service
    
    async def persist_data(
        self,
        table: str,
        data: Dict[str, Any],
        provider: str = "supabase",
    ) -> IntegrationResult:
        """
        硬约束:
        - 前置: 服务已初始化
        - 失败: 返回 IntegrationResult(success=False, error=...)
        - 异常: 向上传播，不吞掉
        """
        service = self._ensure_initialized()  # 硬检查
        
        # 不捕获异常，让调用者处理
        mutation = MutationSpec(table=table, operation="insert", data=data)
        result = await service.execute(
            domain=CapabilityDomain.DATA_PERSISTENCE,
            operation="mutate",
            params={"mutation": mutation},
        )
        
        return IntegrationResult(
            success=result.success,
            data=result.data,
            error=result.error,
            provider=provider,
        )
```

---

## 6. CI/CD 集成

### 6.1 GitHub Actions 工作流

```yaml
name: Hard Constraints Check

on:
  pull_request:
    branches: [main]

jobs:
  hard-constraints:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install mypy flake8 pydocstyle pytest
          pip install -e ./eco-platform-integrations
          pip install -e ./eco-backend
      
      - name: Type check
        run: mypy --strict --warn-return-any eco-backend/app/
      
      - name: Lint check
        run: flake8 --select=E722 eco-backend/app/  # 禁止裸 except
      
      - name: Hard constraints check
        run: |
          chmod +x hard_constraints_check.sh
          ./hard_constraints_check.sh
      
      - name: Run tests
        run: pytest eco-backend/tests/ -v --tb=short
      
      - name: Coverage check
        run: |
          pytest eco-backend/tests/ --cov=eco-backend/app --cov-report=xml
          coverage report --fail-under=80
```

---

## 7. 验收清单 (Acceptance Checklist)

### 每个 PR 必须通过

- [ ] 所有类型检查通过 (`mypy --strict`)
- [ ] 无禁止模式 (`hard_constraints_check.sh`)
- [ ] 所有测试通过 (`pytest`)
- [ ] 覆盖率 >= 80%
- [ ] 无测试 skip
- [ ] 代码审查通过（人工）

### 人工审查重点

1. **是否有软失败模式？**
   - `if not initialized: return None/False/[]`
   - `try: ... except: return None`

2. **是否有教学注释？**
   - `# 实际实现...`
   - `# 需要...`
   - `# TODO:`

3. **是否有模拟数据？**
   - `mock_data = [...]`
   - `example_xxx = {...}`

4. **接口是否有契约定义？**
   - 输入参数类型
   - 返回值类型
   - 异常类型

---

**执行原则**: 任何不符合硬约束的代码，CI 必须阻断，不能合并到 main。
