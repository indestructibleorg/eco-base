# @ECO-governed
# @ECO-layer: GL90-99
# @ECO-semantic: archive-tools
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
# GL Unified Architecture Governance Framework Activated
#
# @ECO-governed
# @ECO-layer: gl-platform.governance
# @ECO-semantic: generation_engine
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
Generation Engine - 生成全自動化引擎
100% 自主執行的生成引擎，負責：
- 自動生成 Playbook
- 自動生成索引
- 自動生成報告
- 自動生成文檔
Version: 1.0.0
"""
# MNGA-002: Import organization needs review
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from engine_base import (
    BaseEngine,
    EngineConfig,
    EngineType,
    ExecutionMode,
    TaskResult,
)
sys.path.insert(0, str(Path(__file__).parent.parent))
BASE_PATH = Path(__file__).parent.parent.parent.parent
PLAYBOOKS_PATH = BASE_PATH / "docs" / "refactor_playbooks"
class GenerationEngine(BaseEngine):
    """生成全自動化引擎"""
    ENGINE_TYPE = EngineType.GENERATION
    def __init__(self, config: Optional[EngineConfig] = None):
        config = config or EngineConfig(
            engine_name="GenerationEngine",
            engine_type=EngineType.GENERATION,
            execution_mode=ExecutionMode.AUTONOMOUS,
        )
        super().__init__(config)
        self._target_path = PLAYBOOKS_PATH
        self._templates_path = PLAYBOOKS_PATH / "templates"
    async def _initialize(self) -> bool:
        self._logger.info("初始化生成引擎...")
        self._templates_path.mkdir(parents=True, exist_ok=True)
        return True
    async def _execute(self, task: Dict[str, Any]) -> TaskResult:
        task_id = task.get("task_id", "")
        operation = task.get("operation", "")
        try:
            if operation == "generate_playbook":
                result = await self._generate_playbook(task.get("params", {}))
            elif operation == "generate_index":
                result = await self._generate_index(task.get("target_dir"))
            elif operation == "generate_report":
                result = await self._generate_report(task.get("report_type", "summary"))
            elif operation == "generate_readme":
                result = await self._generate_readme(task.get("target_dir"))
            elif operation == "generate_all":
                result = await self._generate_all()
            else:
                return TaskResult(
                    task_id=task_id, success=False, error=f"未知操作: {operation}"
                )
            return TaskResult(task_id=task_id, success=True, result=result)
        except Exception as e:
            return TaskResult(task_id=task_id, success=False, error=str(e))
    async def _shutdown(self) -> bool:
        return True
    def _get_capabilities(self) -> Dict[str, Any]:
        return {
            "operations": [
                "generate_playbook",
                "generate_index",
                "generate_report",
                "generate_readme",
                "generate_all",
            ],
            "templates": (
                list(self._templates_path.glob("*.md"))
                if self._templates_path.exists()
                else []
            ),
        }
    async def _generate_playbook(self, params: Dict) -> Dict:
        """生成 Playbook"""
        name = params.get("name", "unnamed")
        cluster_id = params.get("cluster_id", name.lower().replace(" ", "_"))
        playbook_content = f"""# {name} Refactor Playbook
> 自動生成於 {datetime.now().strftime('%Y-%m-%d %H:%M')}
## 概述
{params.get('description', '待補充描述')}
## 目標
- [ ] 目標 1
- [ ] 目標 2
## 步驟
### 階段 1: 分析
1. 分析現有結構
2. 識別問題點
### 階段 2: 規劃
1. 設計目標結構
2. 制定遷移計畫
### 階段 3: 執行
1. 執行重構操作
2. 更新引用
### 階段 4: 驗證
1. 驗證結構正確性
2. 驗證功能完整性
## 風險
| 風險 | 等級 | 緩解措施 |
|------|------|----------|
| - | - | - |
## 回滾計畫
如需回滾，執行：
```bash
python tools/refactor/refactor_engine.py rollback --checkpoint latest
```
---
#此文件由 GenerationEngine 自動生成*
"""
        output_path = self._target_path / "03_refactor" / f"{cluster_id}__playbook.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(playbook_content, encoding="utf-8")
        return {"generated": str(output_path), "cluster_id": cluster_id}
    async def _generate_index(self, target_dir: str = None) -> Dict:
        """生成索引"""
        target = Path(target_dir) if target_dir else self._target_path / "03_refactor"
        # 掃描 playbook 檔案
        playbooks = list(target.glob("*__playbook.md")) + list(
            target.glob("*_playbook.md")
        )
        clusters = []
        for pb in playbooks:
            cluster_id = pb.stem.replace("__playbook", "").replace("_playbook", "")
            clusters.append(
                {
                    "cluster_id": cluster_id,
                    "name": cluster_id.replace("_", " ").title(),
                    "playbook_path": str(pb.relative_to(self._target_path)),
                    "status": "active",
                }
            )
        index_data = {
            "version": "1.0.0",
            "generated_at": datetime.now().isoformat(),
            "refactor_clusters": clusters,
        }
        index_path = target / "index.yaml"
        with open(index_path, "w", encoding="utf-8") as f:
            yaml.dump(index_data, f, allow_unicode=True, default_flow_style=False)
        return {"generated": str(index_path), "clusters": len(clusters)}
    async def _generate_report(self, report_type: str) -> Dict:
        """生成報告"""
        if report_type == "summary":
            report = await self._generate_summary_report()
        elif report_type == "detailed":
            report = await self._generate_detailed_report()
        else:
            report = await self._generate_summary_report()
        output_path = (
            self._target_path
            / "reports"
            / f"{report_type}_report_{datetime.now().strftime('%Y%m%d')}.md"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        return {"generated": str(output_path)}
    async def _generate_summary_report(self) -> str:
        """生成摘要報告"""
        # 統計
        total_files = len(list(self._target_path.rglob("*")))
        total_dirs = len([d for d in self._target_path.rglob("*") if d.is_dir()])
        playbooks = len(list(self._target_path.rglob("*playbook*.md")))
        return f"""# Refactor Playbooks 摘要報告
> 生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M')}
## 統計
| 指標 | 數值 |
|------|------|
| 總檔案數 | {total_files} |
| 總目錄數 | {total_dirs} |
| Playbook 數 | {playbooks} |
## 結構
```
docs/refactor_playbooks/
├── 01_deconstruction/
├── 02_integration/
├── 03_refactor/
├── config/
└── templates/
```
---
#此報告由 GenerationEngine 自動生成*
"""
    async def _generate_detailed_report(self) -> str:
        """生成詳細報告"""
        return await self._generate_summary_report()  # 暫時使用摘要
    async def _generate_readme(self, target_dir: str = None) -> Dict:
        """生成 README"""
        target = Path(target_dir) if target_dir else self._target_path
        readme_content = f"""# {target.name.replace('_', ' ').title()}
> 自動生成於 {datetime.now().strftime('%Y-%m-%d %H:%M')}
## 目錄結構
"""
        # 列出子目錄
        for item in sorted(target.iterdir()):
            if item.is_dir() and not item.name.startswith("."):
                readme_content += (
                    f"- `{item.name}/` - {item.name.replace('_', ' ').title()}\n"
                )
        readme_content += """
## 使用方式
請參考各子目錄的 README 文件。
---
#此文件由 GenerationEngine 自動生成*
"""
        readme_path = target / "README.md"
        readme_path.write_text(readme_content, encoding="utf-8")
        return {"generated": str(readme_path)}
    async def _generate_all(self) -> Dict:
        """生成所有內容"""
        results = {
            "indexes": [],
            "reports": [],
            "readmes": [],
        }
        # 生成索引
        for subdir in ["01_deconstruction", "02_integration", "03_refactor"]:
            target = self._target_path / subdir
            if target.exists():
                result = await self._generate_index(str(target))
                results["indexes"].append(result)
        # 生成報告
        result = await self._generate_report("summary")
        results["reports"].append(result)
        # 生成 README
        result = await self._generate_readme()
        results["readmes"].append(result)
        return results
