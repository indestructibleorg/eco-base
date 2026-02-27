#
# @ECO-governed
# @ECO-layer: gl-platform.gl-platform.governance
# @ECO-semantic: auto-fix-issues
# @ECO-audit-trail: ../../engine/gl-platform.gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
自動修復程式碼問題工具
Automated Code Issue Fix Tool
此腳本自動修復 PR-1-REVIEW-REPORT.md 中識別的可自動修復問題
This script automatically fixes issues identified in PR-1-REVIEW-REPORT.md
"""
import subprocess
from pathlib import Path
import argparse
class AutoFixer:
    """自動修復器"""
    def __init__(self, repo_root: Path, dry_run: bool = False):
        self.repo_root = repo_root
        self.dry_run = dry_run
        self.fixed_count = 0
    def fix_all(self):
        """執行所有自動修復"""
        print("🔧 開始自動修復程式碼問題...")
        if self.dry_run:
            print("⚠️  DRY RUN 模式 - 不會實際修改檔案")
        self.format_python_code()
        self.fix_import_sorting()
        self.add_gitignore_for_env()
        self.create_env_example()
        print(f"\n✅ 完成！共修復 {self.fixed_count} 個問題")
    def format_python_code(self):
        """使用 Black 格式化 Python 程式碼"""
        print("\n🎨 格式化 Python 程式碼...")
        try:
            cmd = ["black", "--line-length", "100"]
            if self.dry_run:
                cmd.append("--check")
            cmd.extend(["workspace/src/", "ns-root/"])
            result = subprocess.run(
                cmd,
                cwd=self.repo_root,
                capture_output=True,
                text=True
            )
            if not self.dry_run and result.returncode == 0:
                self.fixed_count += 1
                print("✅ Python 程式碼已格式化")
            elif self.dry_run:
                print("ℹ️  將格式化 Python 程式碼")
        except FileNotFoundError:
            print("⚠️  Black 未安裝，請執行: pip install black")
    def fix_import_sorting(self):
        """使用 isort 排序 imports"""
        print("\n📦 排序 Python imports...")
        try:
            cmd = ["isort", "--profile", "black", "--line-length", "100"]
            if self.dry_run:
                cmd.append("--check")
            cmd.extend(["workspace/src/", "ns-root/"])
            result = subprocess.run(
                cmd,
                cwd=self.repo_root,
                capture_output=True,
                text=True
            )
            if not self.dry_run and result.returncode == 0:
                self.fixed_count += 1
                print("✅ Imports 已排序")
            elif self.dry_run:
                print("ℹ️  將排序 imports")
        except FileNotFoundError:
            print("⚠️  isort 未安裝，請執行: pip install isort")
    def add_gitignore_for_env(self):
        """確保 .env 檔案在 .gitignore 中"""
        print("\n🔒 更新 .gitignore...")
        gitignore_path = self.repo_root / ".gitignore"
        env_patterns = [
            ".env",
            ".env.local",
            ".env.*.local",
            "*.env",
        ]
        if gitignore_path.exists():
            content = gitignore_path.read_text()
            missing_patterns = [p for p in env_patterns if p not in content]
            if missing_patterns and not self.dry_run:
                with open(gitignore_path, "a", encoding='utf-8') as f:
                    f.write("\n# Environment variables (added by auto-fix)\n")
                    for pattern in missing_patterns:
                        f.write(f"{pattern}\n")
                self.fixed_count += 1
                print(f"✅ 已添加 {len(missing_patterns)} 個模式到 .gitignore")
            elif missing_patterns:
                print(f"ℹ️  將添加 {len(missing_patterns)} 個模式到 .gitignore")
            else:
                print("✅ .gitignore 已包含環境變數模式")
        else:
            if not self.dry_run:
                with open(gitignore_path, "w", encoding='utf-8') as f:
                    f.write("# Environment variables\n")
                    for pattern in env_patterns:
                        f.write(f"{pattern}\n")
                self.fixed_count += 1
                print("✅ 已建立 .gitignore")
            else:
                print("ℹ️  將建立 .gitignore")
    def create_env_example(self):
        """建立 .env.example 範本"""
        print("\n📝 建立 .env.example...")
        env_example_path = self.repo_root / ".env.example"
        if not env_example_path.exists():
            example_content = """# 環境變數範本
# Environment Variables Template
# 資料庫設定 / Database Configuration
# DB_HOST=localhost
# DB_PORT=5432
# DB_NAME=your_database
# DB_USER=your_user
# DB_PASSWORD=your_password
# API 金鑰 / API Keys
# API_KEY=your_api_key_here
# SECRET_KEY=your_secret_key_here
# Jira 整合 / Jira Integration
# JIRA_API_KEY=your_jira_api_key
# JIRA_BASE_URL=https://your-domain.atlassian.net
# JIRA_USERNAME=your_email@example.com
# Slack 整合 / Slack Integration
# SLACK_TOKEN=your_slack_token
# SLACK_WEBHOOK_URL=your_webhook_url
# 日誌設定 / Logging Configuration
# LOG_LEVEL=INFO
# LOG_FILE=logs/app.log
# 其他設定 / Other Settings
# NODE_ENV=development
"""
            if not self.dry_run:
                env_example_path.write_text(example_content)
                self.fixed_count += 1
                print("✅ 已建立 .env.example")
            else:
                print("ℹ️  將建立 .env.example")
        else:
            print("✅ .env.example 已存在")
def main():
    parser = argparse.ArgumentParser(description="自動修復程式碼問題")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="倉庫根目錄路徑"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="預覽模式，不實際修改檔案"
    )
    args = parser.parse_args()
    fixer = AutoFixer(args.repo_root, args.dry_run)
    fixer.fix_all()
if __name__ == "__main__":
    main()
