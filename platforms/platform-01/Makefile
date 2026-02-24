# Makefile - 閉環系統一鍵驗收
# 任何人、任何時間、任何重跑都能得到同樣結論

.PHONY: verify verify-json verify-junit verify-continuous verify-all clean help \
    test-contract test-state test-crash test-idem test-verify test-audit test-e2e acceptance \
    verify-artifacts verify-manifest

# 默認目標
.DEFAULT_GOAL := verify

# Python 解釋器
PYTHON := python3
PIP := pip3

# 輸出目錄
OUTPUT_DIR := verification_reports
ARTIFACTS_DIR := artifacts

# 安裝依賴
install:
	@echo "📦 安裝依賴..."
	$(PIP) install -q jsonschema semver pytest pytest-asyncio 2>/dev/null || true
	@echo "✅ 依賴安裝完成"

# 一鍵驗收 - 默認輸出 Markdown 格式
verify: install
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════╗"
	@echo "║           閉環系統 6 層驗證矩陣 - 一鍵驗收                    ║"
	@echo "╚══════════════════════════════════════════════════════════════╝"
	@echo ""
	@mkdir -p $(OUTPUT_DIR)
	$(PYTHON) verify.py --output $(OUTPUT_DIR)/verify_report_$(shell date +%Y%m%d_%H%M%S).md
	@echo ""
	@echo "✅ 驗證完成，報告保存在 $(OUTPUT_DIR)/"

# JSON 格式輸出
verify-json: install
	@mkdir -p $(OUTPUT_DIR)
	$(PYTHON) verify.py --output-format json --output $(OUTPUT_DIR)/verify_report_$(shell date +%Y%m%d_%H%M%S).json

# JUnit XML 格式輸出（CI 使用）
verify-junit: install
	@mkdir -p $(OUTPUT_DIR)
	$(PYTHON) verify.py --output-format junit --output $(OUTPUT_DIR)/verify_report_$(shell date +%Y%m%d_%H%M%S).xml

# 連續跑 3 次驗證（最終驗收標準）
verify-continuous: install
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════╗"
	@echo "║        連續 3 次驗證（最終驗收標準）                          ║"
	@echo "╚══════════════════════════════════════════════════════════════╝"
	@echo ""
	@mkdir -p $(OUTPUT_DIR)
	$(PYTHON) verify.py --continuous 3 --output $(OUTPUT_DIR)/verify_continuous_$(shell date +%Y%m%d_%H%M%S).md
	@echo ""
	@echo "✅ 連續驗證完成"

# 完整驗證（所有層）
verify-all: install
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════╗"
	@echo "║              完整 6 層驗證                                    ║"
	@echo "╚══════════════════════════════════════════════════════════════╝"
	@echo ""
	@mkdir -p $(OUTPUT_DIR)
	$(PYTHON) verify.py --output $(OUTPUT_DIR)/verify_all_$(shell date +%Y%m%d_%H%M%S).md

# 單層驗證
verify-layer-1: install
	@echo "🔍 執行第 1 層驗證：合約（Contract）驗證"
	$(PYTHON) verify.py --layer 1

verify-layer-2: install
	@echo "🔍 執行第 2 層驗證：狀態機（State Machine）驗證"
	$(PYTHON) verify.py --layer 2

verify-layer-3: install
	@echo "🔍 執行第 3 層驗證：幂等（Idempotency）驗證"
	$(PYTHON) verify.py --layer 3

verify-layer-4: install
	@echo "🔍 執行第 4 層驗證：Crash-Resume 驗證"
	$(PYTHON) verify.py --layer 4

verify-layer-5: install
	@echo "🔍 執行第 5 層驗證：Verification Gate 驗證"
	$(PYTHON) verify.py --layer 5

verify-layer-6: install
	@echo "🔍 執行第 6 層驗證：稽核/可重播（Audit/Reproducibility）驗證"
	$(PYTHON) verify.py --layer 6

# ============================================================================
# Closed-Loop Acceptance Kit - 驗收測試目標
# ============================================================================

# 契約驗證 - 驗證 JSON Schema 和產物完整性
test-contract: install
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════╗"
	@echo "║              契約驗證 (Contract Validation)                   ║"
	@echo "╚══════════════════════════════════════════════════════════════╝"
	@echo ""
	$(PYTHON) scripts/validate_contracts.py \
		--schemas-dir schemas/ \
		--artifacts-glob "artifacts/*.json" \
		--fail-on-missing
	@echo "✅ 契約驗證通過"

# 驗證產物完整性
verify-artifacts: install
	@echo "🔍 驗證產物完整性..."
	$(PYTHON) scripts/verify_artifacts_required.py \
		--artifacts-dir artifacts/ \
		--strict
	@echo "✅ 產物完整性驗證通過"

# 驗證 manifest hash
verify-manifest: install
	@echo "🔍 驗證 manifest hash..."
	$(PYTHON) scripts/verify_manifest.py \
		--manifest artifacts/manifest.json \
		--verify
	@echo "✅ Manifest hash 驗證通過"

# 狀態機測試
test-state: install
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════╗"
	@echo "║            狀態機測試 (State Machine Test)                    ║"
	@echo "╚══════════════════════════════════════════════════════════════╝"
	@echo ""
	pytest tests/test_state_machine.py -v --tb=short
	@echo "✅ 狀態機測試通過"

# Crash-Resume E2E 測試
test-crash: install
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════╗"
	@echo "║         Crash-Resume E2E 測試                                 ║"
	@echo "╚══════════════════════════════════════════════════════════════╝"
	@echo ""
	pytest tests/e2e/test_crash_resume_e2e.py -v --tb=short
	@echo "✅ Crash-Resume 測試通過"

# 幂等性 E2E 測試
test-idem: install
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════╗"
	@echo "║           幂等性 E2E 測試 (Idempotency Test)                  ║"
	@echo "╚══════════════════════════════════════════════════════════════╝"
	@echo ""
	pytest tests/e2e/test_idempotency_e2e.py -v --tb=short
	@echo "✅ 幂等性測試通過"

# Verify-Fail E2E 測試
test-verify: install
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════╗"
	@echo "║         Verify-Fail E2E 測試                                  ║"
	@echo "╚══════════════════════════════════════════════════════════════╝"
	@echo ""
	pytest tests/e2e/test_verify_fail_e2e.py -v --tb=short
	@echo "✅ Verify-Fail 測試通過"

# Audit E2E 測試
test-audit: install
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════╗"
	@echo "║           Audit E2E 測試                                      ║"
	@echo "╚══════════════════════════════════════════════════════════════╝"
	@echo ""
	pytest tests/e2e/test_audit_e2e.py -v --tb=short
	@echo "✅ Audit 測試通過"

# 所有 E2E 測試
test-e2e: test-crash test-idem test-verify test-audit
	@echo ""
	@echo "✅ 所有 E2E 測試通過"

# 單元測試（scripts 驗證）
test-unit: install
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════╗"
	@echo "║           單元測試 (Unit Tests)                               ║"
	@echo "╚══════════════════════════════════════════════════════════════╝"
	@echo ""
	pytest tests/test_scripts_contracts_and_audit.py -v --tb=short
	@echo "✅ 單元測試通過"

# 完整驗收套件 - 6 層全部驗證
acceptance: install
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════╗"
	@echo "║                                                              ║"
	@echo "║        Closed-Loop Acceptance Kit - 完整驗收套件              ║"
	@echo "║                                                              ║"
	@echo "║     任何人、任何時間、任何重跑都能得到同樣結論：              ║"
	@echo "║        不中斷、可續跑、可回滾、可稽核                         ║"
	@echo "║                                                              ║"
	@echo "╚══════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════"
	@echo "Layer 1: 契約驗證 (Contract)"
	@echo "═══════════════════════════════════════════════════════════════"
	$(MAKE) test-contract
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════"
	@echo "Layer 2: 狀態機測試 (State Machine)"
	@echo "═══════════════════════════════════════════════════════════════"
	$(MAKE) test-state
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════"
	@echo "Layer 3: Crash-Resume E2E"
	@echo "═══════════════════════════════════════════════════════════════"
	$(MAKE) test-crash
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════"
	@echo "Layer 4: 幂等性 E2E (Idempotency)"
	@echo "═══════════════════════════════════════════════════════════════"
	$(MAKE) test-idem
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════"
	@echo "Layer 5: Verify-Fail E2E"
	@echo "═══════════════════════════════════════════════════════════════"
	$(MAKE) test-verify
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════"
	@echo "Layer 6: Audit E2E"
	@echo "═══════════════════════════════════════════════════════════════"
	$(MAKE) test-audit
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════"
	@echo "產物完整性驗證"
	@echo "═══════════════════════════════════════════════════════════════"
	$(MAKE) verify-artifacts
	$(MAKE) verify-manifest
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════╗"
	@echo "║                                                              ║"
	@echo "║              ✅ 完整驗收通過 (ALL ACCEPTANCE PASSED)          ║"
	@echo "║                                                              ║"
	@echo "║     系統滿足以下不變量：                                      ║"
	@echo "║     • 不中斷 (Crash-Resume): 任意中斷都能續跑                 ║"
	@echo "║     • 可續跑 (Idempotency): 重放不產生副作用                  ║"
	@echo "║     • 可回滾 (Verify-Fail): 驗證失敗必回滾                    ║"
	@echo "║     • 可稽核 (Audit): hash 一致、證據齊全                     ║"
	@echo "║                                                              ║"
	@echo "╚══════════════════════════════════════════════════════════════╝"

# Schema 驗證（CI 必跑）
verify-schema:
	@echo "🔍 驗證 JSON Schema..."
	$(PYTHON) -c "import json, jsonschema; \
		schemas = ['contracts/decision.schema.json', 'contracts/action.schema.json', 'contracts/run.schema.json']; \
		[jsonschema.Draft7Validator.check_schema(json.load(open(s))) for s in schemas]; \
		print('✅ All schemas valid')"

# 版本號驗證（CI 必跑）
verify-versions:
	@echo "🔍 驗證版本號 semver 格式..."
	$(PYTHON) -c "import json, semver; \
		schemas = ['contracts/decision.schema.json', 'contracts/action.schema.json', 'contracts/run.schema.json']; \
		[semver.VersionInfo.parse(json.load(open(s))['version']) for s in schemas]; \
		print('✅ All versions valid semver')"

# 清理
 clean:
	@echo "🧹 清理..."
	rm -rf $(OUTPUT_DIR)
	rm -rf $(ARTIFACTS_DIR)
	rm -rf .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ 清理完成"

# 幫助
help:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════╗"
	@echo "║           閉環系統一鍵驗收 - 可用命令                         ║"
	@echo "╚══════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "一鍵驗收:"
	@echo "  make verify              - 執行完整驗證（默認 Markdown 格式）"
	@echo "  make verify-json         - JSON 格式輸出"
	@echo "  make verify-junit        - JUnit XML 格式輸出（CI 使用）"
	@echo "  make verify-continuous   - 連續跑 3 次（最終驗收標準）"
	@echo ""
	@echo "單層驗證:"
	@echo "  make verify-layer-1      - 第 1 層：合約驗證"
	@echo "  make verify-layer-2      - 第 2 層：狀態機驗證"
	@echo "  make verify-layer-3      - 第 3 層：幂等驗證"
	@echo "  make verify-layer-4      - 第 4 層：Crash-Resume 驗證"
	@echo "  make verify-layer-5      - 第 5 層：Verification Gate 驗證"
	@echo "  make verify-layer-6      - 第 6 層：稽核驗證"
	@echo ""
	@echo "CI 驗證:"
	@echo "  make verify-schema       - Schema 有效性驗證"
	@echo "  make verify-versions     - 版本號 semver 驗證"
	@echo "  make verify-artifacts    - 產物完整性驗證"
	@echo "  make verify-manifest     - Manifest hash 驗證"
	@echo ""
	@echo "驗收測試套件 (Acceptance Kit):"
	@echo "  make acceptance          - 完整 6 層驗收套件"
	@echo "  make test-unit           - 單元測試（scripts 驗證）"
	@echo "  make test-contract       - 契約驗證 (Layer 1)"
	@echo "  make test-state          - 狀態機測試 (Layer 2)"
	@echo "  make test-crash          - Crash-Resume E2E (Layer 3)"
	@echo "  make test-idem           - 幂等性 E2E (Layer 4)"
	@echo "  make test-verify         - Verify-Fail E2E (Layer 5)"
	@echo "  make test-audit          - Audit E2E (Layer 6)"
	@echo "  make test-e2e            - 所有 E2E 測試"
	@echo ""
	@echo "其他:"
	@echo "  make install             - 安裝依賴"
	@echo "  make clean               - 清理臨時文件"
	@echo "  make help                - 顯示此幫助"
	@echo ""
	@echo "驗證層說明:"
	@echo "  Layer 1: 合約（Contract）驗證 - Schema + 版本鎖"
	@echo "  Layer 2: 狀態機（State Machine）驗證 - 轉移表 + 性質測試"
	@echo "  Layer 3: 幂等（Idempotency）驗證 - 重放不產生副作用"
	@echo "  Layer 4: Crash-Resume 驗證 - 任意中斷都能續跑"
	@echo "  Layer 5: Verification Gate 驗證 - 驗證失敗必回滾/升級"
	@echo "  Layer 6: 稽核/可重播（Audit/Reproducibility）驗證 - hash 一致、證據齊全"
	@echo ""
