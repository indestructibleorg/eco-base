# Platforms 重構強制檢索工作流

## 啟動（定義問題）

```yaml
task_id: <required>
scope: platforms/<target>
objective: <required>
constraints:
  - minimal_change_only
  - security_redline_enforced
  - output_must_be_actionable
```

P0 執行指令：

```bash
TASK_ID=P0-platforms-refactor-retrieval TARGET=platforms OUT=.tmp/refactor-retrieval ./scripts/platforms_refactor_retrieval.sh
```

P1 執行指令：

```bash
PHASE=P1 TASK_ID=P1-platforms-refactor-retrieval TARGET=platforms OUT=.tmp/refactor-retrieval ./scripts/platforms_refactor_retrieval.sh
```

P2 執行指令：

```bash
PHASE=P2 TASK_ID=P2-platforms-refactor-retrieval TARGET=platforms OUT=.tmp/refactor-retrieval ./scripts/platforms_refactor_retrieval.sh
```

## 階段 1：內網檢索與剖析（強制）

### 執行指令

```bash
set -euo pipefail

TARGET="${TARGET:-platforms}"
OUT="${OUT:-.tmp/refactor-retrieval}"
mkdir -p "$OUT"

# 1) 基準事實
find "$TARGET" -maxdepth 4 -type f \( -name "*.md" -o -name "*.yaml" -o -name "*.yml" -o -name "*.json" -o -name "*.ts" -o -name "*.py" \) | sort > "$OUT/facts.files.txt"

# 2) 內部假設
grep -RInE "TODO|FIXME|HACK|DEPRECATED|legacy|archived|暫時|待補" "$TARGET" > "$OUT/assumptions.raw.txt" || true

# 3) 資訊缺口
grep -RInE "TBD|TBA|unknown|未定|待確認|缺失|missing" "$TARGET" > "$OUT/gaps.raw.txt" || true

# 4) 安全紅線
grep -RInE "secret|token|password|private key|KMS|cosign|slsa|kyverno|policy|admission" "$TARGET" > "$OUT/security-redline.raw.txt" || true

wc -l "$OUT"/*.txt > "$OUT/summary.counts.txt"
```

### 關鍵產出（必出）

- `facts.files.txt`
- `assumptions.raw.txt`
- `gaps.raw.txt`
- `security-redline.raw.txt`
- `summary.counts.txt`

## 決策點 1：缺口是否明確且合規

```bash
set -euo pipefail
OUT="${OUT:-.tmp/refactor-retrieval}"

gaps=$(wc -l < "$OUT/gaps.raw.txt" || echo 0)
redlines=$(wc -l < "$OUT/security-redline.raw.txt" || echo 0)

if [ "$gaps" -gt 0 ] && [ "$redlines" -ge 0 ]; then
  echo "DECISION_1=NO"
  echo "ACTION=expand_internal_access_or_clarify_question"
else
  echo "DECISION_1=YES"
  echo "ACTION=proceed_external_retrieval"
fi
```

`DECISION_1=NO` 時執行：

```bash
grep -RInE "owner|maintainer|contact|runbook|ADR|architecture" platforms docs | sort > .tmp/refactor-retrieval/clarify.targets.txt
```

## 階段 2：外網專業檢索（深度/權威）

```bash
set -euo pipefail
OUT="${OUT:-.tmp/refactor-retrieval}"

cat > "$OUT/external.professional.sources.txt" <<'EOF'
https://kubernetes.io/docs/
https://argo-cd.readthedocs.io/
https://kyverno.io/docs/
https://slsa.dev/
https://docs.sigstore.dev/
EOF
```

## 階段 3：全球開放檢索（廣度/即時）

```bash
set -euo pipefail
OUT="${OUT:-.tmp/refactor-retrieval}"

cat > "$OUT/external.open.sources.txt" <<'EOF'
https://github.com/topics/platform-engineering
https://github.com/topics/gitops
https://github.com/search?q=argocd+applicationset+monorepo&type=code
EOF
```

> 階段 2/3 可在安全紅線內並行。

## 核心：交叉驗證與綜合推理（驗證矩陣）

建立 `verification-matrix.csv`：

```csv
claim_id,claim_text,internal_evidence,external_professional,external_open,status
C1,platform boundary is enforced,YES,YES,YES,PASS
C2,supply chain gate is mandatory,YES,YES,YES,PASS
C3,legacy path still referenced,YES,NO,YES,REVIEW
```

## 決策點 2：資訊是否收斂/足以決策

```bash
set -euo pipefail
OUT="${OUT:-.tmp/refactor-retrieval}"

pass_count=$(grep -c ",PASS$" "$OUT/verification-matrix.csv" || true)
review_count=$(grep -c ",REVIEW$" "$OUT/verification-matrix.csv" || true)

if [ "$pass_count" -ge 2 ] && [ "$review_count" -eq 0 ]; then
  echo "DECISION_2=YES"
  echo "ACTION=emit_actions"
else
  echo "DECISION_2=NO"
  echo "ACTION=generate_next_focused_questions"
fi
```

`DECISION_2=NO` 時輸出下一輪問題：

```bash
cat > .tmp/refactor-retrieval/next-questions.txt <<'EOF'
Q1: 哪些 platforms/* 仍引用 legacy 目錄？
Q2: 哪些部署流程未經 supply-chain-gate 覆蓋？
Q3: 哪些文件的平台映射與實際目錄不一致？
EOF
```

## 產出洞見與行動方案（直接可執行）

```bash
set -euo pipefail
OUT="${OUT:-.tmp/refactor-retrieval}"

cat > "$OUT/action-plan.md" <<'EOF'
# Action Plan

1. 修正 `platforms/README.md` 的平台映射與命名索引。
2. 對齊 `ARCHITECTURE.md` 的 platforms 結構引用。
3. 移除已淘汰 legacy 路徑的對外入口連結。
4. 以最小差異提交（僅文檔與映射）。
EOF
```

## 最小執行入口

```bash
TARGET=platforms OUT=.tmp/refactor-retrieval bash -c '
set -euo pipefail
mkdir -p "$OUT"
find "$TARGET" -maxdepth 4 -type f | sort > "$OUT/facts.files.txt"
grep -RInE "TODO|FIXME|legacy|TBD|secret|token|kyverno|cosign" "$TARGET" > "$OUT/signals.txt" || true
echo "READY"
'
```

## 一鍵執行（建議）

```bash
./scripts/platforms_refactor_retrieval.sh
```

```bash
make platforms-refactor-retrieval
```
