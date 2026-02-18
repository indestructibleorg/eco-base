# 🔒 Format Validation Pipeline - Complete Guide

## Overview

This is an **absolute zero-tolerance format validation system** that ensures your entire codebase never has format errors. The pipeline is multi-layered:

1. **Pre-Commit Validation** - Local validation before committing
2. **Auto-Fix Tool** - Automatic repair of format issues
3. **CI/CD Integration** - GitHub Actions validation on push/PR
4. **Real-time Monitoring** - Continuous format compliance checking

---

## 📋 What Gets Validated

### File Formats
- ✅ **JSON** - `.json` files (package.json, tsconfig.json, etc.)
- ✅ **YAML** - `.yaml` and `.yml` files (CI/CD, Kubernetes, configs)
- ✅ **TOML** - `.toml` files (wrangler.toml, config.toml, etc.)
- ✅ **SQL** - `.sql` files (database migrations)

### Validation Checks
- ✅ **Syntax Validation** - Valid JSON/YAML/TOML/SQL syntax
- ✅ **Encoding** - Must be UTF-8 (no Latin-1 or others)
- ✅ **Line Endings** - Must be LF (Unix), not CRLF (Windows)
- ✅ **Trailing Whitespace** - No spaces/tabs at line ends
- ✅ **BOM (Byte Order Mark)** - No BOM in files
- ✅ **Final Newline** - Every file must end with newline
- ✅ **Indentation** - Consistent 2-space indentation
- ✅ **YAML Tabs** - No tabs allowed in YAML (spaces only)
- ✅ **File Structure** - No empty files or malformed data

---

## 🚀 Quick Start

### 1. First-Time Setup

```bash
# Make scripts executable
chmod +x .claude/hooks/*.sh

# Run auto-fixer to clean up existing issues
bash .claude/hooks/auto-fix-format.sh

# Verify validation passes
bash .claude/hooks/pre-commit-validator.sh
```

### 2. Before Every Commit

```bash
# Validate format (blocking if errors)
bash .claude/hooks/pre-commit-validator.sh

# If validation fails, auto-fix and rerun
bash .claude/hooks/auto-fix-format.sh
bash .claude/hooks/pre-commit-validator.sh
```

### 3. CI/CD Pipeline

GitHub Actions automatically validates on:
- Push to `main`, `develop`, `claude/*` branches
- Pull requests to `main`, `develop`
- Changes to config files

No additional setup needed - just commit and push!

---

## 📦 Tools & Scripts

### Pre-Commit Validator (Blocking)
**File:** `.claude/hooks/pre-commit-validator.sh`

Validates all format files and **blocks commit if errors found**.

```bash
bash .claude/hooks/pre-commit-validator.sh
```

**Output Example:**
```
╔════════════════════════════════════════════════════════════════╗
║     🔒 ABSOLUTE FORMAT VALIDATION PIPELINE (v1.0)             ║
║     No errors. No exceptions. Zero tolerance.                 ║
╚════════════════════════════════════════════════════════════════╝

→ Validating JSON files...
✓ JSON valid: package.json
✓ JSON valid: tsconfig.json
...
✓ YAML valid: .circleci/config.yml
...

✅ ALL VALIDATION PASSED
```

### Auto-Fix Tool
**File:** `.claude/hooks/auto-fix-format.sh`

Automatically fixes all format issues that can be corrected.

```bash
bash .claude/hooks/auto-fix-format.sh
```

**Fixes automatically:**
- JSON/YAML/TOML indentation
- Line endings (CRLF → LF)
- Trailing whitespace
- BOM markers
- File encoding
- Missing final newlines
- Prettier formatting

### Configuration File
**File:** `.claude/config/format-validation.json`

Master configuration for validation settings:
- Validation levels
- File format rules
- Auto-fix options
- Error reporting
- Hook settings

---

## 🔧 Integration with Git Hooks

### Option 1: Manual (Recommended for Claude Code)

Run before commit:
```bash
bash .claude/hooks/pre-commit-validator.sh && git commit ...
```

### Option 2: Automatic Git Hooks (Optional)

```bash
# Create git pre-commit hook
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
bash .claude/hooks/pre-commit-validator.sh
EOF

chmod +x .git/hooks/pre-commit
```

### Option 3: GitHub Actions (Already Set Up)

Automatic validation in CI/CD pipeline:
- File: `.github/workflows/format-validation.yml`
- Runs on every push and PR
- Blocks merge if validation fails

---

## 📊 Validation Report Example

```
╔════════════════════════════════════════════════════════════════╗
║                   VALIDATION REPORT                           ║
╚════════════════════════════════════════════════════════════════╝

Total Files Checked:  45
Passed:              45
Failed:              0
Warnings:            2

✅ ALL VALIDATION PASSED

Warnings:
⚠ Missing newline at end of config.yml
⚠ Trailing whitespace in package.json
```

---

## 🆘 Troubleshooting

### Issue: "Invalid JSON: Unexpected token"

**Solution:** Run auto-fixer
```bash
bash .claude/hooks/auto-fix-format.sh
```

### Issue: "YAML contains tabs"

**Solution:** Tabs are not allowed in YAML. Replace with spaces:
```bash
# Manual fix
sed -i 's/\t/  /g' file.yaml

# Or use auto-fixer
bash .claude/hooks/auto-fix-format.sh
```

### Issue: "File has DOS line endings (CRLF)"

**Solution:** Convert to Unix line endings:
```bash
# Manual fix
dos2unix filename

# Or use auto-fixer
bash .claude/hooks/auto-fix-format.sh
```

### Issue: "Invalid encoding (must be UTF-8)"

**Solution:** Convert file to UTF-8:
```bash
# Manual fix
iconv -f ISO-8859-1 -t UTF-8 input.json > output.json
mv output.json input.json

# Or use auto-fixer
bash .claude/hooks/auto-fix-format.sh
```

### Issue: Validator keeps failing on same file

**Solution:**
1. Run auto-fixer multiple times
2. Manually review the file
3. Delete and recreate if corrupted

---

## 📋 File-by-File Validation

### JSON Files
- **Standard:** JSON 5 compatible
- **Indentation:** 2 spaces
- **Extensions:** `.json`
- **Examples:** `package.json`, `tsconfig.json`, `components.json`

### YAML Files
- **Standard:** YAML 1.2
- **Indentation:** 2 spaces (NO TABS)
- **Extensions:** `.yaml`, `.yml`
- **Examples:** `.circleci/config.yml`, `.github/workflows/*.yml`

### TOML Files
- **Standard:** TOML 1.0.0
- **Indentation:** 2 spaces
- **Extensions:** `.toml`
- **Examples:** `supabase/config.toml`, `wrangler.toml`

### SQL Files
- **Standard:** PostgreSQL/Supabase compatible
- **Indentation:** 2 spaces
- **Extensions:** `.sql`
- **Examples:** `supabase/migrations/*.sql`

---

## 🔄 CI/CD Pipeline Details

### GitHub Actions Workflow
**File:** `.github/workflows/format-validation.yml`

Triggered on:
- Push to protected branches
- Pull requests
- Changes to config files

Validates:
1. JSON syntax and structure
2. YAML syntax (no tabs)
3. TOML syntax
4. File encoding (UTF-8)
5. Line endings (LF)
6. Trailing whitespace
7. Prettier formatting
8. TypeScript types

**Result:** Blocks PR/push if validation fails

---

## 🎯 Best Practices

### ✅ DO

```bash
# Run validator before committing
bash .claude/hooks/pre-commit-validator.sh

# Auto-fix when validation fails
bash .claude/hooks/auto-fix-format.sh

# Review changes after auto-fix
git diff

# Commit with confidence
git commit -m "..."
```

### ❌ DON'T

```bash
# Don't skip validation
# ❌ git commit -m "..." --no-verify

# Don't edit config files manually if unsure
# ❌ vim package.json (without knowing JSON syntax)

# Don't ignore warnings
# ⚠️  Fix trailing whitespace and BOM issues
```

---

## 📚 Additional Resources

### Configuration
- Main config: `.claude/config/format-validation.json`
- Prettier config: `.prettierrc`
- TypeScript: `tsconfig.json`

### Scripts Location
- Validator: `.claude/hooks/pre-commit-validator.sh`
- Auto-fixer: `.claude/hooks/auto-fix-format.sh`
- CI/CD: `.github/workflows/format-validation.yml`

### Support Files
- This guide: `FORMAT_VALIDATION_PIPELINE.md`

---

## 📞 Support & Issues

If validation fails unexpectedly:

1. **Check error messages:**
   ```bash
   bash .claude/hooks/pre-commit-validator.sh 2>&1 | tee validation-error.log
   ```

2. **Try auto-fix:**
   ```bash
   bash .claude/hooks/auto-fix-format.sh
   ```

3. **Manual validation:**
   ```bash
   # Check specific file
   jq empty package.json
   python3 -c "import yaml; yaml.safe_load(open('.circleci/config.yml'))"
   ```

4. **Review logs:**
   - Error logs: `/tmp/format-validation-errors-*.log`
   - Warning logs: `/tmp/format-validation-warnings-*.log`

---

## 🎉 Summary

Your repository now has **absolute format validation**:

| Level | Tool | When | Status |
|-------|------|------|--------|
| 🔴 Local | Pre-commit validator | Before commit | ⚠️ Blocking |
| 🟡 Auto-fix | Auto-fix tool | On demand | ✅ Available |
| 🟢 CI/CD | GitHub Actions | On push/PR | 🔄 Automated |

**Never worry about format errors again!** ✨

