# ⚡ Quick Format Reference - TL;DR

## 🚨 One Command to Rule Them All

```bash
# Fix ALL format issues automatically
bash .claude/hooks/auto-fix-format.sh

# Verify everything passes
bash .claude/hooks/pre-commit-validator.sh
```

---

## 📋 Validation Checklist

Before every commit, run:

```bash
bash .claude/hooks/pre-commit-validator.sh
```

If it **FAILS**, run:

```bash
bash .claude/hooks/auto-fix-format.sh
bash .claude/hooks/pre-commit-validator.sh
```

---

## ✅ What Gets Checked

| File Type | Checks |
|-----------|--------|
| **JSON** | Syntax, indentation (2 spaces), UTF-8, LF line endings, no trailing spaces |
| **YAML** | Syntax, indentation (2 spaces, NO tabs), UTF-8, LF line endings, multi-doc support |
| **TOML** | Syntax, indentation (2 spaces), UTF-8, LF line endings |
| **SQL** | Syntax, UTF-8, LF line endings |

---

## 🆘 Common Issues & Fixes

### Issue: Validator Shows Errors

```bash
# Auto-fix everything
bash .claude/hooks/auto-fix-format.sh

# Re-validate
bash .claude/hooks/pre-commit-validator.sh
```

### Issue: JSON Won't Format

```bash
# Let jq format it
jq . package.json > package.json.tmp
mv package.json.tmp package.json
```

### Issue: YAML Has Tabs

```bash
# Convert tabs to spaces (auto-fixer does this)
sed -i 's/\t/  /g' file.yaml
```

### Issue: File Has CRLF Line Endings

```bash
# Convert to LF (auto-fixer does this)
dos2unix filename
```

---

## 🔄 Git Workflow

```bash
# 1. Make your changes
vim package.json

# 2. Validate format
bash .claude/hooks/pre-commit-validator.sh

# 3. If validation fails:
bash .claude/hooks/auto-fix-format.sh

# 4. Review changes
git diff

# 5. Commit
git commit -m "message"

# 6. Push
git push origin claude/your-branch
```

---

## 📊 GitHub Actions

Automatic validation runs on:
- Every push to `main`, `develop`, `claude/*`
- Every PR to `main`, `develop`
- Any config file changes

**Status**: Check GitHub Actions tab for results

---

## 🎯 Key Files

| File | Purpose |
|------|---------|
| `.claude/hooks/pre-commit-validator.sh` | Validation (blocking) |
| `.claude/hooks/auto-fix-format.sh` | Auto-fix tool |
| `.claude/config/format-validation.json` | Configuration |
| `.github/workflows/format-validation.yml` | CI/CD automation |
| `FORMAT_VALIDATION_PIPELINE.md` | Full documentation |

---

## 💡 Pro Tips

✅ **Do This:**
```bash
# Before every commit
bash .claude/hooks/pre-commit-validator.sh
```

✅ **Use Auto-Fix:**
```bash
# Fixes 99% of issues automatically
bash .claude/hooks/auto-fix-format.sh
```

✅ **Trust the Tools:**
- Validator catches errors before commit
- Auto-fixer repairs most issues
- CI/CD validates on push

---

❌ **Don't Do This:**
```bash
# ❌ Don't manually edit JSON/YAML without validation
# ❌ Don't commit without running validator
# ❌ Don't skip validation on CI/CD
```

---

## 🚀 TL;DR For Busy Developers

```bash
# Before committing:
bash .claude/hooks/pre-commit-validator.sh

# If it fails:
bash .claude/hooks/auto-fix-format.sh

# Then:
bash .claude/hooks/pre-commit-validator.sh

# Done! 🎉
```

That's it. No format errors will ever reach production.

---

📖 **Need more details?** Read `FORMAT_VALIDATION_PIPELINE.md`
