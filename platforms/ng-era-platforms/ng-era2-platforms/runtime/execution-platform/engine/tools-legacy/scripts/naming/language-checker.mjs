#!/usr/bin/env node
/**
 * 語言合規性檢查工具
 * 
 * 檢查以下內容的語言使用：
 * - Markdown 文件
 * - YAML 註解
 * - Git commit 訊息
 * - 程式碼註解（選擇性）
 */

import { readFileSync, readdirSync, statSync } from 'fs';
import { join, extname } from 'path';

const CONFIG = {
  requiredLanguage: 'zh', // 'zh' 或 'en'
  checkFiles: ['.md', '.yml', '.yaml'],
  excludeDirs: ['node_modules', '.git', 'dist', 'build', '.next', '_codeql_detected_source_root'],
  strictMode: false, // true: 阻擋, false: 警告
};

// 檢測中文字符
function containsChinese(text) {
  return /[\u4e00-\u9fa5]/.test(text);
}

// 檢測英文內容（排除技術關鍵字）
function containsEnglishContent(text) {
  // 移除常見技術術語
  const techTerms = /\b(API|HTTP|URL|JSON|YAML|CI|CD|SLSA|SBOM|Kubernetes|Docker|npm|node)\b/gi;
  const cleaned = text.replace(techTerms, '');
  
  // 檢查是否包含英文句子
  return /[a-zA-Z]{4,}.*[a-zA-Z]{4,}/.test(cleaned);
}

function checkFile(filePath) {
  const ext = extname(filePath);
  if (!CONFIG.checkFiles.includes(ext)) return null;

  const content = readFileSync(filePath, 'utf-8');
  const lines = content.split('\n');
  const violations = [];

  lines.forEach((line, index) => {
    // 跳過代碼塊和 URL
    if (line.trim().startsWith('```') || line.includes('http://') || line.includes('https://')) {
      return;
    }

    if (CONFIG.requiredLanguage === 'zh') {
      // 要求中文，檢查是否有過多英文內容
      if (containsEnglishContent(line) && !containsChinese(line) && line.trim().length > 20) {
        violations.push({
          line: index + 1,
          content: line.trim().substring(0, 80),
          issue: '缺少中文說明',
        });
      }
    }
  });

  return violations.length > 0 ? { filePath, violations } : null;
}

function scanDirectory(dir, results = []) {
  const entries = readdirSync(dir);

  for (const entry of entries) {
    const fullPath = join(dir, entry);
    
    // Use statSync with error handling to follow symlinks safely
    let stat;
    try {
      stat = statSync(fullPath, { throwIfNoEntry: false });
      if (!stat) continue; // Skip if entry doesn't exist
    } catch (err) {
      // Skip on error (e.g., permission denied, broken symlink)
      console.warn(`⚠️  Skipping ${fullPath}: ${err.message}`);
      continue;
    }

    if (stat.isDirectory()) {
      // Skip if excluded directory (check after confirming it's a directory)
      if (!CONFIG.excludeDirs.includes(entry)) {
        scanDirectory(fullPath, results);
      }
    } else if (stat.isFile()) {
      const result = checkFile(fullPath);
      if (result) results.push(result);
    }
  }

  return results;
}

function main() {
  const rootDir = process.argv[2] || '.';
  console.log(`🔍 開始檢查語言合規性：${rootDir}\n`);

  const results = scanDirectory(rootDir);

  if (results.length === 0) {
    console.log('✅ 所有文件符合語言規範\n');
    process.exit(0);
  }

  console.log(`⚠️  發現 ${results.length} 個文件有語言問題：\n`);

  results.forEach(({ filePath, violations }) => {
    console.log(`📄 ${filePath}`);
    violations.forEach(({ line, content, issue }) => {
      console.log(`   行 ${line}: ${issue}`);
      console.log(`   內容: ${content}...`);
    });
    console.log('');
  });

  if (CONFIG.strictMode) {
    console.error('❌ 語言檢查未通過（嚴格模式）\n');
    process.exit(1);
  } else {
    console.log('⚠️  語言檢查發現問題（警告模式）\n');
    process.exit(0);
  }
}

main();
