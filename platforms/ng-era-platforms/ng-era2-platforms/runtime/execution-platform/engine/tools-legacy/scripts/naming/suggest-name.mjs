#!/usr/bin/env node
/**
 * 命名建議工具
 * 根據規範提供命名建議和唯一性檢查
 */

import { readFileSync, readdirSync, statSync } from 'fs';
import { join } from 'path';


// 命名規範
const NAMING_PATTERNS = {
  namespace: /^(team|tenant|feature)-[a-z0-9-]+$/,
  deployment: /^[a-z0-9]([-a-z0-9]*[a-z0-9])?$/,
  service: /^[a-z0-9]([-a-z0-9]*[a-z0-9])?$/,
};



class NamingSuggester {
  constructor() {
    this.existingNames = new Set();
    this.violations = [];
  }

  /**
   * 掃描現有名稱
   */
  scanExistingNames(dir = 'deploy') {
    console.log(`🔍 掃描現有名稱...`);
    this._scanDirectory(dir);
    console.log(`✅ 找到 ${this.existingNames.size} 個現有資源名稱`);
  }

  _scanDirectory(dir) {
    try {
      const files = readdirSync(dir);
      for (const file of files) {
        const fullPath = join(dir, file);
        try {
          const stats = statSync(fullPath);
          if (stats.isDirectory()) {
            this._scanDirectory(fullPath);
          } else if (file.endsWith('.yaml') || file.endsWith('.yml')) {
            this._extractNames(fullPath);
          }
        } catch (err) {
          // 跳過無法存取的檔案
        }
      }
    } catch (err) {
      // 目錄不存在時忽略
    }
  }

  _extractNames(filePath) {
    try {
      const content = readFileSync(filePath, 'utf8');
      const nameMatch = content.match(/^\s*name:\s*(.+)$/m);
      if (nameMatch) {
        this.existingNames.add(nameMatch[1].trim());
      }
    } catch (err) {
      // 無法讀取時靜默忽略
    }
  }

  /**
   * 驗證名稱
   */
  validateName(name, type = 'deployment') {
    const pattern = NAMING_PATTERNS[type];
    if (!pattern) {
      return { valid: false, reason: `未知的資源類型：${type}` };
    }

    if (!pattern.test(name)) {
      return {
        valid: false,
        reason: `名稱不符合 ${type} 的命名規範：${pattern}`,
      };
    }

    if (name.length > 63) {
      return {
        valid: false,
        reason: `名稱長度超過 63 字元限制（當前：${name.length}）`,
      };
    }

    return { valid: true };
  }

  /**
   * 檢查唯一性
   */
  checkUniqueness(name) {
    if (this.existingNames.has(name)) {
      return {
        unique: false,
        reason: `名稱 '${name}' 已存在`,
      };
    }
    return { unique: true };
  }

  /**
   * 生成建議名稱
   */
  suggestName(baseName, type = 'deployment') {
    // 清理基礎名稱
    let cleaned = baseName
      .toLowerCase()
      .replace(/[^a-z0-9-]/g, '-')
      .replace(/^-+|-+$/g, '')
      .replace(/-+/g, '-');

    // 確保符合規範
    if (type === 'namespace' && !cleaned.match(/^(team|tenant|feature)-/)) {
      cleaned = `team-${cleaned}`;
    }

    // 檢查長度
    if (cleaned.length > 63) {
      cleaned = cleaned.substring(0, 63);
    }

    // 檢查唯一性
    let suggested = cleaned;
    let counter = 1;
    while (this.existingNames.has(suggested)) {
      const suffix = `-${counter}`;
      const maxLength = 63 - suffix.length;
      suggested = cleaned.substring(0, maxLength) + suffix;
      counter++;
    }

    return {
      original: baseName,
      suggested,
      type,
      validation: this.validateName(suggested, type),
      uniqueness: this.checkUniqueness(suggested),
    };
  }

  /**
   * 生成標籤建議
   */
  suggestLabels(resourceName, team = 'platform', environment = 'production') {
    return {
      'namespace.io/team': team,
      'namespace.io/environment': environment,
      'namespace.io/lifecycle': 'active',
      'namespace.io/managed-by': 'gitops',
      app: resourceName,
    };
  }

  /**
   * 生成完整資源建議
   */
  suggestResource(input) {
    const {
      name,
      type = 'deployment',
      team = 'platform',
      environment = 'production',
    } = input;

    const nameResult = this.suggestName(name, type);
    const labels = this.suggestLabels(nameResult.suggested, team, environment);

    return {
      name: nameResult,
      labels,
      valid: nameResult.validation.valid && nameResult.uniqueness.unique,
    };
  }

  /**
   * 輸出報告
   */
  printReport(result) {
    console.log('\n📋 命名建議報告');
    console.log('─'.repeat(50));
    console.log(`原始名稱: ${result.name.original}`);
    console.log(`建議名稱: ${result.name.suggested}`);
    console.log(`資源類型: ${result.name.type}`);
    console.log(`驗證狀態: ${result.name.validation.valid ? '✅ 通過' : '❌ 失敗'}`);
    if (!result.name.validation.valid) {
      console.log(`失敗原因: ${result.name.validation.reason}`);
    }
    console.log(`唯一性: ${result.name.uniqueness.unique ? '✅ 唯一' : '⚠️ 重複'}`);
    if (!result.name.uniqueness.unique) {
      console.log(`重複說明: ${result.name.uniqueness.reason}`);
    }
    
    console.log('\n🏷️ 建議標籤:');
    for (const [key, value] of Object.entries(result.labels)) {
      console.log(`  ${key}: ${value}`);
    }
    
    console.log('\n' + '─'.repeat(50));
    console.log(result.valid ? '✅ 所有檢查通過' : '❌ 存在問題需要修正');
  }
}

// CLI 介面
// 檢查是否為直接執行（跨平台相容）
import { fileURLToPath } from 'url';
const __filename = fileURLToPath(import.meta.url);
if (process.argv[1] === __filename) {
  const args = process.argv.slice(2);
  
  if (args.length === 0) {
    console.log(`
使用方式: suggest-name.mjs <name> [type] [team] [environment]

參數:
  name        - 資源名稱（必填）
  type        - 資源類型（可選，預設：deployment）
                可選值：namespace, deployment, service
  team        - 團隊名稱（可選，預設：platform）
  environment - 環境名稱（可選，預設：production）

範例:
  suggest-name.mjs my-service
  suggest-name.mjs "My Service" deployment contracts-team production
  suggest-name.mjs team-platform namespace platform production
    `);
    process.exit(1);
  }

  const [name, type = 'deployment', team = 'platform', environment = 'production'] = args;

  const suggester = new NamingSuggester();
  suggester.scanExistingNames();
  
  const result = suggester.suggestResource({
    name,
    type,
    team,
    environment,
  });

  suggester.printReport(result);

  process.exit(result.valid ? 0 : 1);
}

export default NamingSuggester;
