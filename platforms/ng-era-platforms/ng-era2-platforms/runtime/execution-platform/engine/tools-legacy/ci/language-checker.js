//# @GL-governed
//# @GL-layer: GL90-99
//# @GL-semantic: archive-tools
//# @GL-audit-trail: ../../engine/governance/GL_SEMANTIC_ANCHOR.json
//#
//# GL Unified Charter Activated
// @GL-status: active

/**
 * ==============================================================================
 * Language Checker - 語言合規性檢查工具
 * ==============================================================================
 * 用途: 檢查文件、註解、提交訊息是否符合語言治理策略
 * 語言: 繁體中文註解
 * ==============================================================================
 */

const fs = require('fs');
const path = require('path');

class LanguageChecker {
  constructor(options = {}) {
    this.config = options.config || 'governance/language-policy.yml';
    this.stagedOnly = options.stagedOnly || false;
    this.violations = [];
    this.warnings = [];
  }

  /**
   * 檢查字符串中中文字符的比例
   */
  getChineseRatio(text) {
    const chineseChars = text.match(/[\u4e00-\u9fa5]/g) || [];
    const totalChars = text.replace(/\s/g, '').length;
    return totalChars > 0 ? chineseChars.length / totalChars : 0;
  }

  /**
   * 檢查文件是否包含足夠的中文內容
   */
  checkFile(filePath) {
    if (!fs.existsSync(filePath)) {
      return;
    }

    const content = fs.readFileSync(filePath, 'utf8');
    const ratio = this.getChineseRatio(content);

    // Markdown 文件至少應包含 30% 中文
    if (filePath.endsWith('.md')) {
      if (ratio < 0.3) {
        this.violations.push({
          file: filePath,
          type: 'documentation',
          severity: ratio < 0.1 ? 'critical' : 'high',
          message: `文檔中文比例不足: ${(ratio * 100).toFixed(1)}% (建議 ≥30%)`,
          ratio: ratio
        });
      } else {
        console.log(`✅ ${filePath}: 中文比例 ${(ratio * 100).toFixed(1)}%`);
      }
    }
  }

  /**
   * 檢查提交訊息
   */
  checkCommitMessage(message) {
    // 檢查是否包含中文
    const ratio = this.getChineseRatio(message);
    
    if (ratio < 0.3) {
      this.violations.push({
        type: 'commit_message',
        severity: 'critical',
        message: `提交訊息缺少中文描述: ${message.substring(0, 50)}...`,
        ratio: ratio
      });
      return false;
    }

    // 檢查是否符合 conventional commits 格式
    const pattern = /^(feat|fix|docs|style|refactor|test|chore)(\([^)]+\))?: .+/;
    if (!pattern.test(message)) {
      this.warnings.push({
        type: 'commit_message',
        severity: 'medium',
        message: '提交訊息不符合 Conventional Commits 格式'
      });
    }

    return true;
  }

  /**
   * 掃描目錄
   */
  async scanDirectory(dir = '.') {
    const files = [];
    
    const scan = (directory) => {
      const items = fs.readdirSync(directory);
      
      for (const item of items) {
        const fullPath = path.join(directory, item);
        const stats = fs.statSync(fullPath);
        
        // 跳過排除的目錄
        if (stats.isDirectory()) {
          if (item === 'node_modules' || item === '.git' || item === 'vendor') {
            continue;
          }
          scan(fullPath);
        } else if (stats.isFile()) {
          // 只檢查 Markdown 文件
          if (item.endsWith('.md')) {
            files.push(fullPath);
          }
        }
      }
    };
    
    scan(dir);
    return files;
  }

  /**
   * 執行檢查
   */
  async run() {
    console.log('🔍 開始語言合規性檢查...\n');
    
    // 掃描文件
    const files = await this.scanDirectory('.');
    console.log(`📁 找到 ${files.length} 個 Markdown 文件\n`);
    
    for (const file of files) {
      this.checkFile(file);
    }
    
    // 生成報告
    const report = {
      timestamp: new Date().toISOString(),
      summary: {
        total_files_checked: files.length,
        files_compliant: files.length - this.violations.filter(v => v.file).length,
        files_with_violations: new Set(this.violations.filter(v => v.file).map(v => v.file)).size,
        compliance_rate: ((files.length - this.violations.filter(v => v.file).length) / files.length * 100).toFixed(1),
        violations_by_severity: {
          critical: this.violations.filter(v => v.severity === 'critical').length,
          high: this.violations.filter(v => v.severity === 'high').length,
          medium: [...this.violations, ...this.warnings].filter(v => v.severity === 'medium').length,
          low: this.warnings.filter(v => v.severity === 'low').length
        }
      },
      violations: this.violations,
      warnings: this.warnings
    };
    
    // 輸出結果
    console.log('\n' + '='.repeat(80));
    console.log('📊 語言合規性檢查報告');
    console.log('='.repeat(80));
    console.log(`檢查文件數: ${report.summary.total_files_checked}`);
    console.log(`合規文件數: ${report.summary.files_compliant}`);
    console.log(`違規文件數: ${report.summary.files_with_violations}`);
    console.log(`合規率: ${report.summary.compliance_rate}%`);
    console.log(`\n違規分佈:`);
    console.log(`  ❗ 嚴重: ${report.summary.violations_by_severity.critical}`);
    console.log(`  ⚠️  高: ${report.summary.violations_by_severity.high}`);
    console.log(`  ⚡ 中: ${report.summary.violations_by_severity.medium}`);
    console.log(`  💡 低: ${report.summary.violations_by_severity.low}`);
    
    if (this.violations.length > 0) {
      console.log(`\n違規詳情:`);
      this.violations.forEach(v => {
        console.log(`  - [${v.severity}] ${v.message}`);
        if (v.file) {
          console.log(`    文件: ${v.file}`);
        }
      });
    }
    
    // 保存報告
    const reportDir = 'reports';
    if (!fs.existsSync(reportDir)) {
      fs.mkdirSync(reportDir, { recursive: true });
    }
    
    const reportPath = path.join(reportDir, 'language-compliance.json');
    fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
    console.log(`\n📄 報告已保存: ${reportPath}`);
    
    // 返回是否有嚴重違規
    const hasCritical = this.violations.filter(v => v.severity === 'critical').length > 0;
    
    if (hasCritical) {
      console.log('\n❌ 發現嚴重語言違規');
      return false;
    } else if (this.violations.length > 0) {
      console.log('\n⚠️  發現語言違規警告');
      return true;
    } else {
      console.log('\n✅ 所有文件符合語言治理策略');
      return true;
    }
  }
}

// CLI 介面
if (require.main === module) {
  const args = process.argv.slice(2);
  const options = {
    stagedOnly: args.includes('--staged-only'),
    config: args.find(arg => arg.startsWith('--config='))?.split('=')[1]
  };
  
  const checker = new LanguageChecker(options);
  
  checker.run().then(passed => {
    process.exit(passed ? 0 : 1);
  }).catch(error => {
    console.error(`❌ 檢查過程發生錯誤: ${error.message}`);
    process.exit(1);
  });
}

module.exports = LanguageChecker;

// ==============================================================================
// 使用範例
// ==============================================================================
// const LanguageChecker = require('./language-checker');
//
// const checker = new LanguageChecker();
// const passed = await checker.run();
// ==============================================================================
