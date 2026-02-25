//# @GL-governed
//# @GL-layer: GL90-99
//# @GL-semantic: archive-tools
//# @GL-audit-trail: ../../engine/governance/GL_SEMANTIC_ANCHOR.json
//#
//# GL Unified Charter Activated
// @GL-status: active

/**
 * ==============================================================================
 * Contract Checker - 契約驗證工具
 * ==============================================================================
 * 用途: 驗證外部 API 契約是否符合 SLA 要求
 * 語言: 繁體中文註解
 * ==============================================================================
 */

const fs = require('fs');
const path = require('path');

class ContractChecker {
  constructor(contractPath) {
    this.contractPath = contractPath;
    this.contract = null;
    this.violations = [];
    this.warnings = [];
  }

  /**
   * 載入契約文件
   */
  loadContract() {
    try {
      const content = fs.readFileSync(this.contractPath, 'utf8');
      this.contract = JSON.parse(content);
      console.log(`✅ 載入契約: ${this.contract.contractId}`);
      return true;
    } catch (error) {
      console.error(`❌ 載入契約失敗: ${error.message}`);
      return false;
    }
  }

  /**
   * 驗證契約結構
   */
  validateStructure() {
    console.log('\n📋 驗證契約結構...');
    
    const required = ['contractId', 'contractVersion', 'endpoints'];
    const missing = required.filter(field => !this.contract[field]);
    
    if (missing.length > 0) {
      this.violations.push({
        type: 'structure',
        severity: 'critical',
        message: `缺少必要欄位: ${missing.join(', ')}`
      });
      return false;
    }
    
    console.log('✅ 契約結構驗證通過');
    return true;
  }

  /**
   * 驗證 SLA 定義
   */
  validateSLA() {
    console.log('\n📊 驗證 SLA 定義...');
    
    if (!this.contract.endpoints || this.contract.endpoints.length === 0) {
      this.violations.push({
        type: 'sla',
        severity: 'critical',
        message: '沒有定義任何端點'
      });
      return false;
    }
    
    this.contract.endpoints.forEach((endpoint, index) => {
      if (!endpoint.sla) {
        this.violations.push({
          type: 'sla',
          severity: 'high',
          endpoint: endpoint.name,
          message: `端點 ${endpoint.name} 缺少 SLA 定義`
        });
        return;
      }
      
      // 驗證可用性
      if (!endpoint.sla.availability) {
        this.warnings.push({
          type: 'sla',
          severity: 'medium',
          endpoint: endpoint.name,
          message: '缺少可用性 SLA'
        });
      }
      
      // 驗證延遲
      if (!endpoint.sla.latency) {
        this.warnings.push({
          type: 'sla',
          severity: 'medium',
          endpoint: endpoint.name,
          message: '缺少延遲 SLA'
        });
      }
      
      // 驗證錯誤率
      if (!endpoint.sla.error_rate) {
        this.warnings.push({
          type: 'sla',
          severity: 'medium',
          endpoint: endpoint.name,
          message: '缺少錯誤率 SLA'
        });
      }
    });
    
    if (this.violations.filter(v => v.type === 'sla').length === 0) {
      console.log('✅ SLA 定義驗證通過');
      return true;
    }
    
    return false;
  }

  /**
   * 驗證 API Schema
   */
  validateSchema() {
    console.log('\n🔍 驗證 API Schema...');
    
    this.contract.endpoints.forEach(endpoint => {
      if (!endpoint.contract) {
        this.warnings.push({
          type: 'schema',
          severity: 'high',
          endpoint: endpoint.name,
          message: '缺少 API 契約定義'
        });
        return;
      }
      
      // 驗證請求 schema
      if (!endpoint.contract.request) {
        this.warnings.push({
          type: 'schema',
          severity: 'medium',
          endpoint: endpoint.name,
          message: '缺少請求 schema'
        });
      }
      
      // 驗證回應 schema
      if (!endpoint.contract.response) {
        this.warnings.push({
          type: 'schema',
          severity: 'medium',
          endpoint: endpoint.name,
          message: '缺少回應 schema'
        });
      } else {
        // 驗證成功回應
        if (!endpoint.contract.response.success) {
          this.warnings.push({
            type: 'schema',
            severity: 'medium',
            endpoint: endpoint.name,
            message: '缺少成功回應定義'
          });
        }
        
        // 驗證錯誤回應
        if (!endpoint.contract.response.errors) {
          this.warnings.push({
            type: 'schema',
            severity: 'low',
            endpoint: endpoint.name,
            message: '缺少錯誤回應定義'
          });
        }
      }
    });
    
    console.log('✅ Schema 驗證完成');
    return true;
  }

  /**
   * 生成報告
   */
  generateReport() {
    const report = {
      timestamp: new Date().toISOString(),
      contractId: this.contract.contractId,
      contractVersion: this.contract.contractVersion,
      validation_result: {
        passed: this.violations.length === 0,
        total_violations: this.violations.length,
        total_warnings: this.warnings.length
      },
      violations: this.violations,
      warnings: this.warnings,
      summary: {
        critical: this.violations.filter(v => v.severity === 'critical').length,
        high: this.violations.filter(v => v.severity === 'high').length,
        medium: [...this.violations, ...this.warnings].filter(v => v.severity === 'medium').length,
        low: this.warnings.filter(v => v.severity === 'low').length
      }
    };
    
    return report;
  }

  /**
   * 執行完整驗證
   */
  async run() {
    console.log('🚀 開始契約驗證...\n');
    console.log(`契約文件: ${this.contractPath}`);
    
    if (!this.loadContract()) {
      return false;
    }
    
    this.validateStructure();
    this.validateSLA();
    this.validateSchema();
    
    const report = this.generateReport();
    
    // 輸出報告
    console.log('\n' + '='.repeat(80));
    console.log('📊 驗證報告');
    console.log('='.repeat(80));
    console.log(`契約 ID: ${report.contractId}`);
    console.log(`版本: ${report.contractVersion}`);
    console.log(`狀態: ${report.validation_result.passed ? '✅ 通過' : '❌ 失敗'}`);
    console.log(`\n摘要:`);
    console.log(`  ❗ 嚴重: ${report.summary.critical}`);
    console.log(`  ⚠️  高: ${report.summary.high}`);
    console.log(`  ⚡ 中: ${report.summary.medium}`);
    console.log(`  💡 低: ${report.summary.low}`);
    
    if (report.violations.length > 0) {
      console.log(`\n違規項目:`);
      report.violations.forEach(v => {
        console.log(`  - [${v.severity}] ${v.message}`);
      });
    }
    
    if (report.warnings.length > 0) {
      console.log(`\n警告項目:`);
      report.warnings.forEach(w => {
        console.log(`  - [${w.severity}] ${w.message}`);
      });
    }
    
    // 保存報告
    const reportPath = path.join(process.cwd(), 'reports', 'contract-validation.json');
    const reportDir = path.dirname(reportPath);
    
    if (!fs.existsSync(reportDir)) {
      fs.mkdirSync(reportDir, { recursive: true });
    }
    
    fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
    console.log(`\n📄 報告已保存: ${reportPath}`);
    
    return report.validation_result.passed;
  }
}

// CLI 介面
if (require.main === module) {
  const args = process.argv.slice(2);
  
  if (args.length === 0) {
    console.log('用法: node contract-checker.js <contract-file>');
    console.log('範例: node contract-checker.js contracts/external-api.json');
    process.exit(1);
  }
  
  const contractPath = args[0];
  
  if (!fs.existsSync(contractPath)) {
    console.error(`❌ 契約文件不存在: ${contractPath}`);
    process.exit(1);
  }
  
  const checker = new ContractChecker(contractPath);
  
  checker.run().then(passed => {
    process.exit(passed ? 0 : 1);
  }).catch(error => {
    console.error(`❌ 驗證過程發生錯誤: ${error.message}`);
    process.exit(1);
  });
}

module.exports = ContractChecker;

// ==============================================================================
// 使用範例
// ==============================================================================
// const ContractChecker = require('./contract-checker');
//
// const checker = new ContractChecker('contracts/external-api.json');
// const passed = await checker.run();
//
// if (!passed) {
//   console.log('契約驗證失敗');
//   process.exit(1);
// }
// ==============================================================================
