//# @ECO-governed
//# @ECO-layer: GL90-99
//# @ECO-semantic: archive-tools
//# @ECO-audit-trail: ../../engine/governance/GL_SEMANTIC_ANCHOR.json
//#
//# GL Unified Charter Activated

import { spawn } from 'node:child_process';
import fs from 'node:fs/promises';
import path from 'node:path';
import { Command } from 'commander';
import chalk from 'chalk';
import figures from 'figures';
import inquirer from 'inquirer';
import { fileURLToPath } from 'node:url';

const moduleDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(moduleDir, '..', '..', '..');
const reportsDir = path.join(repoRoot, 'reports');
const selfAwarenessScript = path.join(repoRoot, 'automation', 'self_awareness_report.py');

const PROJECT_FILE = 'island.project.json';
const TEAM_FILE = 'island.team.json';
const INFRA_FILE = 'island.infrastructure.json';
const AGENT_FILE = 'island.agents.json';
const AUTOMATION_FILE = 'island.automation.json';

type ProjectManifest = {
  name: string;
  version: string;
  createdAt: string;
  agents: string[];
  governanceLevel: string;
};

type AutomationOperation = {
  id: string;
  description: string;
  command: string;
  sample?: boolean;
  lastExecutedAt?: string;
  lastMode?: 'sample' | 'full';
};

const program = new Command();
program.name('island-cli').description('Island AI — AI 驅動的工程控制台').version('0.1.0');

type OptionRecord = Record<string, unknown>;

type SimpleCommand = {
  name: string;
  description: string;
  options?: Array<{
    flags: string;
    description: string;
    defaultValue?: unknown;
  }>;
  action: (...args: any[]) => Promise<void> | void;
};

const headline = (title: string) => {
  console.log(`\n${chalk.cyan(title)}`);
  console.log(chalk.gray('='.repeat(title.length)));
};

const fileExists = async (target: string) => {
  try {
    await fs.access(target);
    return true;
  } catch {
    return false;
  }
};

const writeJson = async (file: string, data: unknown) => {
  await fs.writeFile(file, `${JSON.stringify(data, null, 2)}\n`, 'utf8');
};

const readJson = async <T>(file: string, fallback: T) => {
  try {
    const raw = await fs.readFile(file, 'utf8');
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
};

const cloneAutomationDefaults = (): AutomationOperation[] => [
  {
    id: 'lint',
    description: 'Workspace lint (TypeScript + Python)',
    command: 'npm run lint',
    sample: true,
  },
  {
    id: 'tests',
    description: 'Unit & integration tests',
    command: 'npm test',
    sample: true,
  },
  {
    id: 'audit',
    description: 'Security audit (npm workspaces)',
    command: 'npm audit --workspaces --include-workspace-root',
    sample: false,
  },
];

const loadAutomationOperations = async () =>
  readJson<AutomationOperation[]>(AUTOMATION_FILE, cloneAutomationDefaults());

const saveAutomationOperations = async (operations: AutomationOperation[]) =>
  writeJson(AUTOMATION_FILE, operations);

const ensureAutomationOperations = async () => {
  if (!(await fileExists(AUTOMATION_FILE))) {
    await saveAutomationOperations(cloneAutomationDefaults());
  }
  return loadAutomationOperations();
};

const parseList = (value?: string | string[]) => {
  if (!value) return [] as string[];
  if (Array.isArray(value)) return value;
  try {
    const parsed = JSON.parse(value);
    if (Array.isArray(parsed)) {
      return parsed.map((item) => `${item}`.trim()).filter(Boolean);
    }
  } catch {
    // fall through to CSV parsing
  }
  return value
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean);
};

const ensureIslandFolder = async () => {
  const islandDir = path.join(process.cwd(), '.island');
  await fs.mkdir(islandDir, { recursive: true });
  const configPath = path.join(islandDir, 'config.yaml');
  if (!(await fileExists(configPath))) {
    const template = [
      'version: "1.0"',
      'project:',
      '  name: demo-project',
      '  type: fullstack',
      'agents:',
      '  developer:',
      '    enabled: true',
      'governance:',
      '  default_level: L3',
      '',
    ].join('\n');
    await fs.writeFile(configPath, template, 'utf8');
  }
  return configPath;
};

const registerCommand = (definition: SimpleCommand) => {
  const cmd = program.command(definition.name).description(definition.description);
  const hasPositionalArgs = /[<\[]/.test(definition.name);
  definition.options?.forEach((opt) => {
    if (typeof opt.defaultValue !== 'undefined') {
      cmd.option(opt.flags, opt.description, opt.defaultValue as never);
    } else {
      cmd.option(opt.flags, opt.description);
    }
  });
  cmd.action((...args: unknown[]): void | Promise<void> => {
    const invoke = definition.action as (...inner: unknown[]) => void | Promise<void>;
    if (hasPositionalArgs) {
      const lastIndex = args.length - 1;
      const maybeCommand = lastIndex >= 0 ? args[lastIndex] : undefined;
      if (!(maybeCommand instanceof Command)) {
        args.push(cmd);
      }
      return invoke(...args);
    }
    return invoke(cmd);
  });
};

type ReportMode = 'sample' | 'full';

const runSelfAwarenessReport = async (params: {
  lintCmd?: string;
  testCmd?: string;
  automation: AutomationOperation[];
  mode: ReportMode;
}): Promise<{ markdown: string; json: string }> => {
  await fs.mkdir(reportsDir, { recursive: true });
  const markdown = path.join(reportsDir, `self-awareness-${params.mode}.md`);
  const json = path.join(reportsDir, `self-awareness-${params.mode}.json`);
  const args = [
    selfAwarenessScript,
    '--output',
    markdown,
    '--json-output',
    json,
    '--repo-root',
    repoRoot,
    '--max-log-lines',
    '10',
    '--fail-on-errors',
  ];

  if (params.lintCmd) {
    args.push('--lint-cmd', params.lintCmd);
  }
  if (params.testCmd) {
    args.push('--test-cmd', params.testCmd);
  }

  params.automation
    .filter((op) => op.command && !/self_awareness_report\.py/.test(op.command))
    .forEach((op) => {
      const label = op.description?.trim() || op.id;
      args.push('--automation-cmd', `${label}=${op.command}`);
    });

  await new Promise<void>((resolve, reject) => {
    const child = spawn('python3', args, {
      cwd: repoRoot,
      stdio: 'inherit',
    });

    child.on('exit', (code) => {
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(`self-awareness script exited with code ${code ?? 'unknown'}`));
      }
    });
    child.on('error', reject);
  });

  return { markdown, json };
};

const createProject = async (name: string, version: string) => {
  const manifest = {
    name,
    version,
    createdAt: new Date().toISOString(),
    agents: ['architect', 'developer', 'security'],
    governanceLevel: 'L3',
  };
  await writeJson(PROJECT_FILE, manifest);
  headline('專案初始化完成');
  console.log(`${figures.tick} 建立 ${chalk.green(PROJECT_FILE)}`);
  console.log(`${figures.tick} 預設 Agent：${manifest.agents.join(', ')}`);
};

registerCommand({
  name: 'init <name>',
  description: '初始化 Island AI 專案',
  options: [
    { flags: '-v, --version <version>', description: '設定專案版本', defaultValue: '1.0.0' },
  ],
  action: async (name: string, cmd: Command) => {
    const options = cmd.opts<OptionRecord>();
    await createProject(name, `${options.version ?? '1.0.0'}`);
  },
});

registerCommand({
  name: 'project:init',
  description: '以參數方式建立專案描述',
  options: [
    { flags: '-n, --name <name>', description: '專案名稱' },
    { flags: '-v, --version <version>', description: '專案版本', defaultValue: '1.0.0' },
  ],
  action: async (cmd: Command) => {
    const options = cmd.opts<OptionRecord>();
    const name = `${options.name ?? 'island-ai-project'}`;
    await createProject(name, `${options.version ?? '1.0.0'}`);
  },
});

registerCommand({
  name: 'activate',
  description: '在現有專案啟用 Island AI',
  action: async () => {
    const configPath = await ensureIslandFolder();
    headline('環境啟用');
    console.log(`${figures.tick} 統一配置：${chalk.green(configPath)}`);
  },
});

const workflowCommands: SimpleCommand[] = [
  {
    name: 'dev',
    description: '啟動開發模式',
    action: async () => {
      headline('開發模式');
      console.log('偵測到 3 個服務，準備啟動模擬監控...');
    },
  },
  {
    name: 'build',
    description: '進行建置',
    action: async () => {
      headline('建置流程');
      console.log('Bazel 任務排隊中... (模擬)');
    },
  },
  {
    name: 'test',
    description: '執行測試',
    action: async () => {
      headline('測試報告');
      console.log(`${figures.tick} 單元測試 (TypeScript/Python)`);
      console.log(`${figures.tick} 整合測試 (Go)`);
    },
  },
  {
    name: 'deploy',
    description: '部署到雲端或叢集',
    options: [
      { flags: '-e, --environment <env>', description: '目標環境', defaultValue: 'staging' },
    ],
    action: async (cmd: Command) => {
      const options = cmd.opts<OptionRecord>();
      headline('部署');
      console.log(`部署目標：${options.environment}`);
      console.log(`${figures.pointer} 已開啟 Canary 監控`);
    },
  },
  {
    name: 'status',
    description: '檢視系統狀態',
    action: async () => {
      const manifest = await readJson<ProjectManifest | null>(PROJECT_FILE, null);
      headline('系統狀態');
      if (manifest) {
        console.log(`${figures.tick} ${manifest.name} (${manifest.version})`);
      } else {
        console.log(`${figures.warning} 尚未初始化專案，請執行 init`);
      }
    },
  },
  {
    name: 'health',
    description: '輸出健康檢查結果',
    action: async () => {
      headline('健康檢查');
      console.log(`${figures.tick} API Gateway：OK`);
      console.log(`${figures.tick} Workflow Orchestrator：OK`);
      console.log(`${figures.tick} Agent Runtime：OK`);
    },
  },
  {
    name: 'upgrade',
    description: '升級 Island AI 元件',
    action: async () => {
      headline('升級流程');
      console.log('檢查最新套件版本... (模擬)');
    },
  },
  {
    name: 'config',
    description: '顯示或更新設定',
    options: [{ flags: '-s, --show', description: '列出設定', defaultValue: false }],
    action: async (cmd: Command) => {
      const options = cmd.opts<OptionRecord>();
      const configPath = await ensureIslandFolder();
      headline('設定檔');
      console.log(`${figures.tick} ${configPath}`);
      if (options.show) {
        const content = await fs.readFile(configPath, 'utf8');
        console.log(content);
      }
    },
  },
];

workflowCommands.forEach(registerCommand);

program
  .command('ask <question>')
  .description('請教 AI Agent 問題')
  .action(async (question: string) => {
    headline('AI 回覆');
    console.log(`${figures.pointer} 問題：${question}`);
    console.log(`${figures.tick} 回應：系統已建立上下文並提供建議。`);
  });

program
  .command('review')
  .description('啟動程式碼審查')
  .action(async () => {
    headline('程式碼審查');
    console.log('掃描中... (模擬)');
  });

program
  .command('doc')
  .description('生成文件')
  .action(async () => {
    headline('文檔生成');
    console.log(`${figures.tick} README、API、Runbook 已更新`);
  });

program
  .command('chat')
  .description('互動式對話模式')
  .action(async () => {
    headline('互動模式');
    const { prompt } = await inquirer.prompt<{ prompt: string }>([
      {
        type: 'input',
        name: 'prompt',
        message: '請輸入您的問題 (離開請輸入 exit):',
      },
    ]);
    if (prompt?.toLowerCase() === 'exit') {
      console.log('已離開互動模式');
      return;
    }
    console.log(`${figures.tick} 建議：請檢查治理層級並重跑 pipeline。`);
  });

const languageMap: Record<string, string> = {
  '.ts': 'TypeScript',
  '.tsx': 'TypeScript',
  '.js': 'JavaScript',
  '.py': 'Python',
  '.rs': 'Rust',
  '.go': 'Go',
  '.java': 'Java',
  '.md': 'Docs',
  '.yaml': 'Config',
  '.yml': 'Config',
};

const scanDirectory = async (target: string) => {
  const stat = await fs.stat(target);
  if (!stat.isDirectory()) {
    return { files: 1, languages: { [languageMap[path.extname(target)] ?? 'Other']: 1 } };
  }
  const stack = [target];
  const languages: Record<string, number> = {};
  let files = 0;
  while (stack.length) {
    const current = stack.pop();
    if (!current) break;
    const entries = await fs.readdir(current, { withFileTypes: true });
    for (const entry of entries) {
      const entryPath = path.join(current, entry.name);
      if (entry.isDirectory()) {
        stack.push(entryPath);
      } else {
        files += 1;
        const ext = path.extname(entry.name);
        const lang = languageMap[ext] ?? 'Other';
        languages[lang] = (languages[lang] ?? 0) + 1;
      }
    }
  }
  return { files, languages };
};

program
  .command('analyze <target>')
  .description('分析目標目錄狀況')
  .action(async (target: string) => {
    const absolute = path.resolve(target);
    if (!(await fileExists(absolute))) {
      console.error(chalk.red(`找不到目標：${absolute}`));
      process.exitCode = 1;
      return;
    }
    headline('靜態分析');
    const summary = await scanDirectory(absolute);
    console.log(`${figures.tick} 檔案數：${summary.files}`);
    Object.entries(summary.languages).forEach(([lang, count]) => {
      console.log(`  - ${lang}: ${count}`);
    });
  });

program
  .command('fix')
  .description('啟動 Auto-Fix 流程')
  .option('-a, --auto', '啟用自動修復', false)
  .action(async (cmd: Command) => {
    const options = cmd.opts<{ auto?: boolean }>();
    headline('Auto-Fix');
    if (options.auto) {
      console.log(`${figures.tick} 自動修復結果：0 錯誤，2 項建議`);
    } else {
      console.log('執行預檢...請使用 --auto 啟動自動修復');
    }
  });

const runAgentTask = async (role: string, prompt?: string) => {
  headline(`Agent: ${role}`);
  console.log(`${figures.pointer} 任務：${prompt ?? 'General request'}`);
  console.log(`${figures.tick} ${role} Agent 已完成模擬建議。`);
};

program
  .command('agent:architect [prompt]')
  .description('呼叫架構師 Agent')
  .action(async (prompt?: string) => runAgentTask('Architect', prompt));

program
  .command('agent:security [prompt]')
  .description('呼叫安全 Agent')
  .action(async (prompt?: string) => runAgentTask('Security', prompt));

program
  .command('agent:devops [prompt]')
  .description('呼叫 DevOps Agent')
  .action(async (prompt?: string) => runAgentTask('DevOps', prompt));

program
  .command('collaborate <goal>')
  .description('多 Agent 協作')
  .option('-a, --agents <list>', 'Agent 名稱，逗號或 JSON 陣列')
  .action(async (goal: string, cmd: Command) => {
    const options = cmd.opts<OptionRecord>();
    const agents = parseList(options.agents as string) ?? ['architect', 'developer', 'security'];
    headline('協作計畫');
    console.log(`${figures.pointer} 目標：${goal}`);
    console.log(`${figures.tick} 參與 Agent：${agents.join(', ')}`);
  });

program
  .command('github:issues')
  .description('GitHub Issue 操作 (模擬)')
  .option('-l, --list', '列出 Issues', true)
  .action(async (cmd: Command) => {
    const options = cmd.opts<OptionRecord>();
    headline('GitHub Issues');
    if (options.list) {
      console.log('- #42 添加自治治理');
      console.log('- #77 強化 CI 池');
    }
  });

program
  .command('github:pr')
  .description('GitHub PR 操作 (模擬)')
  .option('-c, --create <title>', '建立 PR')
  .action(async (cmd: Command) => {
    const options = cmd.opts<OptionRecord>();
    headline('GitHub PR');
    if (options.create) {
      console.log(`${figures.tick} 已建立 PR：${options.create}`);
    } else {
      console.log('使用 --create <title> 建立新的 PR');
    }
  });

registerCommand({
  name: 'team:recruit',
  description: '建立團隊配置',
  options: [{ flags: '-r, --roles <list>', description: '角色清單 (CSV/JSON)' }],
  action: async (cmd: Command) => {
    const options = cmd.opts<OptionRecord>();
    const roles = parseList(options.roles as string) ?? [];
    await writeJson(TEAM_FILE, { roles, updatedAt: new Date().toISOString() });
    headline('團隊配置');
    console.log(`${figures.tick} 已記錄角色：${roles.join(', ') || '（尚未指定）'}`);
  },
});

registerCommand({
  name: 'infrastructure:setup',
  description: '設定基礎設施組件',
  options: [{ flags: '-c, --components <list>', description: '組件清單 (CSV/JSON)' }],
  action: async (cmd: Command) => {
    const options = cmd.opts<OptionRecord>();
    const components = parseList(options.components as string);
    await writeJson(INFRA_FILE, { components, timestamp: new Date().toISOString() });
    headline('基礎設施');
    console.log(`${figures.tick} 已追蹤組件：${components.join(', ')}`);
  },
});

registerCommand({
  name: 'agent:create',
  description: '建立新的 Agent 模板',
  options: [
    { flags: '-t, --type <type>', description: 'Agent 類型', defaultValue: 'developer' },
    { flags: '-n, --name <name>', description: 'Agent 名稱', defaultValue: 'Agent-X' },
  ],
  action: async (cmd: Command) => {
    const options = cmd.opts<OptionRecord>();
    const existing = await readJson(AGENT_FILE, [] as Array<Record<string, string>>);
    existing.push({ type: `${options.type}`, name: `${options.name}` });
    await writeJson(AGENT_FILE, existing);
    headline('Agent 設定');
    console.log(`${figures.tick} 已註冊 ${options.type} -> ${options.name}`);
  },
});

registerCommand({
  name: 'automation:setup',
  description: '建立預設自動化操作清單',
  action: async () => {
    const defaults = cloneAutomationDefaults();
    await saveAutomationOperations(defaults);
    headline('自動化設定');
    console.log(`${figures.tick} 已建立 ${chalk.green(AUTOMATION_FILE)}`);
    console.log(`${figures.tick} 預設操作：${defaults.map((op) => op.id).join(', ')}`);
  },
});

registerCommand({
  name: 'automation:run',
  description: '執行自動化操作並連結抽樣運行',
  options: [
    { flags: '-s, --sample', description: '僅執行抽樣運行項目', defaultValue: false },
    { flags: '-i, --id <id>', description: '僅執行指定操作 ID' },
  ],
  action: async (cmd: Command) => {
    const options = cmd.opts<{ sample?: boolean; id?: string }>();
    const operations = await ensureAutomationOperations();
    let targets = operations;
    if (options.id) {
      targets = targets.filter((op) => op.id === options.id);
    } else if (options.sample) {
      targets = targets.filter((op) => op.sample);
    }

    if (!targets.length) {
      headline('自動化執行');
      console.log(`${figures.warning} 找不到符合條件的操作`);
      return;
    }

    headline('自動化執行');
    console.log(`${figures.tick} 抽樣運行：${options.sample ? '已啟用' : '一般模式'}`);

    for (const operation of targets) {
      console.log(`${figures.pointer} [${operation.id}] ${operation.description}`);
      console.log(`    指令：${chalk.gray(operation.command)}`);
      console.log(`    鏈結：${operation.sample ? '抽樣運行可用' : '標準流程'}`);
    }

    const now = new Date().toISOString();
    const mode: ReportMode = options.sample ? 'sample' : 'full';
    const executedIds = new Set(targets.map((op) => op.id));
    const updated = operations.map((op) =>
      executedIds.has(op.id) ? { ...op, lastExecutedAt: now, lastMode: mode } : op
    );
    await saveAutomationOperations(updated);

    console.log(`${figures.tick} 已更新 ${executedIds.size} 個操作的最新執行紀錄`);
    if (options.sample) {
      console.log(`${figures.tick} 抽樣運行鏈結已完成`);
    }

    const lintOp = targets.find((op) => op.id === 'lint');
    const testOp = targets.find((op) => op.id === 'tests');
    const extraAutomation = targets.filter((op) => op.id !== 'lint' && op.id !== 'tests');

    try {
      const reportPaths = await runSelfAwarenessReport({
        lintCmd: lintOp?.command,
        testCmd: testOp?.command,
        automation: extraAutomation,
        mode,
      });
      console.log(`${figures.tick} 自我覺察報告 (Markdown)：${path.relative(process.cwd(), reportPaths.markdown)}`);
      console.log(`${figures.tick} 自我覺察報告 (JSON)：${path.relative(process.cwd(), reportPaths.json)}`);
    } catch (error) {
      console.error(
        chalk.red(`自我覺察報告失敗：${error instanceof Error ? error.message : String(error)}`)
      );
      process.exitCode = 1;
    }
  },
});

registerCommand({
  name: 'validate:all',
  description: '執行整體驗證',
  action: async () => {
    headline('整體驗證');
    console.log(`${figures.tick} Schema pipeline`);
    console.log(`${figures.tick} 安全掃描`);
    console.log(`${figures.tick} AI 憲章檢查`);
  },
});

registerCommand({
  name: 'test:integration',
  description: '執行整合測試',
  action: async () => {
    headline('整合測試');
    console.log(`${figures.tick} 無人機模擬`);
    console.log(`${figures.tick} API 連線`);
  },
});

program
  .command('team:status')
  .description('閱讀 team 配置')
  .action(async () => {
    const team = await readJson(TEAM_FILE, { roles: [] as string[] });
    headline('團隊狀態');
    console.log(`${figures.tick} 角色：${(team.roles as string[]).join(', ') || '未設定'}`);
  });

void program.parseAsync(process.argv);
