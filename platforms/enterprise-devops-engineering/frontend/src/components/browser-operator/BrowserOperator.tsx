import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Bot,
  Globe,
  Play,
  Pause,
  RotateCcw,
  Check,
  X,
  AlertCircle,
  Loader2,
  Eye,
  EyeOff,
  Lock,
  User,
  MousePointer,
  Keyboard,
  Sparkles,
  Plus,
  FileText,
  Code,
  Camera,
  Timer,
} from 'lucide-react';
import { RippleButton } from '@/components/ui/ripple-button';
import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { VirtualBrowser } from '@/components/virtual-browser/VirtualBrowser';
import { cn } from '@/lib/utils';

// Task Types
interface BrowserTask {
  id: string;
  name: string;
  description: string;
  status: 'idle' | 'running' | 'paused' | 'completed' | 'failed';
  progress: number;
  steps: TaskStep[];
  credentials?: {
    username: string;
    password: string;
    domain: string;
  };
  createdAt: Date;
}

interface TaskStep {
  id: string;
  type: 'navigate' | 'click' | 'type' | 'wait' | 'screenshot' | 'extract' | 'login' | 'execute';
  description: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  data?: Record<string, any>;
  result?: any;
  timestamp?: Date;
  screenshot?: string;
}

// Predefined Tasks
const PREDEFINED_TASKS = [
  {
    id: 'login-github',
    name: 'Login to GitHub',
    description: 'Automate GitHub login and navigate to repositories',
    steps: [
      { type: 'navigate', description: 'Navigate to github.com/login' },
      { type: 'type', description: 'Enter username' },
      { type: 'type', description: 'Enter password' },
      { type: 'click', description: 'Click Sign in button' },
      { type: 'wait', description: 'Wait for login to complete' },
      { type: 'screenshot', description: 'Capture dashboard' },
    ],
  },
  {
    id: 'deploy-vercel',
    name: 'Deploy to Vercel',
    description: 'Login to Vercel and deploy a project',
    steps: [
      { type: 'navigate', description: 'Navigate to vercel.com/login' },
      { type: 'login', description: 'Login with GitHub' },
      { type: 'click', description: 'Select project' },
      { type: 'click', description: 'Click Deploy' },
      { type: 'wait', description: 'Wait for deployment' },
      { type: 'extract', description: 'Get deployment URL' },
    ],
  },
  {
    id: 'research-topic',
    name: 'Web Research',
    description: 'Search and extract information from the web',
    steps: [
      { type: 'navigate', description: 'Navigate to search engine' },
      { type: 'type', description: 'Enter search query' },
      { type: 'click', description: 'Submit search' },
      { type: 'extract', description: 'Extract search results' },
      { type: 'click', description: 'Open first result' },
      { type: 'extract', description: 'Extract page content' },
    ],
  },
  {
    id: 'custom-script',
    name: 'Custom Script',
    description: 'Execute a custom browser automation script',
    steps: [
      { type: 'navigate', description: 'Navigate to target URL' },
      { type: 'execute', description: 'Execute custom actions' },
    ],
  },
];

export function BrowserOperator() {
  const [tasks, setTasks] = useState<BrowserTask[]>([]);
  const [activeTask, setActiveTask] = useState<BrowserTask | null>(null);
  const [showNewTaskDialog, setShowNewTaskDialog] = useState(false);
  const [selectedTemplate, setSelectedTemplate] = useState<string | null>(null);
  const [taskName, setTaskName] = useState('');
  const [credentials, setCredentials] = useState({ username: '', password: '', domain: '' });
  const [showPassword, setShowPassword] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);

  // Add log
  const addLog = (message: string) => {
    const timestamp = new Date().toLocaleTimeString();
    setLogs((prev) => [...prev, `[${timestamp}] ${message}`]);
  };

  // Create new task
  const createTask = () => {
    if (!taskName || !selectedTemplate) return;

    const template = PREDEFINED_TASKS.find((t) => t.id === selectedTemplate);
    if (!template) return;

    const newTask: BrowserTask = {
      id: `task-${Date.now()}`,
      name: taskName,
      description: template.description,
      status: 'idle',
      progress: 0,
      steps: template.steps.map((step, index) => ({
        id: `step-${index}`,
        type: step.type as TaskStep['type'],
        description: step.description,
        status: 'pending',
      })),
      credentials: credentials.username ? credentials : undefined,
      createdAt: new Date(),
    };

    setTasks((prev) => [...prev, newTask]);
    setActiveTask(newTask);
    setShowNewTaskDialog(false);
    setTaskName('');
    setSelectedTemplate(null);
    setCredentials({ username: '', password: '', domain: '' });
  };

  // Execute task
  const executeTask = async (task: BrowserTask) => {
    setActiveTask((prev) => (prev ? { ...prev, status: 'running' } : null));
    addLog(`Starting task: ${task.name}`);

    for (let i = 0; i < task.steps.length; i++) {
      const step = task.steps[i];
      
      // Update step status
      setActiveTask((prev) =>
        prev
          ? {
              ...prev,
              steps: prev.steps.map((s, idx) =>
                idx === i ? { ...s, status: 'running', timestamp: new Date() } : s
              ),
              progress: (i / task.steps.length) * 100,
            }
          : null
      );

      addLog(`Executing: ${step.description}`);

      // Simulate step execution
      await executeStep(step, task.credentials);

      // Update step as completed
      setActiveTask((prev) =>
        prev
          ? {
              ...prev,
              steps: prev.steps.map((s, idx) =>
                idx === i ? { ...s, status: 'completed' } : s
              ),
            }
          : null
      );

      addLog(`Completed: ${step.description}`);
    }

    setActiveTask((prev) =>
      prev
        ? { ...prev, status: 'completed', progress: 100 }
        : null
    );
    addLog(`Task completed: ${task.name}`);
  };

  // Execute individual step
  const executeStep = async (step: TaskStep, _credentials?: BrowserTask['credentials']) => {
    const delay = (ms: number) => new Promise((r) => setTimeout(r, ms));

    switch (step.type) {
      case 'navigate':
        await delay(1500);
        break;
      case 'type':
        await delay(800);
        break;
      case 'click':
        await delay(500);
        break;
      case 'wait':
        await delay(2000);
        break;
      case 'screenshot':
        await delay(500);
        break;
      case 'login':
        await delay(2000);
        break;
      case 'extract':
        await delay(1000);
        break;
      case 'execute':
        await delay(3000);
        break;
    }
  };

  // Pause task
  const pauseTask = () => {
    setActiveTask((prev) => (prev ? { ...prev, status: 'paused' } : null));
    addLog('Task paused');
  };

  // Resume task
  const resumeTask = () => {
    if (activeTask) {
      executeTask(activeTask);
    }
  };

  // Reset task
  const resetTask = () => {
    if (activeTask) {
      setActiveTask({
        ...activeTask,
        status: 'idle',
        progress: 0,
        steps: activeTask.steps.map((s) => ({ ...s, status: 'pending' })),
      });
      setLogs([]);
    }
  };

  return (
    <div className="flex h-full bg-white">
      {/* Left Panel - Task List */}
      <div className="w-80 border-r border-gray-200 flex flex-col">
        <div className="h-14 border-b border-gray-200 flex items-center justify-between px-4">
          <div className="flex items-center gap-2">
            <Bot className="w-5 h-5 text-[#FFD700]" />
            <span className="font-semibold text-gray-900">Browser Operator</span>
          </div>
          <RippleButton size="sm" className="bg-[#FFD700] text-black hover:bg-[#FFC700]" onClick={() => setShowNewTaskDialog(true)}>
            <Plus className="w-4 h-4 mr-1" />
            New
          </RippleButton>
        </div>

        <ScrollArea className="flex-1 p-3">
          <div className="space-y-2">
            {tasks.length === 0 ? (
              <div className="text-center py-8">
                <Bot className="w-12 h-12 text-gray-400 mx-auto mb-3" />
                <p className="text-sm text-gray-600">No tasks yet</p>
                <p className="text-xs text-gray-500 mt-1">Create a new automation task</p>
              </div>
            ) : (
              tasks.map((task) => (
                <TaskCard
                  key={task.id}
                  task={task}
                  isActive={activeTask?.id === task.id}
                  onClick={() => setActiveTask(task)}
                />
              ))
            )}
          </div>
        </ScrollArea>
      </div>

      {/* Center Panel - Browser View */}
      <div className="flex-1 flex flex-col">
        {activeTask ? (
          <>
            {/* Task Header */}
            <div className="h-14 border-b border-gray-200 flex items-center justify-between px-4">
              <div className="flex items-center gap-3">
                <div
                  className={cn(
                    'w-2 h-2 rounded-full',
                    activeTask.status === 'running' && 'bg-[#FFD700] animate-pulse',
                    activeTask.status === 'completed' && 'bg-green-500',
                    activeTask.status === 'failed' && 'bg-red-500',
                    activeTask.status === 'idle' && 'bg-gray-400',
                    activeTask.status === 'paused' && 'bg-yellow-500'
                  )}
                />
                <div>
                  <span className="font-medium text-gray-900">{activeTask.name}</span>
                  <span className="text-xs text-gray-500 ml-2">
                    {activeTask.steps.filter((s) => s.status === 'completed').length} / {activeTask.steps.length} steps
                  </span>
                </div>
              </div>
              <div className="flex items-center gap-2">
                {activeTask.status === 'running' ? (
                  <RippleButton variant="outline" size="sm" className="border-gray-300" onClick={pauseTask}>
                    <Pause className="w-4 h-4 mr-1" />
                    Pause
                  </RippleButton>
                ) : activeTask.status === 'paused' ? (
                  <RippleButton size="sm" className="bg-[#FFD700] text-black hover:bg-[#FFC700]" onClick={resumeTask}>
                    <Play className="w-4 h-4 mr-1" />
                    Resume
                  </RippleButton>
                ) : activeTask.status === 'idle' ? (
                  <RippleButton size="sm" className="bg-[#FFD700] text-black hover:bg-[#FFC700]" onClick={() => executeTask(activeTask)}>
                    <Play className="w-4 h-4 mr-1" />
                    Start
                  </RippleButton>
                ) : (
                  <RippleButton variant="outline" size="sm" className="border-gray-300" onClick={resetTask}>
                    <RotateCcw className="w-4 h-4 mr-1" />
                    Reset
                  </RippleButton>
                )}
              </div>
            </div>

            {/* Progress Bar */}
            <div className="h-1 bg-gray-200">
              <motion.div
                className="h-full bg-[#FFD700]"
                initial={{ width: 0 }}
                animate={{ width: `${activeTask.progress}%` }}
                transition={{ duration: 0.3 }}
              />
            </div>

            {/* Browser */}
            <div className="flex-1 p-4">
              <VirtualBrowser 
                automationMode={true}
                onAction={(action) => addLog(`Browser action: ${action.type}`)}
              />
            </div>
          </>
        ) : (
          <div className="flex-1 flex flex-col items-center justify-center">
            <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-[#FFD700] to-[#FFA500] flex items-center justify-center mb-6">
              <Bot className="w-10 h-10 text-black" />
            </div>
            <h2 className="text-xl font-bold text-gray-900 mb-2">Browser Operator</h2>
            <p className="text-gray-500 text-center max-w-md mb-6">
              Automate browser tasks with AI. Login to accounts, extract data, and execute scripts.
            </p>
            <RippleButton className="bg-[#FFD700] text-black hover:bg-[#FFC700]" onClick={() => setShowNewTaskDialog(true)}>
              <Sparkles className="w-4 h-4 mr-2" />
              Create New Task
            </RippleButton>
          </div>
        )}
      </div>

      {/* Right Panel - Steps & Logs */}
      {activeTask && (
        <div className="w-80 border-l border-gray-200 flex flex-col">
          <Tabs defaultValue="steps" className="flex-1 flex flex-col">
            <TabsList className="bg-gray-50 border-b border-gray-200 rounded-none h-10">
              <TabsTrigger value="steps" className="text-xs data-[state=active]:bg-[#FFD700]/10 data-[state=active]:text-[#FFD700]">
                Steps
              </TabsTrigger>
              <TabsTrigger value="logs" className="text-xs data-[state=active]:bg-[#FFD700]/10 data-[state=active]:text-[#FFD700]">
                Logs
              </TabsTrigger>
              <TabsTrigger value="settings" className="text-xs data-[state=active]:bg-[#FFD700]/10 data-[state=active]:text-[#FFD700]">
                Settings
              </TabsTrigger>
            </TabsList>

            <TabsContent value="steps" className="flex-1 m-0">
              <ScrollArea className="h-full p-3">
                <div className="space-y-2">
                  {activeTask.steps.map((step, index) => (
                    <StepCard key={step.id} step={step} index={index} />
                  ))}
                </div>
              </ScrollArea>
            </TabsContent>

            <TabsContent value="logs" className="flex-1 m-0">
              <ScrollArea className="h-full p-3">
                <div className="font-mono text-xs space-y-1">
                  {logs.map((log, index) => (
                    <div key={index} className="text-gray-600">
                      {log}
                    </div>
                  ))}
                </div>
              </ScrollArea>
            </TabsContent>

            <TabsContent value="settings" className="flex-1 m-0 p-3">
              <div className="space-y-4">
                <div>
                  <label className="text-sm text-gray-600 mb-2 block">Task Name</label>
                  <Input value={activeTask.name} readOnly className="bg-gray-50 border-gray-200" />
                </div>
                {activeTask.credentials && (
                  <div>
                    <label className="text-sm text-gray-600 mb-2 block">Credentials</label>
                    <div className="p-3 rounded-lg bg-gray-50 border border-gray-200">
                      <div className="flex items-center gap-2 text-sm text-gray-600">
                        <User className="w-4 h-4" />
                        {activeTask.credentials.username}
                      </div>
                      <div className="flex items-center gap-2 text-sm text-gray-600 mt-1">
                        <Lock className="w-4 h-4" />
                        ••••••••
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </TabsContent>
          </Tabs>
        </div>
      )}

      {/* New Task Dialog */}
      <AnimatePresence>
        {showNewTaskDialog && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/50 flex items-center justify-center z-50"
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              className="bg-white border border-gray-200 rounded-xl w-full max-w-lg p-6"
            >
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-lg font-semibold text-gray-900">Create New Task</h2>
                <RippleButton variant="ghost" size="icon" className="text-gray-500" onClick={() => setShowNewTaskDialog(false)}>
                  <X className="w-4 h-4" />
                </RippleButton>
              </div>

              <div className="space-y-4">
                <div>
                  <label className="text-sm text-gray-600 mb-2 block">Task Name</label>
                  <Input
                    value={taskName}
                    onChange={(e) => setTaskName(e.target.value)}
                    placeholder="Enter task name"
                    className="bg-gray-50 border-gray-200"
                  />
                </div>

                <div>
                  <label className="text-sm text-gray-600 mb-2 block">Select Template</label>
                  <div className="grid grid-cols-2 gap-2">
                    {PREDEFINED_TASKS.map((template) => (
                      <button
                        key={template.id}
                        onClick={() => setSelectedTemplate(template.id)}
                        className={cn(
                          'p-3 rounded-lg border text-left transition-all',
                          selectedTemplate === template.id
                            ? 'border-[#FFD700] bg-[#FFD700]/10'
                            : 'border-gray-200 hover:border-gray-300'
                        )}
                      >
                        <div className="font-medium text-sm text-gray-900">{template.name}</div>
                        <div className="text-xs text-gray-500 mt-1">{template.description}</div>
                      </button>
                    ))}
                  </div>
                </div>

                <div>
                  <label className="text-sm text-gray-600 mb-2 block">Credentials (Optional)</label>
                  <div className="space-y-2">
                    <Input
                      value={credentials.username}
                      onChange={(e) => setCredentials({ ...credentials, username: e.target.value })}
                      placeholder="Username or email"
                      className="bg-gray-50 border-gray-200"
                    />
                    <div className="relative">
                      <Input
                        type={showPassword ? 'text' : 'password'}
                        value={credentials.password}
                        onChange={(e) => setCredentials({ ...credentials, password: e.target.value })}
                        placeholder="Password"
                        className="bg-gray-50 border-gray-200 pr-10"
                      />
                      <button
                        onClick={() => setShowPassword(!showPassword)}
                        className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500"
                      >
                        {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                      </button>
                    </div>
                  </div>
                </div>
              </div>

              <div className="flex justify-end gap-2 mt-6">
                <RippleButton variant="outline" className="border-gray-300" onClick={() => setShowNewTaskDialog(false)}>
                  Cancel
                </RippleButton>
                <RippleButton className="bg-[#FFD700] text-black hover:bg-[#FFC700]" onClick={createTask} disabled={!taskName || !selectedTemplate}>
                  Create Task
                </RippleButton>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// Task Card Component
function TaskCard({
  task,
  isActive,
  onClick,
}: {
  task: BrowserTask;
  isActive: boolean;
  onClick: () => void;
}) {
  const statusColors = {
    idle: 'bg-gray-400',
    running: 'bg-[#FFD700] animate-pulse',
    paused: 'bg-yellow-500',
    completed: 'bg-green-500',
    failed: 'bg-red-500',
  };

  return (
    <button
      onClick={onClick}
      className={cn(
        'w-full p-3 rounded-lg border text-left transition-all',
        isActive
          ? 'border-[#FFD700] bg-[#FFD700]/10'
          : 'border-gray-200 bg-gray-50 hover:border-gray-300'
      )}
    >
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-2">
          <div className={cn('w-2 h-2 rounded-full', statusColors[task.status])} />
          <span className="font-medium text-sm text-gray-900">{task.name}</span>
        </div>
      </div>
      <p className="text-xs text-gray-500 mt-1">{task.description}</p>
      <div className="flex items-center gap-2 mt-2">
        <div className="flex-1 h-1 bg-gray-200 rounded-full overflow-hidden">
          <div
            className="h-full bg-[#FFD700] transition-all"
            style={{ width: `${task.progress}%` }}
          />
        </div>
        <span className="text-xs text-gray-500">{Math.round(task.progress)}%</span>
      </div>
    </button>
  );
}

// Step Card Component
function StepCard({ step, index }: { step: TaskStep; index: number }) {
  const statusIcons = {
    pending: <div className="w-4 h-4 rounded-full border border-gray-400" />,
    running: <Loader2 className="w-4 h-4 animate-spin text-[#FFD700]" />,
    completed: <Check className="w-4 h-4 text-green-500" />,
    failed: <AlertCircle className="w-4 h-4 text-red-500" />,
  };

  const typeIcons = {
    navigate: <Globe className="w-3 h-3" />,
    click: <MousePointer className="w-3 h-3" />,
    type: <Keyboard className="w-3 h-3" />,
    wait: <Timer className="w-3 h-3" />,
    screenshot: <Camera className="w-3 h-3" />,
    extract: <FileText className="w-3 h-3" />,
    login: <Lock className="w-3 h-3" />,
    execute: <Code className="w-3 h-3" />,
  };

  return (
    <div
      className={cn(
        'flex items-center gap-3 p-2 rounded-lg transition-all',
        step.status === 'running' && 'bg-[#FFD700]/10',
        step.status === 'completed' && 'bg-green-100/50'
      )}
    >
      <span className="text-xs text-gray-500 w-5">{index + 1}</span>
      {statusIcons[step.status]}
      <span className="text-gray-500">{typeIcons[step.type]}</span>
      <span className="text-sm text-gray-600 flex-1">{step.description}</span>
      {step.timestamp && (
        <span className="text-xs text-gray-500">
          {step.timestamp.toLocaleTimeString()}
        </span>
      )}
    </div>
  );
}


