import { useState } from 'react';
import { motion } from 'framer-motion';
import {
  Bot,
  Target,
  Brain,
  Code,
  Palette,
  FileSpreadsheet,
  Globe,
  MessageSquare,
  Play,
  Pause,
  Check,
  Loader2,
  Settings,
  Plus,
  Clock,
  TrendingUp,
  Send,
  Paperclip,
  Image,
  Smartphone,
  MousePointer,
  Type,
  Shapes,
  PenTool,
  BarChart3,
} from 'lucide-react';
import { RippleButton } from '@/components/ui/ripple-button';
import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';

// Types
interface Agent {
  id: string;
  name: string;
  type: 'max' | 'design' | 'code' | 'data' | 'web' | 'mobile';
  description: string;
  icon: React.ElementType;
  color: string;
  status: 'idle' | 'running' | 'paused' | 'completed' | 'error';
  capabilities: string[];
}

interface Task {
  id: string;
  title: string;
  description: string;
  status: 'pending' | 'planning' | 'executing' | 'reviewing' | 'completed' | 'failed';
  progress: number;
  agentId: string;
  subTasks: SubTask[];
  createdAt: Date;
  updatedAt: Date;
  output?: any;
}

interface SubTask {
  id: string;
  title: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  agent?: string;
  output?: string;
}

interface ChatMessage {
  id: string;
  role: 'user' | 'agent' | 'system';
  content: string;
  timestamp: Date;
  attachments?: Attachment[];
  thinking?: string;
}

interface Attachment {
  type: 'image' | 'file' | 'code';
  name: string;
  url?: string;
  content?: string;
}

// Mock Agents
const agents: Agent[] = [
  {
    id: 'max-agent',
    name: 'Max Agent',
    type: 'max',
    description: '高智能旗舰代理，自动化处理复杂多步骤任务',
    icon: Brain,
    color: '#FFD700',
    status: 'idle',
    capabilities: ['任务规划', '多步骤执行', '数据分析', '决策制定', '自动化流程'],
  },
  {
    id: 'design-agent',
    name: 'Design Agent',
    type: 'design',
    description: '交互式设计检视，图像创建与编辑',
    icon: Palette,
    color: '#8B5CF6',
    status: 'idle',
    capabilities: ['UI设计', '图像生成', '设计审查', '原型制作', '视觉优化'],
  },
  {
    id: 'code-agent',
    name: 'Code Agent',
    type: 'code',
    description: '端到端应用开发，代码生成与优化',
    icon: Code,
    color: '#1E88E5',
    status: 'idle',
    capabilities: ['代码生成', '代码审查', '重构优化', '测试生成', 'Debug'],
  },
  {
    id: 'data-agent',
    name: 'Data Agent',
    type: 'data',
    description: '数据分析与试算表自动化',
    icon: FileSpreadsheet,
    color: '#10B981',
    status: 'idle',
    capabilities: ['数据分析', '财务建模', '报告生成', '可视化', '预测分析'],
  },
  {
    id: 'web-agent',
    name: 'Web Agent',
    type: 'web',
    description: 'Web 开发与 UI 优化',
    icon: Globe,
    color: '#06B6D4',
    status: 'idle',
    capabilities: ['前端开发', '响应式设计', '性能优化', 'SEO优化', '组件开发'],
  },
  {
    id: 'mobile-agent',
    name: 'Mobile Agent',
    type: 'mobile',
    description: '端到端移动应用开发',
    icon: Smartphone,
    color: '#EC4899',
    status: 'idle',
    capabilities: ['iOS开发', 'Android开发', '跨平台', 'UI适配', '性能调优'],
  },
];

// Mock Tasks
const mockTasks: Task[] = [
  {
    id: 'task-1',
    title: '构建电商数据分析仪表板',
    description: '从多个数据源整合销售数据，生成可视化仪表板',
    status: 'executing',
    progress: 65,
    agentId: 'max-agent',
    subTasks: [
      { id: 'st-1', title: '数据源连接与验证', status: 'completed', progress: 100 },
      { id: 'st-2', title: '数据清洗与转换', status: 'completed', progress: 100 },
      { id: 'st-3', title: '生成可视化图表', status: 'running', progress: 60 },
      { id: 'st-4', title: '构建交互式仪表板', status: 'pending', progress: 0 },
    ],
    createdAt: new Date(Date.now() - 1000 * 60 * 30),
    updatedAt: new Date(),
  },
  {
    id: 'task-2',
    title: '设计新的登录页面',
    description: '创建现代化的登录页面设计，包含社交登录选项',
    status: 'completed',
    progress: 100,
    agentId: 'design-agent',
    subTasks: [
      { id: 'st-1', title: '分析需求与竞品研究', status: 'completed', progress: 100 },
      { id: 'st-2', title: '创建线框图', status: 'completed', progress: 100 },
      { id: 'st-3', title: '高保真设计', status: 'completed', progress: 100 },
      { id: 'st-4', title: '设计审查与优化', status: 'completed', progress: 100 },
    ],
    createdAt: new Date(Date.now() - 1000 * 60 * 60 * 2),
    updatedAt: new Date(Date.now() - 1000 * 60 * 15),
  },
];

// Mock Chat Messages
const mockChatMessages: ChatMessage[] = [
  {
    id: 'msg-1',
    role: 'user',
    content: '我需要创建一个数据分析仪表板，包含销售趋势、用户增长和收入分析',
    timestamp: new Date(Date.now() - 1000 * 60 * 35),
  },
  {
    id: 'msg-2',
    role: 'agent',
    content: '我来帮您构建这个数据分析仪表板。我将分步骤完成：\n\n1. 首先连接您的数据源\n2. 清洗和转换数据\n3. 创建可视化图表\n4. 构建交互式仪表板',
    timestamp: new Date(Date.now() - 1000 * 60 * 34),
    thinking: '用户需要数据分析仪表板，涉及多个数据源。我应该使用 Max Agent 来协调 Data Agent 和 Web Agent 完成这个任务。',
  },
  {
    id: 'msg-3',
    role: 'agent',
    content: '✅ 已完成数据源连接\n✅ 数据清洗完成\n🔄 正在生成可视化图表...',
    timestamp: new Date(Date.now() - 1000 * 60 * 5),
  },
];

export function MaxAgentHub() {
  const [activeTab, setActiveTab] = useState('agents');
  const [, setSelectedAgent] = useState<Agent | null>(null);
  const [tasks] = useState<Task[]>(mockTasks);
  const [selectedTask, setSelectedTask] = useState<Task | null>(null);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>(mockChatMessages);
  const [chatInput, setChatInput] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [selectedTool, setSelectedTool] = useState('select');

  // Send message to agent
  const sendMessage = async () => {
    if (!chatInput.trim() || isProcessing) return;

    const userMessage: ChatMessage = {
      id: `msg-${Date.now()}`,
      role: 'user',
      content: chatInput,
      timestamp: new Date(),
    };

    setChatMessages((prev) => [...prev, userMessage]);
    setChatInput('');
    setIsProcessing(true);

    // Simulate agent response
    setTimeout(() => {
      const agentMessage: ChatMessage = {
        id: `msg-${Date.now() + 1}`,
        role: 'agent',
        content: '我已理解您的需求。让我为您创建一个新的任务来执行这个请求。',
        timestamp: new Date(),
        thinking: '分析用户需求，准备创建任务计划...',
      };
      setChatMessages((prev) => [...prev, agentMessage]);
      setIsProcessing(false);
    }, 1500);
  };

  // Render agent card
  const renderAgentCard = (agent: Agent) => (
    <motion.div
      key={agent.id}
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      onClick={() => setSelectedAgent(agent)}
      className="bg-gray-50 border border-gray-200 rounded-xl p-5 cursor-pointer hover:border-[#FFD700]/50 transition-colors"
    >
      <div className="flex items-start justify-between mb-4">
        <div
          className="w-12 h-12 rounded-xl flex items-center justify-center"
          style={{ backgroundColor: `${agent.color}20` }}
        >
          <agent.icon className="w-6 h-6" style={{ color: agent.color }} />
        </div>
        <Badge
          className={cn(
            'border-0 text-[10px]',
            agent.status === 'idle' && 'bg-gray-100 text-gray-600',
            agent.status === 'running' && 'bg-[#FFD700]/20 text-[#FFD700]',
            agent.status === 'completed' && 'bg-green-100 text-green-600'
          )}
        >
          {agent.status}
        </Badge>
      </div>
      <h3 className="font-semibold text-gray-900 mb-1">{agent.name}</h3>
      <p className="text-sm text-gray-500 mb-3">{agent.description}</p>
      <div className="flex flex-wrap gap-1">
        {agent.capabilities.slice(0, 3).map((cap, i) => (
          <Badge key={i} className="bg-white text-gray-600 border-0 text-[9px]">
            {cap}
          </Badge>
        ))}
        {agent.capabilities.length > 3 && (
          <Badge className="bg-white text-gray-600 border-0 text-[9px]">
            +{agent.capabilities.length - 3}
          </Badge>
        )}
      </div>
    </motion.div>
  );

  return (
    <div className="h-full flex bg-white">
      {/* Sidebar */}
      <div className="w-64 border-r border-gray-200 flex flex-col">
        <div className="h-16 border-b border-gray-200 flex items-center px-4">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-[#FFD700] to-[#FFA500] flex items-center justify-center mr-3">
            <Brain className="w-5 h-5 text-black" />
          </div>
          <div>
            <h2 className="font-semibold text-gray-900">Max Agent</h2>
            <p className="text-xs text-gray-500">AI Agent Hub</p>
          </div>
        </div>

        <div className="flex-1 py-4 px-3">
          <nav className="space-y-1">
            {[
              { id: 'agents', label: 'Agents', icon: Bot },
              { id: 'tasks', label: 'Tasks', icon: Target },
              { id: 'design', label: 'Design View', icon: Palette },
              { id: 'chat', label: 'Chat', icon: MessageSquare },
              { id: 'analytics', label: 'Analytics', icon: BarChart3 },
            ].map((item) => (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={cn(
                  'w-full flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all',
                  activeTab === item.id
                    ? 'bg-[#FFD700]/10 text-[#FFD700]'
                    : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
                )}
              >
                <item.icon className="w-5 h-5" />
                <span className="font-medium text-sm">{item.label}</span>
              </button>
            ))}
          </nav>
        </div>

        {/* Quick Stats */}
        <div className="p-4 border-t border-gray-200">
          <div className="grid grid-cols-2 gap-3">
            <div className="bg-gray-50 rounded-lg p-3">
              <div className="text-2xl font-bold text-gray-900">{tasks.length}</div>
              <div className="text-xs text-gray-500">Active Tasks</div>
            </div>
            <div className="bg-gray-50 rounded-lg p-3">
              <div className="text-2xl font-bold text-[#FFD700]">{agents.length}</div>
              <div className="text-xs text-gray-500">Agents</div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <div className="h-16 border-b border-gray-200 flex items-center justify-between px-6">
          <div className="flex items-center gap-4">
            <h1 className="text-xl font-semibold text-gray-900">
              {activeTab === 'agents' && 'AI Agents'}
              {activeTab === 'tasks' && 'Task Manager'}
              {activeTab === 'design' && 'Design View'}
              {activeTab === 'chat' && 'Agent Chat'}
              {activeTab === 'analytics' && 'Analytics'}
            </h1>
          </div>
          <div className="flex items-center gap-2">
            <RippleButton variant="outline" size="sm" className="gap-2 border-gray-300">
              <Settings className="w-4 h-4" />
              Settings
            </RippleButton>
            <RippleButton size="sm" className="gap-2 bg-[#FFD700] text-black hover:bg-[#FFC700]">
              <Plus className="w-4 h-4" />
              New Task
            </RippleButton>
          </div>
        </div>

        {/* Content Area */}
        <div className="flex-1 overflow-hidden">
          {/* Agents Tab */}
          {activeTab === 'agents' && (
            <ScrollArea className="h-full p-6">
              <div className="max-w-6xl mx-auto">
                {/* Featured Agent */}
                <div className="mb-8">
                  <div className="bg-gradient-to-r from-[#FFD700] to-[#FFA500] rounded-2xl p-6">
                    <div className="flex items-start justify-between">
                      <div>
                        <Badge className="bg-white/30 text-black border-0 mb-3">Featured</Badge>
                        <h2 className="text-2xl font-bold text-black mb-2">Max Agent</h2>
                        <p className="text-black/70 max-w-lg mb-4">
                          高智能旗舰代理，具备高成功率与自主性，特别适合需要多步骤规划、
                          数据分析与自动化开发的复杂任务。
                        </p>
                        <div className="flex gap-2">
                          <RippleButton className="bg-white text-black hover:bg-gray-100">
                            <Play className="w-4 h-4 mr-2" />
                            Start Task
                          </RippleButton>
                          <RippleButton variant="outline" className="border-black/30 text-black hover:bg-black/10">
                            Learn More
                          </RippleButton>
                        </div>
                      </div>
                      <div className="w-24 h-24 bg-white/30 rounded-2xl flex items-center justify-center">
                        <Brain className="w-12 h-12 text-black" />
                      </div>
                    </div>
                  </div>
                </div>

                {/* Agent Grid */}
                <div className="grid grid-cols-3 gap-4">
                  {agents.map(renderAgentCard)}
                </div>
              </div>
            </ScrollArea>
          )}

          {/* Tasks Tab */}
          {activeTab === 'tasks' && (
            <div className="h-full flex">
              {/* Task List */}
              <div className="w-96 border-r border-gray-200 flex flex-col">
                <div className="h-14 border-b border-gray-200 flex items-center justify-between px-4">
                  <Input
                    placeholder="Search tasks..."
                    className="bg-gray-50 border-gray-200"
                  />
                </div>
                <ScrollArea className="flex-1 p-3">
                  <div className="space-y-2">
                    {tasks.map((task) => (
                      <button
                        key={task.id}
                        onClick={() => setSelectedTask(task)}
                        className={cn(
                          'w-full p-4 rounded-xl text-left transition-colors',
                          selectedTask?.id === task.id
                            ? 'bg-[#FFD700]/10 border border-[#FFD700]/30'
                            : 'bg-gray-50 border border-gray-200 hover:border-gray-300'
                        )}
                      >
                        <div className="flex items-start justify-between mb-2">
                          <h4 className="font-medium text-gray-900 text-sm">{task.title}</h4>
                          <Badge
                            className={cn(
                              'border-0 text-[10px]',
                              task.status === 'completed' && 'bg-green-100 text-green-600',
                              task.status === 'executing' && 'bg-[#FFD700]/20 text-[#FFD700]',
                              task.status === 'pending' && 'bg-gray-100 text-gray-600'
                            )}
                          >
                            {task.status}
                          </Badge>
                        </div>
                        <div className="flex items-center gap-2 text-xs text-gray-500">
                          <Clock className="w-3 h-3" />
                          {task.updatedAt.toLocaleTimeString()}
                        </div>
                        <div className="mt-3">
                          <div className="h-1.5 bg-gray-200 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-[#FFD700] transition-all"
                              style={{ width: `${task.progress}%` }}
                            />
                          </div>
                        </div>
                      </button>
                    ))}
                  </div>
                </ScrollArea>
              </div>

              {/* Task Detail */}
              <div className="flex-1 p-6 overflow-auto">
                {selectedTask ? (
                  <div className="max-w-3xl">
                    <div className="flex items-start justify-between mb-6">
                      <div>
                        <h2 className="text-xl font-semibold text-gray-900 mb-2">{selectedTask.title}</h2>
                        <p className="text-gray-600">{selectedTask.description}</p>
                      </div>
                      <div className="flex gap-2">
                        {selectedTask.status === 'executing' ? (
                          <RippleButton variant="outline" size="sm" className="border-gray-300">
                            <Pause className="w-4 h-4 mr-2" />
                            Pause
                          </RippleButton>
                        ) : (
                          <RippleButton size="sm" className="bg-[#FFD700] text-black hover:bg-[#FFC700]">
                            <Play className="w-4 h-4 mr-2" />
                            Start
                          </RippleButton>
                        )}
                      </div>
                    </div>

                    {/* Subtasks */}
                    <div className="space-y-3">
                      <h3 className="font-medium text-gray-900 mb-3">执行步骤</h3>
                      {selectedTask.subTasks.map((subTask, index) => (
                        <div
                          key={subTask.id}
                          className="flex items-center gap-4 p-4 bg-gray-50 border border-gray-200 rounded-xl"
                        >
                          <div className="w-8 h-8 rounded-lg bg-gray-200 flex items-center justify-center text-sm text-gray-600">
                            {index + 1}
                          </div>
                          <div className="flex-1">
                            <div className="flex items-center gap-2">
                              <span className="text-gray-900">{subTask.title}</span>
                              {subTask.status === 'completed' && (
                                <Check className="w-4 h-4 text-green-500" />
                              )}
                              {subTask.status === 'running' && (
                                <Loader2 className="w-4 h-4 text-[#FFD700] animate-spin" />
                              )}
                            </div>
                          </div>
                          <Badge
                            className={cn(
                              'border-0 text-[10px]',
                              subTask.status === 'completed' && 'bg-green-100 text-green-600',
                              subTask.status === 'running' && 'bg-[#FFD700]/20 text-[#FFD700]',
                              subTask.status === 'pending' && 'bg-gray-100 text-gray-600'
                            )}
                          >
                            {subTask.status}
                          </Badge>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div className="h-full flex flex-col items-center justify-center text-gray-500">
                    <Target className="w-16 h-16 mb-4 opacity-50" />
                    <p className="text-lg">Select a task to view details</p>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Design View Tab */}
          {activeTab === 'design' && (
            <div className="h-full flex">
              {/* Toolbar */}
              <div className="w-16 border-r border-gray-200 bg-white flex flex-col items-center py-4 gap-2">
                {[
                  { id: 'select', icon: MousePointer },
                  { id: 'text', icon: Type },
                  { id: 'shape', icon: Shapes },
                  { id: 'image', icon: Image },
                  { id: 'pen', icon: PenTool },
                ].map((tool) => (
                  <RippleButton
                    key={tool.id}
                    variant="ghost"
                    size="icon"
                    className={cn(
                      'h-10 w-10',
                      selectedTool === tool.id && 'bg-[#FFD700]/20 text-[#FFD700]'
                    )}
                    onClick={() => setSelectedTool(tool.id)}
                  >
                    <tool.icon className="w-5 h-5" />
                  </RippleButton>
                ))}
              </div>

              {/* Canvas */}
              <div className="flex-1 bg-gray-100 relative overflow-hidden">
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="w-[800px] h-[600px] bg-white rounded-lg shadow-2xl relative overflow-hidden">
                    {/* Grid */}
                    <div
                      className="absolute inset-0 opacity-10"
                      style={{
                        backgroundImage: `
                          linear-gradient(to right, #000 1px, transparent 1px),
                          linear-gradient(to bottom, #000 1px, transparent 1px)
                        `,
                        backgroundSize: '20px 20px',
                      }}
                    />
                    {/* Canvas Content */}
                    <div className="absolute inset-0 p-8">
                      <div className="w-full h-full border-2 border-dashed border-gray-300 rounded-lg flex items-center justify-center">
                        <div className="text-center text-gray-400">
                          <Palette className="w-12 h-12 mx-auto mb-3" />
                          <p>Design Canvas</p>
                          <p className="text-sm">Select a tool to start designing</p>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Properties Panel */}
              <div className="w-72 border-l border-gray-200 bg-white p-4">
                <h3 className="font-medium text-gray-900 mb-4">Properties</h3>
                <div className="space-y-4">
                  <div>
                    <label className="text-xs text-gray-500 mb-1 block">Size</label>
                    <div className="grid grid-cols-2 gap-2">
                      <Input placeholder="W" className="bg-gray-50 border-gray-200" />
                      <Input placeholder="H" className="bg-gray-50 border-gray-200" />
                    </div>
                  </div>
                  <div>
                    <label className="text-xs text-gray-500 mb-1 block">Position</label>
                    <div className="grid grid-cols-2 gap-2">
                      <Input placeholder="X" className="bg-gray-50 border-gray-200" />
                      <Input placeholder="Y" className="bg-gray-50 border-gray-200" />
                    </div>
                  </div>
                  <div>
                    <label className="text-xs text-gray-500 mb-1 block">Background</label>
                    <div className="flex gap-2">
                      {['#FFD700', '#1E88E5', '#10B981', '#8B5CF6', '#EC4899'].map((color) => (
                        <button
                          key={color}
                          className="w-8 h-8 rounded-lg"
                          style={{ backgroundColor: color }}
                        />
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Chat Tab */}
          {activeTab === 'chat' && (
            <div className="h-full flex">
              {/* Chat Area */}
              <div className="flex-1 flex flex-col">
                <ScrollArea className="flex-1 p-6">
                  <div className="max-w-3xl mx-auto space-y-6">
                    {chatMessages.map((message) => (
                      <div
                        key={message.id}
                        className={cn(
                          'flex gap-4',
                          message.role === 'user' && 'flex-row-reverse'
                        )}
                      >
                        <div
                          className={cn(
                            'w-10 h-10 rounded-xl flex items-center justify-center shrink-0',
                            message.role === 'user'
                              ? 'bg-[#FFD700]'
                              : 'bg-gradient-to-br from-[#FFD700] to-[#FFA500]'
                          )}
                        >
                          {message.role === 'user' ? (
                            <span className="text-black font-medium">U</span>
                          ) : (
                            <Brain className="w-5 h-5 text-black" />
                          )}
                        </div>
                        <div className={cn('flex-1 max-w-[80%]', message.role === 'user' && 'text-right')}>
                          {message.thinking && (
                            <div className="mb-2 p-3 bg-gray-50 border border-gray-200 rounded-lg">
                              <div className="flex items-center gap-2 text-xs text-gray-500 mb-1">
                                <Brain className="w-3 h-3" />
                                Thinking
                              </div>
                              <p className="text-sm text-gray-600">{message.thinking}</p>
                            </div>
                          )}
                          <div
                            className={cn(
                              'inline-block rounded-xl px-4 py-3 text-left',
                              message.role === 'user'
                                ? 'bg-[#FFD700] text-black'
                                : 'bg-gray-50 border border-gray-200 text-gray-800'
                            )}
                          >
                            <p className="whitespace-pre-wrap">{message.content}</p>
                          </div>
                        </div>
                      </div>
                    ))}
                    {isProcessing && (
                      <div className="flex gap-4">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-[#FFD700] to-[#FFA500] flex items-center justify-center">
                          <Loader2 className="w-5 h-5 text-black animate-spin" />
                        </div>
                        <div className="bg-gray-50 border border-gray-200 rounded-xl px-4 py-3">
                          <div className="flex gap-1">
                            <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" />
                            <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:0.1s]" />
                            <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:0.2s]" />
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                </ScrollArea>

                {/* Input */}
                <div className="p-4 border-t border-gray-200">
                  <div className="max-w-3xl mx-auto">
                    <div className="relative bg-gray-50 border border-gray-200 rounded-xl overflow-hidden focus-within:border-[#FFD700]">
                      <textarea
                        value={chatInput}
                        onChange={(e) => setChatInput(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter' && !e.shiftKey) {
                            e.preventDefault();
                            sendMessage();
                          }
                        }}
                        placeholder="描述您想要完成的任务..."
                        className="w-full min-h-[80px] max-h-[200px] bg-transparent border-0 text-gray-900 placeholder:text-gray-500 p-4 pr-14 resize-none focus:outline-none"
                      />
                      <div className="absolute right-3 bottom-3 flex items-center gap-2">
                        <RippleButton variant="ghost" size="icon" className="h-9 w-9 text-gray-500">
                          <Paperclip className="w-5 h-5" />
                        </RippleButton>
                        <RippleButton
                          size="icon"
                          className="h-9 w-9 bg-[#FFD700] text-black hover:bg-[#FFC700]"
                          onClick={sendMessage}
                          disabled={!chatInput.trim() || isProcessing}
                        >
                          <Send className="w-5 h-5" />
                        </RippleButton>
                      </div>
                    </div>
                    <div className="flex items-center justify-between mt-2 text-xs text-gray-500">
                      <span>Press Enter to send, Shift+Enter for new line</span>
                      <span>Max Agent is ready to help</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Analytics Tab */}
          {activeTab === 'analytics' && (
            <ScrollArea className="h-full p-6">
              <div className="max-w-6xl mx-auto">
                <div className="grid grid-cols-4 gap-4 mb-6">
                  <div className="bg-gray-50 border border-gray-200 rounded-xl p-5">
                    <div className="text-3xl font-bold text-gray-900 mb-1">24</div>
                    <div className="text-sm text-gray-500">Total Tasks</div>
                    <div className="text-xs text-green-600 mt-2 flex items-center gap-1">
                      <TrendingUp className="w-3 h-3" />
                      +12% this week
                    </div>
                  </div>
                  <div className="bg-gray-50 border border-gray-200 rounded-xl p-5">
                    <div className="text-3xl font-bold text-gray-900 mb-1">94%</div>
                    <div className="text-sm text-gray-500">Success Rate</div>
                    <div className="text-xs text-green-600 mt-2 flex items-center gap-1">
                      <TrendingUp className="w-3 h-3" />
                      +5% this week
                    </div>
                  </div>
                  <div className="bg-gray-50 border border-gray-200 rounded-xl p-5">
                    <div className="text-3xl font-bold text-gray-900 mb-1">156</div>
                    <div className="text-sm text-gray-500">Hours Saved</div>
                    <div className="text-xs text-green-600 mt-2 flex items-center gap-1">
                      <TrendingUp className="w-3 h-3" />
                      +23% this week
                    </div>
                  </div>
                  <div className="bg-gray-50 border border-gray-200 rounded-xl p-5">
                    <div className="text-3xl font-bold text-gray-900 mb-1">6</div>
                    <div className="text-sm text-gray-500">Active Agents</div>
                    <div className="text-xs text-gray-600 mt-2">All systems operational</div>
                  </div>
                </div>

                {/* Charts */}
                <div className="grid grid-cols-2 gap-6">
                  <div className="bg-gray-50 border border-gray-200 rounded-xl p-5">
                    <h3 className="font-medium text-gray-900 mb-4">Task Completion</h3>
                    <div className="h-48 flex items-end justify-between gap-2">
                      {[65, 80, 45, 90, 75, 85, 70].map((height, i) => (
                        <div
                          key={i}
                          className="flex-1 bg-[#FFD700]/20 rounded-t-lg relative group"
                          style={{ height: `${height}%` }}
                        >
                          <div
                            className="absolute bottom-0 left-0 right-0 bg-[#FFD700] rounded-t-lg transition-all group-hover:bg-[#FFA500]"
                            style={{ height: `${height}%` }}
                          />
                        </div>
                      ))}
                    </div>
                    <div className="flex justify-between mt-2 text-xs text-gray-500">
                      <span>Mon</span>
                      <span>Tue</span>
                      <span>Wed</span>
                      <span>Thu</span>
                      <span>Fri</span>
                      <span>Sat</span>
                      <span>Sun</span>
                    </div>
                  </div>

                  <div className="bg-gray-50 border border-gray-200 rounded-xl p-5">
                    <h3 className="font-medium text-gray-900 mb-4">Agent Usage</h3>
                    <div className="space-y-4">
                      {agents.map((agent) => (
                        <div key={agent.id} className="flex items-center gap-3">
                          <div
                            className="w-8 h-8 rounded-lg flex items-center justify-center"
                            style={{ backgroundColor: `${agent.color}20` }}
                          >
                            <agent.icon className="w-4 h-4" style={{ color: agent.color }} />
                          </div>
                          <div className="flex-1">
                            <div className="flex items-center justify-between mb-1">
                              <span className="text-sm text-gray-900">{agent.name}</span>
                              <span className="text-xs text-gray-500">{Math.floor(Math.random() * 50 + 10)} tasks</span>
                            </div>
                            <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                              <div
                                className="h-full rounded-full"
                                style={{ width: `${Math.random() * 60 + 20}%`, backgroundColor: agent.color }}
                              />
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </ScrollArea>
          )}
        </div>
      </div>
    </div>
  );
}
