import { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Sparkles,
  MessageSquare,
  Image,
  Mic,
  FileText,
  FileVideo,
  FileAudio,
  Search,
  Wand2,
  Settings,
  ChevronDown,
  Plus,
  X,
  Send,
  Paperclip,
  Share2,
  Type,
  Languages,
  PenTool,
  GitBranch,
  Cpu,
  BookOpen,
  Headphones,
  BarChart3,
  Check,
  Loader2,
  History,
  Bookmark,
  Trash2,
  Filter,
  Subtitles,
  Crop,
  Palette,
  Scissors,
  FilePlus,
  FileMinus,
  FileSignature,
  Bot,
  Code,
} from 'lucide-react';
import { RippleButton } from '@/components/ui/ripple-button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';

// Types
interface AIModel {
  id: string;
  name: string;
  provider: string;
  icon: string;
  color: string;
  capabilities: string[];
  isPremium: boolean;
}

interface ChatSession {
  id: string;
  title: string;
  modelId: string;
  messages: ChatMessage[];
  createdAt: Date;
  updatedAt: Date;
}

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  model?: string;
  timestamp: Date;
  attachments?: Attachment[];
  isStreaming?: boolean;
}

interface Attachment {
  type: 'image' | 'pdf' | 'audio' | 'video' | 'text';
  name: string;
  url?: string;
  content?: string;
  size?: string;
}

interface AITool {
  id: string;
  name: string;
  description: string;
  icon: React.ElementType;
  color: string;
  category: 'document' | 'image' | 'audio' | 'video' | 'productivity';
}



// AI Models
const aiModels: AIModel[] = [
  {
    id: 'gpt-4o',
    name: 'GPT-4o',
    provider: 'OpenAI',
    icon: '🤖',
    color: '#10A37F',
    capabilities: ['text', 'image', 'code', 'analysis'],
    isPremium: true,
  },
  {
    id: 'claude-3-7',
    name: 'Claude 3.7',
    provider: 'Anthropic',
    icon: '🧠',
    color: '#D97757',
    capabilities: ['text', 'image', 'code', 'long-context'],
    isPremium: true,
  },
  {
    id: 'gemini-1-5',
    name: 'Gemini 1.5',
    provider: 'Google',
    icon: '✨',
    color: '#4285F4',
    capabilities: ['text', 'image', 'video', 'audio'],
    isPremium: true,
  },
  {
    id: 'gpt-4o-mini',
    name: 'GPT-4o Mini',
    provider: 'OpenAI',
    icon: '⚡',
    color: '#10A37F',
    capabilities: ['text', 'image'],
    isPremium: false,
  },
  {
    id: 'claude-3-haiku',
    name: 'Claude 3 Haiku',
    provider: 'Anthropic',
    icon: '🌸',
    color: '#D97757',
    capabilities: ['text', 'image'],
    isPremium: false,
  },
];

// AI Tools
const aiTools: AITool[] = [
  // Document Tools
  { id: 'pdf-merge', name: 'PDF 合并', description: '合并多个PDF文件', icon: FilePlus, color: '#EF4444', category: 'document' },
  { id: 'pdf-split', name: 'PDF 分割', description: '分割PDF页面', icon: FileMinus, color: '#EF4444', category: 'document' },
  { id: 'pdf-convert', name: 'PDF 转换', description: 'PDF转Word/Excel', icon: FileText, color: '#EF4444', category: 'document' },
  { id: 'pdf-sign', name: 'PDF 签名', description: '电子签名PDF', icon: FileSignature, color: '#EF4444', category: 'document' },
  
  // Image Tools
  { id: 'image-remove-bg', name: '去背景', description: 'AI智能去背景', icon: Scissors, color: '#8B5CF6', category: 'image' },
  { id: 'image-enhance', name: '画质增强', description: 'AI图像增强', icon: Sparkles, color: '#8B5CF6', category: 'image' },
  { id: 'image-crop', name: '智能裁剪', description: 'AI智能裁剪', icon: Crop, color: '#8B5CF6', category: 'image' },
  { id: 'image-style', name: '风格转换', description: '艺术风格转换', icon: Palette, color: '#8B5CF6', category: 'image' },
  
  // Audio Tools
  { id: 'audio-transcribe', name: '语音转文字', description: 'AI语音识别', icon: Mic, color: '#10B981', category: 'audio' },
  { id: 'audio-summarize', name: '音频摘要', description: 'AI音频摘要', icon: FileAudio, color: '#10B981', category: 'audio' },
  { id: 'podcast-generate', name: 'AI播客', description: '生成AI播客', icon: Headphones, color: '#10B981', category: 'audio' },
  
  // Video Tools
  { id: 'video-subtitle', name: '自动生成字幕', description: 'AI视频字幕', icon: Subtitles, color: '#F59E0B', category: 'video' },
  { id: 'video-summarize', name: '视频摘要', description: 'AI视频摘要', icon: FileVideo, color: '#F59E0B', category: 'video' },
  
  // Productivity Tools
  { id: 'mindmap', name: '心智图', description: 'AI生成心智图', icon: GitBranch, color: '#06B6D4', category: 'productivity' },
  { id: 'smart-search', name: '智慧搜索', description: 'AI网络搜索', icon: Search, color: '#06B6D4', category: 'productivity' },
  { id: 'text-summary', name: '文章摘要', description: 'AI文章摘要', icon: FileText, color: '#06B6D4', category: 'productivity' },
  { id: 'translation', name: '智能翻译', description: '多语言翻译', icon: Languages, color: '#06B6D4', category: 'productivity' },
];

// Mock Chat Sessions
const mockSessions: ChatSession[] = [
  {
    id: 'session-1',
    title: 'Python数据分析讨论',
    modelId: 'gpt-4o',
    messages: [
      {
        id: 'msg-1',
        role: 'user',
        content: '帮我分析这个销售数据集',
        timestamp: new Date(Date.now() - 1000 * 60 * 30),
      },
      {
        id: 'msg-2',
        role: 'assistant',
        content: '我来帮您分析销售数据。首先，我需要了解数据的结构和您希望分析的重点。',
        model: 'GPT-4o',
        timestamp: new Date(Date.now() - 1000 * 60 * 29),
      },
    ],
    createdAt: new Date(Date.now() - 1000 * 60 * 30),
    updatedAt: new Date(Date.now() - 1000 * 60 * 29),
  },
];

export function AIWorkstation() {
  const [activeTab, setActiveTab] = useState('chat');
  const [selectedModel, setSelectedModel] = useState<AIModel>(aiModels[0]);
  const [showModelSelector, setShowModelSelector] = useState(false);
  const [sessions, setSessions] = useState<ChatSession[]>(mockSessions);
  const [activeSessionId, setActiveSessionId] = useState<string>('session-1');
  const [input, setInput] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const [selectedToolCategory, setSelectedToolCategory] = useState<string>('all');

  const activeSession = sessions.find((s) => s.id === activeSessionId);

  // Send message
  const sendMessage = useCallback(async () => {
    if (!input.trim() && attachments.length === 0) return;
    if (!activeSession) return;

    const userMessage: ChatMessage = {
      id: `msg-${Date.now()}`,
      role: 'user',
      content: input,
      timestamp: new Date(),
      attachments: attachments.length > 0 ? attachments : undefined,
    };

    setSessions((prev) =>
      prev.map((s) =>
        s.id === activeSessionId
          ? { ...s, messages: [...s.messages, userMessage], updatedAt: new Date() }
          : s
      )
    );
    setInput('');
    setAttachments([]);
    setIsGenerating(true);

    // Simulate AI response
    setTimeout(() => {
      const responses = [
        `使用 ${selectedModel.name} 为您分析...`,
        '这是一个很好的问题，让我为您详细解答。',
        '根据您的需求，我建议采用以下方案：',
        '我来帮您处理这个任务，请稍等片刻。',
      ];
      
      const assistantMessage: ChatMessage = {
        id: `msg-${Date.now() + 1}`,
        role: 'assistant',
        content: responses[Math.floor(Math.random() * responses.length)],
        model: selectedModel.name,
        timestamp: new Date(),
      };

      setSessions((prev) =>
        prev.map((s) =>
          s.id === activeSessionId
            ? { ...s, messages: [...s.messages, assistantMessage], updatedAt: new Date() }
            : s
        )
      );
      setIsGenerating(false);
    }, 1500);
  }, [input, attachments, activeSessionId, activeSession, selectedModel]);

  // Create new session
  const createNewSession = () => {
    const newSession: ChatSession = {
      id: `session-${Date.now()}`,
      title: '新对话',
      modelId: selectedModel.id,
      messages: [],
      createdAt: new Date(),
      updatedAt: new Date(),
    };
    setSessions((prev) => [newSession, ...prev]);
    setActiveSessionId(newSession.id);
  };

  // Filter tools by category
  const filteredTools = selectedToolCategory === 'all'
    ? aiTools
    : aiTools.filter((t) => t.category === selectedToolCategory);

  return (
    <div className="h-full flex bg-white">
      {/* Sidebar */}
      <div className="w-72 border-r border-gray-200 flex flex-col bg-white">
        {/* Header */}
        <div className="h-16 border-b border-gray-200 flex items-center justify-between px-4">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-[#FFD700] to-[#FFA500] flex items-center justify-center">
              <Sparkles className="w-4 h-4 text-black" />
            </div>
            <span className="font-semibold text-gray-900">AI 工作台</span>
          </div>
          <RippleButton
            variant="ghost"
            size="icon"
            className="h-8 w-8 text-gray-500"
            onClick={createNewSession}
          >
            <Plus className="w-4 h-4" />
          </RippleButton>
        </div>

        {/* Navigation */}
        <div className="p-2">
          <nav className="space-y-1">
            {[
              { id: 'chat', label: 'AI 对话', icon: MessageSquare },
              { id: 'tools', label: 'AI 工具箱', icon: Wand2 },
              { id: 'agents', label: '智能体', icon: Bot },
              { id: 'history', label: '历史记录', icon: History },
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

        {/* Sessions List */}
        {activeTab === 'chat' && (
          <>
            <div className="px-4 py-2">
              <span className="text-xs text-gray-500 uppercase tracking-wider">最近对话</span>
            </div>
            <ScrollArea className="flex-1 px-2">
              <div className="space-y-1">
                {sessions.map((session) => (
                  <button
                    key={session.id}
                    onClick={() => setActiveSessionId(session.id)}
                    className={cn(
                      'w-full p-3 rounded-lg text-left transition-colors',
                      activeSessionId === session.id
                        ? 'bg-[#FFD700]/10 border border-[#FFD700]/30'
                        : 'hover:bg-gray-100'
                    )}
                  >
                    <div className="flex items-center gap-2 mb-1">
                      <MessageSquare className="w-4 h-4 text-gray-500" />
                      <span className="text-sm text-gray-900 truncate">{session.title}</span>
                    </div>
                    <div className="flex items-center gap-2 text-xs text-gray-500">
                      <span>{aiModels.find((m) => m.id === session.modelId)?.name}</span>
                      <span>•</span>
                      <span>{session.updatedAt.toLocaleDateString()}</span>
                    </div>
                  </button>
                ))}
              </div>
            </ScrollArea>
          </>
        )}

        {/* Model Selector */}
        <div className="p-4 border-t border-gray-200">
          <div className="relative">
            <button
              onClick={() => setShowModelSelector(!showModelSelector)}
              className="w-full flex items-center gap-2 p-2 rounded-lg bg-gray-50 border border-gray-200 hover:border-[#FFD700]/50 transition-colors"
            >
              <span className="text-lg">{selectedModel.icon}</span>
              <div className="flex-1 text-left">
                <div className="text-sm text-gray-900">{selectedModel.name}</div>
                <div className="text-xs text-gray-500">{selectedModel.provider}</div>
              </div>
              <ChevronDown className="w-4 h-4 text-gray-500" />
            </button>

            <AnimatePresence>
              {showModelSelector && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: 10 }}
                  className="absolute bottom-full left-0 right-0 mb-2 bg-white border border-gray-200 rounded-xl overflow-hidden shadow-xl"
                >
                  <div className="p-2">
                    <div className="text-xs text-gray-500 px-2 py-1">Premium Models</div>
                    {aiModels.filter((m) => m.isPremium).map((model) => (
                      <button
                        key={model.id}
                        onClick={() => {
                          setSelectedModel(model);
                          setShowModelSelector(false);
                        }}
                        className={cn(
                          'w-full flex items-center gap-2 p-2 rounded-lg transition-colors',
                          selectedModel.id === model.id
                            ? 'bg-[#FFD700]/10 text-[#FFD700]'
                            : 'hover:bg-gray-100 text-gray-600'
                        )}
                      >
                        <span className="text-lg">{model.icon}</span>
                        <div className="flex-1 text-left">
                          <div className="text-sm">{model.name}</div>
                          <div className="text-xs text-gray-500">{model.provider}</div>
                        </div>
                        {selectedModel.id === model.id && <Check className="w-4 h-4" />}
                      </button>
                    ))}
                    <div className="text-xs text-gray-500 px-2 py-1 mt-2">Standard Models</div>
                    {aiModels.filter((m) => !m.isPremium).map((model) => (
                      <button
                        key={model.id}
                        onClick={() => {
                          setSelectedModel(model);
                          setShowModelSelector(false);
                        }}
                        className={cn(
                          'w-full flex items-center gap-2 p-2 rounded-lg transition-colors',
                          selectedModel.id === model.id
                            ? 'bg-[#FFD700]/10 text-[#FFD700]'
                            : 'hover:bg-gray-100 text-gray-600'
                        )}
                      >
                        <span className="text-lg">{model.icon}</span>
                        <div className="flex-1 text-left">
                          <div className="text-sm">{model.name}</div>
                          <div className="text-xs text-gray-500">{model.provider}</div>
                        </div>
                        {selectedModel.id === model.id && <Check className="w-4 h-4" />}
                      </button>
                    ))}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Chat Tab */}
        {activeTab === 'chat' && (
          <>
            {/* Header */}
            <div className="h-14 border-b border-gray-200 flex items-center justify-between px-4">
              <div className="flex items-center gap-3">
                <h2 className="font-semibold text-gray-900">
                  {activeSession?.title || '新对话'}
                </h2>
                <Badge className="bg-gray-100 text-gray-600 border-0 text-[10px]">
                  {selectedModel.name}
                </Badge>
              </div>
              <div className="flex items-center gap-2">
                <RippleButton variant="ghost" size="icon" className="h-8 w-8 text-gray-500">
                  <Share2 className="w-4 h-4" />
                </RippleButton>
                <RippleButton variant="ghost" size="icon" className="h-8 w-8 text-gray-500">
                  <Settings className="w-4 h-4" />
                </RippleButton>
              </div>
            </div>

            {/* Messages */}
            <ScrollArea className="flex-1 p-4">
              <div className="max-w-3xl mx-auto space-y-4">
                {activeSession?.messages.length === 0 && (
                  <div className="text-center py-12">
                    <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-[#FFD700] to-[#FFA500] flex items-center justify-center mx-auto mb-4">
                      <Sparkles className="w-8 h-8 text-black" />
                    </div>
                    <h3 className="text-xl font-semibold text-gray-900 mb-2">
                      有什么可以帮您的？
                    </h3>
                    <p className="text-gray-500 mb-6">
                      支持文字、图片、PDF、音频等多种格式
                    </p>
                    <div className="grid grid-cols-2 gap-3 max-w-md mx-auto">
                      {[
                        '帮我写一段Python代码',
                        '分析这张图片',
                        '总结这个PDF文档',
                        '翻译这段文字',
                      ].map((suggestion, i) => (
                        <button
                          key={i}
                          onClick={() => setInput(suggestion)}
                          className="p-3 rounded-xl bg-gray-50 border border-gray-200 hover:border-[#FFD700]/50 transition-colors text-left"
                        >
                          <span className="text-sm text-gray-600">{suggestion}</span>
                        </button>
                      ))}
                    </div>
                  </div>
                )}
                {activeSession?.messages.map((message) => (
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
                        <Sparkles className="w-5 h-5 text-black" />
                      )}
                    </div>
                    <div className={cn('flex-1 max-w-[80%]', message.role === 'user' && 'text-right')}>
                      <div
                        className={cn(
                          'inline-block rounded-xl px-4 py-3 text-left',
                          message.role === 'user'
                            ? 'bg-[#FFD700] text-black'
                            : 'bg-gray-50 border border-gray-200 text-gray-800'
                        )}
                      >
                        {message.model && (
                          <div className="flex items-center gap-1 mb-2 text-xs text-gray-500">
                            <Cpu className="w-3 h-3" />
                            {message.model}
                          </div>
                        )}
                        <p className="whitespace-pre-wrap">{message.content}</p>
                        {message.attachments && (
                          <div className="mt-3 space-y-2">
                            {message.attachments.map((att, i) => (
                              <div
                                key={i}
                                className="flex items-center gap-2 p-2 bg-white rounded-lg"
                              >
                                {att.type === 'image' && <Image className="w-4 h-4 text-[#8B5CF6]" />}
                                {att.type === 'pdf' && <FileText className="w-4 h-4 text-[#EF4444]" />}
                                {att.type === 'audio' && <Mic className="w-4 h-4 text-[#10B981]" />}
                                <span className="text-sm text-gray-600">{att.name}</span>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                      <div className="mt-1 text-xs text-gray-500">
                        {message.timestamp.toLocaleTimeString()}
                      </div>
                    </div>
                  </div>
                ))}
                {isGenerating && (
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
                {/* Attachments */}
                {attachments.length > 0 && (
                  <div className="flex gap-2 mb-2 flex-wrap">
                    {attachments.map((att, i) => (
                      <div
                        key={i}
                        className="flex items-center gap-2 px-2 py-1 bg-gray-50 border border-gray-200 rounded-lg"
                      >
                        {att.type === 'image' && <Image className="w-4 h-4 text-[#8B5CF6]" />}
                        <span className="text-xs text-gray-600">{att.name}</span>
                        <button
                          onClick={() => setAttachments((prev) => prev.filter((_, idx) => idx !== i))}
                          className="text-gray-500 hover:text-gray-900"
                        >
                          <X className="w-3 h-3" />
                        </button>
                      </div>
                    ))}
                  </div>
                )}
                
                <div className="relative bg-gray-50 border border-gray-200 rounded-xl overflow-hidden focus-within:border-[#FFD700]">
                  <textarea
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        sendMessage();
                      }
                    }}
                    placeholder="输入消息... 支持 @提及文件"
                    className="w-full min-h-[60px] max-h-[150px] bg-transparent border-0 text-gray-900 placeholder:text-gray-500 p-3 pr-24 resize-none focus:outline-none"
                    rows={1}
                  />
                  <div className="absolute right-2 bottom-2 flex items-center gap-1">
                    <RippleButton variant="ghost" size="icon" className="h-8 w-8 text-gray-500">
                      <Paperclip className="w-4 h-4" />
                    </RippleButton>
                    <RippleButton variant="ghost" size="icon" className="h-8 w-8 text-gray-500">
                      <Mic className="w-4 h-4" />
                    </RippleButton>
                    <RippleButton
                      size="icon"
                      className="h-8 w-8 bg-[#FFD700] text-black"
                      onClick={sendMessage}
                      disabled={!input.trim() && attachments.length === 0}
                    >
                      <Send className="w-4 h-4" />
                    </RippleButton>
                  </div>
                </div>
                <div className="flex items-center justify-between mt-2 text-xs text-gray-500">
                  <span>Enter 发送，Shift+Enter 换行</span>
                  <span>支持多模态输入</span>
                </div>
              </div>
            </div>
          </>
        )}

        {/* Tools Tab */}
        {activeTab === 'tools' && (
          <ScrollArea className="flex-1 p-6">
            <div className="max-w-5xl mx-auto">
              {/* Category Filter */}
              <div className="flex items-center gap-2 mb-6">
                {[
                  { id: 'all', label: '全部' },
                  { id: 'document', label: '文档工具' },
                  { id: 'image', label: '图片工具' },
                  { id: 'audio', label: '音频工具' },
                  { id: 'video', label: '视频工具' },
                  { id: 'productivity', label: '效率工具' },
                ].map((cat) => (
                  <button
                    key={cat.id}
                    onClick={() => setSelectedToolCategory(cat.id)}
                    className={cn(
                      'px-4 py-2 rounded-lg text-sm transition-colors',
                      selectedToolCategory === cat.id
                        ? 'bg-[#FFD700] text-black'
                        : 'bg-gray-50 text-gray-600 hover:bg-gray-100'
                    )}
                  >
                    {cat.label}
                  </button>
                ))}
              </div>

              {/* Tools Grid */}
              <div className="grid grid-cols-4 gap-4">
                {filteredTools.map((tool) => (
                  <motion.div
                    key={tool.id}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    className="bg-gray-50 border border-gray-200 rounded-xl p-5 cursor-pointer hover:border-[#FFD700]/50 transition-colors"
                  >
                    <div
                      className="w-12 h-12 rounded-xl flex items-center justify-center mb-4"
                      style={{ backgroundColor: `${tool.color}20` }}
                    >
                      <tool.icon className="w-6 h-6" style={{ color: tool.color }} />
                    </div>
                    <h3 className="font-medium text-gray-900 mb-1">{tool.name}</h3>
                    <p className="text-sm text-gray-500">{tool.description}</p>
                  </motion.div>
                ))}
              </div>

              {/* Featured Section */}
              <div className="mt-8">
                <h3 className="font-semibold text-gray-900 mb-4">热门功能</h3>
                <div className="grid grid-cols-3 gap-4">
                  <div className="bg-gradient-to-br from-[#FFD700]/20 to-[#FFA500]/20 border border-[#FFD700]/30 rounded-xl p-5">
                    <div className="w-10 h-10 rounded-lg bg-[#FFD700]/20 flex items-center justify-center mb-3">
                      <FileText className="w-5 h-5 text-[#FFD700]" />
                    </div>
                    <h4 className="font-medium text-gray-900 mb-1">PDF 全能处理</h4>
                    <p className="text-sm text-gray-600">合并、分割、转换、签名一站式解决</p>
                  </div>
                  <div className="bg-gradient-to-br from-[#8B5CF6]/20 to-[#A78BFA]/20 border border-[#8B5CF6]/30 rounded-xl p-5">
                    <div className="w-10 h-10 rounded-lg bg-[#8B5CF6]/20 flex items-center justify-center mb-3">
                      <Image className="w-5 h-5 text-[#8B5CF6]" />
                    </div>
                    <h4 className="font-medium text-gray-900 mb-1">AI 图像处理</h4>
                    <p className="text-sm text-gray-600">去背景、增强、风格转换</p>
                  </div>
                  <div className="bg-gradient-to-br from-[#10B981]/20 to-[#34D399]/20 border border-[#10B981]/30 rounded-xl p-5">
                    <div className="w-10 h-10 rounded-lg bg-[#10B981]/20 flex items-center justify-center mb-3">
                      <Headphones className="w-5 h-5 text-[#10B981]" />
                    </div>
                    <h4 className="font-medium text-gray-900 mb-1">AI 播客生成</h4>
                    <p className="text-sm text-gray-600">将文章转换为播客音频</p>
                  </div>
                </div>
              </div>
            </div>
          </ScrollArea>
        )}

        {/* Agents Tab */}
        {activeTab === 'agents' && (
          <ScrollArea className="flex-1 p-6">
            <div className="max-w-5xl mx-auto">
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h2 className="text-xl font-semibold text-gray-900">我的智能体</h2>
                  <p className="text-sm text-gray-500">创建和定制您的专属 AI 助手</p>
                </div>
                <RippleButton className="bg-[#FFD700] text-black gap-2">
                  <Plus className="w-4 h-4" />
                  创建智能体
                </RippleButton>
              </div>

              <div className="grid grid-cols-3 gap-4">
                {[
                  { name: '代码助手', description: '专业的编程助手', icon: Code, color: '#FFD700', usage: '1.2k' },
                  { name: '写作专家', description: '创意写作与文案', icon: Type, color: '#3B82F6', usage: '856' },
                  { name: '翻译大师', description: '多语言精准翻译', icon: Languages, color: '#10B981', usage: '2.3k' },
                  { name: '数据分析', description: '数据洞察与可视化', icon: BarChart3, color: '#8B5CF6', usage: '432' },
                  { name: '学习导师', description: '个性化学习辅导', icon: BookOpen, color: '#F59E0B', usage: '678' },
                  { name: '设计顾问', description: 'UI/UX 设计建议', icon: PenTool, color: '#EC4899', usage: '345' },
                ].map((agent, i) => (
                  <div
                    key={i}
                    className="bg-gray-50 border border-gray-200 rounded-xl p-5 hover:border-[#FFD700]/50 transition-colors"
                  >
                    <div className="flex items-start justify-between mb-4">
                      <div
                        className="w-12 h-12 rounded-xl flex items-center justify-center"
                        style={{ backgroundColor: `${agent.color}20` }}
                      >
                        <agent.icon className="w-6 h-6" style={{ color: agent.color }} />
                      </div>
                      <Badge className="bg-gray-100 text-gray-600 border-0 text-[10px]">
                        {agent.usage} 使用
                      </Badge>
                    </div>
                    <h3 className="font-medium text-gray-900 mb-1">{agent.name}</h3>
                    <p className="text-sm text-gray-500 mb-4">{agent.description}</p>
                    <div className="flex gap-2">
                      <RippleButton variant="outline" size="sm" className="flex-1">
                        编辑
                      </RippleButton>
                      <RippleButton size="sm" className="flex-1 bg-[#FFD700] text-black">
                        使用
                      </RippleButton>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </ScrollArea>
        )}

        {/* History Tab */}
        {activeTab === 'history' && (
          <ScrollArea className="flex-1 p-6">
            <div className="max-w-4xl mx-auto">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-semibold text-gray-900">历史记录</h2>
                <div className="flex items-center gap-2">
                  <RippleButton variant="outline" size="sm" className="gap-2">
                    <Filter className="w-4 h-4" />
                    筛选
                  </RippleButton>
                  <RippleButton variant="outline" size="sm" className="gap-2">
                    <Trash2 className="w-4 h-4" />
                    清空
                  </RippleButton>
                </div>
              </div>

              <div className="space-y-2">
                {sessions.map((session) => (
                  <div
                    key={session.id}
                    className="flex items-center gap-4 p-4 bg-gray-50 border border-gray-200 rounded-xl hover:border-gray-300 transition-colors"
                  >
                    <MessageSquare className="w-5 h-5 text-gray-500" />
                    <div className="flex-1">
                      <h4 className="text-gray-900 font-medium">{session.title}</h4>
                      <div className="flex items-center gap-2 text-xs text-gray-500">
                        <span>{aiModels.find((m) => m.id === session.modelId)?.name}</span>
                        <span>•</span>
                        <span>{session.messages.length} 条消息</span>
                      </div>
                    </div>
                    <div className="text-xs text-gray-500">
                      {session.updatedAt.toLocaleDateString()}
                    </div>
                    <div className="flex items-center gap-1">
                      <RippleButton variant="ghost" size="icon" className="h-8 w-8">
                        <Bookmark className="w-4 h-4" />
                      </RippleButton>
                      <RippleButton variant="ghost" size="icon" className="h-8 w-8">
                        <Trash2 className="w-4 h-4" />
                      </RippleButton>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </ScrollArea>
        )}
      </div>
    </div>
  );
}
