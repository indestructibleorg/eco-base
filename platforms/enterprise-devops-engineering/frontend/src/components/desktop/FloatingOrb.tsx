import { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Sparkles,
  MessageSquare,
  Image,
  Mic,
  FileText,
  Code,
  Languages,
  Wand2,
  X,
  Minimize2,
  Maximize2,
  Send,
  Paperclip,
  ChevronRight,
  Bot,
  Users,
  Zap,
  Type,
} from 'lucide-react';
import { RippleButton } from '@/components/ui/ripple-button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';

// Types
interface OrbMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  type: 'text' | 'image' | 'audio' | 'video';
  timestamp: Date;
  attachments?: OrbAttachment[];
}

interface OrbAttachment {
  type: 'image' | 'audio' | 'file';
  url?: string;
  name: string;
}

interface QuickAction {
  id: string;
  label: string;
  icon: React.ElementType;
  color: string;
}

interface MCPAgent {
  id: string;
  name: string;
  avatar: string;
  role: string;
  status: 'idle' | 'working' | 'completed';
}

const quickActions: QuickAction[] = [
  { id: 'write', label: 'AI写作', icon: Type, color: '#1E88E5' },
  { id: 'translate', label: '翻译', icon: Languages, color: '#10B981' },
  { id: 'code', label: '编程', icon: Code, color: '#FFD700' },
  { id: 'image', label: '生图', icon: Image, color: '#8B5CF6' },
  { id: 'doc', label: '文档', icon: FileText, color: '#06B6D4' },
  { id: 'voice', label: '语音', icon: Mic, color: '#EC4899' },
];

const mcpAgents: MCPAgent[] = [
  { id: 'writer', name: '写作助手', avatar: '✍️', role: '内容创作', status: 'idle' },
  { id: 'coder', name: '代码专家', avatar: '💻', role: '编程开发', status: 'idle' },
  { id: 'analyst', name: '数据分析师', avatar: '📊', role: '数据分析', status: 'idle' },
  { id: 'designer', name: '设计师', avatar: '🎨', role: '视觉设计', status: 'idle' },
];

export function FloatingOrb() {
  const [isOpen, setIsOpen] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);
  const [activeMode, setActiveMode] = useState<'chat' | 'mcp' | 'tools'>('chat');
  const [messages, setMessages] = useState<OrbMessage[]>([]);
  const [input, setInput] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [activeAgents, setActiveAgents] = useState<string[]>([]);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  
  const orbRef = useRef<HTMLDivElement>(null);
  const dragStartRef = useRef({ x: 0, y: 0 });
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Handle drag start
  const handleDragStart = (e: React.MouseEvent | React.TouchEvent) => {
    if (isOpen) return;
    setIsDragging(true);
    const clientX = 'touches' in e ? e.touches[0].clientX : e.clientX;
    const clientY = 'touches' in e ? e.touches[0].clientY : e.clientY;
    dragStartRef.current = { x: clientX - position.x, y: clientY - position.y };
  };

  // Handle drag move
  const handleDragMove = (e: React.MouseEvent | React.TouchEvent) => {
    if (!isDragging) return;
    const clientX = 'touches' in e ? e.touches[0].clientX : e.clientX;
    const clientY = 'touches' in e ? e.touches[0].clientY : e.clientY;
    setPosition({
      x: clientX - dragStartRef.current.x,
      y: clientY - dragStartRef.current.y,
    });
  };

  // Handle drag end
  const handleDragEnd = () => {
    setIsDragging(false);
    // Snap to nearest edge
    const windowWidth = window.innerWidth;
    const windowHeight = window.innerHeight;
    const orbSize = 56;
    
    let newX = position.x;
    let newY = position.y;
    
    // Snap horizontally
    if (position.x < windowWidth / 2) {
      newX = 20;
    } else {
      newX = windowWidth - orbSize - 20;
    }
    
    // Keep within vertical bounds
    newY = Math.max(20, Math.min(windowHeight - orbSize - 20, position.y));
    
    setPosition({ x: newX, y: newY });
  };

  // Send message
  const sendMessage = () => {
    if (!input.trim()) return;

    const userMessage: OrbMessage = {
      id: `msg-${Date.now()}`,
      role: 'user',
      content: input,
      type: 'text',
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');

    // Simulate AI response
    setTimeout(() => {
      const responses = [
        '我来帮您处理这个任务！',
        '这是一个很好的问题，让我为您分析...',
        '我已经理解了您的需求，正在处理中...',
        '根据您的描述，我建议...',
      ];
      const assistantMessage: OrbMessage = {
        id: `msg-${Date.now() + 1}`,
        role: 'assistant',
        content: responses[Math.floor(Math.random() * responses.length)],
        type: 'text',
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, assistantMessage]);
    }, 1000);
  };

  // Toggle MCP agent
  const toggleAgent = (agentId: string) => {
    setActiveAgents((prev) =>
      prev.includes(agentId)
        ? prev.filter((id) => id !== agentId)
        : [...prev, agentId]
    );
  };

  // Handle voice recording
  const toggleRecording = () => {
    if (isRecording) {
      setIsRecording(false);
      // Add voice message
      const voiceMessage: OrbMessage = {
        id: `msg-${Date.now()}`,
        role: 'user',
        content: '🎤 语音消息',
        type: 'audio',
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, voiceMessage]);
    } else {
      setIsRecording(true);
    }
  };

  return (
    <>
      {/* Floating Orb */}
      <AnimatePresence>
        {!isOpen && (
          <motion.div
            ref={orbRef}
            initial={{ scale: 0, opacity: 0 }}
            animate={{ 
              scale: 1, 
              opacity: 1,
              x: position.x,
              y: position.y,
            }}
            exit={{ scale: 0, opacity: 0 }}
            className={cn(
              'fixed z-50 cursor-move',
              isDragging && 'cursor-grabbing'
            )}
            style={{ right: position.x === 0 ? 20 : undefined, bottom: 100 }}
            onMouseDown={handleDragStart}
            onMouseMove={handleDragMove}
            onMouseUp={handleDragEnd}
            onMouseLeave={handleDragEnd}
            onTouchStart={handleDragStart}
            onTouchMove={handleDragMove}
            onTouchEnd={handleDragEnd}
          >
            <motion.button
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => setIsOpen(true)}
              className="w-14 h-14 rounded-full bg-gradient-to-br from-[#FFD700] to-[#FFA500] shadow-lg shadow-[#FFD700]/30 flex items-center justify-center"
            >
              <Sparkles className="w-6 h-6 text-black" />
            </motion.button>
            
            {/* Pulse Effect */}
            <div className="absolute inset-0 rounded-full bg-[#FFD700] animate-ping opacity-20" />
          </motion.div>
        )}
      </AnimatePresence>

      {/* Expanded Panel */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, scale: 0.9, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.9, y: 20 }}
            className={cn(
              'fixed z-50 bg-white border border-gray-200 rounded-2xl shadow-2xl overflow-hidden',
              isExpanded ? 'w-[800px] h-[600px]' : 'w-[400px] h-[500px]'
            )}
            style={{ right: 20, bottom: 100 }}
          >
            {/* Header */}
            <div className="h-14 border-b border-gray-200 flex items-center justify-between px-4">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-[#FFD700] to-[#FFA500] flex items-center justify-center">
                  <Sparkles className="w-4 h-4 text-black" />
                </div>
                <div>
                  <span className="font-medium text-gray-900">AI 助手</span>
                  {activeAgents.length > 0 && (
                    <Badge className="ml-2 bg-[#FFD700]/20 text-[#FFD700] border-0 text-[10px]">
                      {activeAgents.length} 个代理协作中
                    </Badge>
                  )}
                </div>
              </div>
              <div className="flex items-center gap-1">
                <RippleButton
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8 text-gray-500"
                  onClick={() => setIsExpanded(!isExpanded)}
                >
                  {isExpanded ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
                </RippleButton>
                <RippleButton
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8 text-gray-500"
                  onClick={() => setIsOpen(false)}
                >
                  <X className="w-4 h-4" />
                </RippleButton>
              </div>
            </div>

            {/* Mode Tabs */}
            <div className="flex items-center gap-1 px-4 py-2 border-b border-gray-200">
              {[
                { id: 'chat', label: '对话', icon: MessageSquare },
                { id: 'mcp', label: 'MCP协作', icon: Users },
                { id: 'tools', label: '工具', icon: Wand2 },
              ].map((mode) => (
                <button
                  key={mode.id}
                  onClick={() => setActiveMode(mode.id as any)}
                  className={cn(
                    'flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm transition-colors',
                    activeMode === mode.id
                      ? 'bg-[#FFD700]/20 text-[#FFD700]'
                      : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
                  )}
                >
                  <mode.icon className="w-4 h-4" />
                  {mode.label}
                </button>
              ))}
            </div>

            {/* Content */}
            <div className="flex-1 overflow-hidden" style={{ height: 'calc(100% - 120px)' }}>
              {/* Chat Mode */}
              {activeMode === 'chat' && (
                <div className="h-full flex flex-col">
                  <ScrollArea className="flex-1 p-4">
                    {messages.length === 0 ? (
                      <div className="text-center py-8">
                        <Sparkles className="w-12 h-12 text-[#FFD700] mx-auto mb-4" />
                        <h3 className="text-gray-900 font-medium mb-2">有什么可以帮您的？</h3>
                        <p className="text-sm text-gray-500 mb-4">支持文本、语音、图片等多种输入方式</p>
                        
                        {/* Quick Actions */}
                        <div className="grid grid-cols-3 gap-2 px-4">
                          {quickActions.slice(0, 6).map((action) => (
                            <button
                              key={action.id}
                              onClick={() => setInput(action.label)}
                              className="p-3 rounded-xl bg-gray-50 border border-gray-200 hover:border-[#FFD700]/50 transition-colors text-left"
                            >
                              <action.icon className="w-5 h-5 mb-2" style={{ color: action.color }} />
                              <span className="text-xs text-gray-600">{action.label}</span>
                            </button>
                          ))}
                        </div>
                      </div>
                    ) : (
                      <div className="space-y-4">
                        {messages.map((message) => (
                          <div
                            key={message.id}
                            className={cn(
                              'flex gap-3',
                              message.role === 'user' && 'flex-row-reverse'
                            )}
                          >
                            <div
                              className={cn(
                                'w-8 h-8 rounded-lg flex items-center justify-center shrink-0',
                                message.role === 'user'
                                  ? 'bg-[#FFD700]'
                                  : 'bg-gradient-to-br from-[#FFD700] to-[#FFA500]'
                              )}
                            >
                              {message.role === 'user' ? (
                                <span className="text-black text-sm">U</span>
                              ) : (
                                <Sparkles className="w-4 h-4 text-black" />
                              )}
                            </div>
                            <div
                              className={cn(
                                'max-w-[80%] rounded-xl px-3 py-2 text-sm',
                                message.role === 'user'
                                  ? 'bg-[#FFD700] text-black'
                                  : 'bg-gray-50 border border-gray-200 text-gray-800'
                              )}
                            >
                              {message.content}
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </ScrollArea>

                  {/* Input Area */}
                  <div className="p-4 border-t border-gray-200">
                    <div className="relative bg-gray-50 border border-gray-200 rounded-xl overflow-hidden focus-within:border-[#FFD700]">
                      <textarea
                        ref={inputRef}
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter' && !e.shiftKey) {
                            e.preventDefault();
                            sendMessage();
                          }
                        }}
                        placeholder="输入消息..."
                        className="w-full min-h-[60px] max-h-[120px] bg-transparent border-0 text-gray-900 placeholder:text-gray-500 p-3 pr-24 resize-none focus:outline-none"
                        rows={1}
                      />
                      <div className="absolute right-2 bottom-2 flex items-center gap-1">
                        <RippleButton
                          variant="ghost"
                          size="icon"
                          className="h-8 w-8 text-gray-500"
                        >
                          <Paperclip className="w-4 h-4" />
                        </RippleButton>
                        <RippleButton
                          variant="ghost"
                          size="icon"
                          className={cn(
                            'h-8 w-8',
                            isRecording && 'text-red-500 animate-pulse'
                          )}
                          onClick={toggleRecording}
                        >
                          <Mic className="w-4 h-4" />
                        </RippleButton>
                        <RippleButton
                          size="icon"
                          className="h-8 w-8 bg-[#FFD700] text-black hover:bg-[#FFC700]"
                          onClick={sendMessage}
                          disabled={!input.trim()}
                        >
                          <Send className="w-4 h-4" />
                        </RippleButton>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* MCP Mode */}
              {activeMode === 'mcp' && (
                <ScrollArea className="h-full p-4">
                  <div className="space-y-4">
                    <div className="bg-gradient-to-r from-[#FFD700]/20 to-[#FFA500]/20 rounded-xl p-4 border border-[#FFD700]/30">
                      <div className="flex items-center gap-2 mb-2">
                        <Users className="w-5 h-5 text-[#FFD700]" />
                        <span className="font-medium text-gray-900">MCP 多智能体协作</span>
                      </div>
                      <p className="text-sm text-gray-600">
                        选择多个 AI 代理协同工作，共同解决复杂问题
                      </p>
                    </div>

                    <div className="space-y-2">
                      {mcpAgents.map((agent) => (
                        <div
                          key={agent.id}
                          onClick={() => toggleAgent(agent.id)}
                          className={cn(
                            'flex items-center gap-3 p-3 rounded-xl border cursor-pointer transition-all',
                            activeAgents.includes(agent.id)
                              ? 'bg-[#FFD700]/10 border-[#FFD700]/50'
                              : 'bg-gray-50 border-gray-200 hover:border-gray-300'
                          )}
                        >
                          <div className="w-10 h-10 rounded-lg bg-gray-100 flex items-center justify-center text-2xl">
                            {agent.avatar}
                          </div>
                          <div className="flex-1">
                            <div className="flex items-center gap-2">
                              <span className="font-medium text-gray-900">{agent.name}</span>
                              {activeAgents.includes(agent.id) && (
                                <Badge className="bg-[#FFD700]/20 text-[#FFD700] border-0 text-[10px]">
                                  已激活
                                </Badge>
                              )}
                            </div>
                            <span className="text-xs text-gray-500">{agent.role}</span>
                          </div>
                          <div
                            className={cn(
                              'w-5 h-5 rounded border flex items-center justify-center',
                              activeAgents.includes(agent.id)
                                ? 'bg-[#FFD700] border-[#FFD700]'
                                : 'border-gray-400'
                            )}
                          >
                            {activeAgents.includes(agent.id) && (
                              <Zap className="w-3 h-3 text-black" />
                            )}
                          </div>
                        </div>
                      ))}
                    </div>

                    {activeAgents.length > 0 && (
                      <RippleButton className="w-full bg-[#FFD700] text-black hover:bg-[#FFC700] gap-2">
                        <Bot className="w-4 h-4" />
                        启动 MCP 协作 ({activeAgents.length} 个代理)
                      </RippleButton>
                    )}
                  </div>
                </ScrollArea>
              )}

              {/* Tools Mode */}
              {activeMode === 'tools' && (
                <ScrollArea className="h-full p-4">
                  <div className="grid grid-cols-2 gap-3">
                    {quickActions.map((action) => (
                      <button
                        key={action.id}
                        className="p-4 rounded-xl bg-gray-50 border border-gray-200 hover:border-[#FFD700]/50 transition-all text-left group"
                      >
                        <div
                          className="w-10 h-10 rounded-lg flex items-center justify-center mb-3 transition-colors"
                          style={{ backgroundColor: `${action.color}20` }}
                        >
                          <action.icon className="w-5 h-5" style={{ color: action.color }} />
                        </div>
                        <span className="font-medium text-gray-900 group-hover:text-[#FFD700] transition-colors">
                          {action.label}
                        </span>
                        <ChevronRight className="w-4 h-4 text-gray-500 mt-2 group-hover:translate-x-1 transition-transform" />
                      </button>
                    ))}
                  </div>
                </ScrollArea>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
