import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Send, 
  Plus, 
  MessageSquare, 
  Trash2, 
  MoreHorizontal,
  Copy,
  RotateCcw,
  ThumbsUp,
  ThumbsDown,
  ChevronDown,
  Sparkles,
  Cpu,
  Paperclip,
  Mic,
  Image,
  FileCode,
  Play,
  Code,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Avatar, AvatarFallback } from '@/components/ui/avatar';

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { useChatStore, useAppStore } from '@/store/appStore';
import { cn } from '@/lib/utils';
import type { Message } from '@/types';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import type { Components } from 'react-markdown';

export function ChatInterface() {
  const { 
    chats, 
    currentChatId, 
    models, 
    selectedModel, 
    isStreaming,
    createChat, 
    deleteChat, 
    setCurrentChat, 
    addMessage, 
    updateMessage,
    setSelectedModel 
  } = useChatStore();
  const { isSidebarOpen, toggleSidebar } = useAppStore();
  const [input, setInput] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  const currentChat = chats.find((c) => c.id === currentChatId);

  // Auto-scroll to bottom
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [currentChat?.messages, isStreaming]);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
    }
  }, [input]);

  const handleSend = async () => {
    if (!input.trim() || isStreaming) return;

    const chatId = currentChatId || createChat();
    const userMessage: Message = {
      id: `msg-${Date.now()}`,
      role: 'user',
      content: input.trim(),
      timestamp: new Date(),
    };

    addMessage(chatId, userMessage);
    setInput('');

    // Simulate AI response with streaming
    const assistantMessageId = `msg-${Date.now() + 1}`;
    const assistantMessage: Message = {
      id: assistantMessageId,
      role: 'assistant',
      content: '',
      timestamp: new Date(),
      isStreaming: true,
      model: selectedModel,
    };

    addMessage(chatId, assistantMessage);

    // Simulate streaming response
    const responses = [
      '我来帮您分析这个问题。',
      '\n\n',
      '基于我的理解，',
      '这是一个涉及多个方面的复杂问题。',
      '\n\n',
      '以下是详细的解答：',
      '\n\n',
      '1. **核心概念**：首先需要理解基本原理\n',
      '2. **实现方法**：可以采用多种方式\n',
      '3. **最佳实践**：建议遵循行业标准\n',
      '\n',
      '如果您需要更具体的代码示例，请告诉我！',
    ];

    let fullContent = '';
    for (const chunk of responses) {
      await new Promise((r) => setTimeout(r, 100 + Math.random() * 200));
      fullContent += chunk;
      updateMessage(chatId, assistantMessageId, fullContent);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const groupedChats = chats.reduce((acc, chat) => {
    const date = new Date(chat.updatedAt);
    const today = new Date();
    const yesterday = new Date(today);
    yesterday.setDate(yesterday.getDate() - 1);

    let group = 'Earlier';
    if (date.toDateString() === today.toDateString()) {
      group = 'Today';
    } else if (date.toDateString() === yesterday.toDateString()) {
      group = 'Yesterday';
    } else if (today.getTime() - date.getTime() < 7 * 24 * 60 * 60 * 1000) {
      group = 'This Week';
    }

    if (!acc[group]) acc[group] = [];
    acc[group].push(chat);
    return acc;
  }, {} as Record<string, typeof chats>);

  const groupOrder = ['Today', 'Yesterday', 'This Week', 'Earlier'];

  return (
    <div className="flex h-full bg-white">
      {/* Sidebar */}
      <AnimatePresence mode="wait">
        {isSidebarOpen && (
          <motion.div
            initial={{ width: 0, opacity: 0 }}
            animate={{ width: 280, opacity: 1 }}
            exit={{ width: 0, opacity: 0 }}
            transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
            className="border-r border-gray-200 bg-white flex flex-col overflow-hidden"
          >
            <div className="p-4">
              <Button
                onClick={() => createChat()}
                className="w-full justify-start gap-2 bg-[#FFD700] hover:bg-[#FFC700] text-black"
              >
                <Plus className="w-4 h-4" />
                New Chat
              </Button>
            </div>

            <ScrollArea className="flex-1 px-3">
              {groupOrder.map((group) => {
                const groupChats = groupedChats[group];
                if (!groupChats?.length) return null;

                return (
                  <div key={group} className="mb-4">
                    <div className="text-xs font-medium text-gray-500 px-2 mb-2">
                      {group}
                    </div>
                    {groupChats.map((chat) => (
                      <div
                        key={chat.id}
                        onClick={() => setCurrentChat(chat.id)}
                        className={cn(
                          'group flex items-center gap-2 px-2 py-2 rounded-lg cursor-pointer text-sm',
                          'hover:bg-gray-100 transition-colors',
                          currentChatId === chat.id && 'bg-gray-100'
                        )}
                      >
                        <MessageSquare className="w-4 h-4 text-gray-500 shrink-0" />
                        <span className="flex-1 truncate text-gray-600">{chat.title}</span>
                        <DropdownMenu>
                          <DropdownMenuTrigger asChild>
                            <Button
                              variant="ghost"
                              size="icon"
                              className="h-6 w-6 opacity-0 group-hover:opacity-100 text-gray-500 hover:text-gray-900 hover:bg-gray-200"
                              onClick={(e) => e.stopPropagation()}
                            >
                              <MoreHorizontal className="w-3 h-3" />
                            </Button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent align="end" className="bg-white border-gray-200">
                            <DropdownMenuItem
                              onClick={(e) => {
                                e.stopPropagation();
                                deleteChat(chat.id);
                              }}
                              className="text-red-500 focus:text-red-500 focus:bg-red-50"
                            >
                              <Trash2 className="w-4 h-4 mr-2" />
                              Delete
                            </DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      </div>
                    ))}
                  </div>
                );
              })}
            </ScrollArea>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <div className="h-16 border-b border-gray-200 flex items-center justify-between px-4">
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="icon"
              onClick={toggleSidebar}
              className={cn(
                'text-gray-500 hover:text-gray-900 hover:bg-gray-100',
                !isSidebarOpen && 'rotate-180'
              )}
            >
              <ChevronDown className="w-4 h-4 -rotate-90" />
            </Button>
            <span className="font-medium text-gray-900">
              {currentChat?.title || 'New Chat'}
            </span>
          </div>

          <Select value={selectedModel} onValueChange={setSelectedModel}>
            <SelectTrigger className="w-[200px] bg-gray-50 border-gray-200 text-gray-900">
              <Cpu className="w-4 h-4 mr-2 text-[#FFD700]" />
              <SelectValue />
            </SelectTrigger>
            <SelectContent className="bg-white border-gray-200">
              {models.map((model) => (
                <SelectItem 
                  key={model.id} 
                  value={model.id}
                  className="text-gray-900 focus:bg-gray-100 focus:text-gray-900"
                >
                  <div className="flex items-center gap-2">
                    <span>{model.icon}</span>
                    <span>{model.name}</span>
                    <span className="text-xs text-gray-500">
                      {model.provider}
                    </span>
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Messages */}
        <ScrollArea ref={scrollRef} className="flex-1 p-4">
          <div className="max-w-3xl mx-auto space-y-6">
            {!currentChat?.messages.length ? (
              <div className="flex flex-col items-center justify-center h-[60vh] text-center">
                <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-[#FFD700] to-[#FFA500] flex items-center justify-center mb-4">
                  <Sparkles className="w-8 h-8 text-black" />
                </div>
                <h2 className="text-2xl font-bold mb-2 text-gray-900">What can I help you with?</h2>
                <p className="text-gray-500 max-w-md mb-8">
                  I can help you write code, answer questions, analyze data, or brainstorm ideas.
                </p>
                <div className="flex flex-wrap gap-2 justify-center">
                  {[
                    { icon: <Code className="w-4 h-4" />, text: 'Write Python function' },
                    { icon: <FileCode className="w-4 h-4" />, text: 'Explain this code' },
                    { icon: <Sparkles className="w-4 h-4" />, text: 'Brainstorm ideas' },
                    { icon: <Play className="w-4 h-4" />, text: 'Debug my code' },
                  ].map((suggestion) => (
                    <Button
                      key={suggestion.text}
                      variant="outline"
                      size="sm"
                      onClick={() => {
                        setInput(suggestion.text);
                        textareaRef.current?.focus();
                      }}
                      className="border-gray-200 text-gray-600 hover:bg-gray-50 hover:text-gray-900"
                    >
                      {suggestion.icon}
                      <span className="ml-2">{suggestion.text}</span>
                    </Button>
                  ))}
                </div>
              </div>
            ) : (
              currentChat.messages.map((message, index) => (
                <motion.div
                  key={message.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3, delay: index * 0.05 }}
                  className={cn(
                    'flex gap-4',
                    message.role === 'user' && 'flex-row-reverse'
                  )}
                >
                  <Avatar className={cn(
                    'w-8 h-8 shrink-0',
                    message.role === 'assistant' && 'bg-gradient-to-br from-[#FFD700] to-[#FFA500]'
                  )}>
                    <AvatarFallback className={cn(
                      'text-black text-sm',
                      message.role === 'assistant' && 'bg-transparent'
                    )}>
                      {message.role === 'user' ? 'U' : <Sparkles className="w-4 h-4 text-black" />}
                    </AvatarFallback>
                  </Avatar>

                  <div className={cn(
                    'flex-1 max-w-[85%]',
                    message.role === 'user' && 'text-right'
                  )}>
                    <div className={cn(
                      'inline-block rounded-2xl px-4 py-3 text-left',
                      message.role === 'user' 
                        ? 'bg-[#FFD700] text-black ml-auto' 
                        : 'bg-gray-50 border border-gray-200 text-gray-800'
                    )}>
                      {message.role === 'assistant' ? (
                        <MarkdownContent content={message.content} />
                      ) : (
                        <p className="whitespace-pre-wrap">{message.content}</p>
                      )}
                      {message.isStreaming && (
                        <span className="inline-block w-2 h-4 ml-1 bg-current animate-pulse" />
                      )}
                    </div>

                    {message.role === 'assistant' && !message.isStreaming && (
                      <div className="flex items-center gap-1 mt-2">
                        <Button variant="ghost" size="icon" className="h-7 w-7 text-gray-500 hover:text-gray-900 hover:bg-gray-100">
                          <Copy className="w-4 h-4" />
                        </Button>
                        <Button variant="ghost" size="icon" className="h-7 w-7 text-gray-500 hover:text-gray-900 hover:bg-gray-100">
                          <RotateCcw className="w-4 h-4" />
                        </Button>
                        <Button variant="ghost" size="icon" className="h-7 w-7 text-gray-500 hover:text-gray-900 hover:bg-gray-100">
                          <ThumbsUp className="w-4 h-4" />
                        </Button>
                        <Button variant="ghost" size="icon" className="h-7 w-7 text-gray-500 hover:text-gray-900 hover:bg-gray-100">
                          <ThumbsDown className="w-4 h-4" />
                        </Button>
                      </div>
                    )}
                  </div>
                </motion.div>
              ))
            )}
          </div>
        </ScrollArea>

        {/* Input Area */}
        <div className="border-t border-gray-200 p-4">
          <div className="max-w-3xl mx-auto">
            <div className="relative bg-gray-50 border border-gray-200 rounded-xl overflow-hidden focus-within:border-[#FFD700] focus-within:ring-1 focus-within:ring-[#FFD700]/30 transition-all">
              <Textarea
                ref={textareaRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Message AutoEcoops... (Enter to send, Shift+Enter for new line)"
                className="min-h-[56px] max-h-[200px] pr-14 resize-none bg-transparent border-0 text-gray-900 placeholder:text-gray-500 focus-visible:ring-0"
                rows={1}
              />
              <div className="flex items-center justify-between px-3 pb-2">
                <div className="flex items-center gap-1">
                  <Button variant="ghost" size="icon" className="h-8 w-8 text-gray-500 hover:text-gray-900 hover:bg-gray-100">
                    <Paperclip className="w-4 h-4" />
                  </Button>
                  <Button variant="ghost" size="icon" className="h-8 w-8 text-gray-500 hover:text-gray-900 hover:bg-gray-100">
                    <Image className="w-4 h-4" />
                  </Button>
                  <Button variant="ghost" size="icon" className="h-8 w-8 text-gray-500 hover:text-gray-900 hover:bg-gray-100">
                    <Mic className="w-4 h-4" />
                  </Button>
                </div>
                <Button
                  onClick={handleSend}
                  disabled={!input.trim() || isStreaming}
                  size="icon"
                  className="h-8 w-8 bg-[#FFD700] hover:bg-[#FFC700] text-black"
                >
                  <Send className="w-4 h-4" />
                </Button>
              </div>
            </div>
            <p className="text-xs text-gray-500 text-center mt-2">
              AI-generated content may be inaccurate. Please verify important information.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

interface CodeProps {
  inline?: boolean;
  className?: string;
  children?: React.ReactNode;
}

interface HeadingProps {
  children?: React.ReactNode;
}

interface ListProps {
  children?: React.ReactNode;
}

function MarkdownContent({ content }: { content: string }) {
  const components: Components = {
    code({ inline, className, children, ...props }: CodeProps) {
      const match = /language-(\w+)/.exec(className || '');
      return !inline && match ? (
        <div className="relative group my-3">
          <div className="flex items-center justify-between px-3 py-2 bg-[#0E1525] rounded-t-lg border-b border-[#2B3245]">
            <span className="text-xs text-[#5F6775] uppercase">{match[1]}</span>
            <div className="flex items-center gap-2">
              <Button
                variant="ghost"
                size="sm"
                className="h-6 text-xs text-[#5F6775] hover:text-white hover:bg-[#2B3245]"
                onClick={() => navigator.clipboard.writeText(String(children))}
              >
                <Copy className="w-3 h-3 mr-1" />
                Copy
              </Button>
              <Button
                variant="ghost"
                size="sm"
                className="h-6 text-xs text-[#5F6775] hover:text-white hover:bg-[#2B3245]"
              >
                <Play className="w-3 h-3 mr-1" />
                Run
              </Button>
            </div>
          </div>
          <SyntaxHighlighter
            style={vscDarkPlus}
            language={match[1]}
            PreTag="div"
            className="rounded-b-lg !my-0 !bg-[#0E1525]"
            {...props}
          >
            {String(children).replace(/\n$/, '')}
          </SyntaxHighlighter>
        </div>
      ) : (
        <code className="bg-[#2B3245] text-[#F26207] px-1.5 py-0.5 rounded text-sm" {...props}>
          {children}
        </code>
      );
    },
    p({ children }: { children?: React.ReactNode }) {
      return <p className="mb-2 last:mb-0 text-[#E5E7EB]">{children}</p>;
    },
    ul({ children }: ListProps) {
      return <ul className="list-disc pl-4 mb-2 text-[#E5E7EB]">{children}</ul>;
    },
    ol({ children }: ListProps) {
      return <ol className="list-decimal pl-4 mb-2 text-[#E5E7EB]">{children}</ol>;
    },
    li({ children }: { children?: React.ReactNode }) {
      return <li className="mb-1">{children}</li>;
    },
    h1({ children }: HeadingProps) {
      return <h1 className="text-xl font-bold mb-2 text-white">{children}</h1>;
    },
    h2({ children }: HeadingProps) {
      return <h2 className="text-lg font-bold mb-2 text-white">{children}</h2>;
    },
    h3({ children }: HeadingProps) {
      return <h3 className="text-base font-bold mb-2 text-white">{children}</h3>;
    },
    strong({ children }: { children?: React.ReactNode }) {
      return <strong className="font-semibold text-white">{children}</strong>;
    },
  };

  return (
    <ReactMarkdown components={components}>
      {content}
    </ReactMarkdown>
  );
}
