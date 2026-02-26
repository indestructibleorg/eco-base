import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  MessageSquare,
  Code,
  Image,
  Music,
  Video,
  FileText,
  Globe,
  Terminal,
  Sparkles,
  Send,
  Paperclip,
  Mic,
  Maximize2,
  Minimize2,
  Copy,
  Download,
  Share2,
  RefreshCw,
  Check,
  Loader2,
  Cpu,
  Zap,
  Brain,
  Eye,
} from 'lucide-react';
import { RippleButton } from '@/components/ui/ripple-button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Avatar, AvatarFallback } from '@/components/ui/avatar';
import { cn } from '@/lib/utils';

// AI Capability Types
interface AICapability {
  id: string;
  name: string;
  icon: React.ReactNode;
  description: string;
  color: string;
  active: boolean;
}

const capabilities: AICapability[] = [
  { id: 'chat', name: 'Chat', icon: <MessageSquare className="w-4 h-4" />, description: 'General conversation', color: '#FFD700', active: true },
  { id: 'code', name: 'Code', icon: <Code className="w-4 h-4" />, description: 'Programming assistant', color: '#1E88E5', active: false },
  { id: 'image', name: 'Image', icon: <Image className="w-4 h-4" />, description: 'Image generation & analysis', color: '#8B5CF6', active: false },
  { id: 'video', name: 'Video', icon: <Video className="w-4 h-4" />, description: 'Video creation & editing', color: '#EC4899', active: false },
  { id: 'music', name: 'Music', icon: <Music className="w-4 h-4" />, description: 'Music & audio generation', color: '#F59E0B', active: false },
  { id: 'research', name: 'Research', icon: <Globe className="w-4 h-4" />, description: 'Web research & analysis', color: '#10B981', active: false },
  { id: 'document', name: 'Document', icon: <FileText className="w-4 h-4" />, description: 'Document processing', color: '#06B6D4', active: false },
  { id: 'terminal', name: 'Terminal', icon: <Terminal className="w-4 h-4" />, description: 'Command execution', color: '#6B7280', active: false },
];

// Message Types
interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  type: 'text' | 'code' | 'image' | 'video' | 'audio' | 'file';
  timestamp: Date;
  isStreaming?: boolean;
  attachments?: Attachment[];
  metadata?: {
    model?: string;
    tokens?: number;
    processingTime?: number;
  };
}

interface Attachment {
  id: string;
  type: 'image' | 'file' | 'code';
  name: string;
  url?: string;
  content?: string;
  language?: string;
}

export function MultiFunctionAI() {
  const [activeCapability, setActiveCapability] = useState<string>('chat');
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [showCapabilities, setShowCapabilities] = useState(true);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  const currentChat = { messages };

  // Auto-scroll
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

    const userMessage: Message = {
      id: `msg-${Date.now()}`,
      role: 'user',
      content: input.trim(),
      type: 'text',
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsStreaming(true);

    // Simulate AI response
    const assistantMessageId = `msg-${Date.now() + 1}`;
    const assistantMessage: Message = {
      id: assistantMessageId,
      role: 'assistant',
      content: '',
      type: activeCapability === 'code' ? 'code' : 'text',
      timestamp: new Date(),
      isStreaming: true,
      metadata: { model: 'MiniMax-M2.1', tokens: 0 },
    };

    setMessages((prev) => [...prev, assistantMessage]);

    // Simulate streaming
    const responses: Record<string, string[]> = {
      chat: [
        'I understand your request.',
        '\n\n',
        'Let me analyze this for you...',
        '\n\n',
        'Based on my analysis, here are my thoughts:',
        '\n\n',
        'This is a comprehensive answer to your question.',
        'I can provide more details if needed!',
      ],
      code: [
        '```python\n',
        'def analyze_data(data):\n',
        '    """Analyze data with advanced algorithms"""\n',
        '    results = {}\n',
        '    \n',
        '    # Process each item\n',
        '    for item in data:\n',
        '        key = item.get("category")\n',
        '        if key not in results:\n',
        '            results[key] = []\n',
        '        results[key].append(item)\n',
        '    \n',
        '    return results\n',
        '```\n',
        '\n',
        'This function efficiently categorizes and processes your data.',
      ],
      image: [
        '🎨 Generating image based on your description...',
        '\n\n',
        '![Generated Image](https://placehold.co/600x400/1C2333/F26207?text=AI+Generated+Image)',
        '\n\n',
        'Here is the image I created based on your prompt!',
      ],
      research: [
        '🔍 Starting web research...',
        '\n\n',
        '**Research Results:**\n\n',
        '1. [Source 1](https://example.com) - Key findings about the topic\n',
        '2. [Source 2](https://example.com) - Additional insights\n',
        '3. [Source 3](https://example.com) - Related research\n',
        '\n',
        '**Summary:** Based on my research, here are the key points...',
      ],
    };

    const responseChunks = responses[activeCapability] || responses.chat;
    let fullContent = '';

    for (const chunk of responseChunks) {
      await new Promise((r) => setTimeout(r, 80 + Math.random() * 120));
      fullContent += chunk;
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantMessageId ? { ...m, content: fullContent } : m
        )
      );
    }

    setMessages((prev) =>
      prev.map((m) =>
        m.id === assistantMessageId ? { ...m, isStreaming: false } : m
      )
    );
    setIsStreaming(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const activeCap = capabilities.find((c) => c.id === activeCapability);

  return (
    <div className={cn(
      "flex flex-col bg-white h-full",
      isFullscreen && "fixed inset-0 z-50"
    )}>
      {/* Header */}
      <div className="h-14 border-b border-gray-200 flex items-center justify-between px-4">
        <div className="flex items-center gap-3">
          <div 
            className="w-8 h-8 rounded-lg flex items-center justify-center"
            style={{ backgroundColor: activeCap?.color }}
          >
            {activeCap?.icon}
          </div>
          <div>
            <h1 className="font-semibold text-gray-900">MiniMax Agent</h1>
            <p className="text-xs text-gray-500">{activeCap?.description}</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <RippleButton
            variant="ghost"
            size="icon"
            onClick={() => setShowCapabilities(!showCapabilities)}
            className="text-gray-500"
          >
            {showCapabilities ? <span className="text-xs">Hide</span> : <span className="text-xs">Show</span>}
          </RippleButton>
          <RippleButton
            variant="ghost"
            size="icon"
            onClick={() => setIsFullscreen(!isFullscreen)}
            className="text-gray-500"
          >
            {isFullscreen ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
          </RippleButton>
        </div>
      </div>

      {/* Capability Selector */}
      <AnimatePresence>
        {showCapabilities && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="border-b border-gray-200 overflow-hidden"
          >
            <div className="p-3">
              <div className="flex flex-wrap gap-2">
                {capabilities.map((cap) => (
                  <RippleButton
                    key={cap.id}
                    variant={activeCapability === cap.id ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => setActiveCapability(cap.id)}
                    className={cn(
                      "gap-2 transition-all duration-200",
                      activeCapability === cap.id 
                        ? "bg-[#FFD700] text-black border-[#FFD700] shadow-lg" 
                        : "border-gray-300 text-gray-700 hover:bg-gray-50"
                    )}
                  >
                    {cap.icon}
                    <span>{cap.name}</span>
                  </RippleButton>
                ))}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Messages Area */}
      <ScrollArea ref={scrollRef} className="flex-1 p-4">
        <div className="max-w-4xl mx-auto space-y-6">
          {messages.length === 0 ? (
            <WelcomeScreen 
              capability={activeCapability} 
              onSuggestion={(text) => {
                setInput(text);
                textareaRef.current?.focus();
              }}
            />
          ) : (
            messages.map((message, index) => (
              <MessageBubble 
                key={message.id} 
                message={message} 
                index={index}
              />
            ))
          )}
        </div>
      </ScrollArea>

      {/* Input Area */}
      <div className="border-t border-gray-200 p-4">
        <div className="max-w-4xl mx-auto">
          <div className="relative bg-gray-50 border border-gray-200 rounded-xl overflow-hidden focus-within:border-[#FFD700] focus-within:ring-1 focus-within:ring-[#FFD700]/30 transition-all">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={`Ask ${activeCap?.name} anything...`}
              className="min-h-[60px] max-h-[200px] pr-14 resize-none bg-transparent border-0 text-gray-900 placeholder:text-gray-500 focus-visible:ring-0 w-full p-3"
              rows={1}
            />
            <div className="flex items-center justify-between px-3 pb-2">
              <div className="flex items-center gap-1">
                <RippleButton variant="ghost" size="icon" className="h-8 w-8 text-gray-500">
                  <Paperclip className="w-4 h-4" />
                </RippleButton>
                <RippleButton variant="ghost" size="icon" className="h-8 w-8 text-gray-500">
                  <Image className="w-4 h-4" />
                </RippleButton>
                <RippleButton variant="ghost" size="icon" className="h-8 w-8 text-gray-500">
                  <Mic className="w-4 h-4" />
                </RippleButton>
              </div>
              <RippleButton
                onClick={handleSend}
                disabled={!input.trim() || isStreaming}
                size="icon"
                className="h-8 w-8 bg-[#FFD700] text-black hover:bg-[#FFC700]"
              >
                {isStreaming ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Send className="w-4 h-4" />
                )}
              </RippleButton>
            </div>
          </div>
          <p className="text-xs text-gray-500 text-center mt-2">
            MiniMax Agent can make mistakes. Consider checking important information.
          </p>
        </div>
      </div>
    </div>
  );
}

// Welcome Screen Component
function WelcomeScreen({ capability, onSuggestion }: { capability: string; onSuggestion: (text: string) => void }) {
  const suggestions: Record<string, { icon: React.ReactNode; text: string }[]> = {
    chat: [
      { icon: <Brain className="w-4 h-4" />, text: 'Explain quantum computing in simple terms' },
      { icon: <Sparkles className="w-4 h-4" />, text: 'Help me brainstorm ideas for a startup' },
      { icon: <FileText className="w-4 h-4" />, text: 'Write a professional email to my team' },
      { icon: <Globe className="w-4 h-4" />, text: 'What are the latest trends in AI?' },
    ],
    code: [
      { icon: <Code className="w-4 h-4" />, text: 'Create a React component with TypeScript' },
      { icon: <Terminal className="w-4 h-4" />, text: 'Write a Python script for data analysis' },
      { icon: <Zap className="w-4 h-4" />, text: 'Optimize this SQL query for performance' },
      { icon: <Check className="w-4 h-4" />, text: 'Debug this JavaScript error' },
    ],
    image: [
      { icon: <Image className="w-4 h-4" />, text: 'Generate a futuristic cityscape at sunset' },
      { icon: <PaletteIcon className="w-4 h-4" />, text: 'Create a logo for a tech startup' },
      { icon: <Eye className="w-4 h-4" />, text: 'Analyze this image and describe it' },
      { icon: <Sparkles className="w-4 h-4" />, text: 'Transform this photo into anime style' },
    ],
    research: [
      { icon: <Globe className="w-4 h-4" />, text: 'Research the latest AI breakthroughs' },
      { icon: <FileText className="w-4 h-4" />, text: 'Find information about climate change' },
      { icon: <Check className="w-4 h-4" />, text: 'Compare different cloud providers' },
      { icon: <Sparkles className="w-4 h-4" />, text: 'Summarize this article for me' },
    ],
  };

  const capSuggestions = suggestions[capability] || suggestions.chat;

  return (
    <div className="flex flex-col items-center justify-center py-12">
      <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-[#FFD700] to-[#FFA500] flex items-center justify-center mb-6 animate-float">
        <Sparkles className="w-8 h-8 text-black" />
      </div>
      <h2 className="text-2xl font-bold text-gray-900 mb-2">What can I help you with?</h2>
      <p className="text-gray-500 mb-8 text-center max-w-md">
        I can chat, write code, generate images, conduct research, and much more.
      </p>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 w-full max-w-2xl">
        {capSuggestions.map((suggestion, index) => (
          <motion.button
            key={index}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            onClick={() => onSuggestion(suggestion.text)}
            className="flex items-center gap-3 p-4 rounded-xl border border-gray-200 bg-gray-50 hover:border-[#FFD700]/50 hover:bg-white transition-all text-left group"
          >
            <span className="text-gray-500 group-hover:text-[#FFD700] transition-colors">
              {suggestion.icon}
            </span>
            <span className="text-sm text-gray-600 group-hover:text-gray-900 transition-colors">
              {suggestion.text}
            </span>
          </motion.button>
        ))}
      </div>
    </div>
  );
}

// Message Bubble Component
function MessageBubble({ message, index }: { message: Message; index: number }) {
  const isUser = message.role === 'user';

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: index * 0.05 }}
      className={cn('flex gap-4', isUser && 'flex-row-reverse')}
    >
      <Avatar className={cn('w-8 h-8 shrink-0', !isUser && 'bg-gradient-to-br from-[#FFD700] to-[#FFA500]')}>
        <AvatarFallback className={cn('text-black text-sm', !isUser && 'bg-transparent')}>
          {isUser ? 'U' : <Sparkles className="w-4 h-4 text-black" />}
        </AvatarFallback>
      </Avatar>

      <div className={cn('flex-1 max-w-[85%]', isUser && 'text-right')}>
        <div
          className={cn(
            'inline-block rounded-2xl px-4 py-3 text-left',
            isUser
              ? 'bg-[#FFD700] text-black ml-auto'
              : 'bg-gray-50 border border-gray-200 text-gray-800'
          )}
        >
          {message.type === 'code' ? (
            <CodeBlock content={message.content} />
          ) : message.type === 'image' ? (
            <ImageContent content={message.content} />
          ) : (
            <div className="whitespace-pre-wrap">{message.content}</div>
          )}
          {message.isStreaming && (
            <span className="inline-block w-2 h-4 ml-1 bg-current animate-pulse" />
          )}
        </div>

        {!isUser && !message.isStreaming && (
          <div className="flex items-center gap-1 mt-2">
            <RippleButton variant="ghost" size="icon" className="h-7 w-7 text-gray-500">
              <Copy className="w-4 h-4" />
            </RippleButton>
            <RippleButton variant="ghost" size="icon" className="h-7 w-7 text-gray-500">
              <RefreshCw className="w-4 h-4" />
            </RippleButton>
            <RippleButton variant="ghost" size="icon" className="h-7 w-7 text-gray-500">
              <Share2 className="w-4 h-4" />
            </RippleButton>
            <RippleButton variant="ghost" size="icon" className="h-7 w-7 text-gray-500">
              <Download className="w-4 h-4" />
            </RippleButton>
          </div>
        )}

        {message.metadata && (
          <div className="flex items-center gap-2 mt-1 text-xs text-gray-500">
            <Cpu className="w-3 h-3" />
            <span>{message.metadata.model}</span>
            {message.metadata.tokens && (
              <>
                <span>•</span>
                <span>{message.metadata.tokens} tokens</span>
              </>
            )}
          </div>
        )}
      </div>
    </motion.div>
  );
}

// Code Block Component
function CodeBlock({ content }: { content: string }) {
  const codeMatch = content.match(/```(\w+)?\n([\s\S]*?)```/);
  
  if (!codeMatch) {
    return <div className="whitespace-pre-wrap">{content}</div>;
  }

  const language = codeMatch[1] || 'text';
  const code = codeMatch[2];

  return (
    <div className="relative group">
      <div className="flex items-center justify-between px-3 py-2 bg-gray-100 rounded-t-lg border-b border-gray-200">
        <span className="text-xs text-gray-500 uppercase">{language}</span>
        <div className="flex items-center gap-2">
          <RippleButton
            variant="ghost"
            size="sm"
            className="h-6 text-xs text-gray-500 hover:text-gray-900"
            onClick={() => navigator.clipboard.writeText(code)}
          >
            <Copy className="w-3 h-3 mr-1" />
            Copy
          </RippleButton>
          <RippleButton variant="ghost" size="sm" className="h-6 text-xs text-gray-500 hover:text-gray-900">
            <Zap className="w-3 h-3 mr-1" />
            Run
          </RippleButton>
        </div>
      </div>
      <pre className="bg-gray-100 rounded-b-lg p-4 overflow-x-auto">
        <code className="text-sm font-mono text-gray-800">{code}</code>
      </pre>
    </div>
  );
}

// Image Content Component
function ImageContent({ content }: { content: string }) {
  const imageMatch = content.match(/!\[([^\]]*)\]\(([^)]+)\)/);
  
  if (!imageMatch) {
    return <div className="whitespace-pre-wrap">{content}</div>;
  }

  const alt = imageMatch[1];
  const src = imageMatch[2];

  return (
    <div className="space-y-2">
      <div className="relative group rounded-lg overflow-hidden">
        <img src={src} alt={alt} className="w-full max-w-md rounded-lg" />
        <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center gap-2">
          <RippleButton variant="secondary" size="sm" className="bg-white text-gray-900">
            <Download className="w-4 h-4 mr-1" />
            Download
          </RippleButton>
          <RippleButton variant="secondary" size="sm" className="bg-white text-gray-900">
            <Share2 className="w-4 h-4 mr-1" />
            Share
          </RippleButton>
        </div>
      </div>
      <p className="text-sm text-gray-500">{alt}</p>
    </div>
  );
}

// Palette icon component
function PaletteIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="13.5" cy="6.5" r=".5" fill="currentColor" />
      <circle cx="17.5" cy="10.5" r=".5" fill="currentColor" />
      <circle cx="8.5" cy="7.5" r=".5" fill="currentColor" />
      <circle cx="6.5" cy="12.5" r=".5" fill="currentColor" />
      <path d="M12 2C6.5 2 2 6.5 2 12s4.5 10 10 10c.926 0 1.648-.746 1.648-1.688 0-.437-.18-.835-.437-1.125-.29-.289-.438-.652-.438-1.125a1.64 1.64 0 0 1 1.668-1.668h1.996c3.051 0 5.555-2.503 5.555-5.554C21.965 6.01 17.461 2 12 2z" />
    </svg>
  );
}
