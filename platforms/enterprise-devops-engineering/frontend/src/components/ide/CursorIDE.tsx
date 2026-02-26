import { useState, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Folder,
  ChevronRight,
  ChevronDown,
  Plus,
  Play,
  Edit3,
  X,
  Terminal,
  Code2,
  Sparkles,
  MessageSquare,
  Command,
  Wand2,
  Bug,
  RefreshCw,
  Check,
  Copy,
  Send,
  MoreHorizontal,
  History,
  AtSign,
  FileCode,
  Lightbulb,
} from 'lucide-react';
import { RippleButton } from '@/components/ui/ripple-button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';
import Editor from '@monaco-editor/react';

// Types
interface FileNode {
  id: string;
  name: string;
  type: 'file' | 'folder';
  language?: string;
  content?: string;
  children?: FileNode[];
  isOpen?: boolean;
}

interface EditorTab {
  id: string;
  fileId: string;
  name: string;
  language: string;
  content: string;
  isModified: boolean;
}

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  codeBlocks?: CodeBlock[];
  isStreaming?: boolean;
}

interface CodeBlock {
  language: string;
  code: string;
  fileName?: string;
}

interface GhostText {
  id: string;
  text: string;
  position: { line: number; column: number };
}

// Mock Data
const mockFiles: FileNode[] = [
  {
    id: 'root',
    name: 'my-project',
    type: 'folder',
    isOpen: true,
    children: [
      {
        id: 'src',
        name: 'src',
        type: 'folder',
        isOpen: true,
        children: [
          {
            id: 'main-py',
            name: 'main.py',
            type: 'file',
            language: 'python',
            content: `def fibonacci(n):
    """Calculate fibonacci number"""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

def main():
    print("Fibonacci Sequence:")
    for i in range(10):
        print(f"F({i}) = {fibonacci(i)}")

if __name__ == "__main__":
    main()`,
          },
          {
            id: 'utils-py',
            name: 'utils.py',
            type: 'file',
            language: 'python',
            content: `def greet(name: str) -> str:
    """Greet a user"""
    return f"Hello, {name}!"

def format_number(n: int) -> str:
    """Format number with commas"""
    return f"{n:,}"`,
          },
        ],
      },
      {
        id: 'app-ts',
        name: 'app.ts',
        type: 'file',
        language: 'typescript',
        content: `interface User {
  id: string;
  name: string;
  email: string;
}

class UserService {
  private users: User[] = [];
  
  addUser(user: User): void {
    this.users.push(user);
  }
  
  getUser(id: string): User | undefined {
    return this.users.find(u => u.id === id);
  }
}

export { UserService, type User };`,
      },
    ],
  },
];

export function CursorIDE() {
  // State
  const [files, setFiles] = useState<FileNode[]>(mockFiles);
  const [openTabs, setOpenTabs] = useState<EditorTab[]>([]);
  const [activeTabId, setActiveTabId] = useState<string | null>(null);
  const [showChatPanel, setShowChatPanel] = useState(true);
  const [showInlineEdit, setShowInlineEdit] = useState(false);
  const [inlineEditCode, setInlineEditCode] = useState('');
  const [selectedCode, setSelectedCode] = useState('');
  const [ghostText, setGhostText] = useState<GhostText | null>(null);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [contextFiles, setContextFiles] = useState<string[]>([]);
  const [showContextMenu, setShowContextMenu] = useState(false);
  const [cursorPosition, setCursorPosition] = useState({ line: 1, column: 1 });
  const [terminalLines] = useState<{ id: string; type: 'input' | 'output' | 'error' | 'ai'; content: string }[]>([
    { id: '1', type: 'ai', content: '👋 Welcome to Cursor AI IDE! Press Cmd+K for inline edit, Cmd+L for chat.' },
  ]);

  const editorRef = useRef<any>(null);
  const chatInputRef = useRef<HTMLTextAreaElement>(null);
  const inlineEditRef = useRef<HTMLTextAreaElement>(null);

  // Get active tab
  const activeTab = openTabs.find((t) => t.id === activeTabId);

  // Open file
  const openFile = useCallback((file: FileNode) => {
    if (file.type !== 'file') return;
    
    const existingTab = openTabs.find((t) => t.fileId === file.id);
    if (existingTab) {
      setActiveTabId(existingTab.id);
      return;
    }

    const newTab: EditorTab = {
      id: `tab-${Date.now()}`,
      fileId: file.id,
      name: file.name,
      language: file.language || 'text',
      content: file.content || '',
      isModified: false,
    };
    setOpenTabs((prev) => [...prev, newTab]);
    setActiveTabId(newTab.id);
  }, [openTabs]);

  // Close tab
  const closeTab = useCallback((tabId: string) => {
    setOpenTabs((prev) => {
      const newTabs = prev.filter((t) => t.id !== tabId);
      if (activeTabId === tabId) {
        setActiveTabId(newTabs[newTabs.length - 1]?.id || null);
      }
      return newTabs;
    });
  }, [activeTabId]);

  // Update tab content
  const updateTabContent = useCallback((tabId: string, content: string) => {
    setOpenTabs((prev) =>
      prev.map((t) =>
        t.id === tabId ? { ...t, content, isModified: true } : t
      )
    );
  }, []);

  // Handle editor mount
  const handleEditorMount = (editor: any, monaco: any) => {
    editorRef.current = editor;

    // Add keyboard shortcuts
    editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyK, () => {
      const selection = editor.getSelection();
      if (selection) {
        const model = editor.getModel();
        if (model) {
          const selectedText = model.getValueInRange(selection);
          if (selectedText) {
            setSelectedCode(selectedText);
            setShowInlineEdit(true);
          }
        }
      }
    });

    editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyL, () => {
      setShowChatPanel((prev) => !prev);
    });

    // Track cursor position
    editor.onDidChangeCursorPosition((e: any) => {
      setCursorPosition({ line: e.position.lineNumber, column: e.position.column });
      
      // Simulate ghost text (Tab completion)
      if (Math.random() > 0.7) {
        const suggestions = [
          'console.log(',
          'return ',
          'const ',
          'function ',
          'async ',
          'await ',
          '.map(',
          '.filter(',
          '.reduce(',
        ];
        setGhostText({
          id: `ghost-${Date.now()}`,
          text: suggestions[Math.floor(Math.random() * suggestions.length)],
          position: { line: e.position.lineNumber, column: e.position.column },
        });
      } else {
        setGhostText(null);
      }
    });
  };

  // Send chat message
  const sendChatMessage = async () => {
    if (!chatInput.trim() || isGenerating) return;

    const userMessage: ChatMessage = {
      id: `msg-${Date.now()}`,
      role: 'user',
      content: chatInput,
      timestamp: new Date(),
    };

    setChatMessages((prev) => [...prev, userMessage]);
    setChatInput('');
    setIsGenerating(true);

    // Simulate AI response
    setTimeout(() => {
      const responses = [
        {
          content: 'I can help you with that! Here\'s the solution:',
          code: `function optimizedFibonacci(n: number, memo: Record<number, number> = {}): number {
  if (n <= 1) return n;
  if (memo[n]) return memo[n];
  memo[n] = optimizedFibonacci(n - 1, memo) + optimizedFibonacci(n - 2, memo);
  return memo[n];
}`,
        },
        {
          content: 'I notice you\'re working with user data. Here\'s a more type-safe approach:',
          code: `interface CreateUserInput {
  name: string;
  email: string;
  password: string;
}

async function createUser(input: CreateUserInput): Promise<User> {
  // Validate input
  if (!input.email.includes('@')) {
    throw new Error('Invalid email');
  }
  // Create user logic...
}`,
        },
        {
          content: 'You can refactor this code to be more efficient using memoization.',
        },
      ];

      const response = responses[Math.floor(Math.random() * responses.length)];
      
      const assistantMessage: ChatMessage = {
        id: `msg-${Date.now() + 1}`,
        role: 'assistant',
        content: response.content,
        timestamp: new Date(),
        codeBlocks: response.code ? [{ language: 'typescript', code: response.code }] : undefined,
      };

      setChatMessages((prev) => [...prev, assistantMessage]);
      setIsGenerating(false);
    }, 1500);
  };

  // Apply inline edit
  const applyInlineEdit = () => {
    if (!activeTab || !editorRef.current) return;
    
    const selection = editorRef.current.getSelection();
    if (selection) {
      editorRef.current.executeEdits('inline-edit', [
        {
          range: selection,
          text: inlineEditCode,
        },
      ]);
      updateTabContent(activeTab.id, editorRef.current.getValue());
    }
    setShowInlineEdit(false);
    setInlineEditCode('');
  };

  // Render file tree
  const renderFileTree = (nodes: FileNode[], level = 0) => {
    return nodes.map((node) => (
      <div key={node.id}>
        <div
          className={cn(
            'flex items-center gap-1 px-2 py-1 cursor-pointer hover:bg-gray-100 rounded-sm text-sm transition-colors group',
            activeTab?.fileId === node.id && 'bg-[#FFD700]/10 text-[#FFD700]'
          )}
          style={{ paddingLeft: `${level * 12 + 8}px` }}
          onClick={() => {
            if (node.type === 'folder') {
              setFiles((prev) => toggleFolder(prev, node.id));
            } else {
              openFile(node);
            }
          }}
        >
          {node.type === 'folder' && (
            <span className="w-4 h-4 flex items-center justify-center text-gray-500">
              {node.isOpen ? (
                <ChevronDown className="w-3 h-3" />
              ) : (
                <ChevronRight className="w-3 h-3" />
              )}
            </span>
          )}
          {node.type === 'folder' ? (
            <Folder className="w-4 h-4 text-[#FFD700]" />
          ) : (
            <FileCode className="w-4 h-4 text-[#1E88E5]" />
          )}
          <span className="truncate text-gray-600 group-hover:text-gray-900">{node.name}</span>
        </div>
        {node.type === 'folder' && node.isOpen && node.children && (
          <div>{renderFileTree(node.children, level + 1)}</div>
        )}
      </div>
    ));
  };

  // Toggle folder
  const toggleFolder = (nodes: FileNode[], id: string): FileNode[] => {
    return nodes.map((node) => {
      if (node.id === id) {
        return { ...node, isOpen: !node.isOpen };
      }
      if (node.children) {
        return { ...node, children: toggleFolder(node.children, id) };
      }
      return node;
    });
  };

  // AI Actions
  const aiActions = [
    { id: 'explain', label: 'Explain', icon: Lightbulb, shortcut: '⌘E' },
    { id: 'fix', label: 'Fix', icon: Bug, shortcut: '⌘.' },
    { id: 'refactor', label: 'Refactor', icon: RefreshCw, shortcut: '⌘R' },
    { id: 'generate', label: 'Generate', icon: Wand2, shortcut: '⌘G' },
    { id: 'docs', label: 'Add Docs', icon: FileCode, shortcut: '⌘D' },
    { id: 'test', label: 'Generate Test', icon: Check, shortcut: '⌘T' },
  ];

  return (
    <div className="flex h-full bg-white">
      {/* File Explorer */}
      <div className="w-64 border-r border-gray-200 bg-white flex flex-col">
        <div className="h-10 flex items-center justify-between px-3 border-b border-gray-200">
          <span className="text-sm font-medium text-gray-900">Explorer</span>
          <RippleButton variant="ghost" size="icon" className="h-6 w-6 text-gray-500">
            <Plus className="w-4 h-4" />
          </RippleButton>
        </div>
        <ScrollArea className="flex-1 py-2">
          {renderFileTree(files)}
        </ScrollArea>
      </div>

      {/* Main Editor Area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Tabs */}
        {openTabs.length > 0 && (
          <div className="h-10 bg-white flex items-center border-b border-gray-200 overflow-x-auto">
            {openTabs.map((tab) => (
              <div
                key={tab.id}
                onClick={() => setActiveTabId(tab.id)}
                className={cn(
                  'flex items-center gap-2 px-3 py-2 text-sm cursor-pointer border-r border-gray-200 min-w-[120px] max-w-[200px] transition-all group',
                  activeTabId === tab.id
                    ? 'bg-gray-50 border-t-2 border-t-[#FFD700] text-gray-900'
                    : 'text-gray-500 hover:bg-gray-50 hover:text-gray-700'
                )}
              >
                <Code2 className="w-4 h-4" />
                <span className="flex-1 truncate">{tab.name}</span>
                {tab.isModified && <span className="text-[#FFD700]">•</span>}
                <RippleButton
                  variant="ghost"
                  size="icon"
                  className="h-4 w-4 opacity-0 group-hover:opacity-100 text-gray-500 hover:text-gray-900"
                  onClick={(e) => {
                    e.stopPropagation();
                    closeTab(tab.id);
                  }}
                >
                  <X className="w-3 h-3" />
                </RippleButton>
              </div>
            ))}
          </div>
        )}

        {/* Toolbar */}
        <div className="h-10 flex items-center justify-between px-4 border-b border-gray-200 bg-white">
          <div className="flex items-center gap-4">
            {activeTab && (
              <>
                <span className="text-sm text-gray-500">{activeTab.language?.toUpperCase()}</span>
                <span className="text-xs text-gray-500">Ln {cursorPosition.line}, Col {cursorPosition.column}</span>
                {activeTab.isModified && <span className="text-xs text-[#FFD700]">Unsaved</span>}
              </>
            )}
          </div>
          <div className="flex items-center gap-2">
            {/* AI Actions */}
            <div className="flex items-center gap-1 mr-4">
              {aiActions.slice(0, 3).map((action) => (
                <RippleButton
                  key={action.id}
                  variant="ghost"
                  size="sm"
                  className="h-7 text-xs gap-1 text-gray-600 hover:text-gray-900 hover:bg-gray-100"
                >
                  <action.icon className="w-3 h-3" />
                  {action.label}
                </RippleButton>
              ))}
              <RippleButton variant="ghost" size="sm" className="h-7 text-xs text-gray-500">
                <MoreHorizontal className="w-3 h-3" />
              </RippleButton>
            </div>
            <RippleButton
              variant="ghost"
              size="sm"
              onClick={() => setShowChatPanel(!showChatPanel)}
              className={cn(
                'text-gray-500 hover:text-gray-900 hover:bg-gray-100',
                showChatPanel && 'text-[#FFD700] bg-[#FFD700]/10'
              )}
            >
              <MessageSquare className="w-4 h-4 mr-1" />
              Chat
            </RippleButton>
            <RippleButton size="sm" className="bg-[#FFD700] hover:bg-[#FFC700] text-black">
              <Play className="w-4 h-4 mr-1" />
              Run
            </RippleButton>
          </div>
        </div>

        {/* Editor */}
        <div className="flex-1 flex overflow-hidden relative">
          <div className={cn('flex-1 relative', showChatPanel && 'w-[calc(100%-380px)]')}>
            {activeTab ? (
              <>
                <Editor
                  height="100%"
                  language={activeTab.language}
                  value={activeTab.content}
                  onChange={(value) => {
                    if (value !== undefined && activeTabId) {
                      updateTabContent(activeTabId, value);
                    }
                  }}
                  theme="vs-light"
                  onMount={handleEditorMount}
                  options={{
                    minimap: { enabled: true, scale: 1 },
                    fontSize: 14,
                    lineNumbers: 'on',
                    roundedSelection: false,
                    scrollBeyondLastLine: false,
                    readOnly: false,
                    automaticLayout: true,
                    fontFamily: 'JetBrains Mono, Fira Code, monospace',
                    quickSuggestions: true,
                    suggestOnTriggerCharacters: true,
                    acceptSuggestionOnEnter: 'on',
                  }}
                />
                
                {/* Ghost Text (Tab Completion) */}
                {ghostText && (
                  <div
                    className="absolute pointer-events-none text-gray-400 opacity-60 font-mono text-sm"
                    style={{
                      left: `${ghostText.position.column * 8 + 60}px`,
                      top: `${(ghostText.position.line - 1) * 19 + 10}px`,
                    }}
                  >
                    {ghostText.text}
                    <span className="ml-2 text-xs text-gray-400">[Tab]</span>
                  </div>
                )}
              </>
            ) : (
              <div className="flex flex-col items-center justify-center h-full text-gray-500">
                <Code2 className="w-16 h-16 mb-4 opacity-30" />
                <p className="text-lg mb-2 text-gray-900">Select a file to start editing</p>
                <div className="flex gap-4 text-sm">
                  <span className="flex items-center gap-1"><Command className="w-4 h-4" /> + K Inline Edit</span>
                  <span className="flex items-center gap-1"><Command className="w-4 h-4" /> + L Chat</span>
                </div>
              </div>
            )}

            {/* Inline Edit Popup */}
            <AnimatePresence>
              {showInlineEdit && (
                <motion.div
                  initial={{ opacity: 0, y: 10, scale: 0.95 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: 10, scale: 0.95 }}
                  className="absolute z-50 w-[500px] bg-white border border-gray-200 rounded-xl shadow-2xl overflow-hidden"
                  style={{ left: 100, top: 100 }}
                >
                  <div className="flex items-center justify-between px-4 py-2 border-b border-gray-200 bg-gray-50">
                    <div className="flex items-center gap-2">
                      <Sparkles className="w-4 h-4 text-[#FFD700]" />
                      <span className="text-sm font-medium text-gray-900">AI Edit</span>
                    </div>
                    <RippleButton variant="ghost" size="icon" className="h-6 w-6 text-gray-500" onClick={() => setShowInlineEdit(false)}>
                      <X className="w-4 h-4" />
                    </RippleButton>
                  </div>
                  <div className="p-4">
                    <div className="mb-3 p-2 bg-gray-50 rounded-lg">
                      <code className="text-xs text-gray-600 font-mono line-clamp-3">{selectedCode}</code>
                    </div>
                    <textarea
                      ref={inlineEditRef}
                      value={inlineEditCode}
                      onChange={(e) => setInlineEditCode(e.target.value)}
                      placeholder="Describe the changes you want..."
                      className="w-full h-20 bg-gray-50 border border-gray-200 rounded-lg p-3 text-sm text-gray-900 placeholder:text-gray-500 resize-none focus:outline-none focus:border-[#FFD700]"
                      autoFocus
                    />
                    <div className="flex justify-end gap-2 mt-3">
                      <RippleButton variant="outline" size="sm" className="border-gray-300" onClick={() => setShowInlineEdit(false)}>
                        Cancel
                      </RippleButton>
                      <RippleButton size="sm" className="gap-2 bg-[#FFD700] text-black hover:bg-[#FFC700]" onClick={applyInlineEdit}>
                        <Sparkles className="w-4 h-4" />
                        Apply
                      </RippleButton>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Chat Panel */}
          <AnimatePresence>
            {showChatPanel && (
              <motion.div
                initial={{ width: 0, opacity: 0 }}
                animate={{ width: 380, opacity: 1 }}
                exit={{ width: 0, opacity: 0 }}
                className="border-l border-gray-200 bg-white flex flex-col"
              >
                {/* Chat Header */}
                <div className="h-12 border-b border-gray-200 flex items-center justify-between px-4">
                  <div className="flex items-center gap-2">
                    <Sparkles className="w-5 h-5 text-[#FFD700]" />
                    <span className="font-medium text-gray-900">Cursor AI</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <RippleButton variant="ghost" size="icon" className="h-7 w-7 text-gray-500">
                      <History className="w-4 h-4" />
                    </RippleButton>
                    <RippleButton variant="ghost" size="icon" className="h-7 w-7 text-gray-500" onClick={() => setShowChatPanel(false)}>
                      <X className="w-4 h-4" />
                    </RippleButton>
                  </div>
                </div>

                {/* Context Files */}
                {contextFiles.length > 0 && (
                  <div className="px-4 py-2 border-b border-gray-200">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="text-xs text-gray-500">Context:</span>
                      {contextFiles.map((file, i) => (
                        <Badge key={i} className="bg-gray-100 text-gray-600 border-0 text-[10px] gap-1">
                          <FileCode className="w-3 h-3" />
                          {file}
                          <X className="w-3 h-3 cursor-pointer hover:text-gray-900" onClick={() => setContextFiles((prev) => prev.filter((f) => f !== file))} />
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}

                {/* Chat Messages */}
                <ScrollArea className="flex-1 p-4">
                  <div className="space-y-4">
                    {chatMessages.length === 0 && (
                      <div className="text-center py-8">
                        <Sparkles className="w-12 h-12 text-[#FFD700] mx-auto mb-4" />
                        <h3 className="text-gray-900 font-medium mb-2">How can I help you?</h3>
                        <p className="text-sm text-gray-500 mb-4">I can help you write, edit, and understand code.</p>
                        <div className="space-y-2">
                          {[
                            'Explain this code',
                            'Refactor this function',
                            'Add error handling',
                            'Generate unit tests',
                          ].map((suggestion, i) => (
                            <button
                              key={i}
                              onClick={() => setChatInput(suggestion)}
                              className="w-full p-2 text-left text-sm text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors"
                            >
                              {suggestion}
                            </button>
                          ))}
                        </div>
                      </div>
                    )}
                    {chatMessages.map((message) => (
                      <div
                        key={message.id}
                        className={cn(
                          'flex gap-3',
                          message.role === 'user' ? 'flex-row-reverse' : ''
                        )}
                      >
                        <div className={cn(
                          'w-8 h-8 rounded-lg flex items-center justify-center shrink-0',
                          message.role === 'user' ? 'bg-[#FFD700]' : 'bg-gradient-to-br from-[#FFD700] to-[#FFA500]'
                        )}>
                          {message.role === 'user' ? (
                            <span className="text-black text-sm">U</span>
                          ) : (
                            <Sparkles className="w-4 h-4 text-black" />
                          )}
                        </div>
                        <div className={cn(
                          'flex-1 max-w-[85%]',
                          message.role === 'user' && 'text-right'
                        )}>
                          <div className={cn(
                            'inline-block rounded-xl px-4 py-3 text-left',
                            message.role === 'user'
                              ? 'bg-[#FFD700] text-black'
                              : 'bg-gray-50 border border-gray-200 text-gray-800'
                          )}>
                            <p className="text-sm whitespace-pre-wrap">{message.content}</p>
                          </div>
                          {message.codeBlocks?.map((block, i) => (
                            <div key={i} className="mt-3 bg-gray-100 rounded-xl overflow-hidden border border-gray-200">
                              <div className="flex items-center justify-between px-3 py-2 border-b border-gray-200">
                                <span className="text-xs text-gray-500">{block.language}</span>
                                <div className="flex gap-1">
                                  <RippleButton variant="ghost" size="icon" className="h-6 w-6 text-gray-500">
                                    <Copy className="w-3 h-3" />
                                  </RippleButton>
                                  <RippleButton variant="ghost" size="icon" className="h-6 w-6 text-gray-500">
                                    <Edit3 className="w-3 h-3" />
                                  </RippleButton>
                                </div>
                              </div>
                              <pre className="p-3 text-xs font-mono text-gray-600 overflow-auto">
                                <code>{block.code}</code>
                              </pre>
                            </div>
                          ))}
                        </div>
                      </div>
                    ))}
                    {isGenerating && (
                      <div className="flex gap-3">
                        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-[#FFD700] to-[#FFA500] flex items-center justify-center">
                          <Sparkles className="w-4 h-4 text-black animate-pulse" />
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

                {/* Chat Input */}
                <div className="p-4 border-t border-gray-200">
                  <div className="relative bg-gray-50 border border-gray-200 rounded-xl overflow-hidden focus-within:border-[#FFD700] transition-colors">
                    <textarea
                      ref={chatInputRef}
                      value={chatInput}
                      onChange={(e) => setChatInput(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' && !e.shiftKey) {
                          e.preventDefault();
                          sendChatMessage();
                        }
                      }}
                      placeholder="Ask anything... (Cmd+K for inline edit)"
                      className="w-full min-h-[60px] max-h-[150px] bg-transparent border-0 text-gray-900 placeholder:text-gray-500 focus-visible:ring-0 p-3 pr-12 resize-none"
                      rows={1}
                    />
                    <div className="absolute right-2 bottom-2 flex items-center gap-1">
                      <RippleButton
                        variant="ghost"
                        size="icon"
                        className="h-8 w-8 text-gray-500"
                        onClick={() => setShowContextMenu(!showContextMenu)}
                      >
                        <AtSign className="w-4 h-4" />
                      </RippleButton>
                      <RippleButton
                        size="icon"
                        className="h-8 w-8 bg-[#FFD700] text-black hover:bg-[#FFC700]"
                        onClick={sendChatMessage}
                        disabled={!chatInput.trim() || isGenerating}
                      >
                        <Send className="w-4 h-4" />
                      </RippleButton>
                    </div>
                  </div>
                  <div className="flex items-center justify-between mt-2 text-xs text-gray-500">
                    <span>Press Enter to send, Shift+Enter for new line</span>
                    <span>Context: {contextFiles.length} files</span>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Terminal */}
        <div className="h-40 border-t border-gray-200 bg-gray-50 flex flex-col">
          <div className="h-8 flex items-center justify-between px-4 bg-gray-100 border-b border-gray-200">
            <div className="flex items-center gap-2">
              <Terminal className="w-4 h-4 text-gray-500" />
              <span className="text-sm text-gray-600">Terminal</span>
            </div>
            <div className="flex items-center gap-1">
              <RippleButton variant="ghost" size="sm" className="h-6 text-xs gap-1 text-gray-500">
                <Sparkles className="w-3 h-3" />
                AI Assist
              </RippleButton>
            </div>
          </div>
          <ScrollArea className="flex-1 p-4 font-mono text-sm">
            {terminalLines.map((line) => (
              <div
                key={line.id}
                className={cn(
                  'mb-1',
                  line.type === 'input' && 'text-gray-600',
                  line.type === 'output' && 'text-green-600',
                  line.type === 'error' && 'text-red-500',
                  line.type === 'ai' && 'text-[#1E88E5]'
                )}
              >
                {line.type === 'input' && <span className="text-gray-500">$ </span>}
                {line.type === 'ai' && <span className="text-[#FFD700]">🤖 </span>}
                {line.content}
              </div>
            ))}
          </ScrollArea>
        </div>
      </div>
    </div>
  );
}
