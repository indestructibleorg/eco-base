import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Globe,
  FolderOpen,
  FileText,
  Terminal,
  MessageSquare,
  Settings,
  X,
  Minus,
  Maximize2,
  Minimize2,
  Cpu,
  Wifi,
  Battery,
  Volume2,
  Search,
  Sparkles,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useDesktopStore } from '@/store/appStore';
import { cn } from '@/lib/utils';
import { FloatingOrb } from './FloatingOrb';

const appComponents: Record<string, React.FC> = {
  browser: BrowserApp,
  files: FilesApp,
  editor: EditorApp,
  terminal: TerminalApp,
  chat: ChatApp,
  settings: SettingsApp,
};

export function DesktopInterface() {
  const { apps, windows, activeWindowId, launchApp, closeWindow, minimizeWindow, maximizeWindow, restoreWindow, focusWindow, updateWindowPosition } = useDesktopStore();
  const [currentTime, setCurrentTime] = useState(new Date());
  const dragRef = useRef<{ windowId: string; offsetX: number; offsetY: number } | null>(null);

  // Update time
  useEffect(() => {
    const interval = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(interval);
  }, []);

  const handleMouseDown = (e: React.MouseEvent, windowId: string) => {
    const window = windows.find((w) => w.id === windowId);
    if (!window || window.isMaximized) return;

    dragRef.current = {
      windowId,
      offsetX: e.clientX - window.x,
      offsetY: e.clientY - window.y,
    };
    focusWindow(windowId);
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!dragRef.current) return;

    const { windowId, offsetX, offsetY } = dragRef.current;
    updateWindowPosition(windowId, e.clientX - offsetX, e.clientY - offsetY);
  };

  const handleMouseUp = () => {
    dragRef.current = null;
  };

  return (
    <div 
      className="h-full relative overflow-hidden"
      style={{
        background: 'linear-gradient(135deg, #1e3a5f 0%, #0f172a 50%, #1e1b4b 100%)',
      }}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
    >
      {/* Desktop Icons */}
      <div className="absolute top-4 left-4 grid grid-cols-1 gap-4">
        {apps.map((app) => (
          <motion.div
            key={app.id}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => launchApp(app.id)}
            className="flex flex-col items-center gap-2 p-3 rounded-xl cursor-pointer hover:bg-white/10 transition-colors"
          >
            <div className={cn('w-14 h-14 rounded-2xl flex items-center justify-center text-2xl shadow-lg', app.color)}>
              {app.icon === '🌐' && <Globe className="w-7 h-7 text-white" />}
              {app.icon === '📁' && <FolderOpen className="w-7 h-7 text-white" />}
              {app.icon === '📝' && <FileText className="w-7 h-7 text-white" />}
              {app.icon === '💻' && <Terminal className="w-7 h-7 text-white" />}
              {app.icon === '💬' && <MessageSquare className="w-7 h-7 text-white" />}
              {app.icon === '⚙️' && <Settings className="w-7 h-7 text-white" />}
            </div>
            <span className="text-white text-xs font-medium drop-shadow-md">{app.name}</span>
          </motion.div>
        ))}
      </div>

      {/* Windows */}
      <AnimatePresence>
        {windows.map((window) => {
          const app = apps.find((a) => a.id === window.appId);
          if (!app || window.isMinimized) return null;

          const AppComponent = appComponents[app.id];

          return (
            <motion.div
              key={window.id}
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ 
                opacity: 1, 
                scale: 1,
                x: window.isMaximized ? 0 : window.x,
                y: window.isMaximized ? 0 : window.y,
                width: window.isMaximized ? '100%' : window.width,
                height: window.isMaximized ? 'calc(100% - 48px)' : window.height,
              }}
              exit={{ opacity: 0, scale: 0.8 }}
              transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
              className={cn(
                'absolute bg-background rounded-xl shadow-2xl overflow-hidden flex flex-col',
                activeWindowId === window.id && 'ring-2 ring-primary/50'
              )}
              style={{ zIndex: window.zIndex }}
              onClick={() => focusWindow(window.id)}
            >
              {/* Window Title Bar */}
              <div
                className="h-10 bg-muted/50 flex items-center justify-between px-3 cursor-move"
                onMouseDown={(e) => handleMouseDown(e, window.id)}
              >
                <div className="flex items-center gap-2">
                  <div className={cn('w-5 h-5 rounded-md flex items-center justify-center', app.color)}>
                    {app.icon === '🌐' && <Globe className="w-3 h-3 text-white" />}
                    {app.icon === '📁' && <FolderOpen className="w-3 h-3 text-white" />}
                    {app.icon === '📝' && <FileText className="w-3 h-3 text-white" />}
                    {app.icon === '💻' && <Terminal className="w-3 h-3 text-white" />}
                    {app.icon === '💬' && <MessageSquare className="w-3 h-3 text-white" />}
                    {app.icon === '⚙️' && <Settings className="w-3 h-3 text-white" />}
                  </div>
                  <span className="text-sm font-medium">{window.title}</span>
                </div>
                <div className="flex items-center gap-1">
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-6 w-6"
                    onClick={() => minimizeWindow(window.id)}
                  >
                    <Minus className="w-3 h-3" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-6 w-6"
                    onClick={() => window.isMaximized ? restoreWindow(window.id) : maximizeWindow(window.id)}
                  >
                    {window.isMaximized ? (
                      <Minimize2 className="w-3 h-3" />
                    ) : (
                      <Maximize2 className="w-3 h-3" />
                    )}
                  </Button>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-6 w-6 hover:bg-red-500 hover:text-white"
                    onClick={() => closeWindow(window.id)}
                  >
                    <X className="w-3 h-3" />
                  </Button>
                </div>
              </div>

              {/* Window Content */}
              <div className="flex-1 overflow-hidden">
                {AppComponent && <AppComponent />}
              </div>
            </motion.div>
          );
        })}
      </AnimatePresence>

      {/* AI Assistant Button in Taskbar */}
      <div className="absolute bottom-16 left-1/2 -translate-x-1/2">
        <motion.button
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.95 }}
          className="w-12 h-12 rounded-full bg-gradient-to-br from-[#FFD700] to-[#FFA500] shadow-lg shadow-[#FFD700]/30 flex items-center justify-center"
        >
          <Sparkles className="w-5 h-5 text-black" />
        </motion.button>
      </div>

      {/* Floating Orb */}
      <FloatingOrb />

      {/* Taskbar */}
      <div className="absolute bottom-0 left-0 right-0 h-12 bg-slate-900/90 backdrop-blur-lg flex items-center justify-between px-4">
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="icon"
            className="h-9 w-9 text-white hover:bg-white/10"
          >
            <Cpu className="w-5 h-5" />
          </Button>
          <div className="h-6 w-px bg-white/20" />
          {apps.map((app) => (
            <Button
              key={app.id}
              variant="ghost"
              size="icon"
              className={cn(
                'h-9 w-9 rounded-lg transition-all',
                app.isOpen 
                  ? 'bg-white/20 text-white' 
                  : 'text-white/70 hover:bg-white/10 hover:text-white'
              )}
              onClick={() => {
                if (app.isOpen) {
                  const window = windows.find((w) => w.appId === app.id && !w.isMinimized);
                  if (window) {
                    if (activeWindowId === window.id) {
                      minimizeWindow(window.id);
                    } else {
                      focusWindow(window.id);
                    }
                  }
                } else {
                  launchApp(app.id);
                }
              }}
            >
              {app.icon === '🌐' && <Globe className="w-4 h-4" />}
              {app.icon === '📁' && <FolderOpen className="w-4 h-4" />}
              {app.icon === '📝' && <FileText className="w-4 h-4" />}
              {app.icon === '💻' && <Terminal className="w-4 h-4" />}
              {app.icon === '💬' && <MessageSquare className="w-4 h-4" />}
              {app.icon === '⚙️' && <Settings className="w-4 h-4" />}
            </Button>
          ))}
        </div>

        <div className="flex items-center gap-4 text-white/80">
          <Wifi className="w-4 h-4" />
          <Volume2 className="w-4 h-4" />
          <Battery className="w-4 h-4" />
          <div className="text-sm">
            {currentTime.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })}
          </div>
        </div>
      </div>
    </div>
  );
}

// App Components
function BrowserApp() {
  const [url, setUrl] = useState('https://www.google.com');
  const [inputUrl, setInputUrl] = useState('https://www.google.com');

  return (
    <div className="h-full flex flex-col bg-white">
      <div className="h-12 bg-muted flex items-center gap-2 px-3">
        <div className="flex gap-1">
          <Button variant="ghost" size="icon" className="h-8 w-8">
            <span className="text-lg">←</span>
          </Button>
          <Button variant="ghost" size="icon" className="h-8 w-8">
            <span className="text-lg">→</span>
          </Button>
          <Button variant="ghost" size="icon" className="h-8 w-8">
            <span className="text-lg">↻</span>
          </Button>
        </div>
        <div className="flex-1 flex items-center gap-2 bg-background rounded-lg px-3 py-1.5">
          <Search className="w-4 h-4 text-muted-foreground" />
          <input
            type="text"
            value={inputUrl}
            onChange={(e) => setInputUrl(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                setUrl(inputUrl);
              }
            }}
            className="flex-1 bg-transparent outline-none text-sm"
          />
        </div>
      </div>
      <div className="flex-1 bg-slate-50 flex items-center justify-center">
        <div className="text-center">
          <Globe className="w-16 h-16 text-muted-foreground mx-auto mb-4" />
          <p className="text-muted-foreground">浏览器模拟器</p>
          <p className="text-sm text-muted-foreground">当前 URL: {url}</p>
        </div>
      </div>
    </div>
  );
}

function FilesApp() {
  const folders = [
    { name: '文档', icon: '📄', count: 12 },
    { name: '图片', icon: '🖼️', count: 48 },
    { name: '视频', icon: '🎬', count: 5 },
    { name: '下载', icon: '⬇️', count: 23 },
    { name: '音乐', icon: '🎵', count: 156 },
  ];

  return (
    <div className="h-full flex bg-white">
      <div className="w-48 bg-muted/30 p-4">
        <div className="space-y-1">
          {['主页', '文档', '图片', '下载', '回收站'].map((item) => (
            <div
              key={item}
              className="px-3 py-2 rounded-lg hover:bg-accent cursor-pointer text-sm"
            >
              {item}
            </div>
          ))}
        </div>
      </div>
      <div className="flex-1 p-6">
        <h2 className="text-lg font-medium mb-4">文件夹</h2>
        <div className="grid grid-cols-4 gap-4">
          {folders.map((folder) => (
            <div
              key={folder.name}
              className="flex flex-col items-center p-4 rounded-xl hover:bg-accent cursor-pointer transition-colors"
            >
              <span className="text-4xl mb-2">{folder.icon}</span>
              <span className="text-sm font-medium">{folder.name}</span>
              <span className="text-xs text-muted-foreground">{folder.count} 项</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function EditorApp() {
  const [content, setContent] = useState('欢迎使用 AutoEcoops 编辑器\n\n这是一个简单的文本编辑器。');

  return (
    <div className="h-full flex flex-col bg-white">
      <div className="h-10 bg-muted flex items-center px-3 gap-2">
        <Button variant="ghost" size="sm">文件</Button>
        <Button variant="ghost" size="sm">编辑</Button>
        <Button variant="ghost" size="sm">查看</Button>
      </div>
      <textarea
        value={content}
        onChange={(e) => setContent(e.target.value)}
        className="flex-1 p-4 resize-none outline-none font-mono text-sm"
        placeholder="在此输入文本..."
      />
    </div>
  );
}

function TerminalApp() {
  const [lines, setLines] = useState([
    { type: 'output', content: 'AutoEcoops Terminal v1.0' },
    { type: 'output', content: 'Type "help" for available commands' },
  ]);
  const [input, setInput] = useState('');

  const handleCommand = () => {
    const newLines = [...lines, { type: 'input', content: `$ ${input}` }];
    
    const command = input.trim().toLowerCase();
    if (command === 'help') {
      newLines.push(
        { type: 'output', content: 'Available commands:' },
        { type: 'output', content: '  help    - Show this help message' },
        { type: 'output', content: '  clear   - Clear terminal' },
        { type: 'output', content: '  echo    - Echo text' },
        { type: 'output', content: '  date    - Show current date' },
        { type: 'output', content: '  whoami  - Show current user' }
      );
    } else if (command === 'clear') {
      setLines([]);
      setInput('');
      return;
    } else if (command.startsWith('echo ')) {
      newLines.push({ type: 'output', content: command.slice(5) });
    } else if (command === 'date') {
      newLines.push({ type: 'output', content: new Date().toString() });
    } else if (command === 'whoami') {
      newLines.push({ type: 'output', content: 'user@autocoops' });
    } else if (command) {
      newLines.push({ type: 'error', content: `Command not found: ${command}` });
    }

    setLines(newLines);
    setInput('');
  };

  return (
    <div className="h-full flex flex-col bg-slate-900 text-slate-100 p-4 font-mono text-sm">
      <div className="flex-1 overflow-auto space-y-1">
        {lines.map((line, i) => (
          <div
            key={i}
            className={cn(
              line.type === 'input' && 'text-slate-300',
              line.type === 'error' && 'text-red-400'
            )}
          >
            {line.content}
          </div>
        ))}
      </div>
      <div className="flex items-center gap-2 mt-2">
        <span className="text-green-400">$</span>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleCommand()}
          className="flex-1 bg-transparent outline-none text-slate-100"
          autoFocus
        />
      </div>
    </div>
  );
}

function ChatApp() {
  const [messages, setMessages] = useState([
    { role: 'assistant', content: '你好！我是 AutoEcoops AI 助手。有什么可以帮您的？' },
  ]);
  const [input, setInput] = useState('');

  const handleSend = () => {
    if (!input.trim()) return;
    
    setMessages([...messages, { role: 'user', content: input }]);
    setInput('');
    
    setTimeout(() => {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: '收到您的消息！这是一个模拟回复。' },
      ]);
    }, 1000);
  };

  return (
    <div className="h-full flex flex-col bg-white">
      <div className="flex-1 overflow-auto p-4 space-y-4">
        {messages.map((msg, i) => (
          <div
            key={i}
            className={cn(
              'flex',
              msg.role === 'user' && 'justify-end'
            )}
          >
            <div
              className={cn(
                'max-w-[80%] rounded-2xl px-4 py-2',
                msg.role === 'user'
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-muted'
              )}
            >
              {msg.content}
            </div>
          </div>
        ))}
      </div>
      <div className="p-4 border-t">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
            placeholder="输入消息..."
            className="flex-1 px-4 py-2 border rounded-lg outline-none focus:ring-2 focus:ring-primary"
          />
          <Button onClick={handleSend}>发送</Button>
        </div>
      </div>
    </div>
  );
}

function SettingsApp() {
  const settings = [
    { name: '显示', icon: '🖥️' },
    { name: '声音', icon: '🔊' },
    { name: '网络', icon: '🌐' },
    { name: '隐私', icon: '🔒' },
    { name: '系统', icon: '⚙️' },
  ];

  return (
    <div className="h-full flex bg-white">
      <div className="w-48 bg-muted/30 p-4">
        {settings.map((setting) => (
          <div
            key={setting.name}
            className="flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-accent cursor-pointer"
          >
            <span>{setting.icon}</span>
            <span className="text-sm">{setting.name}</span>
          </div>
        ))}
      </div>
      <div className="flex-1 p-6">
        <h2 className="text-xl font-medium mb-4">系统设置</h2>
        <div className="space-y-4">
          <div className="flex items-center justify-between p-4 border rounded-lg">
            <div>
              <div className="font-medium">深色模式</div>
              <div className="text-sm text-muted-foreground">启用系统级深色主题</div>
            </div>
            <Button variant="outline">切换</Button>
          </div>
          <div className="flex items-center justify-between p-4 border rounded-lg">
            <div>
              <div className="font-medium">自动更新</div>
              <div className="text-sm text-muted-foreground">自动下载并安装更新</div>
            </div>
            <Button variant="outline">启用</Button>
          </div>
        </div>
      </div>
    </div>
  );
}
