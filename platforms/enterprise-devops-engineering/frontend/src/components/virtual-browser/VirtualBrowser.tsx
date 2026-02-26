import { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Globe,
  ArrowLeft,
  ArrowRight,
  RefreshCw,
  Home,
  Lock,
  Search,
  Plus,
  X,
  Maximize2,
  Minimize2,
  Bookmark,
  Terminal,
  MousePointer,
  Play,
  Check,
  AlertCircle,
  Loader2,
} from 'lucide-react';
import { RippleButton } from '@/components/ui/ripple-button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { cn } from '@/lib/utils';

// Browser Tab Type
interface BrowserTab {
  id: string;
  url: string;
  title: string;
  favicon?: string;
  isLoading: boolean;
}

// Browser Action Type (for automation recording)
interface BrowserAction {
  id: string;
  type: 'navigate' | 'click' | 'type' | 'scroll' | 'screenshot' | 'wait';
  timestamp: Date;
  data: Record<string, any>;
  status: 'pending' | 'running' | 'completed' | 'failed';
}

// Automation Session Type
interface AutomationSession {
  id: string;
  name: string;
  description: string;
  actions: BrowserAction[];
  isRecording: boolean;
  isPlaying: boolean;
}

// Virtual Browser Props
interface VirtualBrowserProps {
  initialUrl?: string;
  onAction?: (action: BrowserAction) => void;
  automationMode?: boolean;
}

export function VirtualBrowser({ 
  initialUrl = 'https://www.google.com',
  onAction,
  automationMode = false 
}: VirtualBrowserProps) {
  const [tabs, setTabs] = useState<BrowserTab[]>([
    { id: 'tab-1', url: initialUrl, title: 'Google', isLoading: false },
  ]);
  const [activeTabId, setActiveTabId] = useState('tab-1');
  const [urlInput, setUrlInput] = useState(initialUrl);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showDevTools, setShowDevTools] = useState(false);
  const [automationSession, setAutomationSession] = useState<AutomationSession | null>(null);
  const browserRef = useRef<HTMLDivElement>(null);

  const activeTab = tabs.find((t) => t.id === activeTabId);

  // Navigation functions
  const navigateTo = (url: string) => {
    const newUrl = url.startsWith('http') ? url : `https://${url}`;
    setTabs((prev) =>
      prev.map((t) =>
        t.id === activeTabId ? { ...t, url: newUrl, isLoading: true } : t
      )
    );
    setUrlInput(newUrl);
    
    // Record action if in automation mode
    if (automationSession?.isRecording) {
      recordAction('navigate', { url: newUrl });
    }
  };

  const addTab = () => {
    const newTab: BrowserTab = {
      id: `tab-${Date.now()}`,
      url: 'https://www.google.com',
      title: 'New Tab',
      isLoading: false,
    };
    setTabs((prev) => [...prev, newTab]);
    setActiveTabId(newTab.id);
    setUrlInput(newTab.url);
  };

  const closeTab = (tabId: string) => {
    if (tabs.length === 1) return;
    const newTabs = tabs.filter((t) => t.id !== tabId);
    setTabs(newTabs);
    if (activeTabId === tabId) {
      setActiveTabId(newTabs[newTabs.length - 1].id);
    }
  };

  // Automation functions
  const startRecording = () => {
    setAutomationSession({
      id: `session-${Date.now()}`,
      name: 'New Session',
      description: '',
      actions: [],
      isRecording: true,
      isPlaying: false,
    });
  };

  const stopRecording = () => {
    setAutomationSession((prev) =>
      prev ? { ...prev, isRecording: false } : null
    );
  };

  const recordAction = (type: BrowserAction['type'], data: Record<string, any>) => {
    const action: BrowserAction = {
      id: `action-${Date.now()}`,
      type,
      timestamp: new Date(),
      data,
      status: 'completed',
    };
    setAutomationSession((prev) =>
      prev
        ? { ...prev, actions: [...prev.actions, action] }
        : null
    );
    onAction?.(action);
  };

  const playAutomation = async () => {
    if (!automationSession) return;
    setAutomationSession((prev) => (prev ? { ...prev, isPlaying: true } : null));

    for (const action of automationSession.actions) {
      setAutomationSession((prev) =>
        prev
          ? {
              ...prev,
              actions: prev.actions.map((a) =>
                a.id === action.id ? { ...a, status: 'running' } : a
              ),
            }
          : null
      );

      // Execute action
      await executeAction(action);

      setAutomationSession((prev) =>
        prev
          ? {
              ...prev,
              actions: prev.actions.map((a) =>
                a.id === action.id ? { ...a, status: 'completed' } : a
              ),
            }
          : null
      );

      await new Promise((r) => setTimeout(r, 500));
    }

    setAutomationSession((prev) => (prev ? { ...prev, isPlaying: false } : null));
  };

  const executeAction = async (action: BrowserAction) => {
    switch (action.type) {
      case 'navigate':
        navigateTo(action.data.url);
        break;
      case 'click':
        // Simulate click on element
        break;
      case 'type':
        // Simulate typing
        break;
      case 'wait':
        await new Promise((r) => setTimeout(r, action.data.duration || 1000));
        break;
    }
  };

  // Handle key events for automation
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && e.target === document.activeElement) {
      navigateTo(urlInput);
    }
  };

  return (
    <div
      ref={browserRef}
      className={cn(
        'flex flex-col bg-white h-full rounded-xl overflow-hidden border border-gray-200',
        isFullscreen && 'fixed inset-0 z-50 rounded-none'
      )}
    >
      {/* Browser Toolbar */}
      <div className="h-12 bg-gray-50 border-b border-gray-200 flex items-center px-3 gap-2">
        {/* Navigation Buttons */}
        <div className="flex items-center gap-1">
          <RippleButton variant="ghost" size="icon" className="h-8 w-8 text-gray-500">
            <ArrowLeft className="w-4 h-4" />
          </RippleButton>
          <RippleButton variant="ghost" size="icon" className="h-8 w-8 text-gray-500">
            <ArrowRight className="w-4 h-4" />
          </RippleButton>
          <RippleButton variant="ghost" size="icon" className="h-8 w-8 text-gray-500">
            <RefreshCw className="w-4 h-4" />
          </RippleButton>
          <RippleButton variant="ghost" size="icon" className="h-8 w-8 text-gray-500">
            <Home className="w-4 h-4" />
          </RippleButton>
        </div>

        {/* Address Bar */}
        <div className="flex-1 flex items-center gap-2 bg-white rounded-lg px-3 py-1.5 border border-gray-200 focus-within:border-[#FFD700] transition-colors">
          <Lock className="w-3 h-3 text-green-500" />
          <input
            type="text"
            value={urlInput}
            onChange={(e) => setUrlInput(e.target.value)}
            onKeyDown={handleKeyDown}
            className="flex-1 bg-transparent outline-none text-sm text-gray-900 placeholder:text-gray-500"
            placeholder="Enter URL or search..."
          />
        </div>

        {/* Right Actions */}
        <div className="flex items-center gap-1">
          {automationMode && (
            <RippleButton
              variant={automationSession?.isRecording ? 'destructive' : 'outline'}
              size="sm"
              onClick={automationSession?.isRecording ? stopRecording : startRecording}
              className="gap-1"
            >
              {automationSession?.isRecording ? (
                <>
                  <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
                  Stop
                </>
              ) : (
                <>
                  <Play className="w-3 h-3" />
                  Record
                </>
              )}
            </RippleButton>
          )}
          <RippleButton
            variant="ghost"
            size="icon"
            className={cn('h-8 w-8', showDevTools && 'text-[#FFD700] bg-[#FFD700]/10')}
            onClick={() => setShowDevTools(!showDevTools)}
          >
            <Terminal className="w-4 h-4" />
          </RippleButton>
          <RippleButton variant="ghost" size="icon" className="h-8 w-8 text-gray-500">
            <Bookmark className="w-4 h-4" />
          </RippleButton>
          <RippleButton
            variant="ghost"
            size="icon"
            className="h-8 w-8 text-gray-500"
            onClick={() => setIsFullscreen(!isFullscreen)}
          >
            {isFullscreen ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
          </RippleButton>
        </div>
      </div>

      {/* Tabs */}
      <div className="h-9 bg-white flex items-center px-2 gap-1 border-b border-gray-200">
        {tabs.map((tab) => (
          <div
            key={tab.id}
            onClick={() => {
              setActiveTabId(tab.id);
              setUrlInput(tab.url);
            }}
            className={cn(
              'flex items-center gap-2 px-3 py-1.5 rounded-t-lg cursor-pointer min-w-[120px] max-w-[200px] transition-all',
              activeTabId === tab.id
                ? 'bg-gray-50 text-gray-900'
                : 'text-gray-500 hover:bg-gray-50 hover:text-gray-700'
            )}
          >
            {tab.isLoading ? (
              <Loader2 className="w-3 h-3 animate-spin text-[#FFD700]" />
            ) : (
              <Globe className="w-3 h-3" />
            )}
            <span className="flex-1 truncate text-sm">{tab.title}</span>
            <button
              onClick={(e) => {
                e.stopPropagation();
                closeTab(tab.id);
              }}
              className="opacity-0 group-hover:opacity-100 hover:opacity-100 text-gray-500 hover:text-gray-900"
            >
              <X className="w-3 h-3" />
            </button>
          </div>
        ))}
        <RippleButton variant="ghost" size="icon" className="h-7 w-7 text-gray-500" onClick={addTab}>
          <Plus className="w-4 h-4" />
        </RippleButton>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 flex overflow-hidden">
        {/* Browser Viewport */}
        <div className="flex-1 relative bg-white">
          {activeTab?.isLoading && (
            <div className="absolute inset-0 flex items-center justify-center bg-white z-10">
              <div className="flex flex-col items-center gap-4">
                <Loader2 className="w-8 h-8 animate-spin text-[#FFD700]" />
                <span className="text-sm text-gray-500">Loading...</span>
              </div>
            </div>
          )}
          
          {/* Simulated Browser Content */}
          <div className="w-full h-full overflow-auto">
            <BrowserContent url={activeTab?.url || ''} />
          </div>


        </div>

        {/* DevTools Panel */}
        <AnimatePresence>
          {showDevTools && (
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: 400 }}
              exit={{ width: 0 }}
              className="border-l border-gray-200 bg-white overflow-hidden"
            >
              <DevToolsPanel 
                automationSession={automationSession}
                onPlayAutomation={playAutomation}
              />
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}

// Simulated Browser Content
function BrowserContent({ url }: { url: string }) {
  const hostname = new URL(url.startsWith('http') ? url : `https://${url}`).hostname;

  if (hostname.includes('google')) {
    return (
      <div className="flex flex-col items-center justify-center min-h-full p-8">
        <div className="text-6xl font-bold mb-8">
          <span className="text-blue-500">G</span>
          <span className="text-red-500">o</span>
          <span className="text-yellow-500">o</span>
          <span className="text-blue-500">g</span>
          <span className="text-green-500">l</span>
          <span className="text-red-500">e</span>
        </div>
        <div className="w-full max-w-xl">
          <div className="flex items-center gap-2 px-4 py-3 border rounded-full shadow-lg hover:shadow-xl transition-shadow">
            <Search className="w-5 h-5 text-gray-400" />
            <input
              type="text"
              className="flex-1 outline-none"
              placeholder="Search Google or type a URL"
            />
          </div>
        </div>
        <div className="flex gap-3 mt-8">
          <button className="px-4 py-2 bg-gray-100 hover:bg-gray-200 rounded text-sm">Google Search</button>
          <button className="px-4 py-2 bg-gray-100 hover:bg-gray-200 rounded text-sm">I'm Feeling Lucky</button>
        </div>
      </div>
    );
  }

  if (hostname.includes('github')) {
    return (
      <div className="min-h-full bg-white">
        <div className="bg-[#24292f] text-white px-4 py-3 flex items-center gap-4">
          <div className="font-bold text-xl">GitHub</div>
          <div className="flex-1">
            <input
              type="text"
              className="w-full max-w-md px-3 py-1.5 bg-[#24292f] border border-gray-600 rounded text-sm"
              placeholder="Search or jump to..."
            />
          </div>
        </div>
        <div className="p-8">
          <h1 className="text-2xl font-bold mb-4">Welcome to GitHub</h1>
          <p className="text-gray-600">The world's leading software development platform.</p>
        </div>
      </div>
    );
  }

  // Default content
  return (
    <div className="flex flex-col items-center justify-center min-h-full p-8 text-center">
      <Globe className="w-16 h-16 text-gray-300 mb-4" />
      <h1 className="text-xl font-semibold text-gray-700 mb-2">{hostname}</h1>
      <p className="text-gray-500">This is a simulated browser environment.</p>
      <p className="text-gray-400 text-sm mt-2">URL: {url}</p>
    </div>
  );
}

// DevTools Panel
function DevToolsPanel({ 
  automationSession, 
  onPlayAutomation 
}: { 
  automationSession: AutomationSession | null;
  onPlayAutomation: () => void;
}) {
  const [activeTab, setActiveTab] = useState('console');

  return (
    <div className="h-full flex flex-col">
      <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1 flex flex-col">
        <TabsList className="bg-gray-50 border-b border-gray-200 rounded-none h-9">
          <TabsTrigger value="console" className="text-xs data-[state=active]:bg-[#FFD700]/10 data-[state=active]:text-[#FFD700]">
            Console
          </TabsTrigger>
          <TabsTrigger value="elements" className="text-xs data-[state=active]:bg-[#FFD700]/10 data-[state=active]:text-[#FFD700]">
            Elements
          </TabsTrigger>
          <TabsTrigger value="automation" className="text-xs data-[state=active]:bg-[#FFD700]/10 data-[state=active]:text-[#FFD700]">
            Automation
            {automationSession && automationSession.actions.length > 0 && (
              <span className="ml-1 px-1.5 py-0.5 bg-[#FFD700] text-black text-[10px] rounded-full">
                {automationSession.actions.length}
              </span>
            )}
          </TabsTrigger>
        </TabsList>

        <TabsContent value="console" className="flex-1 m-0">
          <ScrollArea className="h-full p-2">
            <div className="font-mono text-xs space-y-1">
              <div className="text-gray-500">[Virtual Browser] Console initialized</div>
              <div className="text-green-600">[Virtual Browser] Connected to automation engine</div>
              <div className="text-blue-600">[Virtual Browser] Ready for commands</div>
            </div>
          </ScrollArea>
        </TabsContent>

        <TabsContent value="elements" className="flex-1 m-0">
          <ScrollArea className="h-full p-2">
            <div className="font-mono text-xs text-gray-600">
              &lt;html&gt;<br/>
              &nbsp;&nbsp;&lt;head&gt;...&lt;/head&gt;<br/>
              &nbsp;&nbsp;&lt;body&gt;<br/>
              &nbsp;&nbsp;&nbsp;&nbsp;&lt;div id="root"&gt;...&lt;/div&gt;<br/>
              &nbsp;&nbsp;&lt;/body&gt;<br/>
              &lt;/html&gt;
            </div>
          </ScrollArea>
        </TabsContent>

        <TabsContent value="automation" className="flex-1 m-0 flex flex-col">
          {automationSession ? (
            <>
              <div className="p-3 border-b border-gray-200 flex items-center justify-between">
                <div>
                  <div className="font-medium text-sm text-gray-900">{automationSession.name}</div>
                  <div className="text-xs text-gray-500">
                    {automationSession.actions.length} actions recorded
                  </div>
                </div>
                <div className="flex gap-2">
                  {!automationSession.isRecording && (
                    <RippleButton
                      size="sm"
                      onClick={onPlayAutomation}
                      disabled={automationSession.isPlaying}
                      className="gap-1 bg-[#FFD700] text-black hover:bg-[#FFC700]"
                    >
                      {automationSession.isPlaying ? (
                        <Loader2 className="w-3 h-3 animate-spin" />
                      ) : (
                        <Play className="w-3 h-3" />
                      )}
                      Play
                    </RippleButton>
                  )}
                </div>
              </div>
              <ScrollArea className="flex-1 p-2">
                <div className="space-y-1">
                  {automationSession.actions.map((action, index) => (
                    <div
                      key={action.id}
                      className={cn(
                        'flex items-center gap-2 p-2 rounded text-xs',
                        action.status === 'running' && 'bg-[#FFD700]/20',
                        action.status === 'completed' && 'bg-green-100',
                        action.status === 'failed' && 'bg-red-100'
                      )}
                    >
                      <span className="text-gray-500 w-5">{index + 1}</span>
                      <span className="capitalize font-medium text-gray-900">{action.type}</span>
                      <span className="text-gray-500 flex-1 truncate">
                        {JSON.stringify(action.data).slice(0, 50)}...
                      </span>
                      {action.status === 'completed' && <Check className="w-3 h-3 text-green-500" />}
                      {action.status === 'failed' && <AlertCircle className="w-3 h-3 text-red-500" />}
                      {action.status === 'running' && <Loader2 className="w-3 h-3 animate-spin text-[#FFD700]" />}
                    </div>
                  ))}
                </div>
              </ScrollArea>
            </>
          ) : (
            <div className="flex flex-col items-center justify-center h-full text-center p-4">
              <MousePointer className="w-12 h-12 text-gray-400 mb-4" />
              <p className="text-sm text-gray-600">No automation session active</p>
              <p className="text-xs text-gray-500 mt-1">Click Record to start capturing actions</p>
            </div>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}


