import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { 
  AppState, 
  Chat, 
  Message, 
  FileNode, 
  EditorTab, 
  TerminalLine,
  DesktopApp,
  AppWindow,
  AgentNode,
  AgentConnection,
  Workflow,
  Integration,
  IntegrationType,
  Team,
  Activity,
  Deployment,
  Notification,
} from '@/types';

// Chat Store
interface ChatStore {
  chats: Chat[];
  currentChatId: string | null;
  models: { id: string; name: string; provider: string; icon: string }[];
  selectedModel: string;
  isStreaming: boolean;
  createChat: () => string;
  deleteChat: (id: string) => void;
  setCurrentChat: (id: string) => void;
  addMessage: (chatId: string, message: Message) => void;
  updateMessage: (chatId: string, messageId: string, content: string) => void;
  setSelectedModel: (model: string) => void;
  setIsStreaming: (streaming: boolean) => void;
}

export const useChatStore = create<ChatStore>()(
  persist(
    (set) => ({
      chats: [],
      currentChatId: null,
      models: [
        { id: 'gpt-4', name: 'GPT-4', provider: 'OpenAI', icon: '🤖' },
        { id: 'gpt-4o', name: 'GPT-4o', provider: 'OpenAI', icon: '⚡' },
        { id: 'claude-3-opus', name: 'Claude 3 Opus', provider: 'Anthropic', icon: '🧠' },
        { id: 'claude-3-sonnet', name: 'Claude 3 Sonnet', provider: 'Anthropic', icon: '✨' },
        { id: 'llama-3-70b', name: 'Llama 3 70B', provider: 'Meta', icon: '🦙' },
        { id: 'gemini-pro', name: 'Gemini Pro', provider: 'Google', icon: '💎' },
      ],
      selectedModel: 'gpt-4o',
      isStreaming: false,

      createChat: () => {
        const newChat: Chat = {
          id: `chat-${Date.now()}`,
          title: '新对话',
          messages: [],
          createdAt: new Date(),
          updatedAt: new Date(),
          model: 'gpt-4o',
        };
        set((state) => ({
          chats: [newChat, ...state.chats],
          currentChatId: newChat.id,
        }));
        return newChat.id;
      },

      deleteChat: (id) => {
        set((state) => ({
          chats: state.chats.filter((c) => c.id !== id),
          currentChatId: state.currentChatId === id 
            ? (state.chats.find((c) => c.id !== id)?.id || null)
            : state.currentChatId,
        }));
      },

      setCurrentChat: (id) => {
        set({ currentChatId: id });
      },

      addMessage: (chatId, message) => {
        set((state) => ({
          chats: state.chats.map((chat) =>
            chat.id === chatId
              ? { 
                  ...chat, 
                  messages: [...chat.messages, message],
                  updatedAt: new Date(),
                  title: chat.messages.length === 0 && message.role === 'user' 
                    ? message.content.slice(0, 30) + (message.content.length > 30 ? '...' : '')
                    : chat.title
                }
              : chat
          ),
        }));
      },

      updateMessage: (chatId, messageId, content) => {
        set((state) => ({
          chats: state.chats.map((chat) =>
            chat.id === chatId
              ? {
                  ...chat,
                  messages: chat.messages.map((m) =>
                    m.id === messageId ? { ...m, content } : m
                  ),
                }
              : chat
          ),
        }));
      },

      setSelectedModel: (model) => set({ selectedModel: model }),
      setIsStreaming: (streaming) => set({ isStreaming: streaming }),
    }),
    {
      name: 'chat-store',
    }
  )
);

// IDE Store
interface IDEStore {
  files: FileNode[];
  openTabs: EditorTab[];
  activeTabId: string | null;
  terminalLines: TerminalLine[];
  selectedFileId: string | null;
  createFile: (name: string, parentId?: string | null) => void;
  createFolder: (name: string, parentId?: string | null) => void;
  deleteFile: (id: string) => void;
  renameFile: (id: string, newName: string) => void;
  openFile: (file: FileNode) => void;
  closeTab: (tabId: string) => void;
  updateTabContent: (tabId: string, content: string) => void;
  setActiveTab: (tabId: string) => void;
  addTerminalLine: (line: TerminalLine) => void;
  clearTerminal: () => void;
}

const defaultFiles: FileNode[] = [
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
        parentId: 'root',
        children: [
          {
            id: 'main-py',
            name: 'main.py',
            type: 'file',
            language: 'python',
            parentId: 'src',
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
            parentId: 'src',
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
        id: 'tests',
        name: 'tests',
        type: 'folder',
        isOpen: false,
        parentId: 'root',
        children: [
          {
            id: 'test-main',
            name: 'test_main.py',
            type: 'file',
            language: 'python',
            parentId: 'tests',
            content: `import unittest
from src.main import fibonacci

class TestFibonacci(unittest.TestCase):
    def test_base_cases(self):
        self.assertEqual(fibonacci(0), 0)
        self.assertEqual(fibonacci(1), 1)
    
    def test_recursive(self):
        self.assertEqual(fibonacci(5), 5)
        self.assertEqual(fibonacci(10), 55)

if __name__ == '__main__':
    unittest.main()`,
          },
        ],
      },
      {
        id: 'readme',
        name: 'README.md',
        type: 'file',
        language: 'markdown',
        parentId: 'root',
        content: `# My Project

A sample Python project with fibonacci calculations.

## Getting Started

\`\`\`bash
python src/main.py
\`\`\`

## Running Tests

\`\`\`bash
python -m unittest discover tests
\`\`\`

## Features

- Fibonacci sequence calculator
- Utility functions
- Unit tests`,
      },
      {
        id: 'requirements',
        name: 'requirements.txt',
        type: 'file',
        language: 'text',
        parentId: 'root',
        content: '# Dependencies\npytest>=7.0.0\nblack>=22.0.0\nflake8>=4.0.0',
      },
    ],
  },
];

export const useIDEStore = create<IDEStore>()((set, get) => ({
  files: defaultFiles,
  openTabs: [],
  activeTabId: null,
  terminalLines: [
    { id: '1', type: 'info', content: 'AutoEcoops IDE v2.0 Pro', timestamp: new Date() },
    { id: '2', type: 'info', content: 'Type "help" for available commands', timestamp: new Date() },
  ],
  selectedFileId: null,

  createFile: (name, parentId = null) => {
    const newFile: FileNode = {
      id: `file-${Date.now()}`,
      name,
      type: 'file',
      language: name.split('.').pop() || 'text',
      parentId,
      content: '',
      lastModified: new Date(),
    };
    set((state) => ({
      files: addNodeToTree(state.files, newFile, parentId),
    }));
  },

  createFolder: (name, parentId = null) => {
    const newFolder: FileNode = {
      id: `folder-${Date.now()}`,
      name,
      type: 'folder',
      parentId,
      isOpen: true,
      children: [],
    };
    set((state) => ({
      files: addNodeToTree(state.files, newFolder, parentId),
    }));
  },

  deleteFile: (id) => {
    set((state) => ({
      files: removeNodeFromTree(state.files, id),
      openTabs: state.openTabs.filter((t) => t.fileId !== id),
    }));
  },

  renameFile: (id, newName) => {
    set((state) => ({
      files: updateNodeInTree(state.files, id, (node) => ({ 
        ...node, 
        name: newName,
        lastModified: new Date(),
      })),
    }));
  },

  openFile: (file) => {
    if (file.type !== 'file') return;
    
    const state = get();
    const existingTab = state.openTabs.find((t) => t.fileId === file.id);
    
    if (existingTab) {
      set({ activeTabId: existingTab.id });
    } else {
      const newTab: EditorTab = {
        id: `tab-${Date.now()}`,
        fileId: file.id,
        name: file.name,
        language: file.language || 'text',
        content: file.content || '',
        isModified: false,
        lastSaved: new Date(),
      };
      set((s) => ({
        openTabs: [...s.openTabs, newTab],
        activeTabId: newTab.id,
      }));
    }
  },

  closeTab: (tabId) => {
    set((state) => {
      const newTabs = state.openTabs.filter((t) => t.id !== tabId);
      return {
        openTabs: newTabs,
        activeTabId: state.activeTabId === tabId 
          ? (newTabs[newTabs.length - 1]?.id || null)
          : state.activeTabId,
      };
    });
  },

  updateTabContent: (tabId, content) => {
    set((state) => ({
      openTabs: state.openTabs.map((t) =>
        t.id === tabId ? { ...t, content, isModified: true } : t
      ),
    }));
  },

  setActiveTab: (tabId) => set({ activeTabId: tabId }),

  addTerminalLine: (line) => {
    set((state) => ({
      terminalLines: [...state.terminalLines, line],
    }));
  },

  clearTerminal: () => set({ terminalLines: [] }),
}));

// Desktop Store
interface DesktopStore {
  apps: DesktopApp[];
  windows: AppWindow[];
  activeWindowId: string | null;
  nextZIndex: number;
  launchApp: (appId: string) => void;
  closeWindow: (windowId: string) => void;
  minimizeWindow: (windowId: string) => void;
  maximizeWindow: (windowId: string) => void;
  restoreWindow: (windowId: string) => void;
  focusWindow: (windowId: string) => void;
  updateWindowPosition: (windowId: string, x: number, y: number) => void;
}

const defaultApps: DesktopApp[] = [
  { id: 'browser', name: '浏览器', icon: '🌐', color: 'bg-blue-500', isOpen: false },
  { id: 'files', name: '文件管理', icon: '📁', color: 'bg-yellow-500', isOpen: false },
  { id: 'editor', name: '编辑器', icon: '📝', color: 'bg-green-500', isOpen: false },
  { id: 'terminal', name: '终端', icon: '💻', color: 'bg-gray-700', isOpen: false },
  { id: 'chat', name: 'AI 助手', icon: '💬', color: 'bg-purple-500', isOpen: false },
  { id: 'settings', name: '设置', icon: '⚙️', color: 'bg-gray-500', isOpen: false },
];

export const useDesktopStore = create<DesktopStore>()((set, get) => ({
  apps: defaultApps,
  windows: [],
  activeWindowId: null,
  nextZIndex: 100,

  launchApp: (appId) => {
    const state = get();
    const app = state.apps.find((a) => a.id === appId);
    if (!app) return;

    const existingWindow = state.windows.find((w) => w.appId === appId && !w.isMinimized);
    if (existingWindow) {
      get().focusWindow(existingWindow.id);
      return;
    }

    const newWindow: AppWindow = {
      id: `win-${Date.now()}`,
      appId,
      title: app.name,
      x: 100 + state.windows.length * 30,
      y: 50 + state.windows.length * 30,
      width: 900,
      height: 650,
      isMinimized: false,
      isMaximized: false,
      zIndex: state.nextZIndex,
    };

    set((s) => ({
      windows: [...s.windows, newWindow],
      activeWindowId: newWindow.id,
      nextZIndex: s.nextZIndex + 1,
      apps: s.apps.map((a) => (a.id === appId ? { ...a, isOpen: true } : a)),
    }));
  },

  closeWindow: (windowId) => {
    set((state) => {
      const window = state.windows.find((w) => w.id === windowId);
      return {
        windows: state.windows.filter((w) => w.id !== windowId),
        apps: window 
          ? state.apps.map((a) => (a.id === window.appId ? { ...a, isOpen: false } : a))
          : state.apps,
        activeWindowId: state.activeWindowId === windowId 
          ? (state.windows.find((w) => w.id !== windowId && !w.isMinimized)?.id || null)
          : state.activeWindowId,
      };
    });
  },

  minimizeWindow: (windowId) => {
    set((state) => ({
      windows: state.windows.map((w) =>
        w.id === windowId ? { ...w, isMinimized: true } : w
      ),
      activeWindowId: state.activeWindowId === windowId 
        ? (state.windows.find((w) => w.id !== windowId && !w.isMinimized)?.id || null)
        : state.activeWindowId,
    }));
  },

  maximizeWindow: (windowId) => {
    set((state) => ({
      windows: state.windows.map((w) =>
        w.id === windowId ? { ...w, isMaximized: true } : w
      ),
    }));
  },

  restoreWindow: (windowId) => {
    set((state) => ({
      windows: state.windows.map((w) =>
        w.id === windowId ? { ...w, isMaximized: false, isMinimized: false } : w
      ),
      activeWindowId: windowId,
    }));
  },

  focusWindow: (windowId) => {
    set((state) => ({
      windows: state.windows.map((w) =>
        w.id === windowId ? { ...w, zIndex: state.nextZIndex } : w
      ),
      activeWindowId: windowId,
      nextZIndex: state.nextZIndex + 1,
    }));
  },

  updateWindowPosition: (windowId, x, y) => {
    set((state) => ({
      windows: state.windows.map((w) =>
        w.id === windowId ? { ...w, x, y } : w
      ),
    }));
  },
}));

// Agent Store
interface AgentStore {
  workflows: Workflow[];
  currentWorkflowId: string | null;
  nodes: AgentNode[];
  connections: AgentConnection[];
  selectedNodeId: string | null;
  isRunning: boolean;
  createWorkflow: (name: string) => string;
  addNode: (agentId: string, x: number, y: number) => void;
  removeNode: (nodeId: string) => void;
  updateNodePosition: (nodeId: string, x: number, y: number) => void;
  updateNodeConfig: (nodeId: string, config: Record<string, any>) => void;
  addConnection: (fromNodeId: string, toNodeId: string) => void;
  setSelectedNode: (nodeId: string | null) => void;
  runWorkflow: () => void;
  stopWorkflow: () => void;
}

export const useAgentStore = create<AgentStore>()((set, get) => ({
  workflows: [],
  currentWorkflowId: null,
  nodes: [],
  connections: [],
  selectedNodeId: null,
  isRunning: false,

  createWorkflow: (name) => {
    const newWorkflow: Workflow = {
      id: `wf-${Date.now()}`,
      name,
      description: '',
      nodes: [],
      connections: [],
      createdAt: new Date(),
      updatedAt: new Date(),
      createdBy: 'current-user',
    };
    set((state) => ({
      workflows: [...state.workflows, newWorkflow],
      currentWorkflowId: newWorkflow.id,
      nodes: [],
      connections: [],
    }));
    return newWorkflow.id;
  },

  addNode: (agentId, x, y) => {
    const newNode: AgentNode = {
      id: `node-${Date.now()}`,
      agentId,
      x,
      y,
      config: {},
      status: 'idle',
    };
    set((state) => ({
      nodes: [...state.nodes, newNode],
    }));
  },

  removeNode: (nodeId) => {
    set((state) => ({
      nodes: state.nodes.filter((n) => n.id !== nodeId),
      connections: state.connections.filter(
        (c) => c.fromNodeId !== nodeId && c.toNodeId !== nodeId
      ),
    }));
  },

  updateNodePosition: (nodeId, x, y) => {
    set((state) => ({
      nodes: state.nodes.map((n) =>
        n.id === nodeId ? { ...n, x, y } : n
      ),
    }));
  },

  updateNodeConfig: (nodeId, config) => {
    set((state) => ({
      nodes: state.nodes.map((n) =>
        n.id === nodeId ? { ...n, config: { ...n.config, ...config } } : n
      ),
    }));
  },

  addConnection: (fromNodeId, toNodeId) => {
    const newConnection: AgentConnection = {
      id: `conn-${Date.now()}`,
      fromNodeId,
      toNodeId,
    };
    set((state) => ({
      connections: [...state.connections, newConnection],
    }));
  },

  setSelectedNode: (nodeId) => set({ selectedNodeId: nodeId }),

  runWorkflow: async () => {
    set({ isRunning: true });
    const nodes = get().nodes;
    for (const node of nodes) {
      set((state) => ({
        nodes: state.nodes.map((n) =>
          n.id === node.id ? { ...n, status: 'running' } : n
        ),
      }));
      await new Promise((r) => setTimeout(r, 1000));
      set((state) => ({
        nodes: state.nodes.map((n) =>
          n.id === node.id ? { ...n, status: 'success' } : n
        ),
      }));
    }
    set({ isRunning: false });
  },

  stopWorkflow: () => set({ isRunning: false }),
}));

// Integration Store
interface IntegrationStore {
  integrations: Integration[];
  connectIntegration: (type: IntegrationType) => void;
  disconnectIntegration: (type: IntegrationType) => void;
}

const defaultIntegrations: Integration[] = [
  { id: 'github', type: 'github', name: 'GitHub', description: '代码仓库集成', icon: '⬛', color: '#333', isConnected: true, connectedAt: new Date() },
  { id: 'gitlab', type: 'gitlab', name: 'GitLab', description: '代码仓库与CI/CD', icon: '🦊', color: '#FC6D26', isConnected: false },
  { id: 'notion', type: 'notion', name: 'Notion', description: '文档协作', icon: '📄', color: '#000', isConnected: true, connectedAt: new Date() },
  { id: 'slack', type: 'slack', name: 'Slack', description: '团队通讯', icon: '💬', color: '#4A154B', isConnected: true, connectedAt: new Date() },
  { id: 'discord', type: 'discord', name: 'Discord', description: '社区集成', icon: '🎮', color: '#5865F2', isConnected: false },
  { id: 'figma', type: 'figma', name: 'Figma', description: '设计协作', icon: '🎨', color: '#F24E1E', isConnected: false },
  { id: 'vercel', type: 'vercel', name: 'Vercel', description: '前端部署', icon: '▲', color: '#000', isConnected: true, connectedAt: new Date() },
  { id: 'netlify', type: 'netlify', name: 'Netlify', description: '静态托管', icon: '◆', color: '#00C7B7', isConnected: false },
  { id: 'stripe', type: 'stripe', name: 'Stripe', description: '支付处理', icon: '💳', color: '#635BFF', isConnected: false },
  { id: 'openai', type: 'openai', name: 'OpenAI', description: 'AI API', icon: '🤖', color: '#10A37F', isConnected: true, connectedAt: new Date() },
  { id: 'anthropic', type: 'anthropic', name: 'Anthropic', description: 'Claude API', icon: '🧠', color: '#D4A574', isConnected: false },
  { id: 'google', type: 'google', name: 'Google Workspace', description: '文档协作', icon: '🔍', color: '#4285F4', isConnected: false },
];

export const useIntegrationStore = create<IntegrationStore>()((set) => ({
  integrations: defaultIntegrations,
  connectIntegration: (type) => {
    set((state) => ({
      integrations: state.integrations.map((i) =>
        i.type === type ? { ...i, isConnected: true, connectedAt: new Date() } : i
      ),
    }));
  },
  disconnectIntegration: (type) => {
    set((state) => ({
      integrations: state.integrations.map((i) =>
        i.type === type ? { ...i, isConnected: false, connectedAt: undefined } : i
      ),
    }));
  },
}));

// Team Store
interface TeamStore {
  team: Team | null;
  activities: Activity[];
  notifications: Notification[];
  inviteMember: (email: string, role: string) => void;
  removeMember: (id: string) => void;
  markNotificationRead: (id: string) => void;
}

const mockTeam: Team = {
  id: 'team-1',
  name: 'Engineering',
  description: 'Core engineering team',
  members: [
    { id: '1', userId: 'u1', name: 'Alice Chen', email: 'alice@autocoops.io', avatar: 'https://i.pravatar.cc/150?u=alice', role: 'owner', status: 'online', joinedAt: new Date('2024-01-01') },
    { id: '2', userId: 'u2', name: 'Bob Wang', email: 'bob@autocoops.io', avatar: 'https://i.pravatar.cc/150?u=bob', role: 'admin', status: 'online', joinedAt: new Date('2024-01-15') },
    { id: '3', userId: 'u3', name: 'Carol Liu', email: 'carol@autocoops.io', avatar: 'https://i.pravatar.cc/150?u=carol', role: 'member', status: 'away', lastSeen: new Date(Date.now() - 1000 * 60 * 15), joinedAt: new Date('2024-02-01') },
    { id: '4', userId: 'u4', name: 'David Zhang', email: 'david@autocoops.io', avatar: 'https://i.pravatar.cc/150?u=david', role: 'member', status: 'offline', lastSeen: new Date(Date.now() - 1000 * 60 * 60 * 2), joinedAt: new Date('2024-02-15') },
  ],
  createdAt: new Date('2024-01-01'),
  plan: 'pro',
};

const mockActivities: Activity[] = [
  { id: '1', type: 'commit', user: mockTeam.members[0], title: 'Pushed to main', description: 'Fixed authentication bug', timestamp: new Date(Date.now() - 1000 * 60 * 5) },
  { id: '2', type: 'deploy', user: mockTeam.members[1], title: 'Deployed v1.2.3', description: 'Production deployment successful', timestamp: new Date(Date.now() - 1000 * 60 * 30) },
  { id: '3', type: 'comment', user: mockTeam.members[2], title: 'Commented on PR #42', description: 'LGTM! Great work.', timestamp: new Date(Date.now() - 1000 * 60 * 60) },
  { id: '4', type: 'pr', user: mockTeam.members[3], title: 'Opened PR #45', description: 'Feature: Add dark mode', timestamp: new Date(Date.now() - 1000 * 60 * 60 * 2) },
];

const mockNotifications: Notification[] = [
  { id: '1', type: 'success', title: '部署成功', message: 'v1.2.3 已成功部署到生产环境', timestamp: new Date(Date.now() - 1000 * 60 * 5), read: false },
  { id: '2', type: 'info', title: '新的评论', message: 'Bob 评论了您的 PR #42', timestamp: new Date(Date.now() - 1000 * 60 * 30), read: false },
  { id: '3', type: 'warning', title: '构建警告', message: '依赖包存在安全漏洞', timestamp: new Date(Date.now() - 1000 * 60 * 60), read: true },
];

export const useTeamStore = create<TeamStore>()((set) => ({
  team: mockTeam,
  activities: mockActivities,
  notifications: mockNotifications,
  inviteMember: (_email, _role) => {
    // Mock implementation
  },
  removeMember: (id) => {
    set((state) => ({
      team: state.team ? {
        ...state.team,
        members: state.team.members.filter((m) => m.id !== id),
      } : null,
    }));
  },
  markNotificationRead: (id) => {
    set((state) => ({
      notifications: state.notifications.map((n) =>
        n.id === id ? { ...n, read: true } : n
      ),
    }));
  },
}));

// Deployment Store
interface DeploymentStore {
  deployments: Deployment[];
  createDeployment: (projectId: string, version: string) => void;
}

const mockDeployments: Deployment[] = [
  {
    id: 'd1',
    projectId: 'p1',
    version: 'v1.2.3',
    status: 'success',
    url: 'https://myapp.autocoops.io',
    customDomain: 'myapp.com',
    createdAt: new Date(Date.now() - 1000 * 60 * 60 * 2),
    completedAt: new Date(Date.now() - 1000 * 60 * 60 * 1),
    createdBy: 'Alice',
    logs: [
      { id: 'l1', timestamp: new Date(), level: 'info', message: 'Build started' },
      { id: 'l2', timestamp: new Date(), level: 'info', message: 'Installing dependencies...' },
      { id: 'l3', timestamp: new Date(), level: 'info', message: 'Build completed' },
      { id: 'l4', timestamp: new Date(), level: 'info', message: 'Deploying to production...' },
      { id: 'l5', timestamp: new Date(), level: 'info', message: 'Deployment successful!' },
    ],
    stats: {
      visits: 12500,
      uniqueVisitors: 8400,
      avgLoadTime: 2.3,
      countries: 45,
      last24h: 1200,
    },
  },
  {
    id: 'd2',
    projectId: 'p1',
    version: 'v1.2.2',
    status: 'failed',
    createdAt: new Date(Date.now() - 1000 * 60 * 60 * 5),
    completedAt: new Date(Date.now() - 1000 * 60 * 60 * 4),
    createdBy: 'Bob',
    logs: [
      { id: 'l1', timestamp: new Date(), level: 'info', message: 'Build started' },
      { id: 'l2', timestamp: new Date(), level: 'info', message: 'Installing dependencies...' },
      { id: 'l3', timestamp: new Date(), level: 'error', message: 'Build failed: TypeScript error' },
    ],
  },
];

export const useDeploymentStore = create<DeploymentStore>()((set) => ({
  deployments: mockDeployments,
  createDeployment: (projectId, version) => {
    const newDeployment: Deployment = {
      id: `d-${Date.now()}`,
      projectId,
      version,
      status: 'pending',
      createdAt: new Date(),
      createdBy: 'current-user',
      logs: [],
    };
    set((state) => ({
      deployments: [newDeployment, ...state.deployments],
    }));
  },
}));

// Main App Store
export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      currentView: 'chat',
      isSidebarOpen: true,
      isDarkMode: true,
      currentUser: {
        id: 'u1',
        name: 'Alice Chen',
        email: 'alice@autocoops.io',
        avatar: 'https://i.pravatar.cc/150?u=alice',
        plan: 'pro',
        integrations: ['github', 'notion', 'slack', 'vercel', 'openai'],
      },
      currentTeam: mockTeam,
      setCurrentView: (view) => set({ currentView: view }),
      toggleSidebar: () => set((state) => ({ isSidebarOpen: !state.isSidebarOpen })),
      toggleDarkMode: () => set((state) => ({ isDarkMode: !state.isDarkMode })),
      setCurrentUser: (user) => set({ currentUser: user }),
      setCurrentTeam: (team) => set({ currentTeam: team }),
    }),
    {
      name: 'app-store',
    }
  )
);

// Helper functions for file tree
function addNodeToTree(nodes: FileNode[], newNode: FileNode, parentId: string | null): FileNode[] {
  if (!parentId) return [...nodes, newNode];
  
  return nodes.map((node) => {
    if (node.id === parentId) {
      return {
        ...node,
        children: [...(node.children || []), newNode],
        isOpen: true,
      };
    }
    if (node.children) {
      return {
        ...node,
        children: addNodeToTree(node.children, newNode, parentId),
      };
    }
    return node;
  });
}

function removeNodeFromTree(nodes: FileNode[], id: string): FileNode[] {
  return nodes
    .filter((node) => node.id !== id)
    .map((node) => ({
      ...node,
      children: node.children ? removeNodeFromTree(node.children, id) : undefined,
    }));
}

function updateNodeInTree(
  nodes: FileNode[],
  id: string,
  updater: (node: FileNode) => FileNode
): FileNode[] {
  return nodes.map((node) => {
    if (node.id === id) return updater(node);
    if (node.children) {
      return {
        ...node,
        children: updateNodeInTree(node.children, id, updater),
      };
    }
    return node;
  });
}
