// Chat Types
export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  isStreaming?: boolean;
  model?: string;
  attachments?: Attachment[];
}

export interface Attachment {
  id: string;
  type: 'image' | 'file' | 'code';
  name: string;
  url: string;
  size?: number;
}

export interface Chat {
  id: string;
  title: string;
  messages: Message[];
  createdAt: Date;
  updatedAt: Date;
  model: string;
}

export interface ChatModel {
  id: string;
  name: string;
  provider: string;
  description: string;
  maxTokens: number;
  icon: string;
}

// IDE Types
export interface FileNode {
  id: string;
  name: string;
  type: 'file' | 'folder';
  language?: string;
  content?: string;
  children?: FileNode[];
  isOpen?: boolean;
  parentId?: string | null;
  lastModified?: Date;
  size?: number;
}

export interface EditorTab {
  id: string;
  fileId: string;
  name: string;
  language: string;
  content: string;
  isModified: boolean;
  lastSaved?: Date;
}

export interface TerminalLine {
  id: string;
  type: 'input' | 'output' | 'error' | 'info';
  content: string;
  timestamp: Date;
}

// Virtual Computer Types
export interface DesktopApp {
  id: string;
  name: string;
  icon: string;
  color: string;
  isOpen: boolean;
  window?: AppWindow;
}

export interface AppWindow {
  id: string;
  appId: string;
  title: string;
  x: number;
  y: number;
  width: number;
  height: number;
  isMinimized: boolean;
  isMaximized: boolean;
  zIndex: number;
}

// Multi-Agent Types
export interface Agent {
  id: string;
  name: string;
  type: 'input' | 'process' | 'output' | 'store' | 'research' | 'write' | 'code' | 'calculate';
  description: string;
  icon: string;
  color: string;
  config: Record<string, any>;
}

export interface AgentNode {
  id: string;
  agentId: string;
  x: number;
  y: number;
  config: Record<string, any>;
  status: 'idle' | 'running' | 'success' | 'error';
  output?: any;
}

export interface AgentConnection {
  id: string;
  fromNodeId: string;
  toNodeId: string;
  fromPort?: string;
  toPort?: string;
}

export interface Workflow {
  id: string;
  name: string;
  description: string;
  nodes: AgentNode[];
  connections: AgentConnection[];
  createdAt: Date;
  updatedAt: Date;
  createdBy: string;
}

// Third-Party Integration Types
export type IntegrationType = 
  | 'github' 
  | 'gitlab' 
  | 'notion' 
  | 'slack' 
  | 'discord' 
  | 'figma' 
  | 'vercel' 
  | 'netlify' 
  | 'stripe' 
  | 'openai' 
  | 'anthropic' 
  | 'google';

export interface Integration {
  id: string;
  type: IntegrationType;
  name: string;
  description: string;
  icon: string;
  color: string;
  isConnected: boolean;
  connectedAt?: Date;
  settings?: Record<string, any>;
}

export interface GitHubConnection {
  id: string;
  username: string;
  avatar: string;
  repos: GitHubRepo[];
  token?: string;
}

export interface GitHubRepo {
  id: string;
  name: string;
  fullName: string;
  description: string;
  stars: number;
  forks: number;
  language: string;
  updatedAt: Date;
}

export interface SlackConnection {
  id: string;
  teamName: string;
  teamIcon: string;
  channels: SlackChannel[];
}

export interface SlackChannel {
  id: string;
  name: string;
  isPrivate: boolean;
  memberCount: number;
}

export interface NotionConnection {
  id: string;
  workspaceName: string;
  pages: NotionPage[];
}

export interface NotionPage {
  id: string;
  title: string;
  url: string;
  lastEdited: Date;
}

// Team Collaboration Types
export interface Team {
  id: string;
  name: string;
  description: string;
  avatar?: string;
  members: TeamMember[];
  createdAt: Date;
  plan: 'free' | 'pro' | 'enterprise';
}

export interface TeamMember {
  id: string;
  userId: string;
  name: string;
  email: string;
  avatar: string;
  role: 'owner' | 'admin' | 'member' | 'viewer';
  status: 'online' | 'away' | 'offline';
  lastSeen?: Date;
  joinedAt: Date;
}

export interface Activity {
  id: string;
  type: 'commit' | 'deploy' | 'comment' | 'pr' | 'issue' | 'member';
  user: TeamMember;
  title: string;
  description?: string;
  timestamp: Date;
  metadata?: Record<string, any>;
}

export interface Comment {
  id: string;
  userId: string;
  user: TeamMember;
  content: string;
  timestamp: Date;
  replies?: Comment[];
}

// Deployment Types
export interface Deployment {
  id: string;
  projectId: string;
  version: string;
  status: 'pending' | 'building' | 'success' | 'failed' | 'cancelled';
  url?: string;
  customDomain?: string;
  createdAt: Date;
  completedAt?: Date;
  createdBy: string;
  logs: DeploymentLog[];
  stats?: DeploymentStats;
}

export interface DeploymentLog {
  id: string;
  timestamp: Date;
  level: 'info' | 'warn' | 'error';
  message: string;
}

export interface DeploymentStats {
  visits: number;
  uniqueVisitors: number;
  avgLoadTime: number;
  countries: number;
  last24h: number;
}

// App State
export type ViewMode = 'chat' | 'ide' | 'desktop' | 'agents' | 'integrations' | 'team' | 'deploy' | 'ai-hub' | 'browser' | 'operator' | 'supabase' | 'ai-workstation' | 'learn' | 'landing';

export interface AppState {
  currentView: ViewMode;
  isSidebarOpen: boolean;
  isDarkMode: boolean;
  currentUser?: User;
  currentTeam?: Team;
  setCurrentView: (view: ViewMode) => void;
  toggleSidebar: () => void;
  toggleDarkMode: () => void;
  setCurrentUser: (user: User) => void;
  setCurrentTeam: (team: Team) => void;
}

export interface User {
  id: string;
  name: string;
  email: string;
  avatar: string;
  plan: 'free' | 'pro' | 'enterprise';
  integrations: IntegrationType[];
}

// Notification Types
export interface Notification {
  id: string;
  type: 'info' | 'success' | 'warning' | 'error';
  title: string;
  message: string;
  timestamp: Date;
  read: boolean;
  action?: {
    label: string;
    url: string;
  };
}

// Project Types
export interface Project {
  id: string;
  name: string;
  description: string;
  type: 'web' | 'mobile' | 'api' | 'ai' | 'other';
  language: string;
  framework?: string;
  repoUrl?: string;
  deployedUrl?: string;
  teamId: string;
  createdAt: Date;
  updatedAt: Date;
}
