import { useState } from 'react';
import { motion } from 'framer-motion';
import {
  Database,
  Key,
  Users,
  Activity,
  Zap,
  Shield,
  Plus,
  ChevronRight,
  Download,
  Upload,
  Copy,
  Check,
  AlertCircle,
  Globe,
  Code,
  Settings,
  Edit,
  Table,
  Eye,
  EyeOff,
  Terminal,
  TrendingUp,
  Box,
  GitBranch,
  MessageSquare,
  Filter,
  ArrowUpDown,
} from 'lucide-react';
import { RippleButton } from '@/components/ui/ripple-button';
import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { cn } from '@/lib/utils';

// Supabase Project Type
interface SupabaseProject {
  id: string;
  name: string;
  ref: string;
  region: string;
  status: 'active' | 'paused' | 'upgrading';
  database: {
    version: string;
    size: string;
    connections: number;
    tables: number;
  };
  api: {
    url: string;
    anonKey: string;
    serviceKey: string;
  };
  auth: {
    providers: string[];
    users: number;
    sessions: number;
  };
  storage: {
    buckets: number;
    size: string;
    files: number;
  };
  realtime: {
    connections: number;
    messages: number;
  };
  createdAt: Date;
}

// Database Table Type
interface DatabaseTable {
  id: string;
  name: string;
  schema: string;
  rowCount: number;
  size: string;
  columns: TableColumn[];
  indexes: TableIndex[];
  policies: RowPolicy[];
  lastModified: Date;
}

interface TableColumn {
  name: string;
  type: string;
  nullable: boolean;
  default?: string;
  isPrimary: boolean;
  isForeign: boolean;
}

interface TableIndex {
  name: string;
  columns: string[];
  type: string;
}

interface RowPolicy {
  name: string;
  action: 'SELECT' | 'INSERT' | 'UPDATE' | 'DELETE' | 'ALL';
  using?: string;
  withCheck?: string;
}

// Auth User Type
interface AuthUser {
  id: string;
  email: string;
  phone?: string;
  emailConfirmed: boolean;
  phoneConfirmed: boolean;
  providers: string[];
  lastSignIn?: Date;
  createdAt: Date;
  appMetadata: Record<string, any>;
  userMetadata: Record<string, any>;
}

// Storage Bucket Type
interface StorageBucket {
  id: string;
  name: string;
  public: boolean;
  fileSizeLimit?: number;
  allowedMimeTypes: string[];
  createdAt: Date;
  updatedAt: Date;
  size: string;
  fileCount: number;
}

// API Endpoint Type
interface APIEndpoint {
  id: string;
  table: string;
  method: 'GET' | 'POST' | 'PATCH' | 'DELETE';
  path: string;
  description: string;
  parameters: APIParameter[];
  responses: APIResponse[];
}

interface APIParameter {
  name: string;
  type: string;
  required: boolean;
  description: string;
}

interface APIResponse {
  status: number;
  description: string;
  example: any;
}

// Mock Data
const mockProject: SupabaseProject = {
  id: 'proj-1',
  name: 'AutoEcoops Backend',
  ref: 'abcdefgh12345678',
  region: 'ap-southeast-1',
  status: 'active',
  database: {
    version: 'PostgreSQL 15.1',
    size: '2.3 GB',
    connections: 47,
    tables: 24,
  },
  api: {
    url: 'https://abcdefgh12345678.supabase.co',
    anonKey: 'eyJhbGciOiJIUzI1NiIs...',
    serviceKey: 'eyJhbGciOiJIUzI1NiIs...',
  },
  auth: {
    providers: ['email', 'github', 'google'],
    users: 12543,
    sessions: 8921,
  },
  storage: {
    buckets: 8,
    size: '156.7 MB',
    files: 3241,
  },
  realtime: {
    connections: 2341,
    messages: 456789,
  },
  createdAt: new Date('2024-01-15'),
};

const mockTables: DatabaseTable[] = [
  {
    id: 'tbl-1',
    name: 'users',
    schema: 'public',
    rowCount: 12543,
    size: '4.2 MB',
    columns: [
      { name: 'id', type: 'uuid', nullable: false, isPrimary: true, isForeign: false },
      { name: 'email', type: 'text', nullable: false, isPrimary: false, isForeign: false },
      { name: 'created_at', type: 'timestamp', nullable: false, default: 'now()', isPrimary: false, isForeign: false },
      { name: 'role', type: 'text', nullable: false, default: '\'user\'', isPrimary: false, isForeign: false },
    ],
    indexes: [
      { name: 'users_pkey', columns: ['id'], type: 'btree' },
      { name: 'users_email_idx', columns: ['email'], type: 'btree' },
    ],
    policies: [
      { name: 'Enable read access for all users', action: 'SELECT', using: 'true' },
      { name: 'Enable insert for authenticated users only', action: 'INSERT', withCheck: 'auth.role() = \'authenticated\'' },
    ],
    lastModified: new Date('2024-03-15'),
  },
  {
    id: 'tbl-2',
    name: 'projects',
    schema: 'public',
    rowCount: 892,
    size: '2.1 MB',
    columns: [
      { name: 'id', type: 'uuid', nullable: false, isPrimary: true, isForeign: false },
      { name: 'name', type: 'text', nullable: false, isPrimary: false, isForeign: false },
      { name: 'owner_id', type: 'uuid', nullable: false, isPrimary: false, isForeign: true },
      { name: 'created_at', type: 'timestamp', nullable: false, default: 'now()', isPrimary: false, isForeign: false },
    ],
    indexes: [
      { name: 'projects_pkey', columns: ['id'], type: 'btree' },
      { name: 'projects_owner_idx', columns: ['owner_id'], type: 'btree' },
    ],
    policies: [
      { name: 'Users can view their own projects', action: 'SELECT', using: 'owner_id = auth.uid()' },
    ],
    lastModified: new Date('2024-03-14'),
  },
  {
    id: 'tbl-3',
    name: 'documents',
    schema: 'public',
    rowCount: 45678,
    size: '156.3 MB',
    columns: [
      { name: 'id', type: 'uuid', nullable: false, isPrimary: true, isForeign: false },
      { name: 'title', type: 'text', nullable: false, isPrimary: false, isForeign: false },
      { name: 'content', type: 'text', nullable: true, isPrimary: false, isForeign: false },
      { name: 'embedding', type: 'vector(1536)', nullable: true, isPrimary: false, isForeign: false },
      { name: 'project_id', type: 'uuid', nullable: false, isPrimary: false, isForeign: true },
      { name: 'created_at', type: 'timestamp', nullable: false, default: 'now()', isPrimary: false, isForeign: false },
    ],
    indexes: [
      { name: 'documents_pkey', columns: ['id'], type: 'btree' },
      { name: 'documents_embedding_idx', columns: ['embedding'], type: 'ivfflat' },
    ],
    policies: [
      { name: 'Enable all for project members', action: 'ALL', using: 'project_id IN (SELECT id FROM projects WHERE owner_id = auth.uid())' },
    ],
    lastModified: new Date('2024-03-13'),
  },
];

const mockAuthUsers: AuthUser[] = [
  {
    id: 'user-1',
    email: 'alice@autecoops.io',
    emailConfirmed: true,
    phoneConfirmed: false,
    providers: ['email', 'github'],
    lastSignIn: new Date(Date.now() - 1000 * 60 * 5),
    createdAt: new Date('2024-01-15'),
    appMetadata: { provider: 'github' },
    userMetadata: { full_name: 'Alice Chen', avatar_url: 'https://github.com/alice.png' },
  },
  {
    id: 'user-2',
    email: 'bob@autecoops.io',
    emailConfirmed: true,
    phoneConfirmed: false,
    providers: ['email', 'google'],
    lastSignIn: new Date(Date.now() - 1000 * 60 * 30),
    createdAt: new Date('2024-01-20'),
    appMetadata: { provider: 'google' },
    userMetadata: { full_name: 'Bob Wang', avatar_url: 'https://lh3.google.com/bob.png' },
  },
];

const mockBuckets: StorageBucket[] = [
  {
    id: 'bucket-1',
    name: 'avatars',
    public: true,
    fileSizeLimit: 5242880,
    allowedMimeTypes: ['image/png', 'image/jpeg', 'image/gif'],
    createdAt: new Date('2024-01-15'),
    updatedAt: new Date('2024-03-10'),
    size: '45.2 MB',
    fileCount: 12543,
  },
  {
    id: 'bucket-2',
    name: 'documents',
    public: false,
    fileSizeLimit: 104857600,
    allowedMimeTypes: ['application/pdf', 'text/plain', 'application/json'],
    createdAt: new Date('2024-01-15'),
    updatedAt: new Date('2024-03-12'),
    size: '89.5 MB',
    fileCount: 3241,
  },
  {
    id: 'bucket-3',
    name: 'vector-store',
    public: false,
    allowedMimeTypes: ['*/*'],
    createdAt: new Date('2024-02-01'),
    updatedAt: new Date('2024-03-14'),
    size: '21.9 MB',
    fileCount: 156,
  },
];

const mockEndpoints: APIEndpoint[] = [
  {
    id: 'api-1',
    table: 'users',
    method: 'GET',
    path: '/rest/v1/users',
    description: 'Retrieve all users or filter by query parameters',
    parameters: [
      { name: 'select', type: 'string', required: false, description: 'Columns to select' },
      { name: 'eq', type: 'string', required: false, description: 'Equal filter' },
      { name: 'order', type: 'string', required: false, description: 'Order by column' },
    ],
    responses: [
      { status: 200, description: 'Success', example: [{ id: '...', email: '...' }] },
      { status: 401, description: 'Unauthorized', example: { message: 'JWT expired' } },
    ],
  },
  {
    id: 'api-2',
    table: 'users',
    method: 'POST',
    path: '/rest/v1/users',
    description: 'Insert a new user record',
    parameters: [
      { name: 'body', type: 'object', required: true, description: 'User data' },
    ],
    responses: [
      { status: 201, description: 'Created', example: { id: '...', email: '...' } },
      { status: 409, description: 'Conflict', example: { message: 'Duplicate key' } },
    ],
  },
];

export function SupabaseDashboard() {
  const [activeTab, setActiveTab] = useState('overview');
  const [project] = useState<SupabaseProject>(mockProject);
  const [selectedTable, setSelectedTable] = useState<DatabaseTable | null>(null);
  const [showApiKey, setShowApiKey] = useState(false);
  const [copiedKey, setCopiedKey] = useState(false);

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedKey(true);
    setTimeout(() => setCopiedKey(false), 2000);
  };

  return (
    <div className="h-full flex flex-col bg-white">
      {/* Header */}
      <div className="h-16 border-b border-gray-200 flex items-center justify-between px-6">
        <div className="flex items-center gap-4">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-[#FFD700] to-[#FFA500] flex items-center justify-center">
            <Database className="w-5 h-5 text-black" />
          </div>
          <div>
            <h1 className="font-semibold text-gray-900 text-lg">{project.name}</h1>
            <div className="flex items-center gap-2 text-xs text-gray-500">
              <span className="font-mono">{project.ref}</span>
              <span>•</span>
              <span>{project.region}</span>
              <Badge className="bg-green-100 text-green-600 border-0 text-[10px]">
                {project.status}
              </Badge>
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <RippleButton variant="outline" size="sm" className="gap-2">
            <Settings className="w-4 h-4" />
            Settings
          </RippleButton>
          <RippleButton size="sm" className="gap-2 bg-[#FFD700] text-black hover:bg-[#FFC700]">
            <Zap className="w-4 h-4" />
            Upgrade
          </RippleButton>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="border-b border-gray-200">
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="bg-transparent h-12 px-4 gap-1">
            {[
              { id: 'overview', label: 'Overview', icon: Activity },
              { id: 'database', label: 'Database', icon: Database },
              { id: 'auth', label: 'Authentication', icon: Shield },
              { id: 'storage', label: 'Storage', icon: Box },
              { id: 'api', label: 'API', icon: Code },
              { id: 'realtime', label: 'Realtime', icon: Zap },
            ].map((tab) => (
              <TabsTrigger
                key={tab.id}
                value={tab.id}
                className="data-[state=active]:bg-[#FFD700]/10 data-[state=active]:text-[#FFD700] text-gray-600 gap-2"
              >
                <tab.icon className="w-4 h-4" />
                {tab.label}
              </TabsTrigger>
            ))}
          </TabsList>
        </Tabs>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-hidden">
        <Tabs value={activeTab} className="h-full">
          {/* Overview Tab */}
          <TabsContent value="overview" className="h-full m-0">
            <ScrollArea className="h-full p-6">
              <div className="max-w-6xl mx-auto space-y-6">
                {/* Quick Stats */}
                <div className="grid grid-cols-4 gap-4">
                  <StatCard
                    icon={Database}
                    label="Database Size"
                    value={project.database.size}
                    subValue={`${project.database.tables} tables`}
                    trend="+12%"
                    trendUp={true}
                  />
                  <StatCard
                    icon={Users}
                    label="Total Users"
                    value={project.auth.users.toLocaleString()}
                    subValue={`${project.auth.sessions.toLocaleString()} active sessions`}
                    trend="+8%"
                    trendUp={true}
                  />
                  <StatCard
                    icon={Box}
                    label="Storage Used"
                    value={project.storage.size}
                    subValue={`${project.storage.files.toLocaleString()} files`}
                    trend="+23%"
                    trendUp={true}
                  />
                  <StatCard
                    icon={Zap}
                    label="Realtime Messages"
                    value={project.realtime.messages.toLocaleString()}
                    subValue={`${project.realtime.connections.toLocaleString()} connections`}
                    trend="+45%"
                    trendUp={true}
                  />
                </div>

                {/* API Credentials */}
                <div className="bg-gray-50 border border-gray-200 rounded-xl p-6">
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-3">
                      <Key className="w-5 h-5 text-[#FFD700]" />
                      <h3 className="font-semibold text-gray-900">API Credentials</h3>
                    </div>
                    <RippleButton
                      variant="ghost"
                      size="sm"
                      onClick={() => setShowApiKey(!showApiKey)}
                    >
                      {showApiKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                    </RippleButton>
                  </div>
                  <div className="space-y-4">
                    <div>
                      <label className="text-xs text-gray-500 mb-1 block">Project URL</label>
                      <div className="flex gap-2">
                        <code className="flex-1 bg-white border border-gray-200 rounded-lg px-3 py-2 text-sm font-mono text-gray-600">
                          {project.api.url}
                        </code>
                        <RippleButton
                          variant="outline"
                          size="icon"
                          onClick={() => copyToClipboard(project.api.url)}
                        >
                          {copiedKey ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
                        </RippleButton>
                      </div>
                    </div>
                    <div>
                      <label className="text-xs text-gray-500 mb-1 block">Anon Key</label>
                      <div className="flex gap-2">
                        <code className="flex-1 bg-white border border-gray-200 rounded-lg px-3 py-2 text-sm font-mono text-gray-600">
                          {showApiKey ? project.api.anonKey : '••••••••••••••••••••••••••'}
                        </code>
                        <RippleButton
                          variant="outline"
                          size="icon"
                          onClick={() => copyToClipboard(project.api.anonKey)}
                        >
                          {copiedKey ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
                        </RippleButton>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Recent Activity */}
                <div className="bg-gray-50 border border-gray-200 rounded-xl p-6">
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-3">
                      <Activity className="w-5 h-5 text-[#FFD700]" />
                      <h3 className="font-semibold text-gray-900">Recent Activity</h3>
                    </div>
                    <RippleButton variant="ghost" size="sm">
                      View All
                    </RippleButton>
                  </div>
                  <div className="space-y-3">
                    {[
                      { icon: Database, text: 'Table "users" updated', time: '2 min ago', color: 'text-blue-500' },
                      { icon: Users, text: 'New user registered', time: '5 min ago', color: 'text-green-500' },
                      { icon: Box, text: 'File uploaded to "avatars"', time: '12 min ago', color: 'text-purple-500' },
                      { icon: Zap, text: 'Realtime connection established', time: '15 min ago', color: 'text-yellow-500' },
                    ].map((activity, i) => (
                      <div key={i} className="flex items-center gap-3 p-3 rounded-lg bg-white">
                        <activity.icon className={`w-4 h-4 ${activity.color}`} />
                        <span className="text-sm text-gray-600 flex-1">{activity.text}</span>
                        <span className="text-xs text-gray-500">{activity.time}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </ScrollArea>
          </TabsContent>

          {/* Database Tab */}
          <TabsContent value="database" className="h-full m-0">
            <DatabasePanel tables={mockTables} selectedTable={selectedTable} onSelectTable={setSelectedTable} />
          </TabsContent>

          {/* Auth Tab */}
          <TabsContent value="auth" className="h-full m-0">
            <AuthPanel users={mockAuthUsers} />
          </TabsContent>

          {/* Storage Tab */}
          <TabsContent value="storage" className="h-full m-0">
            <StoragePanel buckets={mockBuckets} />
          </TabsContent>

          {/* API Tab */}
          <TabsContent value="api" className="h-full m-0">
            <APIPanel endpoints={mockEndpoints} projectUrl={project.api.url} />
          </TabsContent>

          {/* Realtime Tab */}
          <TabsContent value="realtime" className="h-full m-0">
            <RealtimePanel connections={project.realtime.connections} messages={project.realtime.messages} />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}

// Stat Card Component
function StatCard({
  icon: Icon,
  label,
  value,
  subValue,
  trend,
  trendUp,
}: {
  icon: React.ElementType;
  label: string;
  value: string;
  subValue: string;
  trend: string;
  trendUp: boolean;
}) {
  return (
    <div className="bg-gray-50 border border-gray-200 rounded-xl p-5">
      <div className="flex items-start justify-between mb-3">
        <div className="w-10 h-10 rounded-lg bg-gray-100 flex items-center justify-center">
          <Icon className="w-5 h-5 text-gray-500" />
        </div>
        <div className={cn('flex items-center gap-1 text-xs', trendUp ? 'text-green-500' : 'text-red-500')}>
          <TrendingUp className="w-3 h-3" />
          {trend}
        </div>
      </div>
      <div className="text-2xl font-bold text-gray-900 mb-1">{value}</div>
      <div className="text-xs text-gray-500">{label}</div>
      <div className="text-xs text-gray-600 mt-1">{subValue}</div>
    </div>
  );
}

// Database Panel Component
function DatabasePanel({
  tables,
  selectedTable,
  onSelectTable,
}: {
  tables: DatabaseTable[];
  selectedTable: DatabaseTable | null;
  onSelectTable: (table: DatabaseTable | null) => void;
}) {
  const [activeDbTab, setActiveDbTab] = useState('tables');

  if (selectedTable) {
    return (
      <div className="h-full flex">
        {/* Table Detail View */}
        <div className="flex-1 flex flex-col">
          <div className="h-14 border-b border-gray-200 flex items-center justify-between px-4">
            <div className="flex items-center gap-3">
              <RippleButton variant="ghost" size="icon" onClick={() => onSelectTable(null)}>
                <ChevronRight className="w-4 h-4 rotate-180" />
              </RippleButton>
              <Table className="w-5 h-5 text-[#FFD700]" />
              <span className="font-semibold text-gray-900">{selectedTable.name}</span>
              <Badge className="bg-gray-100 text-gray-600 border-0 text-[10px]">
                {selectedTable.schema}
              </Badge>
            </div>
            <div className="flex items-center gap-2">
              <RippleButton variant="outline" size="sm" className="gap-2">
                <Edit className="w-4 h-4" />
                Edit Table
              </RippleButton>
              <RippleButton variant="outline" size="sm" className="gap-2">
                <Terminal className="w-4 h-4" />
                SQL Editor
              </RippleButton>
            </div>
          </div>

          <Tabs value={activeDbTab} onValueChange={setActiveDbTab} className="flex-1">
            <TabsList className="bg-transparent border-b border-gray-200 rounded-none h-10 px-4">
              <TabsTrigger value="data" className="data-[state=active]:bg-[#FFD700]/10 data-[state=active]:text-[#FFD700]">Data</TabsTrigger>
              <TabsTrigger value="columns" className="data-[state=active]:bg-[#FFD700]/10 data-[state=active]:text-[#FFD700]">Columns</TabsTrigger>
              <TabsTrigger value="indexes" className="data-[state=active]:bg-[#FFD700]/10 data-[state=active]:text-[#FFD700]">Indexes</TabsTrigger>
              <TabsTrigger value="policies" className="data-[state=active]:bg-[#FFD700]/10 data-[state=active]:text-[#FFD700]">Policies</TabsTrigger>
            </TabsList>

            <div className="flex-1 overflow-hidden">
              <TabsContent value="data" className="h-full m-0 p-4">
                <div className="bg-gray-50 border border-gray-200 rounded-lg overflow-hidden">
                  <div className="flex items-center gap-2 p-3 border-b border-gray-200">
                    <Input
                      placeholder="Search rows..."
                      className="bg-white border-gray-200 max-w-xs"
                    />
                    <RippleButton variant="outline" size="sm" className="gap-2">
                      <Filter className="w-4 h-4" />
                      Filter
                    </RippleButton>
                    <RippleButton variant="outline" size="sm" className="gap-2">
                      <ArrowUpDown className="w-4 h-4" />
                      Sort
                    </RippleButton>
                    <div className="flex-1" />
                    <RippleButton size="sm" className="gap-2 bg-[#FFD700] text-black">
                      <Plus className="w-4 h-4" />
                      Insert Row
                    </RippleButton>
                  </div>
                  <div className="p-8 text-center text-gray-500">
                    <Database className="w-12 h-12 mx-auto mb-3 opacity-50" />
                    <p>Table data view with {selectedTable.rowCount.toLocaleString()} rows</p>
                    <p className="text-sm mt-1">Click "Insert Row" to add data</p>
                  </div>
                </div>
              </TabsContent>

              <TabsContent value="columns" className="h-full m-0 p-4">
                <div className="bg-gray-50 border border-gray-200 rounded-lg overflow-hidden">
                  <table className="w-full">
                    <thead className="bg-gray-100">
                      <tr>
                        <th className="text-left px-4 py-3 text-xs font-medium text-gray-600">Name</th>
                        <th className="text-left px-4 py-3 text-xs font-medium text-gray-600">Type</th>
                        <th className="text-left px-4 py-3 text-xs font-medium text-gray-600">Default</th>
                        <th className="text-left px-4 py-3 text-xs font-medium text-gray-600">Nullable</th>
                        <th className="text-left px-4 py-3 text-xs font-medium text-gray-600">Primary</th>
                      </tr>
                    </thead>
                    <tbody>
                      {selectedTable.columns.map((col, i) => (
                        <tr key={i} className="border-b border-gray-200 last:border-0">
                          <td className="px-4 py-3 text-sm text-gray-900">{col.name}</td>
                          <td className="px-4 py-3 text-sm text-gray-600">{col.type}</td>
                          <td className="px-4 py-3 text-sm text-gray-600">{col.default || '-'}</td>
                          <td className="px-4 py-3 text-sm text-gray-600">{col.nullable ? 'Yes' : 'No'}</td>
                          <td className="px-4 py-3">
                            {col.isPrimary && <Check className="w-4 h-4 text-green-500" />}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </TabsContent>

              <TabsContent value="indexes" className="h-full m-0 p-4">
                <div className="bg-gray-50 border border-gray-200 rounded-lg overflow-hidden">
                  <table className="w-full">
                    <thead className="bg-gray-100">
                      <tr>
                        <th className="text-left px-4 py-3 text-xs font-medium text-gray-600">Name</th>
                        <th className="text-left px-4 py-3 text-xs font-medium text-gray-600">Columns</th>
                        <th className="text-left px-4 py-3 text-xs font-medium text-gray-600">Type</th>
                      </tr>
                    </thead>
                    <tbody>
                      {selectedTable.indexes.map((idx, i) => (
                        <tr key={i} className="border-b border-gray-200 last:border-0">
                          <td className="px-4 py-3 text-sm text-gray-900">{idx.name}</td>
                          <td className="px-4 py-3 text-sm text-gray-600">{idx.columns.join(', ')}</td>
                          <td className="px-4 py-3 text-sm text-gray-600">{idx.type}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </TabsContent>

              <TabsContent value="policies" className="h-full m-0 p-4">
                <div className="bg-gray-50 border border-gray-200 rounded-lg overflow-hidden">
                  <table className="w-full">
                    <thead className="bg-gray-100">
                      <tr>
                        <th className="text-left px-4 py-3 text-xs font-medium text-gray-600">Name</th>
                        <th className="text-left px-4 py-3 text-xs font-medium text-gray-600">Action</th>
                        <th className="text-left px-4 py-3 text-xs font-medium text-gray-600">Using</th>
                      </tr>
                    </thead>
                    <tbody>
                      {selectedTable.policies.map((policy, i) => (
                        <tr key={i} className="border-b border-gray-200 last:border-0">
                          <td className="px-4 py-3 text-sm text-gray-900">{policy.name}</td>
                          <td className="px-4 py-3">
                            <Badge className="bg-[#FFD700]/20 text-[#FFD700] border-0 text-[10px]">
                              {policy.action}
                            </Badge>
                          </td>
                          <td className="px-4 py-3 text-sm text-gray-600 font-mono text-xs">{policy.using || policy.withCheck}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </TabsContent>
            </div>
          </Tabs>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex">
      {/* Tables List */}
      <div className="w-80 border-r border-gray-200 flex flex-col">
        <div className="h-14 border-b border-gray-200 flex items-center justify-between px-4">
          <span className="font-semibold text-gray-900">Tables</span>
          <RippleButton size="sm" className="gap-2 bg-[#FFD700] text-black">
            <Plus className="w-4 h-4" />
            New
          </RippleButton>
        </div>
        <ScrollArea className="flex-1 p-3">
          <div className="space-y-1">
            {tables.map((table) => (
              <button
                key={table.id}
                onClick={() => onSelectTable(table)}
                className="w-full flex items-center gap-3 p-3 rounded-lg hover:bg-gray-100 transition-colors text-left"
              >
                <Table className="w-4 h-4 text-[#FFD700]" />
                <div className="flex-1 min-w-0">
                  <div className="text-sm text-gray-900 truncate">{table.name}</div>
                  <div className="text-xs text-gray-500">
                    {table.rowCount.toLocaleString()} rows • {table.size}
                  </div>
                </div>
                <ChevronRight className="w-4 h-4 text-gray-500" />
              </button>
            ))}
          </div>
        </ScrollArea>
      </div>

      {/* Database Overview */}
      <div className="flex-1 p-6 overflow-auto">
        <div className="max-w-4xl mx-auto space-y-6">
          <div className="bg-gray-50 border border-gray-200 rounded-xl p-6">
            <h3 className="font-semibold text-gray-900 mb-4">Database Information</h3>
            <div className="grid grid-cols-2 gap-4">
              <div className="p-4 bg-white rounded-lg">
                <div className="text-xs text-gray-500 mb-1">PostgreSQL Version</div>
                <div className="text-sm text-gray-900">{mockProject.database.version}</div>
              </div>
              <div className="p-4 bg-white rounded-lg">
                <div className="text-xs text-gray-500 mb-1">Total Size</div>
                <div className="text-sm text-gray-900">{mockProject.database.size}</div>
              </div>
              <div className="p-4 bg-white rounded-lg">
                <div className="text-xs text-gray-500 mb-1">Active Connections</div>
                <div className="text-sm text-gray-900">{mockProject.database.connections}</div>
              </div>
              <div className="p-4 bg-white rounded-lg">
                <div className="text-xs text-gray-500 mb-1">Total Tables</div>
                <div className="text-sm text-gray-900">{mockProject.database.tables}</div>
              </div>
            </div>
          </div>

          <div className="bg-gray-50 border border-gray-200 rounded-xl p-6">
            <h3 className="font-semibold text-gray-900 mb-4">Quick Actions</h3>
            <div className="grid grid-cols-3 gap-3">
              <RippleButton variant="outline" className="h-auto py-4 flex flex-col items-center gap-2">
                <Terminal className="w-6 h-6 text-[#FFD700]" />
                <span className="text-sm">SQL Editor</span>
              </RippleButton>
              <RippleButton variant="outline" className="h-auto py-4 flex flex-col items-center gap-2">
                <Upload className="w-6 h-6 text-[#FFD700]" />
                <span className="text-sm">Import Data</span>
              </RippleButton>
              <RippleButton variant="outline" className="h-auto py-4 flex flex-col items-center gap-2">
                <Download className="w-6 h-6 text-gray-500" />
                <span className="text-sm">Backup</span>
              </RippleButton>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// Auth Panel Component
function AuthPanel({ users }: { users: AuthUser[] }) {
  return (
    <div className="h-full flex">
      {/* Users List */}
      <div className="flex-1 flex flex-col">
        <div className="h-14 border-b border-gray-200 flex items-center justify-between px-4">
          <div className="flex items-center gap-3">
            <Input
              placeholder="Search users..."
              className="bg-gray-50 border-gray-200 w-64"
            />
          </div>
          <div className="flex items-center gap-2">
            <RippleButton variant="outline" size="sm" className="gap-2">
              <Filter className="w-4 h-4" />
              Filter
            </RippleButton>
            <RippleButton size="sm" className="gap-2 bg-[#FFD700] text-black">
              <Plus className="w-4 h-4" />
              Add User
            </RippleButton>
          </div>
        </div>

        <div className="flex-1 overflow-auto p-4">
          <div className="bg-gray-50 border border-gray-200 rounded-lg overflow-hidden">
            <table className="w-full">
              <thead className="bg-gray-100">
                <tr>
                  <th className="text-left px-4 py-3 text-xs font-medium text-gray-600">User</th>
                  <th className="text-left px-4 py-3 text-xs font-medium text-gray-600">Providers</th>
                  <th className="text-left px-4 py-3 text-xs font-medium text-gray-600">Status</th>
                  <th className="text-left px-4 py-3 text-xs font-medium text-gray-600">Last Sign In</th>
                  <th className="text-left px-4 py-3 text-xs font-medium text-gray-600">Created</th>
                </tr>
              </thead>
              <tbody>
                {users.map((user) => (
                  <tr key={user.id} className="border-b border-gray-200 last:border-0 hover:bg-gray-100/50">
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-3">
                        <div className="w-8 h-8 rounded-full bg-gradient-to-br from-[#FFD700] to-[#FFA500] flex items-center justify-center text-black text-sm font-medium">
                          {user.email[0].toUpperCase()}
                        </div>
                        <div>
                          <div className="text-sm text-gray-900">{user.email}</div>
                          <div className="text-xs text-gray-500">{user.id.slice(0, 8)}...</div>
                        </div>
                      </div>
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex gap-1">
                        {user.providers.map((provider) => (
                          <Badge key={provider} className="bg-gray-100 text-gray-600 border-0 text-[10px]">
                            {provider}
                          </Badge>
                        ))}
                      </div>
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        {user.emailConfirmed ? (
                          <Badge className="bg-green-100 text-green-600 border-0 text-[10px]">
                            <Check className="w-3 h-3 mr-1" />
                            Verified
                          </Badge>
                        ) : (
                          <Badge className="bg-yellow-100 text-yellow-600 border-0 text-[10px]">
                            <AlertCircle className="w-3 h-3 mr-1" />
                            Pending
                          </Badge>
                        )}
                      </div>
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-600">
                      {user.lastSignIn?.toLocaleString()}
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-600">
                      {user.createdAt.toLocaleDateString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Auth Settings Sidebar */}
      <div className="w-80 border-l border-gray-200 p-4">
        <h3 className="font-semibold text-gray-900 mb-4">Auth Settings</h3>
        <div className="space-y-4">
          <div className="p-4 bg-gray-50 border border-gray-200 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <Shield className="w-4 h-4 text-[#FFD700]" />
              <span className="text-sm font-medium text-gray-900">Email Auth</span>
            </div>
            <div className="text-xs text-gray-500">Enabled with magic link</div>
          </div>
          <div className="p-4 bg-gray-50 border border-gray-200 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <GitBranch className="w-4 h-4 text-[#FFD700]" />
              <span className="text-sm font-medium text-gray-900">GitHub OAuth</span>
            </div>
            <div className="text-xs text-gray-500">Configured and active</div>
          </div>
          <div className="p-4 bg-gray-50 border border-gray-200 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <Globe className="w-4 h-4 text-[#4285F4]" />
              <span className="text-sm font-medium text-gray-900">Google OAuth</span>
            </div>
            <div className="text-xs text-gray-500">Configured and active</div>
          </div>
        </div>
      </div>
    </div>
  );
}

// Storage Panel Component
function StorageBuckets({ buckets }: { buckets: StorageBucket[] }) {
  return (
    <div className="h-full flex">
      {/* Buckets List */}
      <div className="flex-1 p-6 overflow-auto">
        <div className="flex items-center justify-between mb-6">
          <h3 className="font-semibold text-gray-900">Storage Buckets</h3>
          <RippleButton size="sm" className="gap-2 bg-[#FFD700] text-black">
            <Plus className="w-4 h-4" />
            New Bucket
          </RippleButton>
        </div>

        <div className="grid grid-cols-3 gap-4">
          {buckets.map((bucket) => (
            <div
              key={bucket.id}
              className="bg-gray-50 border border-gray-200 rounded-xl p-5 hover:border-[#FFD700]/50 transition-colors cursor-pointer"
            >
              <div className="flex items-start justify-between mb-4">
                <div className="w-12 h-12 rounded-xl bg-gray-100 flex items-center justify-center">
                  <Box className="w-6 h-6 text-[#FFD700]" />
                </div>
                {bucket.public ? (
                  <Badge className="bg-green-100 text-green-600 border-0 text-[10px]">Public</Badge>
                ) : (
                  <Badge className="bg-gray-100 text-gray-600 border-0 text-[10px]">Private</Badge>
                )}
              </div>
              <h4 className="font-medium text-gray-900 mb-1">{bucket.name}</h4>
              <div className="text-xs text-gray-500 mb-3">
                {bucket.fileCount.toLocaleString()} files • {bucket.size}
              </div>
              <div className="flex flex-wrap gap-1">
                {bucket.allowedMimeTypes.slice(0, 2).map((type) => (
                  <Badge key={type} className="bg-white text-gray-600 border-0 text-[9px]">
                    {type.split('/')[1] || type}
                  </Badge>
                ))}
                {bucket.allowedMimeTypes.length > 2 && (
                  <Badge className="bg-white text-gray-600 border-0 text-[9px]">
                    +{bucket.allowedMimeTypes.length - 2}
                  </Badge>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Storage Stats */}
      <div className="w-80 border-l border-gray-200 p-4">
        <h3 className="font-semibold text-gray-900 mb-4">Storage Stats</h3>
        <div className="space-y-4">
          <div className="p-4 bg-gray-50 border border-gray-200 rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-gray-600">Total Size</span>
              <span className="text-sm font-medium text-gray-900">{mockProject.storage.size}</span>
            </div>
            <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
              <div className="h-full bg-[#FFD700] w-[65%]" />
            </div>
            <div className="text-xs text-gray-500 mt-1">65% of 1GB used</div>
          </div>
          <div className="p-4 bg-gray-50 border border-gray-200 rounded-lg">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600">Total Files</span>
              <span className="text-sm font-medium text-gray-900">{mockProject.storage.files.toLocaleString()}</span>
            </div>
          </div>
          <div className="p-4 bg-gray-50 border border-gray-200 rounded-lg">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600">Buckets</span>
              <span className="text-sm font-medium text-gray-900">{mockProject.storage.buckets}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// API Panel Component
function APIPanel({ endpoints, projectUrl }: { endpoints: APIEndpoint[]; projectUrl: string }) {
  const [selectedEndpoint, setSelectedEndpoint] = useState<APIEndpoint | null>(null);

  return (
    <div className="h-full flex">
      {/* Endpoints List */}
      <div className="w-80 border-r border-gray-200 flex flex-col">
        <div className="h-14 border-b border-gray-200 flex items-center px-4">
          <Input
            placeholder="Search endpoints..."
            className="bg-gray-50 border-gray-200"
          />
        </div>
        <ScrollArea className="flex-1 p-3">
          <div className="space-y-1">
            {endpoints.map((endpoint) => (
              <button
                key={endpoint.id}
                onClick={() => setSelectedEndpoint(endpoint)}
                className={cn(
                  'w-full p-3 rounded-lg text-left transition-colors',
                  selectedEndpoint?.id === endpoint.id
                    ? 'bg-[#FFD700]/10 border border-[#FFD700]/30'
                    : 'hover:bg-gray-100'
                )}
              >
                <div className="flex items-center gap-2 mb-1">
                  <Badge
                    className={cn(
                      'text-[10px] border-0',
                      endpoint.method === 'GET' && 'bg-blue-100 text-blue-600',
                      endpoint.method === 'POST' && 'bg-green-100 text-green-600',
                      endpoint.method === 'PATCH' && 'bg-yellow-100 text-yellow-600',
                      endpoint.method === 'DELETE' && 'bg-red-100 text-red-600'
                    )}
                  >
                    {endpoint.method}
                  </Badge>
                  <span className="text-xs text-gray-600 truncate">{endpoint.table}</span>
                </div>
                <div className="text-sm text-gray-900 truncate">{endpoint.path}</div>
              </button>
            ))}
          </div>
        </ScrollArea>
      </div>

      {/* Endpoint Detail */}
      <div className="flex-1 p-6 overflow-auto">
        {selectedEndpoint ? (
          <div className="max-w-3xl space-y-6">
            <div className="flex items-center gap-3">
              <Badge
                className={cn(
                  'text-sm border-0',
                  selectedEndpoint.method === 'GET' && 'bg-blue-100 text-blue-600',
                  selectedEndpoint.method === 'POST' && 'bg-green-100 text-green-600',
                  selectedEndpoint.method === 'PATCH' && 'bg-yellow-100 text-yellow-600',
                  selectedEndpoint.method === 'DELETE' && 'bg-red-100 text-red-600'
                )}
              >
                {selectedEndpoint.method}
              </Badge>
              <code className="text-lg text-gray-900 font-mono">{selectedEndpoint.path}</code>
            </div>

            <p className="text-gray-600">{selectedEndpoint.description}</p>

            <div className="bg-gray-50 border border-gray-200 rounded-xl p-4">
              <div className="flex items-center justify-between mb-3">
                <span className="text-sm font-medium text-gray-900">Request</span>
                <RippleButton variant="ghost" size="sm" onClick={() => navigator.clipboard.writeText(`${projectUrl}${selectedEndpoint.path}`)}>
                  <Copy className="w-4 h-4" />
                </RippleButton>
              </div>
              <code className="block bg-white rounded-lg p-3 text-sm font-mono text-gray-600">
                {projectUrl}{selectedEndpoint.path}
              </code>
            </div>

            <div>
              <h4 className="text-sm font-medium text-gray-900 mb-3">Parameters</h4>
              <div className="bg-gray-50 border border-gray-200 rounded-xl overflow-hidden">
                <table className="w-full">
                  <thead className="bg-gray-100">
                    <tr>
                      <th className="text-left px-4 py-2 text-xs font-medium text-gray-600">Name</th>
                      <th className="text-left px-4 py-2 text-xs font-medium text-gray-600">Type</th>
                      <th className="text-left px-4 py-2 text-xs font-medium text-gray-600">Required</th>
                      <th className="text-left px-4 py-2 text-xs font-medium text-gray-600">Description</th>
                    </tr>
                  </thead>
                  <tbody>
                    {selectedEndpoint.parameters.map((param, i) => (
                      <tr key={i} className="border-b border-gray-200 last:border-0">
                        <td className="px-4 py-2 text-sm text-gray-900">{param.name}</td>
                        <td className="px-4 py-2 text-sm text-gray-600">{param.type}</td>
                        <td className="px-4 py-2">
                          {param.required ? (
                            <Badge className="bg-red-100 text-red-600 border-0 text-[10px]">Required</Badge>
                          ) : (
                            <Badge className="bg-gray-100 text-gray-600 border-0 text-[10px]">Optional</Badge>
                          )}
                        </td>
                        <td className="px-4 py-2 text-sm text-gray-600">{param.description}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div>
              <h4 className="text-sm font-medium text-gray-900 mb-3">Responses</h4>
              <div className="space-y-2">
                {selectedEndpoint.responses.map((response, i) => (
                  <div key={i} className="bg-gray-50 border border-gray-200 rounded-xl p-4">
                    <div className="flex items-center gap-2 mb-2">
                      <Badge
                        className={cn(
                          'border-0 text-[10px]',
                          response.status < 300 && 'bg-green-100 text-green-600',
                          response.status >= 300 && response.status < 400 && 'bg-yellow-100 text-yellow-600',
                          response.status >= 400 && 'bg-red-100 text-red-600'
                        )}
                      >
                        {response.status}
                      </Badge>
                      <span className="text-sm text-gray-600">{response.description}</span>
                    </div>
                    <pre className="bg-white rounded-lg p-3 text-xs font-mono text-gray-600 overflow-auto">
                      {JSON.stringify(response.example, null, 2)}
                    </pre>
                  </div>
                ))}
              </div>
            </div>
          </div>
        ) : (
          <div className="h-full flex flex-col items-center justify-center text-gray-500">
            <Code className="w-16 h-16 mb-4 opacity-50" />
            <p className="text-lg">Select an endpoint to view details</p>
            <p className="text-sm mt-1">Auto-generated REST API from your database schema</p>
          </div>
        )}
      </div>
    </div>
  );
}

// Realtime Panel Component
function RealtimePanel({ connections, messages }: { connections: number; messages: number }) {
  return (
    <div className="h-full p-6 overflow-auto">
      <div className="max-w-4xl mx-auto space-y-6">
        {/* Realtime Stats */}
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-gray-50 border border-gray-200 rounded-xl p-6">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 rounded-lg bg-[#FFD700]/20 flex items-center justify-center">
                <Zap className="w-5 h-5 text-[#FFD700]" />
              </div>
              <div>
                <div className="text-2xl font-bold text-gray-900">{connections.toLocaleString()}</div>
                <div className="text-xs text-gray-500">Active Connections</div>
              </div>
            </div>
            <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
              <motion.div
                className="h-full bg-[#FFD700]"
                initial={{ width: 0 }}
                animate={{ width: '75%' }}
                transition={{ duration: 1 }}
              />
            </div>
          </div>
          <div className="bg-gray-50 border border-gray-200 rounded-xl p-6">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 rounded-lg bg-[#FFD700]/20 flex items-center justify-center">
                <MessageSquare className="w-5 h-5 text-[#FFD700]" />
              </div>
              <div>
                <div className="text-2xl font-bold text-gray-900">{messages.toLocaleString()}</div>
                <div className="text-xs text-gray-500">Messages Today</div>
              </div>
            </div>
            <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
              <motion.div
                className="h-full bg-[#FFD700]"
                initial={{ width: 0 }}
                animate={{ width: '60%' }}
                transition={{ duration: 1 }}
              />
            </div>
          </div>
        </div>

        {/* Channels */}
        <div className="bg-gray-50 border border-gray-200 rounded-xl p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-semibold text-gray-900">Realtime Channels</h3>
            <RippleButton size="sm" className="gap-2 bg-[#FFD700] text-black">
              <Plus className="w-4 h-4" />
              New Channel
            </RippleButton>
          </div>
          <div className="space-y-3">
            {[
              { name: 'public:users', listeners: 892, events: 'INSERT, UPDATE, DELETE' },
              { name: 'public:projects', listeners: 456, events: 'INSERT, UPDATE' },
              { name: 'public:documents', listeners: 1234, events: 'ALL' },
            ].map((channel, i) => (
              <div key={i} className="flex items-center justify-between p-4 bg-white rounded-lg">
                <div className="flex items-center gap-3">
                  <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                  <code className="text-sm text-gray-900">{channel.name}</code>
                </div>
                <div className="flex items-center gap-4 text-sm text-gray-600">
                  <span>{channel.listeners} listeners</span>
                  <Badge className="bg-gray-100 text-gray-600 border-0 text-[10px]">{channel.events}</Badge>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Code Example */}
        <div className="bg-gray-50 border border-gray-200 rounded-xl p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-semibold text-gray-900">Quick Start</h3>
            <RippleButton variant="ghost" size="sm" onClick={() => navigator.clipboard.writeText(`const channel = supabase
  .channel('schema-db-changes')
  .on(
    'postgres_changes',
    { event: '*', schema: 'public', table: 'users' },
    (payload) => console.log(payload)
  )
  .subscribe()`)}>
              <Copy className="w-4 h-4" />
            </RippleButton>
          </div>
          <pre className="bg-white rounded-lg p-4 text-sm font-mono text-gray-600 overflow-auto">
            <code>{`const channel = supabase
  .channel('schema-db-changes')
  .on(
    'postgres_changes',
    { event: '*', schema: 'public', table: 'users' },
    (payload) => console.log(payload)
  )
  .subscribe()`}</code>
          </pre>
        </div>
      </div>
    </div>
  );
}

// Storage Panel (alias for StorageBuckets)
function StoragePanel({ buckets }: { buckets: StorageBucket[] }) {
  return <StorageBuckets buckets={buckets} />;
}
