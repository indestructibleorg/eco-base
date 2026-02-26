import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Plus,
  Check,
  X,
  ExternalLink,
  Settings,
  RefreshCw,
  Github,
  MessageSquare,
  FileText,
  CreditCard,
  Cloud,
  Bot,
  Palette,
  Gamepad2,
  Search,
  Triangle,
  Diamond,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Badge } from '@/components/ui/badge';
import { useIntegrationStore } from '@/store/appStore';
import { cn } from '@/lib/utils';
import type { Integration, IntegrationType } from '@/types';

const integrationIcons: Record<IntegrationType, React.ReactNode> = {
  github: <Github className="w-6 h-6" />,
  gitlab: <span className="text-lg">🦊</span>,
  notion: <FileText className="w-6 h-6" />,
  slack: <MessageSquare className="w-6 h-6" />,
  discord: <Gamepad2 className="w-6 h-6" />,
  figma: <Palette className="w-6 h-6" />,
  vercel: <Triangle className="w-6 h-6" />,
  netlify: <Diamond className="w-6 h-6" />,
  stripe: <CreditCard className="w-6 h-6" />,
  openai: <Bot className="w-6 h-6" />,
  anthropic: <span className="text-lg">🧠</span>,
  google: <Cloud className="w-6 h-6" />,
};

export function IntegrationHub() {
  const { integrations, connectIntegration, disconnectIntegration } = useIntegrationStore();
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedIntegration, setSelectedIntegration] = useState<Integration | null>(null);

  const connectedIntegrations = integrations.filter((i) => i.isConnected);
  const availableIntegrations = integrations.filter((i) => !i.isConnected);

  const filteredConnected = connectedIntegrations.filter((i) =>
    i.name.toLowerCase().includes(searchQuery.toLowerCase())
  );
  const filteredAvailable = availableIntegrations.filter((i) =>
    i.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="flex h-full">
      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="h-16 border-b border-border flex items-center justify-between px-6">
          <div>
            <h1 className="text-xl font-semibold">Integrations</h1>
            <p className="text-sm text-muted-foreground">
              {connectedIntegrations.length} connected, {availableIntegrations.length} available
            </p>
          </div>
          <div className="flex items-center gap-3">
            <div className="relative">
              <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
              <Input
                placeholder="Search integrations..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-64 pl-10 bg-secondary"
              />
            </div>
          </div>
        </div>

        <ScrollArea className="flex-1 p-6">
          {/* Connected Section */}
          {filteredConnected.length > 0 && (
            <div className="mb-8">
              <h2 className="text-sm font-medium text-muted-foreground mb-4 flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-green-500" />
                Connected ({filteredConnected.length})
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                {filteredConnected.map((integration) => (
                  <IntegrationCard
                    key={integration.id}
                    integration={integration}
                    onClick={() => setSelectedIntegration(integration)}
                    onDisconnect={() => disconnectIntegration(integration.type)}
                  />
                ))}
              </div>
            </div>
          )}

          {/* Available Section */}
          {filteredAvailable.length > 0 && (
            <div>
              <h2 className="text-sm font-medium text-muted-foreground mb-4">
                Available ({filteredAvailable.length})
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                {filteredAvailable.map((integration) => (
                  <IntegrationCard
                    key={integration.id}
                    integration={integration}
                    onConnect={() => connectIntegration(integration.type)}
                  />
                ))}
              </div>
            </div>
          )}
        </ScrollArea>
      </div>

      {/* Integration Detail Panel */}
      <AnimatePresence>
        {selectedIntegration && (
          <motion.div
            initial={{ width: 0, opacity: 0 }}
            animate={{ width: 360, opacity: 1 }}
            exit={{ width: 0, opacity: 0 }}
            transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
            className="border-l border-border bg-card overflow-hidden"
          >
            <IntegrationDetail
              integration={selectedIntegration}
              onClose={() => setSelectedIntegration(null)}
              onDisconnect={() => {
                disconnectIntegration(selectedIntegration.type);
                setSelectedIntegration(null);
              }}
            />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

interface IntegrationCardProps {
  integration: Integration;
  onClick?: () => void;
  onConnect?: () => void;
  onDisconnect?: () => void;
}

function IntegrationCard({ integration, onClick, onConnect }: IntegrationCardProps) {
  return (
    <motion.div
      whileHover={{ y: -2 }}
      className={cn(
        'p-4 rounded-xl border cursor-pointer transition-all',
        'bg-card hover:border-primary/50',
        integration.isConnected && 'border-green-500/50'
      )}
      onClick={onClick}
    >
      <div className="flex items-start justify-between mb-3">
        <div
          className="w-12 h-12 rounded-xl flex items-center justify-center text-white"
          style={{ backgroundColor: integration.color }}
        >
          {integrationIcons[integration.type]}
        </div>
        {integration.isConnected ? (
          <div className="flex items-center gap-1 text-green-500 text-xs">
            <Check className="w-3 h-3" />
            Connected
          </div>
        ) : (
          <Button
            size="sm"
            onClick={(e) => {
              e.stopPropagation();
              onConnect?.();
            }}
            className="bg-[#FFD700] text-black hover:bg-[#FFC700]"
          >
            <Plus className="w-3 h-3 mr-1" />
            Connect
          </Button>
        )}
      </div>
      <h3 className="font-medium mb-1">{integration.name}</h3>
      <p className="text-sm text-muted-foreground">{integration.description}</p>
      {integration.isConnected && integration.connectedAt && (
        <p className="text-xs text-muted-foreground mt-2">
          Connected {new Date(integration.connectedAt).toLocaleDateString()}
        </p>
      )}
    </motion.div>
  );
}

interface IntegrationDetailProps {
  integration: Integration;
  onClose: () => void;
  onDisconnect: () => void;
}

function IntegrationDetail({ integration, onClose, onDisconnect }: IntegrationDetailProps) {
  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="h-16 border-b border-border flex items-center justify-between px-4">
        <h3 className="font-medium">Integration Settings</h3>
        <Button variant="ghost" size="icon" onClick={onClose}>
          <X className="w-4 h-4" />
        </Button>
      </div>

      <ScrollArea className="flex-1 p-4">
        {/* Integration Info */}
        <div className="flex items-center gap-4 mb-6">
          <div
            className="w-16 h-16 rounded-xl flex items-center justify-center text-white text-2xl"
            style={{ backgroundColor: integration.color }}
          >
            {integrationIcons[integration.type]}
          </div>
          <div>
            <h2 className="text-lg font-semibold">{integration.name}</h2>
            <p className="text-sm text-muted-foreground">{integration.description}</p>
          </div>
        </div>

        {/* Status */}
        <div className="p-4 rounded-lg bg-secondary mb-6">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm">Status</span>
            <Badge variant={integration.isConnected ? 'default' : 'secondary'}>
              {integration.isConnected ? 'Connected' : 'Disconnected'}
            </Badge>
          </div>
          {integration.isConnected && integration.connectedAt && (
            <p className="text-xs text-muted-foreground">
              Connected on {new Date(integration.connectedAt).toLocaleDateString()}
            </p>
          )}
        </div>

        {/* Settings */}
        <div className="space-y-4 mb-6">
          <h4 className="text-sm font-medium">Configuration</h4>
          
          <div className="space-y-3">
            <div className="flex items-center justify-between p-3 rounded-lg bg-secondary">
              <div>
                <p className="text-sm font-medium">Auto-sync</p>
                <p className="text-xs text-muted-foreground">Automatically sync data</p>
              </div>
              <div className="w-10 h-6 rounded-full bg-primary relative cursor-pointer">
                <div className="w-4 h-4 rounded-full bg-white absolute right-1 top-1" />
              </div>
            </div>

            <div className="flex items-center justify-between p-3 rounded-lg bg-secondary">
              <div>
                <p className="text-sm font-medium">Notifications</p>
                <p className="text-xs text-muted-foreground">Receive notifications</p>
              </div>
              <div className="w-10 h-6 rounded-full bg-primary relative cursor-pointer">
                <div className="w-4 h-4 rounded-full bg-white absolute right-1 top-1" />
              </div>
            </div>
          </div>
        </div>

        {/* Actions */}
        <div className="space-y-3">
          <Button variant="outline" className="w-full">
            <RefreshCw className="w-4 h-4 mr-2" />
            Sync Now
          </Button>
          <Button variant="outline" className="w-full">
            <Settings className="w-4 h-4 mr-2" />
            Advanced Settings
          </Button>
          <Button variant="outline" className="w-full">
            <ExternalLink className="w-4 h-4 mr-2" />
            Open {integration.name}
          </Button>
          <Button variant="destructive" className="w-full" onClick={onDisconnect}>
            <X className="w-4 h-4 mr-2" />
            Disconnect
          </Button>
        </div>
      </ScrollArea>
    </div>
  );
}
