import { useState } from 'react';
import { motion } from 'framer-motion';
import {
  Rocket,
  Globe,
  RefreshCw,
  Check,
  X,
  Clock,
  ChevronDown,
  ChevronUp,
  ExternalLink,
  BarChart3,
  Users,
  Clock4,
  Globe2,
  Terminal,
  Settings,
  RotateCcw,
  AlertCircle,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';

import { useDeploymentStore } from '@/store/appStore';
import { cn } from '@/lib/utils';
import type { Deployment, DeploymentLog } from '@/types';

export function DeployCenter() {
  const { deployments } = useDeploymentStore();
  const [selectedDeployment, setSelectedDeployment] = useState<Deployment | null>(null);

  const latestDeployment = deployments[0];
  const previousDeployments = deployments.slice(1);

  return (
    <div className="flex h-full">
      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="h-16 border-b border-border flex items-center justify-between px-6">
          <div>
            <h1 className="text-xl font-semibold">Deployments</h1>
            <p className="text-sm text-muted-foreground">
              Manage your deployments and monitor performance
            </p>
          </div>
          <div className="flex items-center gap-3">
            <Button variant="outline">
              <Settings className="w-4 h-4 mr-2" />
              Settings
            </Button>
            <Button className="bg-[#FFD700] text-black hover:bg-[#FFC700]">
              <Rocket className="w-4 h-4 mr-2" />
              Deploy Now
            </Button>
          </div>
        </div>

        <ScrollArea className="flex-1">
          <div className="p-6 space-y-6">
            {/* Current Deployment */}
            {latestDeployment && (
              <div className="space-y-4">
                <h2 className="text-sm font-medium text-muted-foreground">Current Deployment</h2>
                <CurrentDeploymentCard deployment={latestDeployment} />
              </div>
            )}

            {/* Analytics */}
            {latestDeployment?.stats && (
              <div className="space-y-4">
                <h2 className="text-sm font-medium text-muted-foreground">Analytics</h2>
                <AnalyticsCards stats={latestDeployment.stats} />
              </div>
            )}

            {/* Deployment History */}
            <div className="space-y-4">
              <h2 className="text-sm font-medium text-muted-foreground">Deployment History</h2>
              <div className="space-y-2">
                {previousDeployments.map((deployment) => (
                  <DeploymentHistoryCard
                    key={deployment.id}
                    deployment={deployment}
                    isSelected={selectedDeployment?.id === deployment.id}
                    onClick={() => setSelectedDeployment(
                      selectedDeployment?.id === deployment.id ? null : deployment
                    )}
                  />
                ))}
              </div>
            </div>
          </div>
        </ScrollArea>
      </div>

      {/* Deployment Detail Panel */}
      {selectedDeployment && (
        <motion.div
          initial={{ width: 0, opacity: 0 }}
          animate={{ width: 400, opacity: 1 }}
          exit={{ width: 0, opacity: 0 }}
          className="border-l border-border bg-card"
        >
          <DeploymentDetailPanel
            deployment={selectedDeployment}
            onClose={() => setSelectedDeployment(null)}
          />
        </motion.div>
      )}
    </div>
  );
}

function CurrentDeploymentCard({ deployment }: { deployment: Deployment }) {
  const statusColors = {
    pending: 'bg-yellow-500/10 text-yellow-500 border-yellow-500/30',
    building: 'bg-blue-500/10 text-blue-500 border-blue-500/30',
    success: 'bg-green-500/10 text-green-500 border-green-500/30',
    failed: 'bg-red-500/10 text-red-500 border-red-500/30',
    cancelled: 'bg-gray-500/10 text-gray-500 border-gray-500/30',
  };

  const statusIcons = {
    pending: <Clock className="w-4 h-4" />,
    building: <RefreshCw className="w-4 h-4 animate-spin" />,
    success: <Check className="w-4 h-4" />,
    failed: <X className="w-4 h-4" />,
    cancelled: <AlertCircle className="w-4 h-4" />,
  };

  return (
    <div className="p-6 rounded-xl border bg-card">
      <div className="flex items-start justify-between mb-4">
        <div>
          <div className="flex items-center gap-3 mb-2">
            <Badge className={cn('border', statusColors[deployment.status])}>
              <span className="flex items-center gap-1">
                {statusIcons[deployment.status]}
                {deployment.status.charAt(0).toUpperCase() + deployment.status.slice(1)}
              </span>
            </Badge>
            <span className="text-sm text-muted-foreground">{deployment.version}</span>
          </div>
          <h3 className="text-lg font-semibold">Production Deployment</h3>
          <p className="text-sm text-muted-foreground">
            Deployed by {deployment.createdBy} • {new Date(deployment.createdAt).toLocaleString()}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {deployment.url && (
            <Button variant="outline" size="sm" asChild>
              <a href={deployment.url} target="_blank" rel="noopener noreferrer">
                <Globe className="w-4 h-4 mr-2" />
                Visit
              </a>
            </Button>
          )}
          <Button variant="outline" size="sm">
            <RotateCcw className="w-4 h-4 mr-2" />
            Redeploy
          </Button>
        </div>
      </div>

      {deployment.customDomain && (
        <div className="flex items-center gap-2 text-sm mb-4">
          <Globe2 className="w-4 h-4 text-muted-foreground" />
          <span className="text-muted-foreground">Custom Domain:</span>
          <a
            href={`https://${deployment.customDomain}`}
            target="_blank"
            rel="noopener noreferrer"
            className="text-primary hover:underline flex items-center gap-1"
          >
            {deployment.customDomain}
            <ExternalLink className="w-3 h-3" />
          </a>
        </div>
      )}

      {/* Deployment Logs */}
      {deployment.logs && deployment.logs.length > 0 && (
        <div className="mt-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium">Deployment Logs</span>
          </div>
          <div className="bg-gray-100 rounded-lg p-4 font-mono text-sm max-h-48 overflow-auto">
            {deployment.logs.map((log) => (
              <LogLine key={log.id} log={log} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function LogLine({ log }: { log: DeploymentLog }) {
  const levelColors = {
    info: 'text-gray-700',
    warn: 'text-yellow-600',
    error: 'text-red-600',
  };

  return (
    <div className={cn('mb-1', levelColors[log.level])}>
      <span className="text-gray-500 text-xs">
        {new Date(log.timestamp).toLocaleTimeString()}
      </span>{' '}
      {log.message}
    </div>
  );
}

function AnalyticsCards({ stats }: { stats: NonNullable<Deployment['stats']> }) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      <AnalyticsCard
        icon={<BarChart3 className="w-5 h-5" />}
        label="Total Visits"
        value={stats.visits.toLocaleString()}
        change="+12.5%"
        positive
      />
      <AnalyticsCard
        icon={<Users className="w-5 h-5" />}
        label="Unique Visitors"
        value={stats.uniqueVisitors.toLocaleString()}
        change="+8.3%"
        positive
      />
      <AnalyticsCard
        icon={<Clock4 className="w-5 h-5" />}
        label="Avg Load Time"
        value={`${stats.avgLoadTime}s`}
        change="-0.2s"
        positive
      />
      <AnalyticsCard
        icon={<Globe2 className="w-5 h-5" />}
        label="Countries"
        value={stats.countries.toString()}
        change="+3"
        positive
      />
    </div>
  );
}

interface AnalyticsCardProps {
  icon: React.ReactNode;
  label: string;
  value: string;
  change: string;
  positive: boolean;
}

function AnalyticsCard({ icon, label, value, change, positive }: AnalyticsCardProps) {
  return (
    <div className="p-4 rounded-xl border bg-card">
      <div className="flex items-center gap-3 mb-3">
        <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center text-primary">
          {icon}
        </div>
        <span className="text-sm text-muted-foreground">{label}</span>
      </div>
      <div className="flex items-end justify-between">
        <span className="text-2xl font-semibold">{value}</span>
        <span className={cn('text-sm', positive ? 'text-green-500' : 'text-red-500')}>
          {change}
        </span>
      </div>
    </div>
  );
}

interface DeploymentHistoryCardProps {
  deployment: Deployment;
  isSelected: boolean;
  onClick: () => void;
}

function DeploymentHistoryCard({ deployment, isSelected, onClick }: DeploymentHistoryCardProps) {
  const statusColors = {
    pending: 'text-yellow-500',
    building: 'text-blue-500',
    success: 'text-green-500',
    failed: 'text-red-500',
    cancelled: 'text-gray-500',
  };

  const statusIcons = {
    pending: <Clock className="w-4 h-4" />,
    building: <RefreshCw className="w-4 h-4" />,
    success: <Check className="w-4 h-4" />,
    failed: <X className="w-4 h-4" />,
    cancelled: <AlertCircle className="w-4 h-4" />,
  };

  return (
    <button
      onClick={onClick}
      className={cn(
        'w-full flex items-center justify-between p-4 rounded-lg border transition-all text-left',
        isSelected ? 'border-primary bg-primary/5' : 'bg-card hover:border-primary/30'
      )}
    >
      <div className="flex items-center gap-4">
        <div className={cn('flex items-center gap-2', statusColors[deployment.status])}>
          {statusIcons[deployment.status]}
        </div>
        <div>
          <div className="flex items-center gap-2">
            <span className="font-medium">{deployment.version}</span>
            <span className="text-xs text-muted-foreground">
              by {deployment.createdBy}
            </span>
          </div>
          <p className="text-sm text-muted-foreground">
            {new Date(deployment.createdAt).toLocaleString()}
          </p>
        </div>
      </div>
      <div className="flex items-center gap-2">
        {deployment.status === 'success' && (
          <Badge variant="outline" className="text-green-500 border-green-500/30">
            Live
          </Badge>
        )}
        {isSelected ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
      </div>
    </button>
  );
}

interface DeploymentDetailPanelProps {
  deployment: Deployment;
  onClose: () => void;
}

function DeploymentDetailPanel({ deployment, onClose }: DeploymentDetailPanelProps) {
  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="h-16 border-b border-border flex items-center justify-between px-4">
        <h3 className="font-medium">Deployment Details</h3>
        <Button variant="ghost" size="icon" onClick={onClose}>
          <X className="w-4 h-4" />
        </Button>
      </div>

      <ScrollArea className="flex-1 p-4">
        <div className="space-y-6">
          {/* Info */}
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-sm text-muted-foreground">Version</span>
              <span className="text-sm font-medium">{deployment.version}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-sm text-muted-foreground">Status</span>
              <Badge
                variant={deployment.status === 'success' ? 'default' : 'secondary'}
                className={cn(
                  deployment.status === 'failed' && 'bg-red-500/10 text-red-500'
                )}
              >
                {deployment.status}
              </Badge>
            </div>
            <div className="flex justify-between">
              <span className="text-sm text-muted-foreground">Created</span>
              <span className="text-sm">{new Date(deployment.createdAt).toLocaleString()}</span>
            </div>
            {deployment.completedAt && (
              <div className="flex justify-between">
                <span className="text-sm text-muted-foreground">Completed</span>
                <span className="text-sm">{new Date(deployment.completedAt).toLocaleString()}</span>
              </div>
            )}
            <div className="flex justify-between">
              <span className="text-sm text-muted-foreground">Created By</span>
              <span className="text-sm">{deployment.createdBy}</span>
            </div>
          </div>

          {/* URL */}
          {deployment.url && (
            <div className="p-3 rounded-lg bg-secondary">
              <span className="text-sm text-muted-foreground block mb-1">Deployment URL</span>
              <a
                href={deployment.url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm text-primary hover:underline flex items-center gap-1"
              >
                {deployment.url}
                <ExternalLink className="w-3 h-3" />
              </a>
            </div>
          )}

          {/* Actions */}
          <div className="space-y-2">
            <Button variant="outline" className="w-full">
              <RotateCcw className="w-4 h-4 mr-2" />
              Redeploy This Version
            </Button>
            <Button variant="outline" className="w-full">
              <Terminal className="w-4 h-4 mr-2" />
              View Full Logs
            </Button>
          </div>
        </div>
      </ScrollArea>
    </div>
  );
}
