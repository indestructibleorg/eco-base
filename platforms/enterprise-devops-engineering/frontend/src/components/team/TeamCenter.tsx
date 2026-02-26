import { useState } from 'react';

import {
  Plus,
  X,
  MoreHorizontal,
  Mail,
  Crown,
  Shield,
  User,
  Eye,
  Clock,
  MessageSquare,
  GitCommit,
  Rocket,
  GitPullRequest,
  AlertCircle,
  Check,
  Bell,
  Search,
  Filter,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from '@/components/ui/dialog';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { useTeamStore, useAppStore } from '@/store/appStore';
import { cn } from '@/lib/utils';
import type { TeamMember, Activity, Notification } from '@/types';

const roleIcons = {
  owner: <Crown className="w-4 h-4 text-yellow-500" />,
  admin: <Shield className="w-4 h-4 text-blue-500" />,
  member: <User className="w-4 h-4 text-muted-foreground" />,
  viewer: <Eye className="w-4 h-4 text-muted-foreground" />,
};

const roleLabels = {
  owner: 'Owner',
  admin: 'Admin',
  member: 'Member',
  viewer: 'Viewer',
};

const activityIcons = {
  commit: <GitCommit className="w-4 h-4" />,
  deploy: <Rocket className="w-4 h-4" />,
  comment: <MessageSquare className="w-4 h-4" />,
  pr: <GitPullRequest className="w-4 h-4" />,
  issue: <AlertCircle className="w-4 h-4" />,
  member: <User className="w-4 h-4" />,
};

export function TeamCenter() {
  const { team, activities, notifications, inviteMember, removeMember, markNotificationRead } = useTeamStore();
  const { currentUser } = useAppStore();
  const [activeTab, setActiveTab] = useState('members');
  const [inviteDialogOpen, setInviteDialogOpen] = useState(false);
  const [inviteEmail, setInviteEmail] = useState('');
  const [inviteRole, setInviteRole] = useState('member');
  const [searchQuery, setSearchQuery] = useState('');

  const unreadCount = notifications.filter((n) => !n.read).length;

  const filteredMembers = team?.members.filter(
    (m) =>
      m.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      m.email.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const handleInvite = () => {
    if (!inviteEmail.trim()) return;
    inviteMember(inviteEmail, inviteRole);
    setInviteEmail('');
    setInviteDialogOpen(false);
  };

  return (
    <div className="flex h-full">
      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="h-16 border-b border-border flex items-center justify-between px-6">
          <div className="flex items-center gap-4">
            <div>
              <h1 className="text-xl font-semibold">{team?.name}</h1>
              <p className="text-sm text-muted-foreground">{team?.description}</p>
            </div>
            <Badge variant="secondary" className="capitalize">
              {team?.plan} Plan
            </Badge>
          </div>
          <div className="flex items-center gap-3">
            <Button onClick={() => setInviteDialogOpen(true)} className="bg-[#FFD700] text-black hover:bg-[#FFC700]">
              <Plus className="w-4 h-4 mr-2" />
              Invite Member
            </Button>
          </div>
        </div>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1 flex flex-col">
          <div className="border-b border-border px-6">
            <TabsList className="bg-transparent">
              <TabsTrigger value="members" className="data-[state=active]:bg-secondary">
                Members
              </TabsTrigger>
              <TabsTrigger value="activity" className="data-[state=active]:bg-secondary">
                Activity
              </TabsTrigger>
              <TabsTrigger value="notifications" className="data-[state=active]:bg-secondary relative">
                Notifications
                {unreadCount > 0 && (
                  <span className="absolute -top-1 -right-1 w-5 h-5 rounded-full bg-primary text-[10px] flex items-center justify-center text-primary-foreground">
                    {unreadCount}
                  </span>
                )}
              </TabsTrigger>
            </TabsList>
          </div>

          <TabsContent value="members" className="flex-1 m-0">
            <div className="p-6">
              <div className="flex items-center gap-4 mb-6">
                <div className="relative flex-1 max-w-md">
                  <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
                  <Input
                    placeholder="Search members..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="pl-10"
                  />
                </div>
                <Button variant="outline" size="icon">
                  <Filter className="w-4 h-4" />
                </Button>
              </div>

              <div className="space-y-2">
                {filteredMembers?.map((member) => (
                  <MemberCard
                    key={member.id}
                    member={member}
                    isCurrentUser={member.userId === currentUser?.id}
                    onRemove={() => removeMember(member.id)}
                  />
                ))}
              </div>
            </div>
          </TabsContent>

          <TabsContent value="activity" className="flex-1 m-0">
            <ScrollArea className="h-[calc(100vh-200px)]">
              <div className="p-6 space-y-4">
                {activities.map((activity) => (
                  <ActivityCard key={activity.id} activity={activity} />
                ))}
              </div>
            </ScrollArea>
          </TabsContent>

          <TabsContent value="notifications" className="flex-1 m-0">
            <ScrollArea className="h-[calc(100vh-200px)]">
              <div className="p-6 space-y-4">
                {notifications.map((notification) => (
                  <NotificationCard
                    key={notification.id}
                    notification={notification}
                    onMarkRead={() => markNotificationRead(notification.id)}
                  />
                ))}
              </div>
            </ScrollArea>
          </TabsContent>
        </Tabs>
      </div>

      {/* Invite Dialog */}
      <Dialog open={inviteDialogOpen} onOpenChange={setInviteDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Invite Team Member</DialogTitle>
          </DialogHeader>
          <div className="space-y-4">
            <div>
              <label className="text-sm font-medium mb-2 block">Email Address</label>
              <Input
                type="email"
                placeholder="colleague@company.com"
                value={inviteEmail}
                onChange={(e) => setInviteEmail(e.target.value)}
              />
            </div>
            <div>
              <label className="text-sm font-medium mb-2 block">Role</label>
              <div className="grid grid-cols-2 gap-2">
                {(['admin', 'member', 'viewer'] as const).map((role) => (
                  <button
                    key={role}
                    onClick={() => setInviteRole(role)}
                    className={cn(
                      'p-3 rounded-lg border text-left transition-all',
                      inviteRole === role
                        ? 'border-primary bg-primary/10'
                        : 'border-border hover:border-primary/50'
                    )}
                  >
                    <div className="flex items-center gap-2 mb-1">
                      {roleIcons[role]}
                      <span className="font-medium capitalize">{role}</span>
                    </div>
                    <p className="text-xs text-muted-foreground">
                      {role === 'admin' && 'Full access to all settings'}
                      {role === 'member' && 'Can edit and deploy projects'}
                      {role === 'viewer' && 'View-only access'}
                    </p>
                  </button>
                ))}
              </div>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setInviteDialogOpen(false)}>
              Cancel
            </Button>
            <Button onClick={handleInvite} className="bg-[#FFD700] text-black hover:bg-[#FFC700]">
              <Mail className="w-4 h-4 mr-2" />
              Send Invite
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

interface MemberCardProps {
  member: TeamMember;
  isCurrentUser: boolean;
  onRemove: () => void;
}

function MemberCard({ member, isCurrentUser, onRemove }: MemberCardProps) {
  return (
    <div className="flex items-center justify-between p-4 rounded-lg bg-card border hover:border-primary/30 transition-all">
      <div className="flex items-center gap-4">
        <div className="relative">
          <Avatar className="w-10 h-10">
            <AvatarImage src={member.avatar} />
            <AvatarFallback>{member.name[0]}</AvatarFallback>
          </Avatar>
          <div
            className={cn(
              'absolute -bottom-0.5 -right-0.5 w-3 h-3 rounded-full border-2 border-card',
              member.status === 'online' && 'bg-green-500',
              member.status === 'away' && 'bg-yellow-500',
              member.status === 'offline' && 'bg-gray-500'
            )}
          />
        </div>
        <div>
          <div className="flex items-center gap-2">
            <span className="font-medium">{member.name}</span>
            {isCurrentUser && (
              <Badge variant="secondary" className="text-xs">
                You
              </Badge>
            )}
          </div>
          <p className="text-sm text-muted-foreground">{member.email}</p>
        </div>
      </div>
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          {roleIcons[member.role]}
          <span>{roleLabels[member.role]}</span>
        </div>
        {member.status !== 'offline' && member.lastSeen && (
          <div className="flex items-center gap-1 text-xs text-muted-foreground">
            <Clock className="w-3 h-3" />
            {formatLastSeen(member.lastSeen)}
          </div>
        )}
        {!isCurrentUser && (
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="icon">
                <MoreHorizontal className="w-4 h-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem onClick={onRemove} className="text-destructive">
                <X className="w-4 h-4 mr-2" />
                Remove
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        )}
      </div>
    </div>
  );
}

interface ActivityCardProps {
  activity: Activity;
}

function ActivityCard({ activity }: ActivityCardProps) {
  return (
    <div className="flex items-start gap-4 p-4 rounded-lg bg-card border hover:bg-secondary/50 transition-all">
      <div className="w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center text-primary">
        {activityIcons[activity.type]}
      </div>
      <div className="flex-1">
        <div className="flex items-center gap-2 mb-1">
          <Avatar className="w-5 h-5">
            <AvatarImage src={activity.user.avatar} />
            <AvatarFallback>{activity.user.name[0]}</AvatarFallback>
          </Avatar>
          <span className="font-medium text-sm">{activity.user.name}</span>
          <span className="text-muted-foreground">•</span>
          <span className="text-xs text-muted-foreground">
            {formatTimeAgo(activity.timestamp)}
          </span>
        </div>
        <h4 className="font-medium">{activity.title}</h4>
        {activity.description && (
          <p className="text-sm text-muted-foreground mt-1">{activity.description}</p>
        )}
      </div>
    </div>
  );
}

interface NotificationCardProps {
  notification: Notification;
  onMarkRead: () => void;
}

function NotificationCard({ notification, onMarkRead }: NotificationCardProps) {
  const typeColors = {
    info: 'bg-blue-500/10 text-blue-500',
    success: 'bg-green-500/10 text-green-500',
    warning: 'bg-yellow-500/10 text-yellow-500',
    error: 'bg-red-500/10 text-red-500',
  };

  const typeIcons = {
    info: <Bell className="w-4 h-4" />,
    success: <Check className="w-4 h-4" />,
    warning: <AlertCircle className="w-4 h-4" />,
    error: <X className="w-4 h-4" />,
  };

  return (
    <div
      className={cn(
        'flex items-start gap-4 p-4 rounded-lg border transition-all',
        notification.read ? 'bg-card opacity-60' : 'bg-card border-primary/30'
      )}
    >
      <div className={cn('w-10 h-10 rounded-full flex items-center justify-center', typeColors[notification.type])}>
        {typeIcons[notification.type]}
      </div>
      <div className="flex-1">
        <div className="flex items-center justify-between mb-1">
          <h4 className="font-medium">{notification.title}</h4>
          <span className="text-xs text-muted-foreground">
            {formatTimeAgo(notification.timestamp)}
          </span>
        </div>
        <p className="text-sm text-muted-foreground">{notification.message}</p>
        {notification.action && (
          <Button variant="link" size="sm" className="p-0 h-auto mt-2">
            {notification.action.label}
          </Button>
        )}
      </div>
      {!notification.read && (
        <Button variant="ghost" size="icon" className="h-6 w-6" onClick={onMarkRead}>
          <Check className="w-3 h-3" />
        </Button>
      )}
    </div>
  );
}

function formatLastSeen(date: Date): string {
  const now = new Date();
  const diff = now.getTime() - new Date(date).getTime();
  const minutes = Math.floor(diff / 60000);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);

  if (minutes < 1) return 'just now';
  if (minutes < 60) return `${minutes}m ago`;
  if (hours < 24) return `${hours}h ago`;
  return `${days}d ago`;
}

function formatTimeAgo(date: Date): string {
  const now = new Date();
  const diff = now.getTime() - new Date(date).getTime();
  const minutes = Math.floor(diff / 60000);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);

  if (minutes < 1) return 'just now';
  if (minutes < 60) return `${minutes}m ago`;
  if (hours < 24) return `${hours}h ago`;
  if (days < 7) return `${days}d ago`;
  return new Date(date).toLocaleDateString();
}
