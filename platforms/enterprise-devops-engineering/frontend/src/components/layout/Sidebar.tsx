import { motion } from 'framer-motion';
import { useTranslation } from 'react-i18next';
import {
  MessageSquare,
  Code2,
  Monitor,
  Bot,
  Plug,
  Users,
  Rocket,
  Settings,
  Moon,
  ChevronLeft,
  ChevronRight,
  Sparkles,
  Globe,
  Cpu,
  Zap,
  Database,
  Wand2,
  GraduationCap,
  Home,
  Languages,
} from 'lucide-react';
import { RippleButton } from '@/components/ui/ripple-button';
import { useAppStore, useTeamStore } from '@/store/appStore';
import { cn } from '@/lib/utils';
import type { ViewMode } from '@/App';

export function Sidebar() {
  const { t, i18n } = useTranslation();
  const { currentView, setCurrentView, isSidebarOpen, toggleSidebar, toggleDarkMode, currentUser } = useAppStore();
  
  const toggleLanguage = () => {
    const newLang = i18n.language === 'en' ? 'zh-TW' : 'en';
    i18n.changeLanguage(newLang);
  };
  const { notifications } = useTeamStore();
  const unreadCount = notifications.filter((n) => !n.read).length;

  const navItems: { id: ViewMode; label: string; icon: React.ElementType; badge?: string; highlight?: boolean }[] = [
    { id: 'landing', label: t('nav.home'), icon: Home, highlight: true },
    { id: 'ai-workstation', label: t('nav.aiWorkstation'), icon: Wand2, highlight: true },
    { id: 'ai-hub', label: t('nav.aiHub'), icon: Zap, highlight: true },
    { id: 'learn', label: t('nav.learningCenter'), icon: GraduationCap, highlight: true },
    { id: 'chat', label: t('nav.aiChat'), icon: MessageSquare },
    { id: 'ide', label: t('nav.ide'), icon: Code2 },
    { id: 'supabase', label: t('nav.supabase'), icon: Database, highlight: true },
    { id: 'operator', label: t('nav.browserOperator'), icon: Cpu, highlight: true },
    { id: 'browser', label: t('nav.virtualBrowser'), icon: Globe },
    { id: 'desktop', label: t('nav.desktop'), icon: Monitor },
    { id: 'agents', label: t('nav.agents'), icon: Bot },
    { id: 'integrations', label: t('nav.integrations'), icon: Plug },
    { id: 'team', label: t('nav.team'), icon: Users, badge: '3' },
    { id: 'deploy', label: t('nav.deploy'), icon: Rocket },
  ];

  return (
    <motion.div
      initial={false}
      animate={{ width: isSidebarOpen ? 260 : 72 }}
      transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
      className="h-full flex flex-col bg-white border-r border-gray-200"
    >
      {/* Logo */}
      <div className="h-16 flex items-center justify-between px-4 border-b border-gray-200">
        {isSidebarOpen ? (
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-[#FFD700] to-[#FFA500] flex items-center justify-center">
              <Sparkles className="w-5 h-5 text-black" />
            </div>
            <div>
              <span className="font-bold text-gray-900">CodeAI</span>
              <span className="text-xs text-[#FFD700] ml-1">Pro</span>
            </div>
          </div>
        ) : (
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-[#FFD700] to-[#FFA500] flex items-center justify-center mx-auto">
            <Sparkles className="w-5 h-5 text-black" />
          </div>
        )}
        <RippleButton
          variant="ghost"
          size="icon"
          className={cn('h-7 w-7 text-gray-500', !isSidebarOpen && 'hidden')}
          onClick={toggleSidebar}
        >
          <ChevronLeft className="w-4 h-4" />
        </RippleButton>
      </div>

      {/* Navigation */}
      <div className="flex-1 py-4 px-3">
        {!isSidebarOpen && (
          <RippleButton
            variant="ghost"
            size="icon"
            className="w-full mb-4 text-gray-500"
            onClick={toggleSidebar}
          >
            <ChevronRight className="w-4 h-4" />
          </RippleButton>
        )}
        <nav className="space-y-1">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = currentView === item.id;
            const itemUnreadCount = item.id === 'team' ? unreadCount : 0;

            return (
              <button
                key={item.id}
                onClick={() => setCurrentView(item.id)}
                className={cn(
                  'w-full flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all duration-200',
                  isActive && 'bg-[#FFD700]/10 text-[#FFD700]',
                  !isActive && 'text-gray-600 hover:text-gray-900 hover:bg-gray-100',
                  item.highlight && !isActive && 'text-[#FFD700]/80',
                  !isSidebarOpen && 'justify-center px-2'
                )}
              >
                <div className="relative">
                  <Icon className={cn('w-5 h-5 shrink-0', isActive && 'text-[#FFD700]')} />
                  {itemUnreadCount > 0 && (
                    <span className="absolute -top-1 -right-1 w-4 h-4 rounded-full bg-[#FFD700] text-[10px] flex items-center justify-center text-black font-medium">
                      {itemUnreadCount}
                    </span>
                  )}
                  {item.highlight && !isActive && (
                    <span className="absolute -top-0.5 -right-0.5 w-2 h-2 rounded-full bg-[#FFD700]" />
                  )}
                </div>
                {isSidebarOpen && (
                  <span className={cn('font-medium text-sm', isActive && 'text-[#FFD700]')}>
                    {item.label}
                  </span>
                )}
              </button>
            );
          })}
        </nav>
      </div>

      {/* Footer */}
      <div className="p-3 border-t border-gray-200">
        {/* User Profile */}
        {isSidebarOpen && currentUser && (
          <div className="flex items-center gap-3 px-3 py-2 mb-2 rounded-lg cursor-pointer transition-colors hover:bg-gray-100">
            <img
              src={currentUser.avatar}
              alt={currentUser.name}
              className="w-8 h-8 rounded-full"
            />
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium truncate text-gray-900">
                {currentUser.name}
              </p>
              <p className="text-xs capitalize text-gray-500">
                {currentUser.plan} Plan
              </p>
            </div>
          </div>
        )}

        {/* Language Toggle */}
        <button
          onClick={toggleLanguage}
          className={cn(
            'w-full flex items-center gap-3 px-3 py-2 rounded-lg transition-all text-gray-600 hover:text-gray-900 hover:bg-gray-100',
            !isSidebarOpen && 'justify-center px-2'
          )}
        >
          <Languages className="w-5 h-5 shrink-0" />
          {isSidebarOpen && (
            <span className="text-sm font-medium">
              {i18n.language === 'en' ? 'English' : '繁體中文'}
            </span>
          )}
        </button>

        {/* Theme Toggle */}
        <button
          onClick={toggleDarkMode}
          className={cn(
            'w-full flex items-center gap-3 px-3 py-2 rounded-lg transition-all mt-1 text-gray-600 hover:text-gray-900 hover:bg-gray-100',
            !isSidebarOpen && 'justify-center px-2'
          )}
        >
          <Moon className="w-5 h-5 shrink-0" />
          {isSidebarOpen && (
            <span className="text-sm font-medium">
              {t('common.darkMode')}
            </span>
          )}
        </button>

        {/* Settings */}
        <button
          className={cn(
            'w-full flex items-center gap-3 px-3 py-2 rounded-lg transition-all mt-1 text-gray-600 hover:text-gray-900 hover:bg-gray-100',
            !isSidebarOpen && 'justify-center px-2'
          )}
        >
          <Settings className="w-5 h-5 shrink-0" />
          {isSidebarOpen && <span className="text-sm font-medium">{t('common.settings')}</span>}
        </button>
      </div>
    </motion.div>
  );
}
