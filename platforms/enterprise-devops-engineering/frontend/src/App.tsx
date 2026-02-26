import { useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Sidebar } from '@/components/layout/Sidebar';
import { ChatInterface } from '@/components/chat/ChatInterface';
import { CursorIDE } from '@/components/ide/CursorIDE';
import { DesktopInterface } from '@/components/desktop/DesktopInterface';
import { MaxAgentHub } from '@/components/agents/MaxAgentHub';
import { IntegrationHub } from '@/components/integrations/IntegrationHub';
import { TeamCenter } from '@/components/team/TeamCenter';
import { DeployCenter } from '@/components/deploy/DeployCenter';
import { MultiFunctionAI } from '@/components/ai-hub/MultiFunctionAI';
import { AIWorkstation } from '@/components/ai-hub/AIWorkstation';
import { LearningCenter } from '@/components/learn/LearningCenter';
import { VirtualBrowser } from '@/components/virtual-browser/VirtualBrowser';
import { BrowserOperator } from '@/components/browser-operator/BrowserOperator';
import { SupabaseDashboard } from '@/components/supabase/SupabaseDashboard';
import { NewLandingPage } from '@/components/landing/NewLandingPage';
import { useAppStore } from '@/store/appStore';
import './i18n';
import './App.css';

// View mode type including new views
export type ViewMode = 
  | 'chat' 
  | 'ide' 
  | 'desktop' 
  | 'agents' 
  | 'integrations' 
  | 'team' 
  | 'deploy'
  | 'ai-hub'
  | 'browser'
  | 'operator'
  | 'supabase'
  | 'ai-workstation'
  | 'learn'
  | 'landing';

function App() {
  const { currentView, isDarkMode } = useAppStore();

  // Apply dark mode
  useEffect(() => {
    if (isDarkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [isDarkMode]);

  const renderContent = () => {
    switch (currentView) {
      case 'chat':
        return <ChatInterface />;
      case 'ide':
        return <CursorIDE />;
      case 'desktop':
        return <DesktopInterface />;
      case 'agents':
        return <MaxAgentHub />;
      case 'integrations':
        return <IntegrationHub />;
      case 'team':
        return <TeamCenter />;
      case 'deploy':
        return <DeployCenter />;
      case 'ai-hub':
        return <MultiFunctionAI />;
      case 'browser':
        return (
          <div className="h-full p-4">
            <VirtualBrowser automationMode={true} />
          </div>
        );
      case 'operator':
        return <BrowserOperator />;
      case 'supabase':
        return <SupabaseDashboard />;
      case 'ai-workstation':
        return <AIWorkstation />;
      case 'learn':
        return <LearningCenter />;
      case 'landing':
        return <NewLandingPage />;
      default:
        return <ChatInterface />;
    }
  };

  return (
    <div className="h-screen w-screen flex overflow-hidden bg-gray-50 text-gray-900">
      <Sidebar />
      <main className="flex-1 overflow-hidden bg-gray-50">
        <AnimatePresence mode="wait">
          <motion.div
            key={currentView}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
            className="h-full"
          >
            {renderContent()}
          </motion.div>
        </AnimatePresence>
      </main>
    </div>
  );
}

export default App;
