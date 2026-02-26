import { useState } from 'react';
import {
  Folder,
  File,
  ChevronRight,
  ChevronDown,
  Plus,
  Play,
  Trash2,
  Edit3,
  X,
  Terminal,
  Layout,
  Code2,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuTrigger,
} from '@/components/ui/context-menu';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';

import { useIDEStore } from '@/store/appStore';
import { cn } from '@/lib/utils';
import type { FileNode } from '@/types';
import Editor from '@monaco-editor/react';

export function IDEInterface() {
  const {
    files,
    openTabs,
    activeTabId,
    terminalLines,
    openFile,
    closeTab,
    setActiveTab,
    updateTabContent,
    createFile,
    createFolder,
    deleteFile,
    renameFile,
    addTerminalLine,
    clearTerminal,
  } = useIDEStore();

  const [newFileDialog, setNewFileDialog] = useState<{ open: boolean; parentId: string | null; type: 'file' | 'folder' }>({
    open: false,
    parentId: null,
    type: 'file',
  });
  const [newFileName, setNewFileName] = useState('');
  const [renamingFile, setRenamingFile] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState('');
  const [showPreview, setShowPreview] = useState(true);

  const activeTab = openTabs.find((t) => t.id === activeTabId);

  const handleCreateFile = () => {
    if (!newFileName.trim()) return;
    if (newFileDialog.type === 'file') {
      createFile(newFileName, newFileDialog.parentId);
    } else {
      createFolder(newFileName, newFileDialog.parentId);
    }
    setNewFileDialog({ open: false, parentId: null, type: 'file' });
    setNewFileName('');
  };

  const handleRename = (fileId: string) => {
    if (!renameValue.trim()) {
      setRenamingFile(null);
      return;
    }
    renameFile(fileId, renameValue);
    setRenamingFile(null);
  };

  const handleRun = () => {
    if (!activeTab) return;
    
    addTerminalLine({
      id: `term-${Date.now()}`,
      type: 'input',
      content: `$ python ${activeTab.name}`,
      timestamp: new Date(),
    });

    setTimeout(() => {
      addTerminalLine({
        id: `term-${Date.now() + 1}`,
        type: 'output',
        content: '55',
        timestamp: new Date(),
      });
    }, 500);
  };

  const renderFileTree = (nodes: FileNode[], level = 0) => {
    return nodes.map((node) => (
      <div key={node.id}>
        <ContextMenu>
          <ContextMenuTrigger>
            <div
              className={cn(
                'flex items-center gap-1 px-2 py-1 cursor-pointer hover:bg-[#2B3245] rounded-sm text-sm transition-colors',
                activeTab?.fileId === node.id && 'bg-[#F26207]/10 text-[#F26207]'
              )}
              style={{ paddingLeft: `${level * 12 + 8}px` }}
              onClick={() => {
                if (node.type === 'folder') {
                  // Toggle folder
                } else {
                  openFile(node);
                }
              }}
            >
              {node.type === 'folder' && (
                <span className="w-4 h-4 flex items-center justify-center text-[#5F6775]">
                  {node.isOpen ? (
                    <ChevronDown className="w-3 h-3" />
                  ) : (
                    <ChevronRight className="w-3 h-3" />
                  )}
                </span>
              )}
              {node.type === 'folder' ? (
                <Folder className="w-4 h-4 text-[#F59E0B]" />
              ) : (
                <File className="w-4 h-4 text-[#3B82F6]" />
              )}
              
              {renamingFile === node.id ? (
                <Input
                  value={renameValue}
                  onChange={(e) => setRenameValue(e.target.value)}
                  onBlur={() => handleRename(node.id)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') handleRename(node.id);
                    if (e.key === 'Escape') setRenamingFile(null);
                  }}
                  className="h-6 py-0 px-1 text-sm bg-[#1C2333] border-[#2B3245] text-white"
                  autoFocus
                  onClick={(e) => e.stopPropagation()}
                />
              ) : (
                <span className="truncate text-[#9AA0A6]">{node.name}</span>
              )}
            </div>
          </ContextMenuTrigger>
          <ContextMenuContent className="bg-[#1C2333] border-[#2B3245]">
            {node.type === 'folder' && (
              <>
                <ContextMenuItem
                  onClick={() => setNewFileDialog({ open: true, parentId: node.id, type: 'file' })}
                  className="text-[#9AA0A6] focus:text-white focus:bg-[#2B3245]"
                >
                  <Plus className="w-4 h-4 mr-2" />
                  New File
                </ContextMenuItem>
                <ContextMenuItem
                  onClick={() => setNewFileDialog({ open: true, parentId: node.id, type: 'folder' })}
                  className="text-[#9AA0A6] focus:text-white focus:bg-[#2B3245]"
                >
                  <Folder className="w-4 h-4 mr-2" />
                  New Folder
                </ContextMenuItem>
              </>
            )}
            <ContextMenuItem
              onClick={() => {
                setRenamingFile(node.id);
                setRenameValue(node.name);
              }}
              className="text-[#9AA0A6] focus:text-white focus:bg-[#2B3245]"
            >
              <Edit3 className="w-4 h-4 mr-2" />
              Rename
            </ContextMenuItem>
            <ContextMenuItem
              onClick={() => deleteFile(node.id)}
              className="text-red-400 focus:text-red-400 focus:bg-red-500/10"
            >
              <Trash2 className="w-4 h-4 mr-2" />
              Delete
            </ContextMenuItem>
          </ContextMenuContent>
        </ContextMenu>
        
        {node.type === 'folder' && node.isOpen && node.children && (
          <div>{renderFileTree(node.children, level + 1)}</div>
        )}
      </div>
    ));
  };

  return (
    <div className="flex h-full">
      {/* File Tree */}
      <div className="w-64 border-r border-[#2B3245] bg-[#0E1525] flex flex-col">
        <div className="h-10 flex items-center justify-between px-3 border-b border-[#2B3245]">
          <span className="text-sm font-medium text-white">Explorer</span>
          <div className="flex gap-1">
            <Button
              variant="ghost"
              size="icon"
              className="h-6 w-6 text-[#5F6775] hover:text-white hover:bg-[#2B3245]"
              onClick={() => setNewFileDialog({ open: true, parentId: null, type: 'file' })}
            >
              <Plus className="w-3 h-3" />
            </Button>
          </div>
        </div>
        <ScrollArea className="flex-1 py-2">
          {renderFileTree(files)}
        </ScrollArea>
      </div>

      {/* Editor Area */}
      <div className="flex-1 flex flex-col min-w-0 bg-[#0E1525]">
        {/* Tabs */}
        {openTabs.length > 0 && (
          <div className="h-10 bg-[#0E1525] flex items-center border-b border-[#2B3245] overflow-x-auto">
            {openTabs.map((tab) => (
              <div
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={cn(
                  'flex items-center gap-2 px-3 py-2 text-sm cursor-pointer border-r border-[#2B3245] min-w-[120px] max-w-[200px] transition-all',
                  activeTabId === tab.id
                    ? 'bg-[#1C2333] border-t-2 border-t-[#F26207] text-white'
                    : 'text-[#5F6775] hover:bg-[#1C2333] hover:text-[#9AA0A6]'
                )}
              >
                <Code2 className="w-4 h-4" />
                <span className="flex-1 truncate">{tab.name}</span>
                {tab.isModified && <span className="text-[#F26207]">•</span>}
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-4 w-4 opacity-0 group-hover:opacity-100 hover:opacity-100 text-[#5F6775] hover:text-white"
                  onClick={(e) => {
                    e.stopPropagation();
                    closeTab(tab.id);
                  }}
                >
                  <X className="w-3 h-3" />
                </Button>
              </div>
            ))}
          </div>
        )}

        {/* Toolbar */}
        <div className="h-10 flex items-center justify-between px-4 border-b border-[#2B3245] bg-[#0E1525]">
          <div className="flex items-center gap-4">
            {activeTab && (
              <>
                <span className="text-sm text-[#5F6775]">
                  {activeTab.language?.toUpperCase()}
                </span>
                {activeTab.isModified && (
                  <span className="text-xs text-[#F26207]">Unsaved</span>
                )}
              </>
            )}
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowPreview(!showPreview)}
              className="text-[#5F6775] hover:text-white hover:bg-[#2B3245]"
            >
              <Layout className="w-4 h-4 mr-1" />
              {showPreview ? 'Hide Preview' : 'Show Preview'}
            </Button>
            <Button
              size="sm"
              onClick={handleRun}
              disabled={!activeTab}
              className="bg-[#F26207] hover:bg-[#D95706] text-white"
            >
              <Play className="w-4 h-4 mr-1" />
              Run
            </Button>
          </div>
        </div>

        {/* Editor + Preview */}
        <div className="flex-1 flex overflow-hidden">
          <div className={cn('flex-1', showPreview && 'w-1/2')}>
            {activeTab ? (
              <Editor
                height="100%"
                language={activeTab.language}
                value={activeTab.content}
                onChange={(value) => {
                  if (value !== undefined) {
                    updateTabContent(activeTab.id, value);
                  }
                }}
                theme="vs-dark"
                options={{
                  minimap: { enabled: false },
                  fontSize: 14,
                  lineNumbers: 'on',
                  roundedSelection: false,
                  scrollBeyondLastLine: false,
                  readOnly: false,
                  automaticLayout: true,
                  fontFamily: 'JetBrains Mono, Fira Code, monospace',
                }}
              />
            ) : (
              <div className="flex flex-col items-center justify-center h-full text-[#5F6775]">
                <Code2 className="w-12 h-12 mb-4 opacity-50" />
                <p>Select a file to start editing</p>
              </div>
            )}
          </div>

          {showPreview && (
            <div className="w-1/2 border-l border-[#2B3245] bg-[#0E1525] p-4">
              <div className="text-sm font-medium text-white mb-2">Preview</div>
              <div className="bg-[#1C2333] rounded-lg p-4 h-[calc(100%-2rem)] overflow-auto border border-[#2B3245]">
                {activeTab?.language === 'markdown' ? (
                  <div className="prose prose-sm max-w-none prose-invert">
                    {activeTab.content}
                  </div>
                ) : activeTab?.language === 'python' ? (
                  <div>
                    <div className="text-sm text-[#5F6775] mb-2">Python Output:</div>
                    <pre className="bg-[#0E1525] text-[#10B981] p-4 rounded-lg text-sm border border-[#2B3245]">
                      55
                    </pre>
                  </div>
                ) : (
                  <div className="text-[#5F6775]">
                    Preview not available
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Terminal */}
        <div className="h-48 border-t border-[#2B3245] bg-[#0E1525] flex flex-col">
          <div className="h-8 flex items-center justify-between px-4 bg-[#1C2333] border-b border-[#2B3245]">
            <div className="flex items-center gap-2">
              <Terminal className="w-4 h-4 text-[#5F6775]" />
              <span className="text-sm text-[#9AA0A6]">Terminal</span>
            </div>
            <Button
              variant="ghost"
              size="sm"
              className="h-6 text-[#5F6775] hover:text-white hover:bg-[#2B3245]"
              onClick={clearTerminal}
            >
              Clear
            </Button>
          </div>
          <ScrollArea className="flex-1 p-4 font-mono text-sm">
            {terminalLines.map((line) => (
              <div
                key={line.id}
                className={cn(
                  'mb-1',
                  line.type === 'input' && 'text-[#9AA0A6]',
                  line.type === 'output' && 'text-[#10B981]',
                  line.type === 'error' && 'text-[#EF4444]',
                  line.type === 'info' && 'text-[#3B82F6]'
                )}
              >
                {line.type === 'input' && <span className="text-[#5F6775]">$ </span>}
                {line.content}
              </div>
            ))}
          </ScrollArea>
        </div>
      </div>

      {/* New File Dialog */}
      <Dialog open={newFileDialog.open} onOpenChange={(open) => setNewFileDialog({ ...newFileDialog, open })}>
        <DialogContent className="bg-[#1C2333] border-[#2B3245]">
          <DialogHeader>
            <DialogTitle className="text-white">
              New {newFileDialog.type === 'file' ? 'File' : 'Folder'}
            </DialogTitle>
          </DialogHeader>
          <Input
            value={newFileName}
            onChange={(e) => setNewFileName(e.target.value)}
            placeholder={newFileDialog.type === 'file' ? 'filename.py' : 'folder name'}
            onKeyDown={(e) => {
              if (e.key === 'Enter') handleCreateFile();
            }}
            className="bg-[#0E1525] border-[#2B3245] text-white placeholder:text-[#5F6775]"
          />
          <DialogFooter>
            <Button 
              variant="outline" 
              onClick={() => setNewFileDialog({ open: false, parentId: null, type: 'file' })}
              className="border-[#2B3245] text-[#9AA0A6] hover:bg-[#2B3245] hover:text-white"
            >
              Cancel
            </Button>
            <Button 
              onClick={handleCreateFile}
              className="bg-[#F26207] hover:bg-[#D95706] text-white"
            >
              Create
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
