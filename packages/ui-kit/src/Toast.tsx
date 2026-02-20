/**
 * Toast notification component — auto-dismiss alerts.
 * URI: indestructibleeco://packages/ui-kit/Toast
 */
import React, { useState, useEffect, useCallback, createContext, useContext } from 'react';

export type ToastType = 'success' | 'error' | 'warning' | 'info';

export interface ToastMessage {
  id: string;
  type: ToastType;
  message: string;
  duration?: number;
}

interface ToastContextType {
  addToast: (type: ToastType, message: string, duration?: number) => void;
  removeToast: (id: string) => void;
}

const ToastContext = createContext<ToastContextType | null>(null);

export const useToast = () => {
  const ctx = useContext(ToastContext);
  if (!ctx) throw new Error('useToast must be used within ToastProvider');
  return ctx;
};

const colorMap: Record<ToastType, string> = {
  success: '#22c55e', error: '#ef4444', warning: '#f59e0b', info: '#6366f1',
};

const ToastItem: React.FC<{ toast: ToastMessage; onRemove: (id: string) => void }> = ({ toast, onRemove }) => {
  useEffect(() => {
    const timer = setTimeout(() => onRemove(toast.id), toast.duration || 4000);
    return () => clearTimeout(timer);
  }, [toast, onRemove]);

  return (
    <div style={{
      padding: '12px 16px', borderRadius: '8px', marginBottom: '8px',
      background: 'var(--bg-secondary, #1e1e2e)',
      borderLeft: `4px solid ${colorMap[toast.type]}`,
      color: 'var(--text-primary, #fff)', fontSize: '0.875rem',
      display: 'flex', justifyContent: 'space-between', alignItems: 'center',
      boxShadow: '0 4px 12px rgba(0,0,0,0.3)', minWidth: '300px',
    }}>
      <span>{toast.message}</span>
      <button onClick={() => onRemove(toast.id)} style={{
        background: 'none', border: 'none', color: 'var(--text-secondary, #888)',
        cursor: 'pointer', marginLeft: '12px', fontSize: '1.2rem',
      }}>&times;</button>
    </div>
  );
};

export const ToastProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [toasts, setToasts] = useState<ToastMessage[]>([]);

  const addToast = useCallback((type: ToastType, message: string, duration?: number) => {
    const id = `toast-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    setToasts(prev => [...prev, { id, type, message, duration }]);
  }, []);

  const removeToast = useCallback((id: string) => {
    setToasts(prev => prev.filter(t => t.id !== id));
  }, []);

  return (
    <ToastContext.Provider value={{ addToast, removeToast }}>
      {children}
      <div style={{
        position: 'fixed', top: '16px', right: '16px', zIndex: 2000,
      }}>
        {toasts.map(t => <ToastItem key={t.id} toast={t} onRemove={removeToast} />)}
      </div>
    </ToastContext.Provider>
  );
};

export default ToastProvider;
