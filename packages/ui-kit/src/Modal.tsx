/**
 * Modal component — accessible overlay dialog.
 * URI: indestructibleeco://packages/ui-kit/Modal
 */
import React, { useEffect, useRef, useCallback } from 'react';

export interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  title?: string;
  size?: 'sm' | 'md' | 'lg';
  children: React.ReactNode;
  closeOnOverlay?: boolean;
  closeOnEsc?: boolean;
}

export const Modal: React.FC<ModalProps> = ({
  isOpen, onClose, title, size = 'md', children,
  closeOnOverlay = true, closeOnEsc = true,
}) => {
  const overlayRef = useRef<HTMLDivElement>(null);

  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if (closeOnEsc && e.key === 'Escape') onClose();
  }, [closeOnEsc, onClose]);

  useEffect(() => {
    if (isOpen) {
      document.addEventListener('keydown', handleKeyDown);
      document.body.style.overflow = 'hidden';
    }
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      document.body.style.overflow = '';
    };
  }, [isOpen, handleKeyDown]);

  if (!isOpen) return null;

  const widthMap = { sm: '400px', md: '600px', lg: '800px' };

  return (
    <div
      ref={overlayRef}
      onClick={(e) => { if (closeOnOverlay && e.target === overlayRef.current) onClose(); }}
      style={{
        position: 'fixed', inset: 0, zIndex: 1000,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        backgroundColor: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(4px)',
      }}
      role="dialog"
      aria-modal="true"
      aria-label={title || 'Dialog'}
    >
      <div style={{
        background: 'var(--bg-secondary, #1e1e2e)', borderRadius: '12px',
        width: widthMap[size], maxWidth: '90vw', maxHeight: '85vh',
        overflow: 'auto', padding: '24px', position: 'relative',
        border: '1px solid var(--border-color, #333)',
      }}>
        {title && (
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '16px' }}>
            <h2 style={{ margin: 0, fontSize: '1.25rem' }}>{title}</h2>
            <button onClick={onClose} style={{
              background: 'none', border: 'none', color: 'var(--text-secondary, #888)',
              cursor: 'pointer', fontSize: '1.5rem', lineHeight: 1,
            }}>&times;</button>
          </div>
        )}
        {children}
      </div>
    </div>
  );
};

export default Modal;
