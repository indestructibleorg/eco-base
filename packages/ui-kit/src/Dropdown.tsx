/**
 * Dropdown component — accessible select menu.
 * URI: indestructibleeco://packages/ui-kit/Dropdown
 */
import React, { useState, useRef, useEffect, useCallback } from 'react';

export interface DropdownOption {
  value: string;
  label: string;
  disabled?: boolean;
}

export interface DropdownProps {
  options: DropdownOption[];
  value?: string;
  onChange: (value: string) => void;
  placeholder?: string;
  disabled?: boolean;
  label?: string;
}

export const Dropdown: React.FC<DropdownProps> = ({
  options, value, onChange, placeholder = 'Select...', disabled = false, label,
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  const handleClickOutside = useCallback((e: MouseEvent) => {
    if (ref.current && !ref.current.contains(e.target as Node)) setIsOpen(false);
  }, []);

  useEffect(() => {
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [handleClickOutside]);

  const selected = options.find(o => o.value === value);

  return (
    <div ref={ref} style={{ position: 'relative', display: 'inline-block', minWidth: '200px' }}>
      {label && <label style={{ display: 'block', marginBottom: '4px', fontSize: '0.875rem', color: 'var(--text-secondary, #888)' }}>{label}</label>}
      <button
        onClick={() => !disabled && setIsOpen(!isOpen)}
        disabled={disabled}
        style={{
          width: '100%', padding: '8px 12px', textAlign: 'left',
          background: 'var(--bg-tertiary, #2a2a3e)', border: '1px solid var(--border-color, #333)',
          borderRadius: '8px', color: selected ? 'var(--text-primary, #fff)' : 'var(--text-secondary, #888)',
          cursor: disabled ? 'not-allowed' : 'pointer', fontSize: '0.875rem',
        }}
        aria-haspopup="listbox"
        aria-expanded={isOpen}
      >
        {selected?.label || placeholder}
        <span style={{ float: 'right' }}>{isOpen ? '▲' : '▼'}</span>
      </button>
      {isOpen && (
        <ul role="listbox" style={{
          position: 'absolute', top: '100%', left: 0, right: 0, zIndex: 100,
          margin: '4px 0 0', padding: 0, listStyle: 'none',
          background: 'var(--bg-secondary, #1e1e2e)', border: '1px solid var(--border-color, #333)',
          borderRadius: '8px', maxHeight: '200px', overflow: 'auto',
        }}>
          {options.map(opt => (
            <li
              key={opt.value}
              role="option"
              aria-selected={opt.value === value}
              onClick={() => { if (!opt.disabled) { onChange(opt.value); setIsOpen(false); } }}
              style={{
                padding: '8px 12px', cursor: opt.disabled ? 'not-allowed' : 'pointer',
                opacity: opt.disabled ? 0.5 : 1,
                background: opt.value === value ? 'var(--accent-color, #6366f1)' : 'transparent',
                color: opt.value === value ? '#fff' : 'var(--text-primary, #fff)',
              }}
            >
              {opt.label}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

export default Dropdown;
