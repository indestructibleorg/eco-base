/**
 * Table component — sortable, paginated data table.
 * URI: indestructibleeco://packages/ui-kit/Table
 */
import React, { useState, useMemo } from 'react';

export interface Column<T> {
  key: string;
  header: string;
  sortable?: boolean;
  render?: (row: T) => React.ReactNode;
  width?: string;
}

export interface TableProps<T> {
  columns: Column<T>[];
  data: T[];
  pageSize?: number;
  onRowClick?: (row: T) => void;
  emptyMessage?: string;
}

export function Table<T extends Record<string, any>>({
  columns, data, pageSize = 10, onRowClick, emptyMessage = 'No data',
}: TableProps<T>) {
  const [sortKey, setSortKey] = useState<string>('');
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('asc');
  const [page, setPage] = useState(0);

  const sorted = useMemo(() => {
    if (!sortKey) return data;
    return [...data].sort((a, b) => {
      const av = a[sortKey], bv = b[sortKey];
      const cmp = av < bv ? -1 : av > bv ? 1 : 0;
      return sortDir === 'asc' ? cmp : -cmp;
    });
  }, [data, sortKey, sortDir]);

  const totalPages = Math.ceil(sorted.length / pageSize);
  const paged = sorted.slice(page * pageSize, (page + 1) * pageSize);

  const handleSort = (key: string) => {
    if (sortKey === key) setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    else { setSortKey(key); setSortDir('asc'); }
  };

  return (
    <div>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.875rem' }}>
        <thead>
          <tr>
            {columns.map(col => (
              <th
                key={col.key}
                onClick={() => col.sortable && handleSort(col.key)}
                style={{
                  padding: '10px 12px', textAlign: 'left', borderBottom: '2px solid var(--border-color, #333)',
                  cursor: col.sortable ? 'pointer' : 'default', userSelect: 'none',
                  width: col.width, color: 'var(--text-secondary, #888)',
                }}
              >
                {col.header}
                {col.sortable && sortKey === col.key && (sortDir === 'asc' ? ' ↑' : ' ↓')}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {paged.length === 0 ? (
            <tr><td colSpan={columns.length} style={{ padding: '20px', textAlign: 'center', color: 'var(--text-secondary, #888)' }}>{emptyMessage}</td></tr>
          ) : paged.map((row, i) => (
            <tr
              key={i}
              onClick={() => onRowClick?.(row)}
              style={{ cursor: onRowClick ? 'pointer' : 'default', borderBottom: '1px solid var(--border-color, #222)' }}
            >
              {columns.map(col => (
                <td key={col.key} style={{ padding: '10px 12px' }}>
                  {col.render ? col.render(row) : String(row[col.key] ?? '')}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {totalPages > 1 && (
        <div style={{ display: 'flex', justifyContent: 'center', gap: '8px', marginTop: '12px' }}>
          <button disabled={page === 0} onClick={() => setPage(p => p - 1)} style={{ padding: '4px 12px' }}>Prev</button>
          <span style={{ padding: '4px 8px', color: 'var(--text-secondary, #888)' }}>{page + 1} / {totalPages}</span>
          <button disabled={page >= totalPages - 1} onClick={() => setPage(p => p + 1)} style={{ padding: '4px 12px' }}>Next</button>
        </div>
      )}
    </div>
  );
}

export default Table;
