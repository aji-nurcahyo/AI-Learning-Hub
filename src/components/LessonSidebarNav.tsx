import { useId, useState } from 'react';
import { Icon } from '../ui/Icon';

export interface LessonNavItem {
  id: string;
  number: string;
  title: string;
  group?: string;
}

interface LessonGroup {
  label: string;
  items: LessonNavItem[];
}

function groupByConfigOrder(items: LessonNavItem[], fallbackGroup: string): LessonGroup[] {
  const buckets = new Map<string, LessonNavItem[]>();
  for (const item of items) {
    const label = item.group?.trim() || fallbackGroup;
    const list = buckets.get(label);
    if (list) list.push(item);
    else buckets.set(label, [item]);
  }
  return Array.from(buckets, ([label, groupItems]) => ({ label, items: groupItems }));
}

/**
 * Reusable data-driven lesson navigation: grouped vertical index on desktop
 * and an expandable selector panel on tablet/mobile viewports.
 */
export function LessonSidebarNav({
  items,
  activeId,
  hrefFor,
  navLabel,
  fallbackGroup = 'Modul',
}: {
  items: LessonNavItem[];
  activeId: string;
  hrefFor: (item: LessonNavItem) => string;
  navLabel: string;
  fallbackGroup?: string;
}) {
  const [expanded, setExpanded] = useState(false);
  const panelId = useId();
  const activeItem = items.find(m => m.id === activeId) || items[0];
  const groups = groupByConfigOrder(items, fallbackGroup);

  return (
    <nav
      className="lesson-nav"
      aria-label={navLabel}
      onKeyDown={e => {
        if (e.key === 'Escape' && expanded) setExpanded(false);
      }}
    >
      <button
        type="button"
        className="lesson-nav-toggle"
        aria-expanded={expanded}
        aria-controls={panelId}
        onClick={() => setExpanded(prev => !prev)}
      >
        <span className="lesson-nav-toggle-main">
          {activeItem && <span className="lesson-nav-code">{activeItem.number}</span>}
          <span className="lesson-nav-toggle-title">{activeItem ? activeItem.title : navLabel}</span>
        </span>
        <span className="lesson-nav-toggle-meta">
          <span>Daftar ({items.length})</span>
          <Icon name={expanded ? 'expand_less' : 'expand_more'} size={18} />
        </span>
      </button>

      <div id={panelId} className={`lesson-nav-panel ${expanded ? 'is-open' : ''}`}>
        {groups.map(group => (
          <div className="lesson-nav-group" key={group.label}>
            <div className="lesson-nav-group-label">{group.label}</div>
            <ul className="lesson-nav-list">
              {group.items.map(item => {
                const isActive = item.id === activeId;
                return (
                  <li key={item.id}>
                    <a
                      href={hrefFor(item)}
                      className={`lesson-nav-link ${isActive ? 'active' : ''}`}
                      aria-current={isActive ? 'page' : undefined}
                      onClick={() => setExpanded(false)}
                    >
                      <span className="lesson-nav-code">{item.number}</span>
                      <span className="lesson-nav-title">{item.title}</span>
                    </a>
                  </li>
                );
              })}
            </ul>
          </div>
        ))}
      </div>
    </nav>
  );
}

