import { useLayoutEffect, useState } from 'react';
import { Icon } from '../ui/Icon';

export type Theme = 'auto' | 'light' | 'dark';
export type Ratio = 'auto' | '16x9' | '4x3';

function read(key: string, choices: string[]): string {
  if (typeof window === 'undefined') return 'auto';
  const query = new URLSearchParams(window.location.search).get(key);
  let stored: string | null = null;
  try {
    stored = localStorage.getItem(`ai-hub.${key}`);
  } catch {
    /* Browser storage is optional. */
  }
  return choices.find((v) => v === query) ?? choices.find((v) => v === stored) ?? (key === 'ratio' ? '16x9' : 'auto');
}

export function usePreferences() {
  const [theme, setTheme] = useState<Theme>(() => {
    const t = read('theme', ['auto', 'light', 'dark']) as Theme;
    if (typeof document !== 'undefined') {
      document.documentElement.dataset.theme = t;
    }
    return t;
  });

  const [ratio, setRatio] = useState<Ratio>(() => {
    return read('ratio', ['auto', '16x9', '4x3']) as Ratio;
  });

  useLayoutEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem('ai-hub.theme', theme);
      localStorage.setItem('ai-hub.ratio', ratio);
    } catch {
      /* Keep in-memory preferences. */
    }
  }, [theme, ratio]);

  return { theme, setTheme, ratio, setRatio };
}

export function ThemeSelect({
  theme,
  onChange,
}: {
  theme: Theme;
  onChange: (t: Theme) => void;
}) {
  const icon = theme === 'dark' ? 'dark_mode' : theme === 'light' ? 'light_mode' : 'contrast';

  return (
    <label className="select-label" title={`Tema aktif: ${theme === 'dark' ? 'Gelap' : theme === 'light' ? 'Terang' : 'Otomatis'}`}>
      <Icon name={icon} size={16} />
      <span>Tema</span>
      <select
        aria-label="Pilih tema tampilan"
        value={theme}
        onChange={(e) => onChange(e.target.value as Theme)}
      >
        <option value="auto">Otomatis</option>
        <option value="light">Terang</option>
        <option value="dark">Gelap</option>
      </select>
    </label>
  );
}
