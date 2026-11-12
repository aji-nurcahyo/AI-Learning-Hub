import katex from 'katex';

export function cleanTex(raw: string): string {
  if (!raw) return '';
  let s = raw.trim();

  // Strip block math delimiters: \[ ... \] or $$ ... $$
  if ((s.startsWith('$$') && s.endsWith('$$') && s.length >= 4) ||
      (s.startsWith('\\[') && s.endsWith('\\]') && s.length >= 4)) {
    s = s.slice(2, -2).trim();
  }
  // Strip inline math delimiters: \( ... \) or $ ... $
  else if (s.startsWith('\\(') && s.endsWith('\\)') && s.length >= 4) {
    s = s.slice(2, -2).trim();
  } else if (s.startsWith('$') && s.endsWith('$') && s.length >= 2) {
    s = s.slice(1, -1).trim();
  }

  return s;
}

const render = (tex: string, displayMode: boolean): string => {
  const cleaned = cleanTex(tex);
  try {
    return katex.renderToString(cleaned, {
      displayMode,
      throwOnError: false,
      output: 'htmlAndMathml',
      strict: false,
    });
  } catch {
    return `<span class="katex-fallback">${cleaned}</span>`;
  }
};

export function M({ tex }: { tex: string }) {
  return <span className="inline-math" dangerouslySetInnerHTML={{ __html: render(tex, false) }} />;
}

export function MathBlock({ tex }: { tex: string }) {
  return <div className="math-block" dangerouslySetInnerHTML={{ __html: render(tex, true) }} />;
}
