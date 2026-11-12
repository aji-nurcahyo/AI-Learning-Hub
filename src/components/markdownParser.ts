export interface TocItem {
  id: string;
  text: string;
  level: number;
}

export type InlineToken =
  | { type: 'text'; value: string }
  | { type: 'math'; value: string }
  | { type: 'code'; value: string }
  | { type: 'link'; label: string; url: string }
  | { type: 'bold'; children: InlineToken[] }
  | { type: 'italic'; children: InlineToken[] };

export interface Block {
  type: 'heading' | 'code' | 'math' | 'table' | 'ul' | 'ol' | 'quote' | 'paragraph' | 'hr';
  content: string;
  lang?: string;
  level?: number;
  id?: string;
  items?: string[];
  headers?: string[];
  rows?: string[][];
}

// Clean raw LaTeX delimiter wrapper if present
export function cleanTex(raw: string): string {
  if (!raw) return '';
  let s = raw.trim();

  // Strip \[ ... \] or $$ ... $$
  if ((s.startsWith('$$') && s.endsWith('$$') && s.length >= 4) ||
      (s.startsWith('\\[') && s.endsWith('\\]') && s.length >= 4)) {
    s = s.slice(2, -2).trim();
  }
  // Strip \( ... \) or $ ... $
  else if (s.startsWith('\\(') && s.endsWith('\\)') && s.length >= 4) {
    s = s.slice(2, -2).trim();
  } else if (s.startsWith('$') && s.endsWith('$') && s.length >= 2) {
    s = s.slice(1, -1).trim();
  }

  return s;
}

// Tokenize inline markdown elements safely
export function parseInlineTokens(text: string): InlineToken[] {
  // Matches escaped symbols, math delimiters, inline code, links, bold, and italic
  const pattern = /(\\\$|\\\*|\\_|\\#|\\\[[\s\S]+?\\\]|\\\([\s\S]+?\\\)|\$\$[^$\n]+?\$\$|\$(?:\\\$|[^$\n])+\$|`[^`\n]+`|\[[^\]]+\]\([^)\s]+\)|\*\*\*[^*\n]+\*\*\*|___[^_\n]+___|\*\*[^*\n]+\*\*|__[^_\n]+__|\*[^*\n]+\*|_[^_\n]+_)/g;
  const parts = text.split(pattern);
  const tokens: InlineToken[] = [];

  for (const part of parts) {
    if (!part) continue;

    // Escaped literal characters
    if (part === '\\$') { tokens.push({ type: 'text', value: '$' }); continue; }
    if (part === '\\*') { tokens.push({ type: 'text', value: '*' }); continue; }
    if (part === '\\_') { tokens.push({ type: 'text', value: '_' }); continue; }
    if (part === '\\#') { tokens.push({ type: 'text', value: '#' }); continue; }

    // Display math inline: \[ ... \]
    if (part.startsWith('\\[') && part.endsWith('\\]') && part.length >= 4) {
      tokens.push({ type: 'math', value: cleanTex(part) });
      continue;
    }

    // Inline math: \( ... \)
    if (part.startsWith('\\(') && part.endsWith('\\)') && part.length >= 4) {
      tokens.push({ type: 'math', value: cleanTex(part) });
      continue;
    }

    // Math: $$...$$
    if (part.startsWith('$$') && part.endsWith('$$') && part.length >= 4) {
      tokens.push({ type: 'math', value: cleanTex(part) });
      continue;
    }

    // Inline math: $...$
    if (part.startsWith('$') && part.endsWith('$') && part.length > 2 && !part.startsWith('$$')) {
      tokens.push({ type: 'math', value: cleanTex(part) });
      continue;
    }

    // Inline code: `...`
    if (part.startsWith('`') && part.endsWith('`') && part.length > 2) {
      tokens.push({ type: 'code', value: part.slice(1, -1) });
      continue;
    }

    // Link: [text](url)
    const linkMatch = /^\[([^\]]+)\]\(([^)]+)\)$/.exec(part);
    if (linkMatch) {
      tokens.push({ type: 'link', label: linkMatch[1], url: linkMatch[2] });
      continue;
    }

    // Bold + Italic: ***...*** or ___...___
    if ((part.startsWith('***') && part.endsWith('***') && part.length > 6) ||
        (part.startsWith('___') && part.endsWith('___') && part.length > 6)) {
      tokens.push({
        type: 'bold',
        children: [{ type: 'italic', children: parseInlineTokens(part.slice(3, -3)) }]
      });
      continue;
    }

    // Bold: **...** or __...__
    if ((part.startsWith('**') && part.endsWith('**') && part.length > 4) ||
        (part.startsWith('__') && part.endsWith('__') && part.length > 4)) {
      tokens.push({ type: 'bold', children: parseInlineTokens(part.slice(2, -2)) });
      continue;
    }

    // Italic: *...* or _..._
    if ((part.startsWith('*') && part.endsWith('*') && part.length > 2) ||
        (part.startsWith('_') && part.endsWith('_') && part.length > 2)) {
      tokens.push({ type: 'italic', children: parseInlineTokens(part.slice(1, -1)) });
      continue;
    }

    tokens.push({ type: 'text', value: part });
  }

  return tokens;
}

// Parse markdown document into typed blocks, ensuring headings inside code fences are ignored
export function parseMarkdownBlocks(text: string): { blocks: Block[]; toc: TocItem[] } {
  const blocks: Block[] = [];
  const toc: TocItem[] = [];
  const lines = text.replace(/\r\n/g, '\n').split('\n');
  let i = 0;
  let headingIndex = 0;

  while (i < lines.length) {
    const line = lines[i];

    // Fenced code block: ```[lang] ... ```
    if (line.trim().startsWith('```')) {
      const lang = line.trim().slice(3).trim() || 'Python';
      const codeLines: string[] = [];
      i++;
      while (i < lines.length && !lines[i].trim().startsWith('```')) {
        codeLines.push(lines[i]);
        i++;
      }
      i++; // skip closing ```
      blocks.push({
        type: 'code',
        lang,
        content: codeLines.join('\n'),
      });
      continue;
    }

    // Math block: $$ ... $$ or \[ ... \]
    if (line.trim().startsWith('$$') || line.trim().startsWith('\\[')) {
      const isBracket = line.trim().startsWith('\\[');
      const endMarker = isBracket ? '\\]' : '$$';
      const mathLines: string[] = [];

      if (line.trim().endsWith(endMarker) && line.trim().length > 4) {
        mathLines.push(line.trim().slice(2, -2).trim());
        i++;
      } else {
        i++;
        while (i < lines.length && !lines[i].trim().startsWith(endMarker)) {
          mathLines.push(lines[i]);
          i++;
        }
        i++; // skip closing marker
      }
      blocks.push({
        type: 'math',
        content: cleanTex(mathLines.join('\n')),
      });
      continue;
    }

    // Horizontal rule: --- or *** or ___
    if (/^\s*(?:---|\*\*\*|___)\s*$/.test(line)) {
      blocks.push({
        type: 'hr',
        content: '',
      });
      i++;
      continue;
    }

    // Heading: # ... ######
    const headingMatch = /^\s*(#{1,6})\s+(.+)$/.exec(line);
    if (headingMatch) {
      const level = headingMatch[1].length;
      const title = headingMatch[2].trim();
      const id = `section-${headingIndex++}`;
      toc.push({ id, text: title, level });
      blocks.push({
        type: 'heading',
        level,
        id,
        content: title,
      });
      i++;
      continue;
    }

    // Table block: starts and ends with |
    if (line.trim().startsWith('|') && line.trim().endsWith('|')) {
      const tableLines: string[] = [];
      while (i < lines.length && lines[i].trim().startsWith('|') && lines[i].trim().endsWith('|')) {
        tableLines.push(lines[i].trim());
        i++;
      }

      if (tableLines.length >= 2) {
        const rawHeaders = tableLines[0].slice(1, -1).split('|').map(c => c.trim());
        const isSeparator = /^[\s|:-]+$/.test(tableLines[1]);
        const dataLines = isSeparator ? tableLines.slice(2) : tableLines.slice(1);
        const rows = dataLines.map(rowLine =>
          rowLine.slice(1, -1).split('|').map(c => c.trim())
        );

        blocks.push({
          type: 'table',
          content: '',
          headers: rawHeaders,
          rows,
        });
        continue;
      }
    }

    // Blockquote: > ...
    if (line.trim().startsWith('>')) {
      const quoteLines: string[] = [];
      while (i < lines.length && lines[i].trim().startsWith('>')) {
        quoteLines.push(lines[i].replace(/^>\s?/, ''));
        i++;
      }
      blocks.push({
        type: 'quote',
        content: quoteLines.join('\n'),
      });
      continue;
    }

    // Unordered list: - or * or + or •
    if (/^\s*[-*+•]\s+/.test(line)) {
      const listItems: string[] = [];
      while (i < lines.length && /^\s*[-*+•]\s+/.test(lines[i])) {
        let itemText = lines[i].replace(/^\s*[-*+•]\s+/, '').trim();
        i++;
        // Check for multi-line continuation lines (indented by 2+ spaces, not a new block)
        while (
          i < lines.length &&
          lines[i].trim() &&
          /^\s{2,}/.test(lines[i]) &&
          !/^\s*[-*+•]\s+/.test(lines[i]) &&
          !/^\s*\d+[.)]\s+/.test(lines[i]) &&
          !lines[i].trim().startsWith('```')
        ) {
          itemText += '\n' + lines[i].trim();
          i++;
        }
        listItems.push(itemText);
      }
      blocks.push({
        type: 'ul',
        content: '',
        items: listItems,
      });
      continue;
    }

    // Ordered list: 1. or 1) etc
    if (/^\s*\d+[.)]\s+/.test(line)) {
      const listItems: string[] = [];
      while (i < lines.length && /^\s*\d+[.)]\s+/.test(lines[i])) {
        let itemText = lines[i].replace(/^\s*\d+[.)]\s+/, '').trim();
        i++;
        while (
          i < lines.length &&
          lines[i].trim() &&
          /^\s{2,}/.test(lines[i]) &&
          !/^\s*\d+[.)]\s+/.test(lines[i]) &&
          !/^\s*[-*+•]\s+/.test(lines[i]) &&
          !lines[i].trim().startsWith('```')
        ) {
          itemText += '\n' + lines[i].trim();
          i++;
        }
        listItems.push(itemText);
      }
      blocks.push({
        type: 'ol',
        content: '',
        items: listItems,
      });
      continue;
    }

    // Blank line
    if (!line.trim()) {
      i++;
      continue;
    }

    // Paragraph: collect lines until another block starts
    const pLines: string[] = [];
    while (
      i < lines.length &&
      lines[i].trim() &&
      !lines[i].trim().startsWith('```') &&
      !lines[i].trim().startsWith('$$') &&
      !lines[i].trim().startsWith('\\[') &&
      !/^\s*(#{1,6})\s+/.test(lines[i]) &&
      !lines[i].trim().startsWith('>') &&
      !/^\s*[-*+•]\s+/.test(lines[i]) &&
      !/^\s*\d+[.)]\s+/.test(lines[i]) &&
      !/^\s*(?:---|\*\*\*|___)\s*$/.test(lines[i]) &&
      !(lines[i].trim().startsWith('|') && lines[i].trim().endsWith('|'))
    ) {
      pLines.push(lines[i]);
      i++;
    }
    if (pLines.length > 0) {
      blocks.push({
        type: 'paragraph',
        content: pLines.join('\n'), // Preserve line breaks for distinct line rendering
      });
    }
  }

  return { blocks, toc };
}
