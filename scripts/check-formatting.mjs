import { readFileSync, existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { walk, root } from './validate.mjs';
import { parseMarkdownBlocks, parseInlineTokens } from '../src/components/markdownParser.ts';

const errors = [];
const warnings = [];

// Audit all published chapters
const metadataFiles = walk(resolve(root, 'content')).filter(f => !f.replaceAll('\\', '/').includes('/tracks/') && f.endsWith('metadata.json'));

for (const metaFile of metadataFiles) {
  const dir = dirname(metaFile);
  const meta = JSON.parse(readFileSync(metaFile, 'utf8'));

  if (meta.status !== 'published') continue;

  const chId = meta.id;

  // 1. Audit reading module (index.md)
  const readPath = meta.resources?.reading?.path;
  if (readPath && existsSync(resolve(dir, readPath))) {
    const rawMd = readFileSync(resolve(dir, readPath), 'utf8');
    const { blocks, toc } = parseMarkdownBlocks(rawMd);

    // Check code fence balance
    const fenceCount = (rawMd.match(/```/g) || []).length;
    if (fenceCount % 2 !== 0) {
      errors.push(`Ch ${chId} [Reading]: Unbalanced code fences (count=${fenceCount})`);
    }

    // Check for unrendered raw heading indicators inside paragraph blocks
    for (const b of blocks) {
      if (b.type === 'paragraph') {
        const lines = b.content.split('\n');
        for (const l of lines) {
          if (/^#{1,6}\s+/.test(l.trim())) {
            warnings.push(`Ch ${chId} [Reading]: Potential unrendered heading inside paragraph: "${l.slice(0, 40)}..."`);
          }
        }
      }
    }

    // Check ToC validity
    if (toc.length < 3) {
      warnings.push(`Ch ${chId} [Reading]: ToC too short (length=${toc.length})`);
    }
  }

  // 2. Audit slide deck (deck.json)
  const slidePath = meta.resources?.slides?.path;
  if (slidePath && existsSync(resolve(dir, slidePath))) {
    const rawDeck = readFileSync(resolve(dir, slidePath), 'utf8');
    try {
      const deck = JSON.parse(rawDeck);
      deck.forEach((slide, sIdx) => {
        if (!slide.title || slide.title.trim() === '') {
          errors.push(`Ch ${chId} Slide ${sIdx}: Missing title`);
        }
        if (slide.body && slide.body.length > 1000) {
          errors.push(`Ch ${chId} Slide ${sIdx} (${slide.title}): Body exceeds 1000 chars (len=${slide.body.length})`);
        }
        if (slide.code && slide.code.split('\n').length > 18) {
          errors.push(`Ch ${chId} Slide ${sIdx} (${slide.title}): Code exceeds 18 lines (lines=${slide.code.split('\n').length})`);
        }
        if (!slide.notes || slide.notes.trim() === '') {
          warnings.push(`Ch ${chId} Slide ${sIdx} (${slide.title}): Missing speaker notes`);
        }
      });
    } catch {
      errors.push(`Ch ${chId} [Slides]: Malformed deck.json`);
    }
  }
}

if (warnings.length > 0) {
  console.log(`FORMAT AUDIT WARNINGS (${warnings.length}):\n${warnings.join('\n')}\n`);
}

if (errors.length > 0) {
  console.error(`FORMAT AUDIT FAILED (${errors.length} errors):\n${errors.join('\n')}`);
  process.exitCode = 1;
} else {
  console.log(`PASS: All published chapters verified for balanced fences, clean headings, valid slides, and proper formatting.`);
}

