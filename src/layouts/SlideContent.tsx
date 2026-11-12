import { type ReactNode } from 'react';
import { Code } from '../ui/Code';
import { MathBlock } from '../ui/Math';
import { renderInline } from '../components/Reading';
import type { SlideData } from '../decks/registry';

interface ColumnSection {
  title: string;
  lead?: string;
  items: string[];
}

// Parse body that contains multiple `###` subheadings into semantic column cards
function parseSplitSections(bodyText: string): ColumnSection[] | null {
  const lines = bodyText.replace(/\r\n/g, '\n').split('\n');
  const sections: ColumnSection[] = [];
  let currentSec: ColumnSection | null = null;

  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line) continue;

    const headingMatch = /^###\s+(.+)$/.exec(line);
    if (headingMatch) {
      if (currentSec) {
        sections.push(currentSec);
      }
      currentSec = { title: headingMatch[1].trim(), items: [] };
      continue;
    }

    if (currentSec) {
      if (/^[•\-*+]\s+/.test(line)) {
        currentSec.items.push(line.replace(/^[•\-*+]\s+/, '').trim());
      } else if (/^\d+[.)]\s+/.test(line)) {
        currentSec.items.push(line.replace(/^\d+[.)]\s+/, '').trim());
      } else {
        if (!currentSec.lead) {
          currentSec.lead = line;
        } else {
          currentSec.lead += ' ' + line;
        }
      }
    }
  }

  if (currentSec) {
    sections.push(currentSec);
  }

  return sections.length >= 2 ? sections : null;
}

// Parse raw slide body into rich structured blocks (subheadings, lists, code fences, math blocks)
function renderStructuredBody(bodyText: string, isSplit: boolean = false): ReactNode {
  if (!bodyText) return null;

  // Auto-detect split columns if slide is 'split' or has 2+ `###` sections
  if (isSplit || bodyText.includes('### ')) {
    const splitSections = parseSplitSections(bodyText);
    if (splitSections) {
      // Find any intro text before the first `###`
      const firstHeadingIdx = bodyText.indexOf('### ');
      const introText = firstHeadingIdx > 0 ? bodyText.slice(0, firstHeadingIdx).trim() : '';

      return (
        <div className="slide-body-flow">
          {introText && (
            <p className="slide-lead">
              {renderInline(introText)}
            </p>
          )}
          <div className={`slide-columns-grid cols-${Math.min(splitSections.length, 3)}`}>
            {splitSections.map((sec, sIdx) => (
              <div className="slide-column-card" key={sIdx}>
                <h3 className="column-title">{renderInline(sec.title)}</h3>
                {sec.lead && <p className="column-body">{renderInline(sec.lead)}</p>}
                {sec.items.length > 0 && (
                  <ul className="column-list">
                    {sec.items.map((item, itemIdx) => (
                      <li key={itemIdx}>{renderInline(item)}</li>
                    ))}
                  </ul>
                )}
              </div>
            ))}
          </div>
        </div>
      );
    }
  }

  const lines = bodyText.replace(/\r\n/g, '\n').split('\n');
  const blocks: ReactNode[] = [];
  let currentBullets: string[] = [];
  let currentNumbers: string[] = [];
  let blockKey = 0;
  let i = 0;

  const flushBullets = () => {
    if (currentBullets.length > 0) {
      blocks.push(
        <ul className="slide-bullets" key={`ul-${blockKey++}`}>
          {currentBullets.map((item, idx) => (
            <li key={idx}>{renderInline(item)}</li>
          ))}
        </ul>
      );
      currentBullets = [];
    }
  };

  const flushNumbers = () => {
    if (currentNumbers.length > 0) {
      blocks.push(
        <ol className="slide-ordered-list" key={`ol-${blockKey++}`}>
          {currentNumbers.map((item, idx) => (
            <li key={idx}>{renderInline(item)}</li>
          ))}
        </ol>
      );
      currentNumbers = [];
    }
  };

  while (i < lines.length) {
    const rawLine = lines[i];
    const line = rawLine.trim();

    if (!line) {
      i++;
      continue;
    }

    // Fenced Code Block inside slide body: ```lang ... ```
    if (line.startsWith('```')) {
      flushBullets();
      flushNumbers();
      const lang = line.slice(3).trim() || 'python';
      const codeLines: string[] = [];
      i++;
      while (i < lines.length && !lines[i].trim().startsWith('```')) {
        codeLines.push(lines[i]);
        i++;
      }
      i++; // skip closing ```
      blocks.push(
        <div className="slide-code-wrap" key={`code-${blockKey++}`}>
          <Code name={lang} code={codeLines.join('\n')} />
        </div>
      );
      continue;
    }

    // Math Block inside slide body: $$ ... $$ or \[ ... \]
    if (line.startsWith('$$') || line.startsWith('\\[')) {
      flushBullets();
      flushNumbers();
      const isBracket = line.startsWith('\\[');
      const endMarker = isBracket ? '\\]' : '$$';
      const mathLines: string[] = [];

      if (line.endsWith(endMarker) && line.length > 4) {
        mathLines.push(line.slice(2, -2).trim());
        i++;
      } else {
        i++;
        while (i < lines.length && !lines[i].trim().startsWith(endMarker)) {
          mathLines.push(lines[i]);
          i++;
        }
        i++; // skip closing marker
      }
      blocks.push(
        <div className="slide-math-wrap" key={`math-${blockKey++}`}>
          <MathBlock tex={mathLines.join('\n')} />
        </div>
      );
      continue;
    }

    // Subheading: ### or ####
    const h3Match = /^###\s+(.+)$/.exec(line);
    if (h3Match) {
      flushBullets();
      flushNumbers();
      blocks.push(
        <h3 className="slide-subheading" key={`h3-${blockKey++}`}>
          {renderInline(h3Match[1].trim())}
        </h3>
      );
      i++;
      continue;
    }

    const h4Match = /^####\s+(.+)$/.exec(line);
    if (h4Match) {
      flushBullets();
      flushNumbers();
      blocks.push(
        <h4 className="slide-subsubheading" key={`h4-${blockKey++}`}>
          {renderInline(h4Match[1].trim())}
        </h4>
      );
      i++;
      continue;
    }

    // Bullet item: • or - or * or +
    if (/^[•\-*+]\s+/.test(line)) {
      flushNumbers();
      let bulletText = line.replace(/^[•\-*+]\s+/, '').trim();
      i++;
      // Capture indented continuation lines
      while (
        i < lines.length &&
        lines[i].trim() &&
        /^\s{2,}/.test(lines[i]) &&
        !/^[•\-*+]\s+/.test(lines[i].trim()) &&
        !/^\d+[.)]\s+/.test(lines[i].trim()) &&
        !lines[i].trim().startsWith('```')
      ) {
        bulletText += ' ' + lines[i].trim();
        i++;
      }
      currentBullets.push(bulletText);
      continue;
    }

    // Numbered item: 1. or 1)
    if (/^\d+[.)]\s+/.test(line)) {
      flushBullets();
      let numText = line.replace(/^\d+[.)]\s+/, '').trim();
      i++;
      while (
        i < lines.length &&
        lines[i].trim() &&
        /^\s{2,}/.test(lines[i]) &&
        !/^[•\-*+]\s+/.test(lines[i].trim()) &&
        !/^\d+[.)]\s+/.test(lines[i].trim()) &&
        !lines[i].trim().startsWith('```')
      ) {
        numText += ' ' + lines[i].trim();
        i++;
      }
      currentNumbers.push(numText);
      continue;
    }

    // Regular narrative or lead paragraph
    flushBullets();
    flushNumbers();
    blocks.push(
      <p className="slide-lead" key={`p-${blockKey++}`}>
        {renderInline(line)}
      </p>
    );
    i++;
  }

  flushBullets();
  flushNumbers();

  return <div className="slide-body-flow">{blocks}</div>;
}

// Render speaker notes with clean paragraph formatting
function renderNotes(notesText: string): ReactNode {
  if (!notesText) return null;
  const paragraphs = notesText.replace(/\r\n/g, '\n').split(/\n\s*\n/);
  return (
    <>
      {paragraphs.map((p, idx) => (
        <p key={idx}>{renderInline(p.trim())}</p>
      ))}
    </>
  );
}

export function SlideContent({
  slide,
  meeting,
}: {
  slide: SlideData;
  meeting: number;
  base: string;
}) {
  if (slide.kind === 'cover') {
    const coverParagraphs = slide.body ? slide.body.replace(/\r\n/g, '\n').split(/\n\s*\n/) : [];

    return (
      <div className="slide-cover-layout">
        <div className="slide-kicker">AI LEARNING HUB / CHAPTER {String(meeting).padStart(2, '0')}</div>
        <h1 className="cover-title">{slide.title}</h1>
        {coverParagraphs.map((cp, idx) => {
          const trimmed = cp.trim();
          if (trimmed.startsWith('### ')) {
            return (
              <h2 className="cover-subtitle" key={idx}>
                {renderInline(trimmed.slice(4))}
              </h2>
            );
          }
          return (
            <p className="cover-body-lead" key={idx}>
              {renderInline(trimmed)}
            </p>
          );
        })}
        <div className="cover-meta-card">
          <div className="author-name">Aji Pamungkas Tri Nurcahyo</div>
          <div className="author-affiliation">Institut Teknologi Sumatera</div>
        </div>
        <aside className="notes">{renderNotes(slide.notes)}</aside>
      </div>
    );
  }

  const isSplitKind = slide.kind === 'split';

  return (
    <div className={`ai-slide slide-kind-${slide.kind || 'content'}`}>
      <header className="slide-header">
        <span className="slide-kicker">CHAPTER {String(meeting).padStart(2, '0')}</span>
        <h2 className="slide-heading">{slide.title}</h2>
      </header>

      <div className="slide-main-content">
        {/* Case box layout (Studi Kasus) */}
        {slide.caseBox && (
          <div className="slide-case-card">
            {slide.caseBox.title && <div className="case-title">{slide.caseBox.title}</div>}
            <p className="case-scenario">{renderInline(slide.caseBox.scenario)}</p>
            {slide.caseBox.goal && (
              <div className="case-goal">
                <strong>Tujuan:</strong> {renderInline(slide.caseBox.goal)}
              </div>
            )}
            {slide.caseBox.options && (
              <div className="case-options">
                {slide.caseBox.options.map((opt, i) => (
                  <span className="case-pill" key={i}>
                    {renderInline(opt)}
                  </span>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Structured main body */}
        {slide.body && renderStructuredBody(slide.body, isSplitKind)}

        {/* Two-column comparison layout (if explicitly defined via columns array) */}
        {slide.columns && slide.columns.length > 0 && (
          <div className={`slide-columns-grid cols-${slide.columns.length}`}>
            {slide.columns.map((col, idx) => (
              <div className="slide-column-card" key={idx}>
                <h3 className="column-title">{renderInline(col.title)}</h3>
                {col.body && <p className="column-body">{renderInline(col.body)}</p>}
                {col.items && (
                  <ul className="column-list">
                    {col.items.map((item, itemIdx) => (
                      <li key={itemIdx}>{renderInline(item)}</li>
                    ))}
                  </ul>
                )}
              </div>
            ))}
          </div>
        )}

        {/* Explicit numbered discussion questions */}
        {slide.questions && slide.questions.length > 0 && (
          <ol className="slide-questions-list">
            {slide.questions.map((q, idx) => (
              <li key={idx}>
                <span className="q-number">{idx + 1}</span>
                <div className="q-text">{renderInline(q)}</div>
              </li>
            ))}
          </ol>
        )}

        {/* KaTeX formula block */}
        {slide.tex && (
          <div className="slide-math-wrap">
            <MathBlock tex={slide.tex} />
          </div>
        )}

        {/* Python Code block */}
        {slide.code && (
          <div className="slide-code-wrap">
            <Code code={slide.code} name="contoh_kode.py" />
          </div>
        )}

        {/* Workflow Diagram */}
        {slide.diagram && slide.diagram.length > 0 && (
          <div className="slide-workflow-strip" aria-label="Diagram Alur Langkah">
            {slide.diagram.map((step, i) => (
              <div className="workflow-step-node" key={step}>
                {i > 0 && <span className="workflow-arrow" aria-hidden="true">→</span>}
                <div className="workflow-pill">
                  <span className="step-idx">{i + 1}</span>
                  <span className="step-label">{step}</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      <aside className="notes">{renderNotes(slide.notes)}</aside>
    </div>
  );
}
