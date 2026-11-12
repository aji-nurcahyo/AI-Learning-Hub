import { Fragment, type ReactNode } from 'react';
import { Code } from '../ui/Code';
import { MathBlock, M } from '../ui/Math';
import {
  parseMarkdownBlocks,
  parseInlineTokens,
  type InlineToken,
} from './markdownParser';

// Render AST inline token into secure React elements
function renderToken(token: InlineToken, key: number | string): ReactNode {
  switch (token.type) {
    case 'math':
      return <M key={key} tex={token.value} />;

    case 'code':
      return (
        <code key={key} className="inline-code">
          {token.value}
        </code>
      );

    case 'link': {
      const isSafe = /^https?:\/\/|^#|^\/|^mailto:/i.test(token.url);
      const isExternal = /^https?:\/\//i.test(token.url);
      if (isSafe) {
        return (
          <a
            key={key}
            href={token.url}
            target={isExternal ? '_blank' : undefined}
            rel={isExternal ? 'noreferrer noopener' : undefined}
          >
            {token.label}
          </a>
        );
      }
      return <span key={key}>{token.label}</span>;
    }

    case 'bold':
      return (
        <strong key={key}>
          {token.children.map((child, idx) => renderToken(child, idx))}
        </strong>
      );

    case 'italic':
      return (
        <em key={key}>
          {token.children.map((child, idx) => renderToken(child, idx))}
        </em>
      );

    case 'text':
    default:
      return <Fragment key={key}>{token.value}</Fragment>;
  }
}

export function renderInline(text: string): ReactNode {
  const tokens = parseInlineTokens(text);
  return tokens.map((token, idx) => renderToken(token, idx));
}

export function Reading({
  text,
  sidebarTop,
  className = '',
}: {
  text: string;
  sidebarTop?: ReactNode;
  className?: string;
}) {
  const { blocks, toc } = parseMarkdownBlocks(text);

  return (
    <div className={`reading-grid ${className}`.trim()}>
      <aside className="reading-toc" aria-label="Navigasi dan daftar isi modul">
        {sidebarTop}
        {sidebarTop && toc.length > 0 && <hr className="reading-toc-divider" />}
        {toc.length > 0 && (
          <>
            <h2>Daftar isi</h2>
            <nav>
              {toc.map(item => (
                <a
                  key={item.id}
                  href={`#${item.id}`}
                  className={`toc-level-${item.level}`}
                  onClick={e => {
                    e.preventDefault();
                    const el = document.getElementById(item.id);
                    if (el) {
                      el.scrollIntoView({ behavior: 'smooth' });
                      window.history.replaceState(null, '', `#${item.id}`);
                    }
                  }}
                >
                  {item.text}
                </a>
              ))}
            </nav>
          </>
        )}
      </aside>

      <article className="reading-article">
        {blocks.map((block, idx) => {
          switch (block.type) {
            case 'heading': {
              const Tag =
                block.level === 1
                  ? 'h1'
                  : block.level === 2
                  ? 'h2'
                  : block.level === 3
                  ? 'h3'
                  : block.level === 4
                  ? 'h4'
                  : block.level === 5
                  ? 'h5'
                  : 'h6';
              return (
                <Tag id={block.id} key={idx} className={`reading-heading reading-h${block.level}`}>
                  {renderInline(block.content)}
                </Tag>
              );
            }

            case 'code':
              return <Code key={idx} name={block.lang} code={block.content} />;

            case 'math':
              return <MathBlock key={idx} tex={block.content} />;

            case 'hr':
              return <hr key={idx} className="reading-divider" />;

            case 'table':
              return (
                <div key={idx} className="table-wrapper">
                  <table className="content-table">
                    {block.headers && block.headers.length > 0 && (
                      <thead>
                        <tr>
                          {block.headers.map((h, hIdx) => (
                            <th key={hIdx}>{renderInline(h)}</th>
                          ))}
                        </tr>
                      </thead>
                    )}
                    {block.rows && block.rows.length > 0 && (
                      <tbody>
                        {block.rows.map((row, rIdx) => (
                          <tr key={rIdx}>
                            {row.map((cell, cIdx) => (
                              <td key={cIdx}>{renderInline(cell)}</td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    )}
                  </table>
                </div>
              );

            case 'quote':
              return (
                <blockquote key={idx} className="reading-quote">
                  {block.content.split('\n').map((ql, qIdx) => (
                    <p key={qIdx}>{renderInline(ql)}</p>
                  ))}
                </blockquote>
              );

            case 'ul':
              return (
                <ul key={idx} className="reading-list">
                  {block.items?.map((item, itemIdx) => {
                    const itemLines = item.split('\n');
                    return (
                      <li key={itemIdx}>
                        {itemLines.map((il, ilIdx) => (
                          <Fragment key={ilIdx}>
                            {ilIdx > 0 && <br />}
                            {renderInline(il)}
                          </Fragment>
                        ))}
                      </li>
                    );
                  })}
                </ul>
              );

            case 'ol':
              return (
                <ol key={idx} className="reading-list-ordered">
                  {block.items?.map((item, itemIdx) => {
                    const itemLines = item.split('\n');
                    return (
                      <li key={itemIdx}>
                        {itemLines.map((il, ilIdx) => (
                          <Fragment key={ilIdx}>
                            {ilIdx > 0 && <br />}
                            {renderInline(il)}
                          </Fragment>
                        ))}
                      </li>
                    );
                  })}
                </ol>
              );

            case 'paragraph':
            default: {
              const pLines = block.content.split('\n');
              return (
                <p key={idx} className="reading-paragraph">
                  {pLines.map((l, lIdx) => (
                    <Fragment key={lIdx}>
                      {lIdx > 0 && <br />}
                      {renderInline(l)}
                    </Fragment>
                  ))}
                </p>
              );
            }
          }
        })}
      </article>
    </div>
  );
}
