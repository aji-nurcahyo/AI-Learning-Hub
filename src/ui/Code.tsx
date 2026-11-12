import { useState } from 'react';
import { tokenize } from './highlight';
import { Icon } from './Icon';

export function Code({
  code,
  name = 'contoh.py',
}: {
  code: string;
  name?: string;
}) {
  const [copied, setCopied] = useState(false);

  async function copy() {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 2500);
    } catch {
      // Fallback
      setCopied(false);
    }
  }

  return (
    <div className="code-card">
      <div className="code-bar">
        <span>
          <Icon name="code" size={16} /> <b>PY</b> {name}
        </span>
        <button
          type="button"
          onClick={copy}
          className={`copy-btn ${copied ? 'copied' : ''}`}
          aria-label={`Salin kode ${name}`}
        >
          <Icon name={copied ? 'check' : 'content_copy'} size={14} />
          <span>{copied ? 'Tersalin' : 'Salin kode'}</span>
        </button>
      </div>
      <pre tabIndex={0} aria-label={`Kode Python ${name}`}>
        <code>
          {tokenize(code).map((line, i) => (
            <span className="code-line" key={i}>
              <span aria-hidden="true" className="line-number">
                {i + 1}
              </span>
              {line.map((token, j) => (
                <span
                  key={j}
                  style={{
                    color: token.color,
                    fontStyle: token.italic ? 'italic' : undefined,
                  }}
                >
                  {token.text}
                </span>
              ))}
              {'\n'}
            </span>
          ))}
        </code>
      </pre>
      <span role="status" className="sr-only">
        {copied ? 'Kode berhasil disalin ke clipboard' : ''}
      </span>
    </div>
  );
}
