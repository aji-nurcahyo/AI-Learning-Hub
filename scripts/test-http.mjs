import { spawn } from 'node:child_process';
import { readFileSync, existsSync } from 'node:fs';
import { resolve, relative } from 'node:path';
import assert from 'node:assert/strict';
import { root, walk } from './validate.mjs';

const child = spawn(process.execPath, ['node_modules/vite/bin/vite.js', 'preview', '--host', '127.0.0.1', '--port', '8944', '--strictPort'], { stdio: 'pipe', windowsHide: true });
let log = '';
child.stdout.on('data', d => log += d);
child.stderr.on('data', d => log += d);
const origin = 'http://127.0.0.1:8944';
const base = '/ai-learning-hub/';

try {
  let ready = false;
  for (let i = 0; i < 60; i++) {
    try {
      const res = await fetch(origin + base);
      if (res.ok) { ready = true; break; }
    } catch {}
    await new Promise(r => setTimeout(r, 100));
  }
  assert.ok(ready, log || 'Preview server did not become ready');

  const html = await (await fetch(origin + base)).text();
  const entryUrls = [...html.matchAll(/(?:src|href)="([^"]+\/assets\/[^"]+)"/g)].map(m => m[1]);

  // Manifest-driven resource discovery across Chapters 01-40
  const metaFiles = walk(resolve(root, 'content')).filter(f => f.endsWith('metadata.json'));
  const gsModules = JSON.parse(readFileSync(resolve(root, 'content/getting-started/modules.json'), 'utf8'));

  const staticResources = [
    base + '404.html',
    base + 'LICENSE.upstream',
    base + 'docs/ADDING_CHAPTER.md',
    base + 'content/templates/slides.json',
    base + 'content/levels.json',
    base + 'content/specializations/tracks.json',
    base + 'content/getting-started/modules.json',
    ...gsModules.map(m => base + 'content/getting-started/' + m.file),
  ];

  const chapterMetaUrls = [];
  const chapterResourceUrls = [];

  for (const mf of metaFiles) {
    const chDir = resolve(mf, '..');
    const relDir = relative(root, chDir).replaceAll('\\', '/');
    const meta = JSON.parse(readFileSync(mf, 'utf8'));
    chapterMetaUrls.push({ id: meta.id, url: base + relDir + '/metadata.json' });

    for (const key of ['reading', 'slides', 'practice']) {
      if (meta.resources?.[key]?.status === 'available' && meta.resources[key].path) {
        chapterResourceUrls.push(base + relDir + '/' + meta.resources[key].path);
      }
    }
    const readmePath = resolve(chDir, 'code/README.md');
    if (existsSync(readmePath)) {
      chapterResourceUrls.push(base + relDir + '/code/README.md');
    }
    const nbPath = resolve(chDir, `notebooks/chapter_${String(meta.id).padStart(2, '0')}_companion.ipynb`);
    if (existsSync(nbPath)) {
      chapterResourceUrls.push(base + relDir + `/notebooks/chapter_${String(meta.id).padStart(2, '0')}_companion.ipynb`);
    }
    const auditPath = resolve(chDir, 'audit.md');
    if (existsSync(auditPath)) {
      chapterResourceUrls.push(base + relDir + '/audit.md');
    }
    for (const sub of ['data', 'experiments']) {
      const subDir = resolve(chDir, sub);
      if (existsSync(subDir)) {
        for (const f of walk(subDir)) {
          if (/\.(json|png|npz|pt|txt)$/i.test(f)) {
            const relFile = relative(root, f).replaceAll('\\', '/');
            chapterResourceUrls.push(base + relFile);
          }
        }
      }
    }
  }

  const allUrls = [...new Set([...entryUrls, ...staticResources, ...chapterMetaUrls.map(x => x.url), ...chapterResourceUrls])];

  for (const url of allUrls) {
    const res = await fetch(origin + url);
    assert.equal(res.status, 200, `Expected HTTP 200 for ${url}`);
    const buf = await res.arrayBuffer();
    assert.ok(buf.byteLength > 0, `Empty body for ${url}`);
  }

  // Verify every Chapter 01-40 manifest reports status === 'published' over HTTP
  assert.equal(chapterMetaUrls.length, 40, 'Expected exactly 40 chapter manifests');
  let publishedCount = 0;
  for (const item of chapterMetaUrls) {
    const c = await (await fetch(origin + item.url)).json();
    assert.equal(c.id, item.id);
    assert.equal(c.status, 'published', `Chapter ${item.id} must be published`);
    publishedCount++;
  }

  console.log(
    `PASS: HTTP ${entryUrls.length} entry assets, ${staticResources.length} platform/docs/getting-started assets, ` +
    `${chapterMetaUrls.length} chapter manifests (${publishedCount}/40 published), and ${chapterResourceUrls.length} ` +
    `chapter module/slide/practice/notebook/data/experiment resources (${allUrls.length} unique URLs verified).`
  );
} finally {
  child.kill();
}
