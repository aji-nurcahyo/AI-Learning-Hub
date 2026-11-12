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

  // Manifest-driven resource discovery across Chapters 01-40 and 5 Specialization Tracks
  const metaFiles = walk(resolve(root, 'content')).filter(f => !f.replaceAll('\\', '/').includes('/tracks/') && f.endsWith('metadata.json'));
  const gsModules = JSON.parse(readFileSync(resolve(root, 'content/getting-started/modules.json'), 'utf8'));
  const tracksList = JSON.parse(readFileSync(resolve(root, 'content/specializations/tracks.json'), 'utf8'));

  const staticResources = [
    base + '404.html',
    base + 'LICENSE.upstream',
    base + 'docs/ADDING_CHAPTER.md',
    base + 'docs/CONTENT_MAINTENANCE_GUIDE.md',
    base + 'docs/THEME_AND_ACCESSIBILITY_GUIDE.md',
    base + 'content/templates/slides.json',
    base + 'content/levels.json',
    base + 'content/specializations/tracks.json',
    base + 'content/getting-started/modules.json',
    ...gsModules.map(m => base + 'content/getting-started/' + m.file),
  ];

  const trackMetaUrls = [];
  const trackResourceUrls = [];
  for (const t of tracksList) {
    trackMetaUrls.push({ id: t.id, url: base + `content/tracks/${t.id}/metadata.json` });
    for (const md of ['overview.md', 'practice.md', 'capstone.md']) {
      trackResourceUrls.push(base + `content/tracks/${t.id}/${md}`);
    }
  }

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

  const allUrls = [...new Set([...entryUrls, ...staticResources, ...trackMetaUrls.map(x => x.url), ...trackResourceUrls, ...chapterMetaUrls.map(x => x.url), ...chapterResourceUrls])];

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

  // Verify all 5 Specialization Track manifests report status === 'published' over HTTP
  assert.equal(trackMetaUrls.length, 5, 'Expected exactly 5 track manifests');
  for (const item of trackMetaUrls) {
    const tm = await (await fetch(origin + item.url)).json();
    assert.equal(tm.id, item.id);
    assert.equal(tm.status, 'published', `Track ${item.id} must be published`);
  }

  console.log(
    `PASS: HTTP ${entryUrls.length} entry assets, ${staticResources.length} platform/docs/getting-started assets, ` +
    `${trackMetaUrls.length}/5 published tracks (${trackResourceUrls.length} md files), ` +
    `${chapterMetaUrls.length} chapter manifests (${publishedCount}/40 published), and ${chapterResourceUrls.length} ` +
    `chapter resources (${allUrls.length} unique URLs verified).`
  );
} finally {
  child.kill();
}
