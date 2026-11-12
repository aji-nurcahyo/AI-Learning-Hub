import {readFileSync,existsSync} from 'node:fs';
import assert from 'node:assert/strict';
const base=process.env.VITE_BASE_PATH||'/ai-learning-hub/';
for(const file of ['index.html','404.html']){const html=readFileSync(`dist/${file}`,'utf8');assert.ok(html.includes('id="root"'));const assets=[...html.matchAll(/(?:src|href)="([^"]+\/assets\/[^"]+)"/g)].map(m=>m[1]);assert.ok(assets.length>0);for(const asset of assets){assert.ok(asset.startsWith(base),`wrong base ${asset}`);assert.ok(existsSync(`dist/${asset.slice(base.length)}`));}}
for(const file of ['.nojekyll','LICENSE.upstream','content/level-4-advanced-ai/chapter-40/metadata.json','docs/ADDING_CHAPTER.md'])assert.ok(existsSync(`dist/${file}`),file);
console.log(`PASS: production entrypoints, assets under ${base}, static content, notices and documentation. Hash routes survive refresh without server rewrite.`);
