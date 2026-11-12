import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { validateChapter, validateAll } from './validate.mjs';
import { parseRoute } from '../src/routes.ts';
import { parseMarkdownBlocks, parseInlineTokens } from '../src/components/markdownParser.ts';

const directory=new URL('../content/level-4-advanced-ai/chapter-40/',import.meta.url);
const directoryPath=fileURLToPath(directory);
const sample=JSON.parse(readFileSync(new URL('metadata.json',directory),'utf8'));
test('complete initial curriculum validates',()=>assert.deepEqual(validateAll(),[]));
test('all chapter routes and views resolve',()=>{for(let id=1;id<=40;id++){for(const view of ['read','slides','task','template','practice'])assert.deepEqual(parseRoute(`#chapter/${id}/${view}`),{type:'chapter',id,view});}});
test('route rejects malformed, trailing and out-of-range IDs',()=>{for(const hash of ['#chapter/0','#chapter/41','#level/5','#chapter/2/slides/garbage','#chapter/1evil','#level/1/read'])assert.equal(parseRoute(hash).type,'missing');});
test('levels, tracks and home anchors resolve',()=>{for(let id=1;id<=4;id++)assert.equal(parseRoute(`#level/${id}`).id,id);assert.equal(parseRoute('#track/robotics').id,'robotics');assert.equal(parseRoute('#curriculum').type,'home');assert.equal(parseRoute('#getting-started').type,'getting-started');assert.equal(parseRoute('#getting-started/gs-02').type,'getting-started');});
test('reject unavailable resource with a path',()=>{const c=structuredClone(sample);c.resources.practice={status:'unavailable',path:'missing.py'};assert.ok(validateChapter(c,directoryPath).some(e=>e.includes('null path')));});
test('reject resource traversal',()=>{const c=structuredClone(sample);c.resources.practice={status:'available',path:'../secret.py'};assert.ok(validateChapter(c,directoryPath).some(e=>e.includes('unsafe path')));});
test('reject nonexistent available resource',()=>{const c=structuredClone(sample);c.resources.practice={status:'available',path:'nonexistent.py'};assert.ok(validateChapter(c,directoryPath).some(e=>e.includes('missing file')));});
test('cannot publish placeholder by changing status alone',()=>{const c=structuredClone(sample);c.status='published';c.validation={reviewed:false,executionVerified:false};c.resources={reading:{status:'unavailable',path:null},slides:{status:'unavailable',path:null},practice:{status:'unavailable',path:null}};assert.ok(validateChapter(c,directoryPath).some(e=>e.includes('publication needs')||e.includes('missing reading module')));});

test('chapter 01 passes full publication audit and validation', () => {
  const ch01Dir = new URL('../content/level-1-ai-foundations/chapter-01/', import.meta.url);
  const ch01Path = fileURLToPath(ch01Dir);
  const ch01Data = JSON.parse(readFileSync(new URL('metadata.json', ch01Dir), 'utf8'));
  const errors = validateChapter(ch01Data, ch01Path);
  assert.deepEqual(errors, []);
  assert.equal(ch01Data.status, 'published');
});

test('chapter 02 passes full publication audit and validation', () => {
  const ch02Dir = new URL('../content/level-1-ai-foundations/chapter-02/', import.meta.url);
  const ch02Path = fileURLToPath(ch02Dir);
  const ch02Data = JSON.parse(readFileSync(new URL('metadata.json', ch02Dir), 'utf8'));
  const errors = validateChapter(ch02Data, ch02Path);
  assert.deepEqual(errors, []);
  assert.equal(ch02Data.status, 'published');
  assert.equal(ch02Data.resources.practice.status, 'unavailable');
});

test('chapter 03 passes full publication audit and validation', () => {
  const ch03Dir = new URL('../content/level-1-ai-foundations/chapter-03/', import.meta.url);
  const ch03Path = fileURLToPath(ch03Dir);
  const ch03Data = JSON.parse(readFileSync(new URL('metadata.json', ch03Dir), 'utf8'));
  const errors = validateChapter(ch03Data, ch03Path);
  assert.deepEqual(errors, []);
  assert.equal(ch03Data.status, 'published');
  assert.equal(ch03Data.resources.practice.status, 'available');
  assert.equal(ch03Data.resources.reading.status, 'available');
  assert.equal(ch03Data.resources.slides.status, 'available');
});

test('chapter 04 passes full publication audit and validation', () => {
  const ch04Dir = new URL('../content/level-1-ai-foundations/chapter-04/', import.meta.url);
  const ch04Path = fileURLToPath(ch04Dir);
  const ch04Data = JSON.parse(readFileSync(new URL('metadata.json', ch04Dir), 'utf8'));
  const errors = validateChapter(ch04Data, ch04Path);
  assert.deepEqual(errors, []);
  assert.equal(ch04Data.status, 'published');
  assert.equal(ch04Data.resources.practice.status, 'available');
  assert.equal(ch04Data.resources.reading.status, 'available');
  assert.equal(ch04Data.resources.slides.status, 'available');
});

test('chapter 05 passes full publication audit and validation', () => {
  const ch05Dir = new URL('../content/level-1-ai-foundations/chapter-05/', import.meta.url);
  const ch05Path = fileURLToPath(ch05Dir);
  const ch05Data = JSON.parse(readFileSync(new URL('metadata.json', ch05Dir), 'utf8'));
  const errors = validateChapter(ch05Data, ch05Path);
  assert.deepEqual(errors, []);
  assert.equal(ch05Data.status, 'published');
  assert.equal(ch05Data.resources.practice.status, 'available');
  assert.equal(ch05Data.resources.reading.status, 'available');
  assert.equal(ch05Data.resources.slides.status, 'available');
});

test('chapter 06 passes full publication audit and validation', () => {
  const ch06Dir = new URL('../content/level-1-ai-foundations/chapter-06/', import.meta.url);
  const ch06Path = fileURLToPath(ch06Dir);
  const ch06Data = JSON.parse(readFileSync(new URL('metadata.json', ch06Dir), 'utf8'));
  const errors = validateChapter(ch06Data, ch06Path);
  assert.deepEqual(errors, []);
  assert.equal(ch06Data.status, 'published');
  assert.equal(ch06Data.resources.practice.status, 'available');
  assert.equal(ch06Data.resources.reading.status, 'available');
  assert.equal(ch06Data.resources.slides.status, 'available');
});

test('chapter 07 passes full publication audit and validation', () => {
  const ch07Dir = new URL('../content/level-1-ai-foundations/chapter-07/', import.meta.url);
  const ch07Path = fileURLToPath(ch07Dir);
  const ch07Data = JSON.parse(readFileSync(new URL('metadata.json', ch07Dir), 'utf8'));
  const errors = validateChapter(ch07Data, ch07Path);
  assert.deepEqual(errors, []);
  assert.equal(ch07Data.status, 'published');
  assert.equal(ch07Data.resources.practice.status, 'available');
  assert.equal(ch07Data.resources.reading.status, 'available');
  assert.equal(ch07Data.resources.slides.status, 'available');
});

test('chapter 08 passes full publication audit and validation', () => {
  const ch08Dir = new URL('../content/level-1-ai-foundations/chapter-08/', import.meta.url);
  const ch08Path = fileURLToPath(ch08Dir);
  const ch08Data = JSON.parse(readFileSync(new URL('metadata.json', ch08Dir), 'utf8'));
  const errors = validateChapter(ch08Data, ch08Path);
  assert.deepEqual(errors, []);
  assert.equal(ch08Data.status, 'published');
  assert.equal(ch08Data.resources.practice.status, 'available');
  assert.equal(ch08Data.resources.reading.status, 'available');
  assert.equal(ch08Data.resources.slides.status, 'available');
});

test('chapter 09 passes full publication audit and validation', () => {
  const ch09Dir = new URL('../content/level-1-ai-foundations/chapter-09/', import.meta.url);
  const ch09Path = fileURLToPath(ch09Dir);
  const ch09Data = JSON.parse(readFileSync(new URL('metadata.json', ch09Dir), 'utf8'));
  const errors = validateChapter(ch09Data, ch09Path);
  assert.deepEqual(errors, []);
  assert.equal(ch09Data.status, 'published');
  assert.equal(ch09Data.resources.practice.status, 'available');
  assert.equal(ch09Data.resources.reading.status, 'available');
  assert.equal(ch09Data.resources.slides.status, 'available');
});

test('chapter 10 passes full publication audit and validation', () => {
  const ch10Dir = new URL('../content/level-1-ai-foundations/chapter-10/', import.meta.url);
  const ch10Path = fileURLToPath(ch10Dir);
  const ch10Data = JSON.parse(readFileSync(new URL('metadata.json', ch10Dir), 'utf8'));
  const errors = validateChapter(ch10Data, ch10Path);
  assert.deepEqual(errors, []);
  assert.equal(ch10Data.status, 'published');
  assert.equal(ch10Data.resources.practice.status, 'available');
  assert.equal(ch10Data.resources.reading.status, 'available');
  assert.equal(ch10Data.resources.slides.status, 'available');
});

test('chapter 11 passes full publication audit and validation', () => {
  const ch11Dir = new URL('../content/level-2-machine-learning/chapter-11/', import.meta.url);
  const ch11Path = fileURLToPath(ch11Dir);
  const ch11Data = JSON.parse(readFileSync(new URL('metadata.json', ch11Dir), 'utf8'));
  const errors = validateChapter(ch11Data, ch11Path);
  assert.deepEqual(errors, []);
  assert.equal(ch11Data.status, 'published');
  assert.equal(ch11Data.resources.practice.status, 'available');
  assert.equal(ch11Data.resources.reading.status, 'available');
  assert.equal(ch11Data.resources.slides.status, 'available');
});

test('chapter 12 passes full publication audit and validation', () => {
  const ch12Dir = new URL('../content/level-2-machine-learning/chapter-12/', import.meta.url);
  const ch12Path = fileURLToPath(ch12Dir);
  const ch12Data = JSON.parse(readFileSync(new URL('metadata.json', ch12Dir), 'utf8'));
  const errors = validateChapter(ch12Data, ch12Path);
  assert.deepEqual(errors, []);
  assert.equal(ch12Data.status, 'published');
  assert.equal(ch12Data.resources.practice.status, 'available');
  assert.equal(ch12Data.resources.reading.status, 'available');
  assert.equal(ch12Data.resources.slides.status, 'available');
});

test('chapter 13 passes full publication audit and validation', () => {
  const ch13Dir = new URL('../content/level-2-machine-learning/chapter-13/', import.meta.url);
  const ch13Path = fileURLToPath(ch13Dir);
  const ch13Data = JSON.parse(readFileSync(new URL('metadata.json', ch13Dir), 'utf8'));
  const errors = validateChapter(ch13Data, ch13Path);
  assert.deepEqual(errors, []);
  assert.equal(ch13Data.status, 'published');
  assert.equal(ch13Data.resources.practice.status, 'available');
  assert.equal(ch13Data.resources.reading.status, 'available');
  assert.equal(ch13Data.resources.slides.status, 'available');
});

test('chapter 14 passes full publication audit and validation', () => {
  const ch14Dir = new URL('../content/level-2-machine-learning/chapter-14/', import.meta.url);
  const ch14Path = fileURLToPath(ch14Dir);
  const ch14Data = JSON.parse(readFileSync(new URL('metadata.json', ch14Dir), 'utf8'));
  const errors = validateChapter(ch14Data, ch14Path);
  assert.deepEqual(errors, []);
  assert.equal(ch14Data.status, 'published');
  assert.equal(ch14Data.resources.practice.status, 'available');
  assert.equal(ch14Data.resources.reading.status, 'available');
  assert.equal(ch14Data.resources.slides.status, 'available');
});

test('chapter 15 passes full publication audit and validation', () => {
  const ch15Dir = new URL('../content/level-2-machine-learning/chapter-15/', import.meta.url);
  const ch15Path = fileURLToPath(ch15Dir);
  const ch15Data = JSON.parse(readFileSync(new URL('metadata.json', ch15Dir), 'utf8'));
  const errors = validateChapter(ch15Data, ch15Path);
  assert.deepEqual(errors, []);
  assert.equal(ch15Data.status, 'published');
  assert.equal(ch15Data.resources.practice.status, 'available');
  assert.equal(ch15Data.resources.reading.status, 'available');
  assert.equal(ch15Data.resources.slides.status, 'available');
});

test('chapter 16 passes full publication audit and validation', () => {
  const ch16Dir = new URL('../content/level-2-machine-learning/chapter-16/', import.meta.url);
  const ch16Path = fileURLToPath(ch16Dir);
  const ch16Data = JSON.parse(readFileSync(new URL('metadata.json', ch16Dir), 'utf8'));
  const errors = validateChapter(ch16Data, ch16Path);
  assert.deepEqual(errors, []);
  assert.equal(ch16Data.status, 'published');
  assert.equal(ch16Data.resources.practice.status, 'available');
  assert.equal(ch16Data.resources.reading.status, 'available');
  assert.equal(ch16Data.resources.slides.status, 'available');
});

test('chapter 17 passes full publication audit and validation', () => {
  const ch17Dir = new URL('../content/level-2-machine-learning/chapter-17/', import.meta.url);
  const ch17Path = fileURLToPath(ch17Dir);
  const ch17Data = JSON.parse(readFileSync(new URL('metadata.json', ch17Dir), 'utf8'));
  const errors = validateChapter(ch17Data, ch17Path);
  assert.deepEqual(errors, []);
  assert.equal(ch17Data.status, 'published');
  assert.equal(ch17Data.resources.practice.status, 'available');
  assert.equal(ch17Data.resources.reading.status, 'available');
  assert.equal(ch17Data.resources.slides.status, 'available');
});

test('chapter 18 passes full publication audit and validation', () => {
  const ch18Dir = new URL('../content/level-2-machine-learning/chapter-18/', import.meta.url);
  const ch18Path = fileURLToPath(ch18Dir);
  const ch18Data = JSON.parse(readFileSync(new URL('metadata.json', ch18Dir), 'utf8'));
  const errors = validateChapter(ch18Data, ch18Path);
  assert.deepEqual(errors, []);
  assert.equal(ch18Data.status, 'published');
  assert.equal(ch18Data.resources.practice.status, 'available');
  assert.equal(ch18Data.resources.reading.status, 'available');
  assert.equal(ch18Data.resources.slides.status, 'available');
});

test('chapter 19 passes full publication audit and validation', () => {
  const ch19Dir = new URL('../content/level-2-machine-learning/chapter-19/', import.meta.url);
  const ch19Path = fileURLToPath(ch19Dir);
  const ch19Data = JSON.parse(readFileSync(new URL('metadata.json', ch19Dir), 'utf8'));
  const errors = validateChapter(ch19Data, ch19Path);
  assert.deepEqual(errors, []);
  assert.equal(ch19Data.status, 'published');
  assert.equal(ch19Data.resources.practice.status, 'available');
  assert.equal(ch19Data.resources.reading.status, 'available');
  assert.equal(ch19Data.resources.slides.status, 'available');
});

test('chapter 20 passes full publication audit and validation', () => {
  const ch20Dir = new URL('../content/level-2-machine-learning/chapter-20/', import.meta.url);
  const ch20Path = fileURLToPath(ch20Dir);
  const ch20Data = JSON.parse(readFileSync(new URL('metadata.json', ch20Dir), 'utf8'));
  const errors = validateChapter(ch20Data, ch20Path);
  assert.deepEqual(errors, []);
  assert.equal(ch20Data.status, 'published');
  assert.equal(ch20Data.resources.practice.status, 'available');
  assert.equal(ch20Data.resources.reading.status, 'available');
  assert.equal(ch20Data.resources.slides.status, 'available');
});

test('chapter 21 passes full publication audit and validation', () => {
  const ch21Dir = new URL('../content/level-3-deep-learning/chapter-21/', import.meta.url);
  const ch21Path = fileURLToPath(ch21Dir);
  const ch21Data = JSON.parse(readFileSync(new URL('metadata.json', ch21Dir), 'utf8'));
  const errors = validateChapter(ch21Data, ch21Path);
  assert.deepEqual(errors, []);
  assert.equal(ch21Data.status, 'published');
  assert.equal(ch21Data.resources.practice.status, 'available');
  assert.equal(ch21Data.resources.reading.status, 'available');
  assert.equal(ch21Data.resources.slides.status, 'available');
});

test('chapter 22 passes full publication audit and validation', () => {
  const ch22Dir = new URL('../content/level-3-deep-learning/chapter-22/', import.meta.url);
  const ch22Path = fileURLToPath(ch22Dir);
  const ch22Data = JSON.parse(readFileSync(new URL('metadata.json', ch22Dir), 'utf8'));
  const errors = validateChapter(ch22Data, ch22Path);
  assert.deepEqual(errors, []);
  assert.equal(ch22Data.status, 'published');
  assert.equal(ch22Data.resources.practice.status, 'available');
  assert.equal(ch22Data.resources.reading.status, 'available');
  assert.equal(ch22Data.resources.slides.status, 'available');
});

test('chapter 23 passes full publication audit and validation', () => {
  const ch23Dir = new URL('../content/level-3-deep-learning/chapter-23/', import.meta.url);
  const ch23Path = fileURLToPath(ch23Dir);
  const ch23Data = JSON.parse(readFileSync(new URL('metadata.json', ch23Dir), 'utf8'));
  const errors = validateChapter(ch23Data, ch23Path);
  assert.deepEqual(errors, []);
  assert.equal(ch23Data.status, 'published');
  assert.equal(ch23Data.resources.practice.status, 'available');
  assert.equal(ch23Data.resources.reading.status, 'available');
  assert.equal(ch23Data.resources.slides.status, 'available');
});

test('chapter 24 passes full publication audit and validation', () => {
  const ch24Dir = new URL('../content/level-3-deep-learning/chapter-24/', import.meta.url);
  const ch24Path = fileURLToPath(ch24Dir);
  const ch24Data = JSON.parse(readFileSync(new URL('metadata.json', ch24Dir), 'utf8'));
  const errors = validateChapter(ch24Data, ch24Path);
  assert.deepEqual(errors, []);
  assert.equal(ch24Data.status, 'published');
  assert.equal(ch24Data.resources.practice.status, 'available');
  assert.equal(ch24Data.resources.reading.status, 'available');
  assert.equal(ch24Data.resources.slides.status, 'available');
});

test('chapter 25 passes full publication audit and validation', () => {
  const ch25Dir = new URL('../content/level-3-deep-learning/chapter-25/', import.meta.url);
  const ch25Path = fileURLToPath(ch25Dir);
  const ch25Data = JSON.parse(readFileSync(new URL('metadata.json', ch25Dir), 'utf8'));
  const errors = validateChapter(ch25Data, ch25Path);
  assert.deepEqual(errors, []);
  assert.equal(ch25Data.status, 'published');
  assert.equal(ch25Data.resources.practice.status, 'available');
  assert.equal(ch25Data.resources.reading.status, 'available');
  assert.equal(ch25Data.resources.slides.status, 'available');
});

test('chapter 26 passes full publication audit and validation', () => {
  const ch26Dir = new URL('../content/level-3-deep-learning/chapter-26/', import.meta.url);
  const ch26Path = fileURLToPath(ch26Dir);
  const ch26Data = JSON.parse(readFileSync(new URL('metadata.json', ch26Dir), 'utf8'));
  const errors = validateChapter(ch26Data, ch26Path);
  assert.deepEqual(errors, []);
  assert.equal(ch26Data.status, 'published');
  assert.equal(ch26Data.resources.practice.status, 'available');
  assert.equal(ch26Data.resources.reading.status, 'available');
  assert.equal(ch26Data.resources.slides.status, 'available');
});

test('chapter 27 passes full publication audit and validation', () => {
  const ch27Dir = new URL('../content/level-3-deep-learning/chapter-27/', import.meta.url);
  const ch27Path = fileURLToPath(ch27Dir);
  const ch27Data = JSON.parse(readFileSync(new URL('metadata.json', ch27Dir), 'utf8'));
  const errors = validateChapter(ch27Data, ch27Path);
  assert.deepEqual(errors, []);
  assert.equal(ch27Data.status, 'published');
  assert.equal(ch27Data.resources.practice.status, 'available');
  assert.equal(ch27Data.resources.reading.status, 'available');
  assert.equal(ch27Data.resources.slides.status, 'available');
});

test('chapter 28 passes full publication audit and validation', () => {
  const ch28Dir = new URL('../content/level-3-deep-learning/chapter-28/', import.meta.url);
  const ch28Path = fileURLToPath(ch28Dir);
  const ch28Data = JSON.parse(readFileSync(new URL('metadata.json', ch28Dir), 'utf8'));
  const errors = validateChapter(ch28Data, ch28Path);
  assert.deepEqual(errors, []);
  assert.equal(ch28Data.status, 'published');
  assert.equal(ch28Data.resources.practice.status, 'available');
  assert.equal(ch28Data.resources.reading.status, 'available');
  assert.equal(ch28Data.resources.slides.status, 'available');
});

test('chapter 29 passes full publication audit and validation', () => {
  const ch29Dir = new URL('../content/level-3-deep-learning/chapter-29/', import.meta.url);
  const ch29Path = fileURLToPath(ch29Dir);
  const ch29Data = JSON.parse(readFileSync(new URL('metadata.json', ch29Dir), 'utf8'));
  const errors = validateChapter(ch29Data, ch29Path);
  assert.deepEqual(errors, []);
  assert.equal(ch29Data.status, 'published');
  assert.equal(ch29Data.resources.practice.status, 'available');
  assert.equal(ch29Data.resources.reading.status, 'available');
  assert.equal(ch29Data.resources.slides.status, 'available');
});

test('chapter 30 passes full publication audit and validation', () => {
  const ch30Dir = new URL('../content/level-3-deep-learning/chapter-30/', import.meta.url);
  const ch30Path = fileURLToPath(ch30Dir);
  const ch30Data = JSON.parse(readFileSync(new URL('metadata.json', ch30Dir), 'utf8'));
  const errors = validateChapter(ch30Data, ch30Path);
  assert.deepEqual(errors, []);
  assert.equal(ch30Data.status, 'published');
  assert.equal(ch30Data.resources.practice.status, 'available');
  assert.equal(ch30Data.resources.reading.status, 'available');
  assert.equal(ch30Data.resources.slides.status, 'available');
});

test('chapter 31 passes full publication audit and validation', () => {
  const ch31Dir = new URL('../content/level-4-advanced-ai/chapter-31/', import.meta.url);
  const ch31Path = fileURLToPath(ch31Dir);
  const ch31Data = JSON.parse(readFileSync(new URL('metadata.json', ch31Dir), 'utf8'));
  const errors = validateChapter(ch31Data, ch31Path);
  assert.deepEqual(errors, []);
  assert.equal(ch31Data.status, 'published');
  assert.equal(ch31Data.resources.practice.status, 'available');
  assert.equal(ch31Data.resources.reading.status, 'available');
  assert.equal(ch31Data.resources.slides.status, 'available');
});

test('chapter 32 passes full publication audit and validation', () => {
  const ch32Dir = new URL('../content/level-4-advanced-ai/chapter-32/', import.meta.url);
  const ch32Path = fileURLToPath(ch32Dir);
  const ch32Data = JSON.parse(readFileSync(new URL('metadata.json', ch32Dir), 'utf8'));
  const errors = validateChapter(ch32Data, ch32Path);
  assert.deepEqual(errors, []);
  assert.equal(ch32Data.status, 'published');
  assert.equal(ch32Data.resources.practice.status, 'available');
  assert.equal(ch32Data.resources.reading.status, 'available');
  assert.equal(ch32Data.resources.slides.status, 'available');
});

test('chapter 33 passes full publication audit and validation', () => {
  const ch33Dir = new URL('../content/level-4-advanced-ai/chapter-33/', import.meta.url);
  const ch33Path = fileURLToPath(ch33Dir);
  const ch33Data = JSON.parse(readFileSync(new URL('metadata.json', ch33Dir), 'utf8'));
  const errors = validateChapter(ch33Data, ch33Path);
  assert.deepEqual(errors, []);
  assert.equal(ch33Data.status, 'published');
  assert.equal(ch33Data.resources.practice.status, 'available');
  assert.equal(ch33Data.resources.reading.status, 'available');
  assert.equal(ch33Data.resources.slides.status, 'available');
});

test('chapter 34 passes full publication audit and validation', () => {
  const ch34Dir = new URL('../content/level-4-advanced-ai/chapter-34/', import.meta.url);
  const ch34Path = fileURLToPath(ch34Dir);
  const ch34Data = JSON.parse(readFileSync(new URL('metadata.json', ch34Dir), 'utf8'));
  const errors = validateChapter(ch34Data, ch34Path);
  assert.deepEqual(errors, []);
  assert.equal(ch34Data.status, 'published');
  assert.equal(ch34Data.resources.practice.status, 'available');
  assert.equal(ch34Data.resources.reading.status, 'available');
  assert.equal(ch34Data.resources.slides.status, 'available');
});

test('chapter 35 passes full publication audit and validation', () => {
  const ch35Dir = new URL('../content/level-4-advanced-ai/chapter-35/', import.meta.url);
  const ch35Path = fileURLToPath(ch35Dir);
  const ch35Data = JSON.parse(readFileSync(new URL('metadata.json', ch35Dir), 'utf8'));
  const errors = validateChapter(ch35Data, ch35Path);
  assert.deepEqual(errors, []);
  assert.equal(ch35Data.status, 'published');
  assert.equal(ch35Data.resources.practice.status, 'available');
  assert.equal(ch35Data.resources.reading.status, 'available');
  assert.equal(ch35Data.resources.slides.status, 'available');
});

test('chapter 36 passes full publication audit and validation', () => {
  const ch36Dir = new URL('../content/level-4-advanced-ai/chapter-36/', import.meta.url);
  const ch36Path = fileURLToPath(ch36Dir);
  const ch36Data = JSON.parse(readFileSync(new URL('metadata.json', ch36Dir), 'utf8'));
  const errors = validateChapter(ch36Data, ch36Path);
  assert.deepEqual(errors, []);
  assert.equal(ch36Data.status, 'published');
  assert.equal(ch36Data.resources.practice.status, 'available');
  assert.equal(ch36Data.resources.reading.status, 'available');
  assert.equal(ch36Data.resources.slides.status, 'available');
});

test('chapter 37 passes full publication audit and validation', () => {
  const ch37Dir = new URL('../content/level-4-advanced-ai/chapter-37/', import.meta.url);
  const ch37Path = fileURLToPath(ch37Dir);
  const ch37Data = JSON.parse(readFileSync(new URL('metadata.json', ch37Dir), 'utf8'));
  const errors = validateChapter(ch37Data, ch37Path);
  assert.deepEqual(errors, []);
  assert.equal(ch37Data.status, 'published');
  assert.equal(ch37Data.resources.practice.status, 'available');
  assert.equal(ch37Data.resources.reading.status, 'available');
  assert.equal(ch37Data.resources.slides.status, 'available');
});

test('chapter 38 passes full publication audit and validation', () => {
  const ch38Dir = new URL('../content/level-4-advanced-ai/chapter-38/', import.meta.url);
  const ch38Path = fileURLToPath(ch38Dir);
  const ch38Data = JSON.parse(readFileSync(new URL('metadata.json', ch38Dir), 'utf8'));
  const errors = validateChapter(ch38Data, ch38Path);
  assert.deepEqual(errors, []);
  assert.equal(ch38Data.status, 'published');
  assert.equal(ch38Data.resources.practice.status, 'available');
  assert.equal(ch38Data.resources.reading.status, 'available');
  assert.equal(ch38Data.resources.slides.status, 'available');
});

test('chapter 39 passes full publication audit and validation', () => {
  const ch39Dir = new URL('../content/level-4-advanced-ai/chapter-39/', import.meta.url);
  const ch39Path = fileURLToPath(ch39Dir);
  const ch39Data = JSON.parse(readFileSync(new URL('metadata.json', ch39Dir), 'utf8'));
  const errors = validateChapter(ch39Data, ch39Path);
  assert.deepEqual(errors, []);
  assert.equal(ch39Data.status, 'published');
  assert.equal(ch39Data.resources.practice.status, 'available');
  assert.equal(ch39Data.resources.reading.status, 'available');
  assert.equal(ch39Data.resources.slides.status, 'available');
});

test('chapter 40 passes full publication audit and validation', () => {
  const ch40Dir = new URL('../content/level-4-advanced-ai/chapter-40/', import.meta.url);
  const ch40Path = fileURLToPath(ch40Dir);
  const ch40Data = JSON.parse(readFileSync(new URL('metadata.json', ch40Dir), 'utf8'));
  const errors = validateChapter(ch40Data, ch40Path);
  assert.deepEqual(errors, []);
  assert.equal(ch40Data.status, 'published');
  assert.equal(ch40Data.resources.practice.status, 'available');
  assert.equal(ch40Data.resources.reading.status, 'available');
  assert.equal(ch40Data.resources.slides.status, 'available');
});

test('markdown parser ignores headings inside code fences and builds clean ToC', () => {
  const md = `# Judul Utama
Pengantar materi.

\`\`\`python
# Ini komentar python bukan heading
def test():
    pass
\`\`\`

## Sub Judul
Penjelasan konsep.`;

  const { blocks, toc } = parseMarkdownBlocks(md);
  assert.equal(toc.length, 2);
  assert.equal(toc[0].text, 'Judul Utama');
  assert.equal(toc[1].text, 'Sub Judul');
  assert.ok(!toc.some(t => t.text.includes('komentar python')));
  assert.equal(blocks.some(b => b.type === 'code'), true);
});

test('markdown parser parses tables, lists, blockquotes, and math blocks', () => {
  const md = `| Kolom A | Kolom B |
|---|---|
| Nilai 1 | Nilai 2 |

- Poin 1
- Poin 2

1. Langkah 1
2. Langkah 2

> Catatan penting

$$
\\hat{y} = w x + b
$$`;

  const { blocks } = parseMarkdownBlocks(md);
  const types = blocks.map(b => b.type);
  assert.ok(types.includes('table'));
  assert.ok(types.includes('ul'));
  assert.ok(types.includes('ol'));
  assert.ok(types.includes('quote'));
  assert.ok(types.includes('math'));

  const tableBlock = blocks.find(b => b.type === 'table');
  assert.deepEqual(tableBlock?.headers, ['Kolom A', 'Kolom B']);
  assert.deepEqual(tableBlock?.rows, [['Nilai 1', 'Nilai 2']]);
});

test('inline tokenizer handles math, code, links, bold, and italic', () => {
  const text = 'Belajar **Machine Learning** dengan *Python* dan rumus $y = wx$ serta `import numpy` di [Google](https://google.com).';
  const tokens = parseInlineTokens(text);
  assert.ok(tokens.some(t => t.type === 'bold'));
  assert.ok(tokens.some(t => t.type === 'italic'));
  assert.ok(tokens.some(t => t.type === 'math' && t.value === 'y = wx'));
  assert.ok(tokens.some(t => t.type === 'code' && t.value === 'import numpy'));
  assert.ok(tokens.some(t => t.type === 'link' && t.url === 'https://google.com'));
});

test('enhanced tokenizer handles LaTeX delimiters, underscores, and escaped currency', () => {
  const text = 'Rumus inline \\( \\Delta DP < 0.10 \\) dan display \\[ f(x) = x^2 \\] serta __tebal__ dan _miring_ dan harga \\$100 bukan math.';
  const tokens = parseInlineTokens(text);
  assert.ok(tokens.some(t => t.type === 'math' && t.value === '\\Delta DP < 0.10'));
  assert.ok(tokens.some(t => t.type === 'math' && t.value === 'f(x) = x^2'));
  assert.ok(tokens.some(t => t.type === 'bold'));
  assert.ok(tokens.some(t => t.type === 'italic'));
  assert.ok(tokens.some(t => t.type === 'text' && t.value === '$'));
});

test('markdown parser handles bracket math blocks, horizontal rules, and deep headings', () => {
  const md = `---
##### Sub-heading level 5
\\[
P(A|B) = \\frac{P(B|A)P(A)}{P(B)}
\\]
***`;
  const { blocks, toc } = parseMarkdownBlocks(md);
  assert.ok(blocks.some(b => b.type === 'hr'));
  assert.ok(blocks.some(b => b.type === 'heading' && b.level === 5));
  assert.ok(blocks.some(b => b.type === 'math'));
  assert.equal(toc.some(t => t.text === 'Sub-heading level 5' && t.level === 5), true);
});

