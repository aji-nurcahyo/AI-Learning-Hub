import { spawn } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { root } from './validate.mjs';

const args = new Set(process.argv.slice(2));
const capturePrefix = [...args].find(a => a.startsWith('--prefix='))?.split('=')[1] || '';
const saveShots = args.has('--screenshots') || Boolean(capturePrefix);
const full40Sweep = args.has('--sweep40') || !args.has('--quick');

const edgeCandidates = [
  'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
  'C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe',
  'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
];
const browserBin = edgeCandidates.find(p => existsSync(p));
if (!browserBin) {
  console.error('FAIL: Neither Microsoft Edge nor Chrome found for headless browser theme audit.');
  process.exit(1);
}

const reportDir = resolve(root, 'reports/theme-audit');
mkdirSync(reportDir, { recursive: true });

const previewPort = 8971;
const cdpPort = 9249;
const preview = spawn(process.execPath, ['node_modules/vite/bin/vite.js', 'preview', '--host', '127.0.0.1', '--port', String(previewPort), '--strictPort'], { stdio: 'pipe', windowsHide: true });
const userDataDir = mkdtempSync(join(tmpdir(), 'agy-theme-'));
const browser = spawn(browserBin, ['--headless=new', '--disable-gpu', '--disable-extensions', '--no-first-run', '--no-default-browser-check', `--remote-debugging-port=${cdpPort}`, `--user-data-dir=${userDataDir}`, 'about:blank'], { stdio: 'pipe', windowsHide: true });

async function waitUp(url) {
  for (let i = 0; i < 60; i++) {
    try { const r = await fetch(url); if (r.ok) return r; } catch {}
    await new Promise(r => setTimeout(r, 150));
  }
  throw new Error('Timeout waiting for ' + url);
}

const representativeRoutes = [
  { name: 'homepage', hash: '#/' },
  { name: 'chapter-01', hash: '#chapter/1' },
  { name: 'chapter-27', hash: '#chapter/27' },
  { name: 'chapter-39', hash: '#chapter/39' },
  { name: 'chapter-40', hash: '#chapter/40' },
  { name: 'chapter-27-read', hash: '#chapter/27/read' },
  { name: 'slide-28', hash: '#chapter/28/slides' },
  { name: 'slide-35', hash: '#chapter/35/slides' },
  { name: 'chapter-37-practice', hash: '#chapter/37/practice' },
  { name: 'chapter-38-read', hash: '#chapter/38/read' },
  { name: 'slide-39', hash: '#chapter/39/slides' },
  { name: 'chapter-40-practice', hash: '#chapter/40/practice' },
  { name: 'track-computer-vision', hash: '#track/computer-vision' },
  { name: 'track-nlp-llm', hash: '#track/nlp-llm' },
  { name: 'track-aiot-edge', hash: '#track/aiot-edge' },
  { name: 'track-telecommunications', hash: '#track/telecommunications' },
  { name: 'track-robotics', hash: '#track/robotics' },
  { name: 'track-telecommunications-practice', hash: '#track/telecommunications/practice' },
  { name: 'track-nlp-llm-practice', hash: '#track/nlp-llm/practice' },
  { name: 'track-computer-vision-capstone', hash: '#track/computer-vision/capstone' },
  { name: 'track-robotics-capstone', hash: '#track/robotics/capstone' },
];

const shotTargets = new Set(['homepage', 'chapter-27', 'track-telecommunications', 'slide-28']);

const viewports = [
  { name: 'desktop', width: 1440, height: 900, mobile: false },
  { name: 'tablet', width: 768, height: 1024, mobile: false },
  { name: 'mobile', width: 390, height: 844, mobile: true },
];

const AUDIT_EXPRESSION = `(() => {
  const parseRgba = (str) => {
    if (!str || str === 'transparent') return [0, 0, 0, 0];
    const m = str.match(/rgba?\\(\\s*(\\d+(?:\\.\\d+)?),\\s*(\\d+(?:\\.\\d+)?),\\s*(\\d+(?:\\.\\d+)?)(?:,\\s*(\\d+(?:\\.\\d+)?))?\\s*\\)/);
    if (!m) return [0, 0, 0, 0];
    return [Number(m[1]), Number(m[2]), Number(m[3]), m[4] === undefined ? 1 : Number(m[4])];
  };
  const compositeOver = (fg, bg) => {
    const a = fg[3];
    if (a >= 0.999) return [fg[0], fg[1], fg[2], 1];
    return [
      fg[0] * a + bg[0] * (1 - a),
      fg[1] * a + bg[1] * (1 - a),
      fg[2] * a + bg[2] * (1 - a),
      1,
    ];
  };
  const relLum = ([r, g, b]) => {
    const conv = v => {
      const c = v / 255;
      return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
    };
    return 0.2126 * conv(r) + 0.7152 * conv(g) + 0.0722 * conv(b);
  };
  const contrastRatio = (c1, c2) => {
    const l1 = relLum(c1);
    const l2 = relLum(c2);
    const hi = Math.max(l1, l2);
    const lo = Math.min(l1, l2);
    return (hi + 0.05) / (lo + 0.05);
  };
  const effectiveBg = (el) => {
    const stack = [];
    let cur = el;
    while (cur && cur.nodeType === 1) {
      const st = getComputedStyle(cur);
      const rgba = parseRgba(st.backgroundColor);
      if (rgba[3] > 0.01) stack.push(rgba);
      cur = cur.parentElement;
    }
    let acc = [248, 250, 252, 1];
    const htmlBg = parseRgba(getComputedStyle(document.documentElement).backgroundColor);
    const bodyBg = parseRgba(getComputedStyle(document.body).backgroundColor);
    if (htmlBg[3] > 0.5) acc = [htmlBg[0], htmlBg[1], htmlBg[2], 1];
    if (bodyBg[3] > 0.5) acc = compositeOver(bodyBg, acc);
    for (let i = stack.length - 1; i >= 0; i--) {
      acc = compositeOver(stack[i], acc);
    }
    return acc;
  };
  const isVisible = (el) => {
    const st = getComputedStyle(el);
    if (st.display === 'none' || st.visibility === 'hidden' || Number(st.opacity) === 0) return false;
    const rect = el.getBoundingClientRect();
    if (rect.width < 1 || rect.height < 1) return false;
    return true;
  };
  const selectorFor = (el) => {
    if (el.id) return '#' + el.id;
    const cls = typeof el.className === 'string' && el.className.trim()
      ? '.' + el.className.trim().split(/\\s+/).slice(0, 2).join('.')
      : '';
    return el.tagName.toLowerCase() + cls;
  };

  const issues = [];
  const sampleRatios = {};

  // Check text-bearing elements & UI controls
  const candidates = document.querySelectorAll('h1,h2,h3,h4,p,a,button,span,small,strong,label,th,td,li,code,select');
  for (const el of candidates) {
    if (!isVisible(el)) continue;
    // Skip Reveal non-present slides
    const parentSlide = el.closest('section.lesson-slide');
    if (parentSlide && !parentSlide.classList.contains('present')) continue;
    // Only evaluate leaf-ish elements or elements with direct text node content
    const hasDirectText = [...el.childNodes].some(n => n.nodeType === 3 && n.textContent.trim().length > 0);
    if (!hasDirectText && !['SELECT'].includes(el.tagName)) continue;

    const st = getComputedStyle(el);
    const fgRaw = parseRgba(st.color);
    const bg = effectiveBg(el);
    const fg = compositeOver(fgRaw, bg);
    const ratio = contrastRatio(fg, bg);
    const fs = parseFloat(st.fontSize) || 16;
    const fw = parseInt(st.fontWeight, 10) || 400;
    const isLarge = fs >= 24 || (fs >= 18.67 && fw >= 700);
    const threshold = isLarge ? 3.0 : 4.5;
    const text = (el.innerText || el.value || '').trim().replace(/\\s+/g, ' ').slice(0, 60);
    if (!text) continue;

    // Classify decorative watermark like .cover-number
    if (el.classList.contains('cover-number')) continue;

    const sel = selectorFor(el);
    if (!sampleRatios[sel]) sampleRatios[sel] = Number(ratio.toFixed(2));

    if (ratio < threshold) {
      const hasComplexBg = st.backgroundImage && st.backgroundImage !== 'none';
      const severity = ratio < 2.2 ? 'critical' : 'high';
      const classification = hasComplexBg ? 'potential_failure' : 'confirmed_failure';
      issues.push({
        selector: sel,
        text,
        fg: fg.slice(0, 3).map(Math.round),
        bg: bg.slice(0, 3).map(Math.round),
        ratio: Number(ratio.toFixed(2)),
        threshold,
        severity,
        classification,
      });
    }
  }

  // Check accessible names for icon-only buttons
  const unlabeledButtons = [];
  for (const btn of document.querySelectorAll('button')) {
    if (!isVisible(btn)) continue;
    const txt = (btn.innerText || '').trim();
    const aria = (btn.getAttribute('aria-label') || btn.getAttribute('title') || '').trim();
    if (!txt && !aria) unlabeledButtons.push(selectorFor(btn));
  }

  const hOverflow = document.documentElement.scrollWidth > window.innerWidth + 1;
  const katexErrors = document.querySelectorAll('.katex-error').length;
  const brokenImages = [...document.querySelectorAll('img')].filter(img => img.complete && img.naturalWidth === 0).length;
  const presentSlide = document.querySelector('.reveal .slides section.present');
  const slideOverflow = presentSlide ? (presentSlide.scrollHeight > presentSlide.clientHeight + 6 || presentSlide.scrollWidth > presentSlide.clientWidth + 6) : false;

  return {
    themeAttr: document.documentElement.dataset.theme,
    hOverflow,
    katexErrors,
    brokenImages,
    slideOverflow,
    unlabeledButtons,
    issues,
    sampleRatios,
  };
})()`;

try {
  await waitUp(`http://127.0.0.1:${previewPort}/ai-learning-hub/`);
  const targets = await (await waitUp(`http://127.0.0.1:${cdpPort}/json/list`)).json();
  const ws = new WebSocket((targets.find(t => t.type === 'page') || targets[0]).webSocketDebuggerUrl);
  await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });

  let seq = 1;
  const pending = new Map();
  let curCtx = '';
  const jsErrors = [];
  ws.onmessage = ev => {
    const msg = JSON.parse(ev.data);
    if (msg.id && pending.has(msg.id)) { pending.get(msg.id)(msg); pending.delete(msg.id); }
    else if (msg.method === 'Runtime.exceptionThrown') {
      const det = msg.params?.exceptionDetails;
      jsErrors.push({ context: curCtx, text: det?.exception?.description || det?.text || 'exception' });
    } else if (msg.method === 'Runtime.consoleAPICalled' && msg.params?.type === 'error') {
      jsErrors.push({ context: curCtx, text: msg.params?.args?.map(a => a.value || a.description).join(' ') });
    }
  };
  const send = (method, params = {}) => new Promise(res => { const id = seq++; pending.set(id, res); ws.send(JSON.stringify({ id, method, params })); });
  const waitRouteSettled = async (isSlide = false, minMs = 280) => {
    await new Promise(r => setTimeout(r, minMs));
    for (let i = 0; i < 20; i++) {
      const st = await send('Runtime.evaluate', {
        expression: `(() => document.readyState === 'complete' && !document.querySelector('[aria-busy="true"]') && (${isSlide} ? Boolean(document.querySelector('.reveal.ready, .reveal .slides section.present')) : Boolean(document.querySelector('main, header, .ai-main, h1'))))()`,
        returnByValue: true,
      });
      if (st.result?.result?.value) break;
      await new Promise(r => setTimeout(r, 100));
    }
    await new Promise(r => setTimeout(r, 120));
  };
  await send('Page.enable');
  await send('Runtime.enable');

  const auditRows = [];
  let confirmedTotal = 0;
  let potentialTotal = 0;

  for (const theme of ['light', 'dark']) {
    for (const vp of viewports) {
      await send('Emulation.setDeviceMetricsOverride', { width: vp.width, height: vp.height, deviceScaleFactor: vp.mobile ? 2 : 1, mobile: vp.mobile });
      // For mobile/tablet check key subset; for desktop check all representative routes
      const routesToCheck = vp.name === 'desktop'
        ? representativeRoutes
        : representativeRoutes.filter(r => ['homepage', 'chapter-27', 'chapter-27-read', 'slide-28', 'track-telecommunications', 'track-robotics-capstone'].includes(r.name));

      for (const r of routesToCheck) {
        curCtx = `${theme}/${vp.name}/${r.hash}`;
        const url = `http://127.0.0.1:${previewPort}/ai-learning-hub/?theme=${theme}${r.hash}`;
        await send('Page.navigate', { url });
        await waitRouteSettled(r.hash.includes('slides'), r.hash.includes('slides') ? 550 : 300);

        const evalRes = await send('Runtime.evaluate', { expression: AUDIT_EXPRESSION, returnByValue: true });
        const val = evalRes.result?.result?.value;
        if (!val) continue;

        const confirmed = val.issues.filter(x => x.classification === 'confirmed_failure');
        const potential = val.issues.filter(x => x.classification === 'potential_failure');
        confirmedTotal += confirmed.length;
        potentialTotal += potential.length;

        auditRows.push({
          route: r.hash,
          theme,
          viewport: `${vp.width}x${vp.height}`,
          hOverflow: val.hOverflow,
          slideOverflow: val.slideOverflow,
          katexErrors: val.katexErrors,
          brokenImages: val.brokenImages,
          unlabeledButtons: val.unlabeledButtons.length,
          confirmedFailures: confirmed,
          potentialFailures: potential,
          sampleRatios: val.sampleRatios,
        });

        if (saveShots && vp.name === 'desktop' && shotTargets.has(r.name)) {
          const shotRes = await send('Page.captureScreenshot', { format: 'png' });
          if (shotRes.result?.data) {
            const fname = capturePrefix ? `${capturePrefix}-${r.name}-${theme}.png` : `${r.name}-${theme}.png`;
            writeFileSync(join(reportDir, fname), Buffer.from(shotRes.result.data, 'base64'));
          }
        }
      }
    }
  }

  // Test Search + Interactive Theme Persistence + Keyboard Focus + Slide Layouts on Desktop
  await send('Emulation.setDeviceMetricsOverride', { width: 1440, height: 900, deviceScaleFactor: 1, mobile: false });
  curCtx = 'interactive-theme-persistence';
  await send('Page.navigate', { url: `http://127.0.0.1:${previewPort}/ai-learning-hub/` });
  await waitRouteSettled(false, 400);
  await send('Input.dispatchKeyEvent', { type: 'keyDown', key: 'Tab', code: 'Tab', windowsVirtualKeyCode: 9 });
  await send('Input.dispatchKeyEvent', { type: 'keyUp', key: 'Tab', code: 'Tab', windowsVirtualKeyCode: 9 });
  const persistenceRes = await send('Runtime.evaluate', {
    expression: `(() => {
      const sel = document.querySelector('select[aria-label="Pilih tema tampilan"]');
      sel.value = 'light';
      sel.dispatchEvent(new Event('change', { bubbles: true }));
      const storedLight = localStorage.getItem('ai-hub.theme');
      const attrLight = document.documentElement.dataset.theme;
      sel.value = 'dark';
      sel.dispatchEvent(new Event('change', { bubbles: true }));
      const storedDark = localStorage.getItem('ai-hub.theme');
      const attrDark = document.documentElement.dataset.theme;
      const inp = document.querySelector('input[type="search"]');
      const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
      setter.call(inp, 'Natural Language Processing');
      inp.dispatchEvent(new Event('input', { bubbles: true }));
      const searchMatch = document.body.innerText.includes('Natural Language Processing Fundamentals');
      const focusedStyle = document.activeElement ? getComputedStyle(document.activeElement).outlineStyle : 'none';
      const hasFocusStyle = focusedStyle !== 'none';
      return { storedLight, attrLight, storedDark, attrDark, searchMatch, hasFocusStyle };
    })()`,
    returnByValue: true,
  });

  // Check multiple Reveal slide layouts on Chapter 28 in Light Mode (content, math, code, columns, case)
  curCtx = 'slide-layouts-ch28';
  await send('Page.navigate', { url: `http://127.0.0.1:${previewPort}/ai-learning-hub/?theme=light#chapter/28/slides` });
  await waitRouteSettled(true, 600);
  const slideAdvanceAudit = await send('Runtime.evaluate', {
    expression: `(() => {
      if (!window.Reveal) return { checkedSlides: 0, failures: 0 };
      const total = window.Reveal.getTotalSlides?.() || 1;
      let checkedSlides = 0;
      let mathRendered = 0;
      let codeRendered = 0;
      let columnsRendered = 0;
      for (let i = 0; i < total; i++) {
        window.Reveal.slide(i);
        checkedSlides++;
        const pres = document.querySelector('.reveal .slides section.present');
        if (pres?.querySelector('.katex')) mathRendered++;
        if (pres?.querySelector('pre, code')) codeRendered++;
        if (pres?.querySelector('.slide-columns-grid')) columnsRendered++;
      }
      return { checkedSlides, mathRendered, codeRendered, columnsRendered, katexErrors: document.querySelectorAll('.katex-error').length };
    })()`,
    returnByValue: true,
  });

  // Lightweight route sweep of all 40 chapters (Desktop Light Mode)
  const sweep40Results = [];
  if (full40Sweep) {
    for (let cid = 1; cid <= 40; cid++) {
      curCtx = `sweep40/ch-${cid}`;
      await send('Page.navigate', { url: `http://127.0.0.1:${previewPort}/ai-learning-hub/?theme=light#chapter/${cid}` });
      await waitRouteSettled(false, 220);
      const sw = await send('Runtime.evaluate', { expression: AUDIT_EXPRESSION, returnByValue: true });
      const v = sw.result?.result?.value;
      const conf = (v?.issues || []).filter(x => x.classification === 'confirmed_failure');
      sweep40Results.push({
        chapter: cid,
        hOverflow: v?.hOverflow,
        confirmedCount: conf.length,
        firstFailure: conf[0] || null,
      });
    }
  }

  const report = {
    timestamp: new Date().toISOString(),
    routesMatrixAudited: auditRows.length,
    confirmedContrastFailures: confirmedTotal,
    potentialContrastFailures: potentialTotal,
    jsErrorsCount: jsErrors.length,
    jsErrors,
    overflowCount: auditRows.filter(r => r.hOverflow || r.slideOverflow).length,
    katexErrorsCount: auditRows.reduce((a, b) => a + b.katexErrors, 0),
    brokenImagesCount: auditRows.reduce((a, b) => a + b.brokenImages, 0),
    unlabeledButtonsCount: auditRows.reduce((a, b) => a + b.unlabeledButtons, 0),
    interactiveChecks: {
      ...(persistenceRes.result?.result?.value || {}),
      slideAdvanceAudit: slideAdvanceAudit.result?.result?.value || null,
    },
    sweep40ChaptersChecked: sweep40Results.length,
    sweep40Failures: sweep40Results.filter(x => x.confirmedCount > 0 || x.hOverflow),
    detailsWithIssues: auditRows.filter(r => r.confirmedFailures.length > 0 || r.potentialFailures.length > 0 || r.hOverflow || r.slideOverflow || r.katexErrors > 0),
    allSamples: auditRows.map(r => ({ route: r.route, theme: r.theme, viewport: r.viewport, ratios: r.sampleRatios })),
  };

  const jsonPath = join(reportDir, capturePrefix ? `theme-audit-${capturePrefix}.json` : 'theme-audit-report.json');
  writeFileSync(jsonPath, JSON.stringify(report, null, 2), 'utf8');

  console.log(`=== THEME AUDIT SUMMARY (${capturePrefix || 'current'}) ===`);
  console.log(`Matrix Checks : ${report.routesMatrixAudited} | Ch01-40 Sweep : ${report.sweep40ChaptersChecked}`);
  console.log(`Confirmed WCAG AA Failures : ${report.confirmedContrastFailures} | Potential : ${report.potentialContrastFailures}`);
  console.log(`Overflow Issues : ${report.overflowCount} | KaTeX Errors : ${report.katexErrorsCount} | JS Errors : ${report.jsErrorsCount}`);
  console.log(`Interactive Checks :`, report.interactiveChecks);
  console.log(`Report saved to : ${jsonPath}`);

  if (report.confirmedContrastFailures > 0 || report.jsErrorsCount > 0 || report.overflowCount > 0 || report.katexErrorsCount > 0 || report.sweep40Failures.length > 0) {
    console.log('\nSample Failures (first 10):');
    const seen = new Set();
    for (const row of report.detailsWithIssues) {
      for (const iss of row.confirmedFailures) {
        const sig = `${row.theme}|${iss.selector}|${iss.fg.join(',')}|${iss.bg.join(',')}`;
        if (seen.has(sig)) continue;
        seen.add(sig);
        console.log(`  [${row.theme} ${row.route}] ${iss.selector} ("${iss.text}") fg=${iss.fg} bg=${iss.bg} ratio=${iss.ratio} < ${iss.threshold} (${iss.severity})`);
        if (seen.size >= 10) break;
      }
      if (seen.size >= 10) break;
    }
    if (report.sweep40Failures.length > 0) {
      console.log(`Sweep40 failing chapters count: ${report.sweep40Failures.length}, example:`, report.sweep40Failures[0]);
    }
    process.exitCode = 1;
  }

  ws.close();
} finally {
  browser.kill();
  preview.kill();
  try { rmSync(userDataDir, { recursive: true, force: true }); } catch {}
}

