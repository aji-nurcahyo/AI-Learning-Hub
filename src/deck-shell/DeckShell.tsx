import { useEffect, useRef, useState, useCallback } from 'react';
import Reveal, { type RevealApi } from 'reveal.js';
import Notes from 'reveal.js/plugin/notes';
import 'reveal.js/reveal.css';
import { SlideContent } from '../layouts/SlideContent';
import type { SlideData } from '../decks/registry';
import { ThemeSelect, usePreferences, type Ratio } from './preferences';
import { Icon } from '../ui/Icon';

export function DeckShell({
  slides,
  meeting,
  base,
  onNavigateChapter,
  onNavigateReading,
  onNavigateTask,
}: {
  slides: SlideData[];
  meeting: number;
  base: string;
  onNavigateChapter?: () => void;
  onNavigateReading?: () => void;
  onNavigateTask?: () => void;
}) {
  const host = useRef<HTMLDivElement>(null);
  const deck = useRef<RevealApi | null>(null);
  const { theme, setTheme, ratio, setRatio } = usePreferences();
  const [ready, setReady] = useState(false);
  const [help, setHelp] = useState(false);
  const [error, setError] = useState('');
  const printing = new URLSearchParams(location.search).has('print-pdf');

  useEffect(() => {
    if (!host.current) return;
    const api = new Reveal(host.current, {
      plugins: [Notes],
      hash: false, // CRITICAL: Disabled to prevent Reveal from overwriting SPA hash route (#/chapter/X/slides)
      respondToHashChanges: false,
      history: false,
      controls: true,
      controlsLayout: 'edges',
      progress: true,
      slideNumber: 'c/t',
      center: false,
      transition: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'none' : 'fade',
      width: 1280,
      height: 720,
      margin: 0.06,
      minScale: 0.15,
      maxScale: 2,
      pdfMaxPagesPerSlide: 1,
      pdfSeparateFragments: false,
    });
    deck.current = api;
    (window as unknown as { Reveal?: RevealApi }).Reveal = api;
    void api
      .initialize()
      .then(() => {
        setReady(true);
      })
      .catch(() => setError('Presentasi gagal dimuat. Silakan muat ulang halaman.'));

    return () => {
      deck.current = null;
      delete (window as unknown as { Reveal?: RevealApi }).Reveal;
      void api.destroy();
    };
  }, []);

  useEffect(() => {
    if (!ready) return;
    const resize = () => {
      const wide = ratio === '16x9' || (ratio === 'auto' && innerWidth / innerHeight >= 1.45);
      deck.current?.configure({ width: 1280, height: wide ? 720 : 960 });
      document.documentElement.dataset.ratio = wide ? '16x9' : '4x3';
      deck.current?.layout();
    };
    resize();
    addEventListener('resize', resize);
    return () => removeEventListener('resize', resize);
  }, [ratio, ready]);

  useEffect(() => {
    if (ready && deck.current) {
      deck.current.layout();
    }
  }, [theme, ready]);

  const toggleFullscreen = useCallback(async () => {
    try {
      if (document.fullscreenElement) {
        await document.exitFullscreen();
      } else {
        await document.documentElement.requestFullscreen();
      }
    } catch {
      setError('Layar penuh tidak didukung di browser ini.');
    }
  }, []);

  const openSpeakerNotes = useCallback(() => {
    const plugin = deck.current?.getPlugin('notes') as { open?: () => void } | undefined;
    plugin?.open?.();
  }, []);

  const toggleOverview = useCallback(() => {
    deck.current?.toggleOverview();
  }, []);

  // Global presentation keyboard shortcuts
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      if (['INPUT', 'SELECT', 'TEXTAREA'].includes(target.tagName)) return;

      if (e.key === 'f' || e.key === 'F') {
        e.preventDefault();
        void toggleFullscreen();
      } else if (e.key === 'o' || e.key === 'O') {
        e.preventDefault();
        toggleOverview();
      } else if (e.key === 's' || e.key === 'S') {
        e.preventDefault();
        openSpeakerNotes();
      } else if (e.key === '?') {
        e.preventDefault();
        setHelp((prev) => !prev);
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [toggleFullscreen, toggleOverview, openSpeakerNotes]);

  function print() {
    const url = new URL(location.href);
    url.searchParams.set('print-pdf', '');
    // Preserve the chapter route when printing.
    location.href = url.href;
  }

  return (
    <div className={`deck-page ${printing ? 'print-mode' : ''}`}>
      <header className="deck-toolbar">
        <a
          className="brand compact"
          href={`${base}index.html#chapter/${meeting}`}
          onClick={(e) => {
            if (onNavigateChapter) {
              e.preventDefault();
              onNavigateChapter();
            }
          }}
          aria-label={`Kembali ke Chapter ${meeting}`}
        >
          <Icon name="arrow_back" size={18} /> <span>Chapter {meeting}</span>
        </a>
        <span className="toolbar-meeting">
          <Icon name="slideshow" size={16} /> Chapter {meeting}
        </span>
        <div className="toolbar-actions">
          <a
            href={`${base}index.html#chapter/${meeting}/read`}
            onClick={(e) => {
              if (onNavigateReading) {
                e.preventDefault();
                onNavigateReading();
              }
            }}
            title="Mode Dokumen Baca Responsif"
          >
            <Icon name="menu_book" size={16} /> <span>Baca</span>
          </a>
          <a
            href={`${base}index.html#chapter/${meeting}/task`}
            onClick={(e) => {
              if (onNavigateTask) {
                e.preventDefault();
                onNavigateTask();
              }
            }}
            title="Lihat Instruksi Tugas"
          >
            <Icon name="assignment" size={16} /> <span>Tugas</span>
          </a>
          <ThemeSelect theme={theme} onChange={setTheme} />
          <label className="select-label">
            <Icon name="aspect_ratio" size={16} />
            <span>Rasio</span>
            <select
              aria-label="Rasio slide"
              value={ratio}
              onChange={(e) => setRatio(e.target.value as Ratio)}
            >
              <option value="auto">Otomatis</option>
              <option value="16x9">16:9</option>
              <option value="4x3">4:3</option>
            </select>
          </label>
          <button onClick={toggleOverview} title="Ikhtisar Semua Slide (O / Esc)">
            <Icon name="grid_view" size={16} /> <span>Ikhtisar</span>
          </button>
          <button onClick={openSpeakerNotes} title="Speaker View & Catatan Pembicara (S)">
            <Icon name="record_voice_over" size={16} /> <span>Pembicara</span>
          </button>
          <button onClick={toggleFullscreen} title="Layar Penuh Proyektor (F)">
            <Icon name="fullscreen" size={16} /> <span>Layar Penuh</span>
          </button>
          <button onClick={print} title="Cetak atau Ekspor PDF">
            <Icon name="picture_as_pdf" size={16} /> <span>PDF</span>
          </button>
          <button aria-label="Bantuan shortcut keyboard (?)" onClick={() => setHelp(!help)}>
            <Icon name="help_outline" size={16} />
          </button>
        </div>
      </header>

      {error && <p className="toolbar-error" role="alert">{error}</p>}

      {help && (
        <div className="help-panel" role="region" aria-label="Bantuan shortcut presentasi">
          <h2>
            <Icon name="keyboard" size={24} /> Shortcut Keyboard Presentasi
          </h2>
          <ul>
            <li><strong>← / → / Spasi</strong> : Berpindah slide</li>
            <li><strong>O / Esc</strong> : Buka / tutup ikhtisar seluruh slide (Slide Overview)</li>
            <li><strong>S</strong> : Buka jendela pembicara (Speaker Notes & View)</li>
            <li><strong>F</strong> : Aktifkan / matikan layar penuh (Fullscreen)</li>
            <li><strong>B / .</strong> : Jeda layar (Blank/Black Screen)</li>
            <li><strong>Home / End</strong> : Langsung ke slide pertama / terakhir</li>
            <li><strong>?</strong> : Tampilkan / sembunyikan bantuan ini</li>
          </ul>
          <p>
            Untuk ekspor PDF, klik tombol <strong>PDF</strong> lalu pilih <em>Save as PDF</em> (orientasi landscape).
          </p>
          <button autoFocus onClick={() => setHelp(false)}>
            Tutup Bantuan
          </button>
        </div>
      )}

      {printing && (
        <div className="print-banner">
          <a href={`${location.pathname}${location.hash}`}>
            <Icon name="arrow_back" size={16} /> Kembali ke slide interaktif
          </a>
          <button onClick={() => window.print()}>
            <Icon name="print" size={16} /> Cetak / Simpan PDF
          </button>
          <span>Tip: Pilih printer &ldquo;Save as PDF&rdquo;, orientasi Landscape, dan hilangkan opsi header/footer.</span>
        </div>
      )}

      <div
        ref={host}
        className="reveal"
        tabIndex={0}
        aria-label={`Presentasi AI Learning Hub Chapter ${meeting}`}
      >
        <div className="slides">
          {slides.map((slide, i) => (
            <section key={i} className={`lesson-slide ${slide.kind}`}>
              <SlideContent slide={slide} meeting={meeting} base={base} />
            </section>
          ))}
        </div>
      </div>
    </div>
  );
}
