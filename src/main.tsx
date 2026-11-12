import {useEffect,useState,lazy,Suspense} from 'react';
import {createRoot} from 'react-dom/client';
import {chapters,levels,tracks,resourceText,gettingStartedModules,gettingStartedText,type Chapter} from './content';
import {parseRoute} from './routes';
import {ThemeSelect,usePreferences} from './deck-shell/preferences';
import {templateSlides,type SlideData} from './decks/registry';
const Reading=lazy(()=>import('./components/Reading').then(m=>({default:m.Reading}))); 
import '@fontsource-variable/google-sans-flex';
import '@fontsource-variable/google-sans-code';
import '@fontsource-variable/material-symbols-rounded';
import 'katex/dist/katex.min.css';
import './theme/styles.css';
import './theme/ai.css';
const DeckShell=lazy(()=>import('./deck-shell/DeckShell').then(m=>({default:m.DeckShell})));
const base=import.meta.env.BASE_URL;

function ChapterCards({items}:{items:Chapter[]}){
  return (
    <div className="chapter-list">
      {items.map(c=>(
        <a className="chapter-row" href={`#chapter/${c.id}`} key={c.id}>
          <span className="chapter-num">{String(c.id).padStart(2,'0')}</span>
          <div>
            <small>LEVEL {c.levelId}</small>
            <h3>{c.title}</h3>
            <p>{c.description}</p>
          </div>
          <span className="status-pill">{c.status==='published'?'Tersedia':'Coming Soon'}</span>
          <span aria-hidden="true">↗</span>
        </a>
      ))}
    </div>
  );
}

function Browser(){
  const [query,setQuery]=useState('');
  const [level,setLevel]=useState('all');
  const [status,setStatus]=useState('all');
  const filtered=chapters.filter(c=>(level==='all'||c.levelId===Number(level))&&(status==='all'||c.status===status)&&`${c.id} ${String(c.id).padStart(2,'0')} ${c.title} ${c.description} ${c.tags.join(' ')}`.toLowerCase().includes(query.trim().toLowerCase()));

  return (
    <section id="curriculum">
      <div className="section-heading">
        <div>
          <div className="eyebrow">PETA PEMBELAJARAN</div>
          <h2>40 chapter. Satu fondasi yang kuat.</h2>
        </div>
        <p>Pilih topik sesuai kebutuhan. Semua materi yang diterbitkan terbuka untuk siapa saja.</p>
      </div>
      <div className="filters">
        <label>
          Cari chapter
          <input type="search" value={query} onChange={e=>setQuery(e.target.value)} placeholder="Search, Neural Network, RAG…"/>
        </label>
        <label>
          Level
          <select value={level} onChange={e=>setLevel(e.target.value)}>
            <option value="all">Semua level</option>
            {levels.map(l=><option key={l.id} value={l.id}>{l.id} · {l.title}</option>)}
          </select>
        </label>
        <label>
          Ketersediaan
          <select value={status} onChange={e=>setStatus(e.target.value)}>
            <option value="all">Semua status</option>
            <option value="published">Tersedia</option>
            <option value="coming-soon">Coming Soon</option>
          </select>
        </label>
      </div>
      <p aria-live="polite">{filtered.length} chapter ditemukan</p>
      <ChapterCards items={filtered}/>
      {!filtered.length&&<div className="notice">Belum ada chapter yang sesuai. Coba kata kunci lain atau ubah filter.</div>}
    </section>
  );
}

function Home(){
  return (
    <>
      <section className="ai-hero">
        <div>
          <div className="eyebrow">TERBUKA UNTUK BELAJAR · DIRANCANG UNTUK BERKARYA</div>
          <h1>Artificial Intelligence<br/><em>Learning Hub</em></h1>
          <h2>From AI Fundamentals to Intelligent Systems</h2>
          <p>Jelajahi Artificial Intelligence, Machine Learning, Deep Learning, dan Generative AI melalui kurikulum terstruktur dan pengembangan intelligent systems yang bertanggung jawab.</p>
          <div className="hero-cta-group">
            <a className="button primary button-lg" href="#curriculum">Jelajahi kurikulum ↗</a>
            <a className="button subtle button-lg" href="#getting-started">Getting Started ↗</a>
            <a className="button subtle button-lg" href="#about">Tentang platform</a>
          </div>
          <div className="hero-facts">
            <span><b>08</b> Modul setup</span>
            <span><b>04</b> Level kompetensi</span>
            <span><b>40</b> Chapter terstruktur</span>
            <span><b>05</b> Jalur spesialisasi</span>
          </div>
        </div>
        <div className="network-art" aria-hidden="true">
          <svg viewBox="0 0 420 420">
            <defs>
              <radialGradient id="glow">
                <stop stopColor="#30bfa5" stopOpacity=".25"/>
                <stop offset="1" stopColor="#30bfa5" stopOpacity="0"/>
              </radialGradient>
            </defs>
            <circle cx="210" cy="210" r="205" fill="url(#glow)"/>
            {[95,150,195].map(r=><circle key={r} cx="210" cy="210" r={r} fill="none" stroke="currentColor" opacity=".18"/>)}
            {[0,1,2,3].flatMap(a=>[0,1,2,3].map(b=><path key={`${a}-${b}`} d={`M 100 ${90+a*80} L 210 ${70+b*90} L 320 ${90+a*80}`} fill="none" stroke="currentColor" opacity=".24"/>))}
            {[100,210,320].flatMap((x,a)=>[0,1,2,3].map(b=><circle key={`${x}-${b}`} cx={x} cy={(a===1?70:90)+b*(a===1?90:80)} r={a===1?10:7} fill="var(--primary)"/>))}
          </svg>
          <span>UNDERSTAND → APPLY → CREATE</span>
        </div>
      </section>

      <div className="notice">
        <strong>Panduan Lingkungan Belajar Tersedia.</strong> Sebelum memulai materi, pelajari panduan instalasi Python, VS Code, virtual environment, dan stack data science pada bagian <a href="#getting-started">Getting Started</a>. Chapter 01 telah tersedia lengkap untuk dipelajari.
      </div>

      <section>
        <div className="section-heading">
          <div>
            <div className="eyebrow">KURIKULUM BERJENJANG</div>
            <h2>Dari konsep ke sistem nyata.</h2>
          </div>
          <p>Empat level untuk membangun kompetensi. Prasyarat adalah rekomendasi, bukan batas akses.</p>
        </div>
        <div className="ai-level-grid">
          {levels.map(l=>(
            <a className={`level-card level-${l.id}`} key={l.id} href={`#level/${l.id}`}>
              <div className="level-card-top">
                <span className="level-badge">LEVEL {l.id}</span>
                <span>{String((l.id-1)*10+1).padStart(2,'0')}—{l.id*10}</span>
              </div>
              <h3>{l.title}</h3>
              <p>{l.description}</p>
              <small>{l.progression}</small>
              <span className="card-link">Lihat 10 chapter →</span>
            </a>
          ))}
        </div>
      </section>

      <Browser/>

      <section id="tracks">
        <div className="section-heading">
          <div>
            <div className="eyebrow">PILIH ARAH PENERAPAN</div>
            <h2>Lima jalur spesialisasi.</h2>
          </div>
          <p>Jalur pilihan untuk memperdalam domain; bukan level wajib.</p>
        </div>
        <div className="ai-track-grid">
          {tracks.map((t,i)=>(
            <a className="track-card" key={t.id} href={`#track/${t.id}`}>
              <span className="track-letter">{String.fromCharCode(65+i)}</span>
              <h3>{t.title}</h3>
              <p>{t.topics.slice(0,3).join(' · ')}</p>
              <span className="status-pill">Coming Soon</span>
            </a>
          ))}
        </div>
      </section>

      <section className="about-strip">
        <div>
          <div className="eyebrow">BELAJAR DENGAN LANDASAN YANG JELAS</div>
          <h2>Terbuka, terstruktur, bertanggung jawab.</h2>
          <p>Adaptasi mandiri dengan rujukan CS2023, UNESCO, NIST AI RMF, dan Google ML Crash Course.</p>
        </div>
        <a className="button subtle" href="#about">Standar & pengembang →</a>
      </section>
    </>
  );
}

function GettingStartedPage({activeSlug}:{activeSlug?:string}){
  const current = gettingStartedModules.find(m=>m.id===activeSlug) || gettingStartedModules[0];
  const currentIndex = gettingStartedModules.findIndex(m=>m.id===current.id);
  const prevMod = currentIndex > 0 ? gettingStartedModules[currentIndex - 1] : null;
  const nextMod = currentIndex < gettingStartedModules.length - 1 ? gettingStartedModules[currentIndex + 1] : null;
  const content = gettingStartedText(current.id);

  return (
    <>
      <div className="page-title">
        <span className="eyebrow">PANDUAN PERSIAPAN LINGKUNGAN · 08 MODUL</span>
        <h1>Getting Started</h1>
        <p>Panduan teknis langkah demi langkah untuk mempersiapkan Python, virtual environment, VS Code, pustaka data science, PyTorch, dan struktur proyek AI.</p>
      </div>

      <nav className="gs-nav" aria-label="Daftar modul Getting Started">
        {gettingStartedModules.map(m=>(
          <a
            key={m.id}
            href={`#getting-started/${m.id}`}
            className={`gs-pill ${m.id===current.id?'active':''}`}
            title={m.title}
          >
            {m.number}: {m.title}
          </a>
        ))}
      </nav>

      <Suspense fallback={<p>Memuat modul Getting Started…</p>}>
        <Reading text={content}/>
      </Suspense>

      <nav className="chapter-nav" aria-label="Navigasi antar modul Getting Started">
        {prevMod ? (
          <a href={`#getting-started/${prevMod.id}`}>
            ← {prevMod.number}: {prevMod.title}
          </a>
        ) : <span/>}
        {nextMod && (
          <a href={`#getting-started/${nextMod.id}`}>
            {nextMod.number}: {nextMod.title} →
          </a>
        )}
      </nav>
    </>
  );
}

function ChapterPage({chapter:c,view}:{chapter:Chapter;view?:string}){
  // Handle views: read, task (backward compatibility, points to reading), practice
  const isReadingView = view === 'read' || view === 'task';
  const isPracticeView = view === 'practice';

  useEffect(() => {
    // If navigated via #chapter/X/task, scroll smoothly to the task section
    if (view === 'task') {
      setTimeout(() => {
        const el = document.getElementById('section-15') || document.querySelector('[id*="tugas"]');
        el?.scrollIntoView({ behavior: 'smooth' });
      }, 100);
    }
  }, [view]);

  return (
    <>
      <a href={`#level/${c.levelId}`}>← Level {c.levelId}</a>
      <div className="page-title">
        <span className="eyebrow">CHAPTER {String(c.id).padStart(2,'0')} / {c.status==='published'?'TERSEDIA':'COMING SOON'}</span>
        <h1>{c.title}</h1>
        <p>{c.description}</p>
      </div>

      {isReadingView && c.resources.reading?.status === 'available' ? (
        <>
          <div className="reading-toolbar">
            <a href={`#chapter/${c.id}`}>← Kembali ke ikhtisar chapter</a>
            <button type="button" onClick={() => window.print()} title="Cetak modul atau simpan sebagai PDF">
              🖨️ Cetak modul (PDF)
            </button>
          </div>
          <Suspense fallback={<p>Memuat modul pembelajaran…</p>}>
            <Reading text={resourceText(c, 'reading')} />
          </Suspense>
        </>
      ) : isPracticeView && c.resources.practice?.status === 'available' ? (
        <>
          <div className="reading-toolbar">
            <a href={`#chapter/${c.id}`}>← Kembali ke ikhtisar chapter</a>
            <button type="button" onClick={() => window.print()} title="Cetak panduan praktik atau simpan sebagai PDF">
              🖨️ Cetak panduan praktik (PDF)
            </button>
          </div>
          <Suspense fallback={<p>Memuat panduan praktik…</p>}>
            <Reading text={resourceText(c, 'practice')} />
          </Suspense>
        </>
      ) : (
        <>
          {c.status === 'coming-soon' ? (
            <div className="notice">
              <strong>Materi Sedang Dipersiapkan.</strong> Rencana pembelajaran dan prasyarat tertera di bawah. Modul lengkap, slide presentasi, dan panduan praktik akan diterbitkan setelah peninjauan akademik selesai.
            </div>
          ) : (
            <div className="notice">
              <strong>Materi Pembelajaran Lengkap Tersedia.</strong> Silakan pilih salah satu dari tiga akses utama di bawah ini untuk memulai belajar, mengikuti presentasi, atau menjalankan praktik kode.
            </div>
          )}

          <div className="chapter-details">
            <section>
              <h2>Tujuan pembelajaran</h2>
              <ul>{c.objectives.map(o=><li key={o}>{o}</li>)}</ul>
            </section>
            <section>
              <h2>Prasyarat yang disarankan</h2>
              {c.prerequisites.length?c.prerequisites.map(id=><p key={id}><a href={`#chapter/${id}`}>{chapters.find(x=>x.id===id)?.title}</a></p>):<p>Literasi komputer dan rasa ingin tahu tentang AI.</p>}
              <small>Seluruh chapter tetap dapat diakses bebas.</small>
            </section>
          </div>

          {/* Three Primary Learning Actions (Only shown if available) */}
          {c.status === 'published' && (
            <>
              <h2>Akses materi pembelajaran</h2>
              <div className="chapter-action-grid">
                {/* 1. Modul Lengkap */}
                {c.resources.reading?.status === 'available' && (
                  <div className="chapter-action-card">
                    <div className="action-card-header">
                      <span className="action-card-kicker">Materi Lengkap</span>
                      <h3>Modul Pembelajaran</h3>
                      <p>Teori komprehensif, studi kasus nyata, lembar latihan terstruktur berjenjang, tugas mini project dengan rubrik 100 poin, dan rujukan kurikulum.</p>
                    </div>
                    <div className="action-card-footer">
                      <a className="action-btn" href={`#chapter/${c.id}/read`}>Baca modul →</a>
                      <a className="action-sub-btn" href={`#chapter/${c.id}/task`}>Buka tugas</a>
                    </div>
                  </div>
                )}

                {/* 2. Slide Reveal.js */}
                {c.resources.slides?.status === 'available' && (
                  <div className="chapter-action-card">
                    <div className="action-card-header">
                      <span className="action-card-kicker">Presentasi Kuliah</span>
                      <h3>Slide Interaktif Reveal.js</h3>
                      <p>Ringkasan visual terstruktur untuk perkuliahan tatap muka, perbandingan konsep, bagan alur, dan speaker notes mendalam untuk pengajar.</p>
                    </div>
                    <div className="action-card-footer">
                      <a className="action-btn" href={`#chapter/${c.id}/slides`}>Buka slide →</a>
                      <a className="action-sub-btn" href={`${base}#chapter/${c.id}/slides?print-pdf`} target="_blank" rel="noreferrer">Cetak PDF</a>
                    </div>
                  </div>
                )}

                {/* 3. Praktik & Laboratorium */}
                {c.resources.practice?.status === 'available' && (
                  <div className="chapter-action-card">
                    <div className="action-card-header">
                      <span className="action-card-kicker">Laboratorium & Kode</span>
                      <h3>Praktik & Notebook</h3>
                      <p>Contoh skrip Python runnable mandiri, interactive companion notebook (.ipynb), panduan eksekusi lokal, dan verifikasi output aktual.</p>
                    </div>
                    <div className="action-card-footer">
                      <a className="action-btn" href={`#chapter/${c.id}/practice`}>Buka praktik →</a>
                      <a className="action-sub-btn" href={`${base}content/level-1-ai-foundations/chapter-01/notebooks/chapter_01_companion.ipynb`} download>Unduh .ipynb</a>
                    </div>
                  </div>
                )}
              </div>
            </>
          )}

          <h2>Rujukan</h2>
          {c.references.map(r=><p key={r.url}><a href={r.url} target="_blank" rel="noreferrer">{r.title} ↗</a> · {r.status==='planned'?'Rujukan rencana':'Rujukan terverifikasi'}</p>)}

          <details>
            <summary>Untuk pengajar: template presentasi</summary>
            <p>Preview kerangka teknis untuk chapter ini. Bukan slide pembelajaran yang selesai.</p>
            <a className="button subtle" href={`#chapter/${c.id}/template`}>Buka preview template</a>
          </details>
        </>
      )}

      <nav className="chapter-nav" aria-label="Chapter sebelumnya dan berikutnya">
        {c.id>1?<a href={`#chapter/${c.id-1}`}>← {chapters[c.id-2].title}</a>:<span/>}
        {c.id<40&&<a href={`#chapter/${c.id+1}`}>{chapters[c.id].title} →</a>}
      </nav>
    </>
  );
}

function About(){
  return (
    <>
      <div className="page-title">
        <div className="eyebrow">TENTANG PLATFORM</div>
        <h1>AI Learning Hub</h1>
        <p>From AI Fundamentals to Intelligent Systems</p>
      </div>
      <section className="about-strip">
        <div>
          <h2>Aji Pamungkas Tri Nurcahyo</h2>
          <p>Creator & Developer · Institut Teknologi Sumatera</p>
          <p>Perpustakaan modul digital berakses terbuka untuk pembelajaran universitas dan mandiri. Materi berbahasa Indonesia dengan istilah teknis English yang lazim.</p>
        </div>
      </section>
      <h2>Landasan kurikulum</h2>
      <p>Struktur 40 chapter merupakan adaptasi mandiri, bukan reproduksi resmi atau sertifikasi ACM, IEEE-CS, AAAI, UNESCO, NIST, maupun Google.</p>
      <div className="resource-grid">
        {[
          ['CS2023','https://csed.acm.org/'],
          ['UNESCO AI Competency Framework for Students','https://www.unesco.org/en/articles/ai-competency-framework-students'],
          ['NIST AI RMF','https://www.nist.gov/itl/ai-risk-management-framework'],
          ['Google ML Crash Course','https://developers.google.com/machine-learning/crash-course/']
        ].map(([title,url])=>(
          <a className="resource-card" key={url} href={url} target="_blank" rel="noreferrer">{title} ↗</a>
        ))}
      </div>
      <h2>Dokumentasi pengajar</h2>
      <div className="resource-grid">
        {['CURRICULUM_STANDARD','REFERENCE_MAPPING','LICENSE_REGISTER','CONTENT_AUDIT','DEVELOPMENT_PLAN','CHAPTER_TEMPLATE','ADDING_CHAPTER','TECHNICAL_LANGUAGE_GUIDE'].map(d=>(
          <a className="resource-card" key={d} href={`${base}docs/${d}.md`}>{d.replaceAll('_',' ')} ↓</a>
        ))}
      </div>
      <h2>Atribusi perangkat lunak</h2>
      <p>Fondasi teknis diadaptasi dari proyek referensi dan komponen upstream. Identitas pengembang platform tidak menghapus hak cipta penulis software asal.</p>
      <a href={`${base}LICENSE.upstream`}>Pemberitahuan lisensi MIT upstream</a>
    </>
  );
}

function App(){
  const [hash,setHash]=useState(location.hash);
  const {theme,setTheme}=usePreferences();
  const route=parseRoute(hash);

  useEffect(()=>{
    const update=()=>{
      setHash(location.hash);
      if(['#curriculum','#tracks'].includes(location.hash))setTimeout(()=>document.getElementById(location.hash.slice(1))?.scrollIntoView(),30);
      else window.scrollTo(0,0);
    };
    addEventListener('hashchange',update);
    return()=>removeEventListener('hashchange',update);
  },[]);

  const chapter=route.type==='chapter'?chapters.find(c=>c.id===route.id):undefined;

  useEffect(()=>{
    document.title=`${chapter?chapter.title+' · ':route.type==='getting-started'?'Getting Started · ':''}AI Learning Hub`;
  },[chapter,route.type]);

  if(chapter&&route.type==='chapter'&&(route.view==='template'||(route.view==='slides'&&chapter.resources.slides.status==='available'))){
    const slides=route.view==='template'?templateSlides(chapter.title):JSON.parse(resourceText(chapter,'slides')) as SlideData[];
    return (
      <Suspense fallback={<p>Memuat presentasi…</p>}>
        <DeckShell key={`${chapter.id}-${route.view}`} slides={slides} meeting={chapter.id} base={base}/>
      </Suspense>
    );
  }

  let content;
  if(route.type==='home') content=<Home/>;
  else if(route.type==='about') content=<About/>;
  else if(route.type==='getting-started') content=<GettingStartedPage activeSlug={route.module}/>;
  else if(chapter&&route.type==='chapter') content=<ChapterPage chapter={chapter} view={route.view}/>;
  else if(route.type==='level'){
    const l=levels.find(l=>l.id===route.id)!;
    content=<>
      <div className="page-title"><span className="eyebrow">LEVEL {l.id} · {l.progression}</span><h1>{l.title}</h1><p>{l.description}</p></div>
      <ChapterCards items={chapters.filter(c=>c.levelId===l.id)}/>
    </>;
  } else if(route.type==='track'){
    const t=tracks.find(t=>t.id===route.id);
    content=t?<>
      <div className="page-title"><span className="eyebrow">SPESIALISASI PILIHAN · COMING SOON</span><h1>{t.title}</h1><p>{t.description}</p></div>
      <h2>Rencana topik</h2>
      <div className="resource-grid">{t.topics.map(topic=><div className="resource-card" key={topic}><strong>{topic}</strong><span>Coming Soon</span></div>)}</div>
      <h2>Chapter fondasi yang disarankan</h2>
      <ChapterCards items={chapters.filter(c=>t.chapters.includes(c.id))}/>
    </>:<Missing/>;
  } else content=<Missing/>;

  return (
    <>
      <a className="skip-link" href="#main" onClick={e=>{e.preventDefault();document.getElementById('main')?.focus();}}>Lewati ke konten</a>
      <header className="site-header">
        <a className="brand-hub" href="#">
          <span className="brand-hub-logo">ai_</span>
          <span className="brand-hub-name">AI Learning Hub</span>
        </a>
        <nav className="hub-nav-links" aria-label="Navigasi utama">
          <a href="#curriculum">Kurikulum</a>
          <a href="#getting-started">Getting Started</a>
          {levels.map(l=><a key={l.id} href={`#level/${l.id}`}>Level {l.id}</a>)}
          <a href="#tracks">Spesialisasi</a>
          <a href="#about">Tentang</a>
        </nav>
        <ThemeSelect theme={theme} onChange={setTheme}/>
      </header>
      <main className="ai-main" id="main" tabIndex={-1}>{content}</main>
      <footer className="site-footer">
        <strong>AI Learning Hub</strong>
        <p>From AI Fundamentals to Intelligent Systems</p>
        <p>Aji Pamungkas Tri Nurcahyo · Institut Teknologi Sumatera</p>
        <a href={`${base}LICENSE.upstream`}>Lisensi upstream</a> · <a href="#about">Standar & atribusi</a>
      </footer>
    </>
  );
}

function Missing(){
  return (
    <div className="page-title">
      <h1>Halaman tidak ditemukan</h1>
      <p>Periksa alamat atau kembali ke kurikulum.</p>
      <a href="#">Kembali ke beranda →</a>
    </div>
  );
}

createRoot(document.getElementById('root')!).render(<App/>);
