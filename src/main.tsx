import {useEffect,useState,lazy,Suspense} from 'react';
import {createRoot} from 'react-dom/client';
import {
  chapters,
  levels,
  tracks,
  trackDetails,
  trackText,
  resourceText,
  chapterNotebookUrl,
  gettingStartedModules,
  gettingStartedText,
  type Chapter,
} from './content';
import {parseRoute} from './routes';
import {ThemeSelect,usePreferences} from './deck-shell/preferences';
import {templateSlides,type SlideData} from './decks/registry';
import {Icon} from './ui/Icon';
import {LessonSidebarNav} from './components/LessonSidebarNav';
import {SITE_COPY} from './copy';
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
          <span className="status-pill">{c.status==='published'?SITE_COPY.homeSections.statusPublished:SITE_COPY.homeSections.statusComingSoon}</span>
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
          <h2>{SITE_COPY.homeSections.catalogHeading}</h2>
        </div>
        <p>{SITE_COPY.homeSections.catalogLead}</p>
      </div>
      <div className="filters">
        <label>
          {SITE_COPY.homeSections.searchLabel}
          <input type="search" value={query} onChange={e=>setQuery(e.target.value)} placeholder={SITE_COPY.homeSections.searchPlaceholder}/>
        </label>
        <label>
          {SITE_COPY.homeSections.levelFilterLabel}
          <select value={level} onChange={e=>setLevel(e.target.value)}>
            <option value="all">{SITE_COPY.homeSections.allLevelsOption}</option>
            {levels.map(l=><option key={l.id} value={l.id}>{l.id} · {l.title}</option>)}
          </select>
        </label>
        <label>
          {SITE_COPY.homeSections.statusFilterLabel}
          <select value={status} onChange={e=>setStatus(e.target.value)}>
            <option value="all">{SITE_COPY.homeSections.allStatusesOption}</option>
            <option value="published">{SITE_COPY.homeSections.statusPublished}</option>
            <option value="coming-soon">{SITE_COPY.homeSections.statusComingSoon}</option>
          </select>
        </label>
      </div>
      <p aria-live="polite">{filtered.length} chapter ditemukan</p>
      <ChapterCards items={filtered}/>
      {!filtered.length&&<div className="notice">{SITE_COPY.homeSections.emptySearch}</div>}
    </section>
  );
}

function Home(){
  return (
    <>
      <section className="ai-hero">
        <div>
          <h1>{SITE_COPY.hero.titlePrefix}<br/><em>{SITE_COPY.hero.titleEmphasis}</em></h1>
          <h2 className="hero-subtitle">{SITE_COPY.hero.subtitle}</h2>
          <p>{SITE_COPY.hero.description}</p>
          <div className="hero-cta-group">
            <a className="button primary button-lg" href="#curriculum">{SITE_COPY.hero.primaryCta}</a>
            <a className="button subtle button-lg" href="#tracks">{SITE_COPY.hero.tracksCta}</a>
            <a className="button subtle button-lg" href="#getting-started">{SITE_COPY.hero.gettingStartedCta}</a>
          </div>
          <div className="hero-facts">
            <span><b>{String(gettingStartedModules.length).padStart(2,'0')}</b> {SITE_COPY.hero.facts[0].label}</span>
            <span><b>{String(levels.length).padStart(2,'0')}</b> {SITE_COPY.hero.facts[1].label}</span>
            <span><b>{String(chapters.length).padStart(2,'0')}</b> {SITE_COPY.hero.facts[2].label}</span>
            <span><b>{String(tracks.length).padStart(2,'0')}</b> {SITE_COPY.hero.facts[3].label}</span>
          </div>
        </div>
        <div className="network-art" aria-hidden="true">
          <svg viewBox="0 0 400 320" width="400" height="320">
            {[
              {y:20,lvl:'L1',label:'AI Foundations',range:'01-10'},
              {y:82,lvl:'L2',label:'Machine Learning',range:'11-20'},
              {y:144,lvl:'L3',label:'Deep Learning & GenAI',range:'21-30'},
              {y:206,lvl:'L4',label:'Applied AI Systems',range:'31-40'}
            ].map((row,idx)=>(
              <g key={row.lvl}>
                <rect x="24" y={row.y} width="352" height="46" rx="8" fill="var(--surface2)" stroke="var(--line)" strokeWidth="1"/>
                <rect x="34" y={row.y+10} width="36" height="26" rx="5" fill="var(--primary-soft)"/>
                <text x="52" y={row.y+27} textAnchor="middle" fontSize="12" fontWeight="700" fontFamily="var(--font-mono)" fill="var(--primary)">{row.lvl}</text>
                <text x="84" y={row.y+27} fontSize="13.5" fontWeight="600" fill="var(--ink)">{row.label}</text>
                <text x="362" y={row.y+27} textAnchor="end" fontSize="12" fontFamily="var(--font-mono)" fill="var(--muted)">{row.range}</text>
                {idx<3&&<line x1="200" y1={row.y+46} x2="200" y2={row.y+62} stroke="var(--line-strong)" strokeWidth="1.5"/>}
              </g>
            ))}
            <line x1="200" y1="252" x2="200" y2="268" stroke="var(--primary)" strokeWidth="1.5"/>
            <rect x="24" y="268" width="352" height="36" rx="8" fill="var(--primary-soft)" stroke="var(--primary)" strokeWidth="1"/>
            <text x="200" y="290" textAnchor="middle" fontSize="12.5" fontWeight="600" fill="var(--primary)">{SITE_COPY.hero.diagramFooter}</text>
          </svg>
          <span className="network-art-caption">{SITE_COPY.hero.diagramCaption}</span>
        </div>
      </section>

      <div className="notice">
        <strong>{SITE_COPY.hero.prepNoticeLabel}</strong> {SITE_COPY.hero.prepNoticeText} <a href="#getting-started">{SITE_COPY.hero.prepNoticeLink}</a>.
      </div>

      <section>
        <div className="section-heading">
          <div>
            <h2>{SITE_COPY.homeSections.levelsHeading}</h2>
          </div>
          <p>{SITE_COPY.homeSections.levelsLead}</p>
        </div>
        <div className="ai-level-grid">
          {levels.map(l=>(
            <a className={`level-card level-${l.id}`} key={l.id} href={`#level/${l.id}`}>
              <div className="level-card-top">
                <span className="level-badge">LEVEL {l.id}</span>
                <span>Chapter {String((l.id-1)*10+1).padStart(2,'0')}–{l.id*10}</span>
              </div>
              <h3>{l.title}</h3>
              <p>{l.description}</p>
              <small>{l.progression}</small>
              <span className="card-link">{SITE_COPY.homeSections.levelCta}</span>
            </a>
          ))}
        </div>
      </section>

      <Browser/>

      <section id="tracks">
        <div className="section-heading">
          <div>
            <h2>{SITE_COPY.homeSections.tracksHeading}</h2>
          </div>
          <p>{SITE_COPY.homeSections.tracksLead}</p>
        </div>
        <div className="ai-track-grid">
          {tracks.map((t,i)=>{
            const letter = (t as {code?:string}).code || String.fromCharCode(65+i);
            const available = t.status === 'published';
            return (
              <a className="track-card" key={t.id} href={`#track/${t.id}`}>
                <div className="level-card-top">
                  <span className="track-letter">Track {letter}</span>
                  <span className="track-meta-count">{available ? `${t.chapters.length} chapter` : SITE_COPY.homeSections.statusComingSoon}</span>
                </div>
                <h3>{t.title}</h3>
                <p>{t.topics.slice(0,3).join(' · ')}</p>
                <span className="card-link">{available ? SITE_COPY.homeSections.trackCtaAvailable : SITE_COPY.homeSections.trackCtaPlanned}</span>
              </a>
            );
          })}
        </div>
      </section>

      <section className="about-strip">
        <div>
          <h2>{SITE_COPY.homeSections.aboutHeading}</h2>
          <p>{SITE_COPY.homeSections.aboutBody}</p>
        </div>
        <a className="button subtle" href="#about">{SITE_COPY.homeSections.aboutCta}</a>
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
      <div className="page-title gs-page-header">
        <span className="eyebrow">{SITE_COPY.gettingStarted.eyebrowPrefix} · {String(gettingStartedModules.length).padStart(2,'0')} MODUL</span>
        <h1>{SITE_COPY.gettingStarted.heading}</h1>
        <p>{SITE_COPY.gettingStarted.lead}</p>
      </div>

      <Suspense fallback={<p>{SITE_COPY.gettingStarted.loading}</p>}>
        <Reading
          text={content}
          className="gs-reading-grid"
          sidebarTop={
            <LessonSidebarNav
              items={gettingStartedModules}
              activeId={current.id}
              hrefFor={m => `#getting-started/${m.id}`}
              navLabel={SITE_COPY.gettingStarted.navLabel}
            />
          }
        />
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
  const isReadingView = view === 'read' || view === 'task';
  const isPracticeView = view === 'practice';
  const levelInfo = levels.find(l => l.id === c.levelId);
  const learningHighlights = (c.highlights && c.highlights.length > 0)
    ? c.highlights
    : c.objectives.slice(0, 5);
  const topicTags = c.tags.slice(0, 3);

  useEffect(() => {
    if (view === 'task') {
      setTimeout(() => {
        const el = document.getElementById('section-15') || document.querySelector('[id*="mini-project"], [id*="tugas"]');
        el?.scrollIntoView({ behavior: 'smooth' });
      }, 100);
    }
  }, [view]);

  return (
    <>
      <a href={`#level/${c.levelId}`}>← Level {c.levelId}{levelInfo ? ` · ${levelInfo.title}` : ''}</a>
      <div className="page-title">
        <span className="eyebrow">
          CHAPTER {String(c.id).padStart(2,'0')} · LEVEL {c.levelId} · {c.status==='published'?'TERSEDIA':'COMING SOON'}
        </span>
        <h1>{c.title}</h1>
        {topicTags.length > 0 && (
          <div className="chapter-tags-row" aria-label="Topik utama chapter">
            {topicTags.map(tag => <span className="status-pill" key={tag}>{tag}</span>)}
          </div>
        )}
        <p>{c.description}</p>
      </div>

      {isReadingView && c.resources.reading?.status === 'available' ? (
        <>
          <div className="reading-toolbar">
            <a href={`#chapter/${c.id}`}>← {SITE_COPY.chapter.backToList}</a>
            <button type="button" onClick={() => window.print()} title={SITE_COPY.chapter.printModule}>
              <Icon name="print" size={16} /> <span>{SITE_COPY.chapter.printModule}</span>
            </button>
          </div>
          <Suspense fallback={<p>{SITE_COPY.chapter.loadingReading}</p>}>
            <Reading text={resourceText(c, 'reading')} />
          </Suspense>
        </>
      ) : isPracticeView && c.resources.practice?.status === 'available' ? (
        <>
          <div className="reading-toolbar">
            <a href={`#chapter/${c.id}`}>← {SITE_COPY.chapter.backToList}</a>
            <button type="button" onClick={() => window.print()} title={SITE_COPY.chapter.printPractice}>
              <Icon name="print" size={16} /> <span>{SITE_COPY.chapter.printPractice}</span>
            </button>
          </div>
          <Suspense fallback={<p>{SITE_COPY.chapter.loadingPractice}</p>}>
            <Reading text={resourceText(c, 'practice')} />
          </Suspense>
        </>
      ) : (
        <>
          {c.status === 'coming-soon' && (
            <div className="notice notice-pending">
              <strong>{SITE_COPY.chapter.comingSoonLabel}</strong> {SITE_COPY.chapter.comingSoonBody}
            </div>
          )}

          <div className="chapter-details">
            <section>
              <h2>{SITE_COPY.chapter.highlightsHeading}</h2>
              <ul>{learningHighlights.map(item=><li key={item}>{item}</li>)}</ul>
            </section>
            <section className="prereq-panel">
              <h2>{SITE_COPY.chapter.prereqHeading}</h2>
              {c.prerequisites.length ? c.prerequisites.map(id => {
                const pre = chapters.find(x => x.id === id);
                return (
                  <p key={id}>
                    <a href={`#chapter/${id}`}>Chapter {String(id).padStart(2,'0')}: {pre?.title}</a>
                  </p>
                );
              }) : <p>{SITE_COPY.chapter.prereqDefault}</p>}
              <small>{SITE_COPY.chapter.prereqNote}</small>
            </section>
          </div>

          {c.status === 'published' && (
            <>
              <h2>{SITE_COPY.chapter.accessHeading}</h2>
              <div className="chapter-action-grid">
                {c.resources.reading?.status === 'available' && (
                  <div className="chapter-action-card primary-pillar">
                    <div className="action-card-header">
                      <span className="action-card-kicker">{SITE_COPY.chapter.resources.reading.kicker}</span>
                      <h3>{SITE_COPY.chapter.resources.reading.title}</h3>
                      <p>{SITE_COPY.chapter.resources.reading.description}</p>
                    </div>
                    <div className="action-card-footer">
                      <a className="action-btn primary-action" href={`#chapter/${c.id}/read`}>{SITE_COPY.chapter.resources.reading.primaryAction}</a>
                      <a className="action-sub-btn" href={`#chapter/${c.id}/task`}>{SITE_COPY.chapter.resources.reading.secondaryAction}</a>
                    </div>
                  </div>
                )}

                {c.resources.slides?.status === 'available' && (
                  <div className="chapter-action-card">
                    <div className="action-card-header">
                      <span className="action-card-kicker">{SITE_COPY.chapter.resources.slides.kicker}</span>
                      <h3>{SITE_COPY.chapter.resources.slides.title}</h3>
                      <p>{SITE_COPY.chapter.resources.slides.description}</p>
                    </div>
                    <div className="action-card-footer">
                      <a className="action-btn" href={`#chapter/${c.id}/slides`}>{SITE_COPY.chapter.resources.slides.primaryAction}</a>
                      <a className="action-sub-btn" href={`${base}#chapter/${c.id}/slides?print-pdf`} target="_blank" rel="noreferrer">{SITE_COPY.chapter.resources.slides.secondaryAction}</a>
                    </div>
                  </div>
                )}

                {c.resources.practice?.status === 'available' && (
                  <div className="chapter-action-card">
                    <div className="action-card-header">
                      <span className="action-card-kicker">{SITE_COPY.chapter.resources.practice.kicker}</span>
                      <h3>{SITE_COPY.chapter.resources.practice.title}</h3>
                      <p>{SITE_COPY.chapter.resources.practice.description}</p>
                    </div>
                    <div className="action-card-footer">
                      <a className="action-btn" href={`#chapter/${c.id}/practice`}>{SITE_COPY.chapter.resources.practice.primaryAction}</a>
                      <a className="action-sub-btn" href={chapterNotebookUrl(c, base)} download>{SITE_COPY.chapter.resources.practice.secondaryAction}</a>
                    </div>
                  </div>
                )}
              </div>
            </>
          )}

          <h2>{SITE_COPY.chapter.referencesHeading}</h2>
          {c.references.map(r=><p key={r.url}><a href={r.url} target="_blank" rel="noreferrer">{r.title} ↗</a>{r.status==='planned'?' (Rujukan rencana)':''}</p>)}

          {c.resources.slides?.status !== 'available' && (
            <details>
              <summary>Untuk pengajar: kerangka presentasi</summary>
              <p>Kerangka awal slide perkuliahan untuk chapter yang sedang disusun.</p>
              <a className="button subtle" href={`#chapter/${c.id}/template`}>Lihat kerangka slide</a>
            </details>
          )}
        </>
      )}

      <nav className="chapter-nav" aria-label="Chapter sebelumnya dan berikutnya">
        {c.id>1?<a href={`#chapter/${c.id-1}`}>← {chapters[c.id-2].title}</a>:<span/>}
        {c.id<40&&<a href={`#chapter/${c.id+1}`}>{chapters[c.id].title} →</a>}
      </nav>
    </>
  );
}

function RoadmapStageCard({title,ids}:{title:string;ids:number[]}){
  return (
    <div className="roadmap-stage-card">
      <h3>{title}</h3>
      <div className="roadmap-list">
        {ids.map(id=>{
          const ch = chapters.find(x => x.id === id);
          if (!ch) return null;
          return (
            <div className="roadmap-chapter-item" key={id}>
              <div>
                <a href={`#chapter/${ch.id}`}><strong>Chapter {String(ch.id).padStart(2,'0')}: {ch.title}</strong></a>
                <p>{ch.description}</p>
              </div>
              <div className="roadmap-quick-links">
                <a href={`#chapter/${ch.id}/read`}>Modul</a>
                <a href={`#chapter/${ch.id}/slides`}>Slide</a>
                {ch.resources.practice?.status === 'available' && <a href={`#chapter/${ch.id}/practice`}>Praktik</a>}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function TrackPage({id,view}:{id:string;view?:'practice'|'capstone'}){
  const summary = tracks.find(t => t.id === id);
  const detail = trackDetails[id];
  if (!summary || !detail) return <Missing/>;

  const activeView = view || 'overview';
  const bodyMd = trackText(id, activeView);

  return (
    <>
      <div className="track-breadcrumb">
        <a href="#tracks">← {SITE_COPY.tracks.backToAll}</a>
        <span>·</span>
        <a href="#curriculum">{SITE_COPY.tracks.backToCurriculum}</a>
      </div>

      <div className="page-title">
        <span className="eyebrow">
          TRACK {detail.code} · JALUR SPESIALISASI · {detail.status === 'published' ? 'TERSEDIA' : 'SEGERA HADIR'}
        </span>
        <h1>{detail.title}</h1>
        <p>{detail.shortDescription}</p>
      </div>

      <nav className="gs-nav" aria-label="Navigasi bagian jalur spesialisasi">
        <a href={`#track/${detail.id}`} className={`gs-pill ${activeView==='overview'?'active':''}`}>
          {SITE_COPY.tracks.tabs.overview}
        </a>
        <a href={`#track/${detail.id}/practice`} className={`gs-pill ${activeView==='practice'?'active':''}`}>
          {SITE_COPY.tracks.tabs.practiceLabel} ({detail.labs.length})
        </a>
        <a href={`#track/${detail.id}/capstone`} className={`gs-pill ${activeView==='capstone'?'active':''}`}>
          {SITE_COPY.tracks.tabs.capstoneLabel} ({detail.capstone.rubricTotal} Poin)
        </a>
      </nav>

      {activeView === 'overview' && (
        <>
          <div className="chapter-details">
            <section>
              <h2>{SITE_COPY.tracks.outcomesHeading}</h2>
              <ul>{detail.learningOutcomes.map(o => <li key={o}>{o}</li>)}</ul>
            </section>
            <section className="prereq-panel">
              <h2>{SITE_COPY.chapter.prereqHeading}</h2>
              {detail.prerequisites.map(cid => {
                const pre = chapters.find(x => x.id === cid);
                return pre ? (
                  <p key={cid}><a href={`#chapter/${cid}`}>Chapter {String(cid).padStart(2,'0')}: {pre.title}</a></p>
                ) : null;
              })}
              <small>{SITE_COPY.tracks.prereqNote}</small>
            </section>
          </div>

          <h2>{SITE_COPY.tracks.roadmapHeading}</h2>
          <div className="track-roadmap-grid">
            <RoadmapStageCard title={SITE_COPY.tracks.roadmapStages.foundation} ids={detail.roadmap.foundation} />
            <RoadmapStageCard title={SITE_COPY.tracks.roadmapStages.core} ids={detail.roadmap.core} />
            <RoadmapStageCard title={SITE_COPY.tracks.roadmapStages.advanced} ids={detail.roadmap.advanced} />
          </div>

          <div className="chapter-action-grid">
            <div className="chapter-action-card primary-pillar">
              <div className="action-card-header">
                <span className="action-card-kicker">{SITE_COPY.tracks.practiceCard.kicker}</span>
                <h3>{SITE_COPY.tracks.practiceCard.title}</h3>
                <p>{detail.labs.length} panduan praktikum pemrograman menggunakan skrip latihan terkait jalur spesialisasi.</p>
              </div>
              <div className="action-card-footer">
                <a className="action-btn primary-action" href={`#track/${detail.id}/practice`}>{SITE_COPY.tracks.practiceCard.cta}</a>
              </div>
            </div>
            <div className="chapter-action-card">
              <div className="action-card-header">
                <span className="action-card-kicker">{SITE_COPY.tracks.capstoneCard.kicker}</span>
                <h3>{detail.capstone.title}</h3>
                <p>{detail.capstone.summary}</p>
              </div>
              <div className="action-card-footer">
                <a className="action-btn" href={`#track/${detail.id}/capstone`}>{SITE_COPY.tracks.capstoneCard.cta}</a>
              </div>
            </div>
          </div>
        </>
      )}

      {activeView === 'practice' && (
        <section>
          <h2>{SITE_COPY.tracks.labsHeading}</h2>
          <div className="lab-grid">
            {detail.labs.map((lab, idx) => (
              <div className="resource-card" key={lab.title}>
                <span className="action-card-kicker">Praktikum {idx + 1} · Chapter {String(lab.chapterId).padStart(2,'0')}</span>
                <strong>{lab.title}</strong>
                <div className="lab-script-list">
                  {lab.scriptPaths.map(sp => <code className="inline-code" key={sp}>{sp}</code>)}
                </div>
                <a href={`#chapter/${lab.chapterId}/practice`}>{SITE_COPY.tracks.openChapterPractice} {String(lab.chapterId).padStart(2,'0')}</a>
              </div>
            ))}
          </div>
        </section>
      )}

      {bodyMd && (
        <Suspense fallback={<p>{SITE_COPY.tracks.loading}</p>}>
          <Reading text={bodyMd} />
        </Suspense>
      )}

      {activeView === 'overview' && detail.references.length > 0 && (
        <section>
          <h2>{SITE_COPY.tracks.referencesHeading}</h2>
          {detail.references.map(r => (
            <p key={r.url}><a href={r.url} target="_blank" rel="noreferrer">{r.title} ↗</a></p>
          ))}
        </section>
      )}
    </>
  );
}

function About(){
  return (
    <>
      <div className="page-title">
        <div className="eyebrow">{SITE_COPY.about.eyebrow}</div>
        <h1>{SITE_COPY.brandName}</h1>
        <p>{SITE_COPY.brandTagline}</p>
      </div>
      <section className="about-strip">
        <div>
          <h2>{SITE_COPY.author}</h2>
          <p>{SITE_COPY.about.roleLine}</p>
          <p>{SITE_COPY.about.bio}</p>
        </div>
      </section>
      <h2>{SITE_COPY.about.curriculumHeading}</h2>
      <p>{SITE_COPY.about.curriculumNote}</p>
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
      <h2>{SITE_COPY.about.docsHeading}</h2>
      <div className="resource-grid">
        {SITE_COPY.about.docsItems.map(d=>(
          <a className="resource-card" key={d.key} href={`${base}docs/${d.key}.md`}>{d.label} ↓</a>
        ))}
      </div>
      <h2>{SITE_COPY.about.attributionHeading}</h2>
      <p>{SITE_COPY.about.attributionNote}</p>
      <a href={`${base}LICENSE.upstream`}>{SITE_COPY.about.attributionLink}</a>
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
    const trackTitle = route.type==='track' ? trackDetails[route.id]?.title : undefined;
    document.title=`${chapter?chapter.title+' · ':trackTitle?trackTitle+' · ':route.type==='getting-started'?'Getting Started · ':''}AI Learning Hub`;
  },[chapter,route]);

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
    content=<TrackPage id={route.id} view={route.view}/>;
  } else content=<Missing/>;

  return (
    <>
      <a className="skip-link" href="#main" onClick={e=>{e.preventDefault();document.getElementById('main')?.focus();}}>{SITE_COPY.skipLink}</a>
      <header className="site-header">
        <a className="brand-hub" href="#">
          <span className="brand-hub-logo" aria-hidden="true">ai_</span>
          <span className="brand-hub-name">{SITE_COPY.brandName}</span>
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
        <strong>{SITE_COPY.brandName}</strong>
        <p>{SITE_COPY.brandTagline}</p>
        <p>{SITE_COPY.author} · {SITE_COPY.institution}</p>
        <a href={`${base}LICENSE.upstream`}>{SITE_COPY.footer.licenseLink}</a> · <a href="#about">{SITE_COPY.footer.aboutLink}</a>
      </footer>
    </>
  );
}

function Missing(){
  return (
    <div className="page-title">
      <h1>{SITE_COPY.notFound.heading}</h1>
      <p>{SITE_COPY.notFound.description}</p>
      <a href="#">{SITE_COPY.notFound.backCta}</a>
    </div>
  );
}

createRoot(document.getElementById('root')!).render(<App/>);
