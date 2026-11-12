import { readdirSync, readFileSync, existsSync, statSync } from 'node:fs';
import { resolve, dirname, relative, isAbsolute } from 'node:path';
import { fileURLToPath } from 'node:url';
export const root=resolve(dirname(fileURLToPath(import.meta.url)),'..');
export function walk(dir){return readdirSync(dir,{withFileTypes:true}).flatMap(e=>e.isDirectory()?walk(resolve(dir,e.name)):[resolve(dir,e.name)]);}
export const resourceKeys=['reading','slides','practice'];

export function validateChapter(c,dir){
 const errors=[];const check=(ok,message)=>{if(!ok)errors.push(`${c.id}: ${message}`);};
 check(Number.isInteger(c.id)&&c.id>=1&&c.id<=40,'invalid id');
 check(c.slug===`chapter-${String(c.id).padStart(2,'0')}`,'invalid slug');
 check(c.levelId===Math.ceil(c.id/10),'invalid level');
 check(typeof c.title==='string'&&c.title.length>5,'missing title');
 check(['coming-soon','published'].includes(c.status),'invalid publication status');
 const desc=typeof c.description==='string'?c.description.trim():'';
 const descWords=desc?desc.split(/\s+/).length:0;
 check(desc.length>20,'missing summary description');
 check(descWords<=65&&desc.length<=480,`overview description too long (${descWords} words / ${desc.length} chars; keep <= 65 words)`);
 if(c.status==='published'){
  check(Array.isArray(c.highlights)&&c.highlights.length>=4&&c.highlights.length<=6&&c.highlights.every(h=>typeof h==='string'&&h.trim().length>5),'published chapter requires 4-6 concise learning highlights');
 }
 check(Array.isArray(c.objectives)&&c.objectives.length>=2&&c.objectives.every(s=>/^(Menjelaskan|Membandingkan|Mengimplementasikan|Menganalisis|Mengevaluasi|Merancang|Menghitung|Menurunkan|Menyiapkan|Membangun|Menyusun|Membedakan|Mengidentifikasi|Merumuskan|Memilih|Merepresentasikan|Menggunakan|Menerapkan|Mengoperasikan|Mengelola|Menghubungkan|Mendeteksi|Memitigasi|Mengintegrasikan|Mendokumentasikan|Mengomunikasikan|Memahami|Mengenali|Menginterpretasikan|Membagi|Melakukan|Mengembangkan|Menangani)/.test(s)),'measurable objectives required');
 check(Array.isArray(c.prerequisites)&&c.prerequisites.every(id=>Number.isInteger(id)&&id>=1&&id<=40&&id!==c.id),'invalid prerequisites');
 check(Array.isArray(c.tags)&&c.tags.length>0,'missing search tags');
 check(c.resources&&Object.keys(c.resources).sort().join()===resourceKeys.toSorted().join(),'resource keys mismatch');
 for(const key of resourceKeys){const r=c.resources?.[key];if(!r)continue;
 check(['available','unavailable'].includes(r.status),`invalid ${key} status`);
 if(r.status==='unavailable'){check(r.path===null,`${key} unavailable must have null path`);continue;}
 const safe=typeof r.path==='string'&&r.path.length>0&&!isAbsolute(r.path)&&!r.path.includes('\\')&&!r.path.includes(':')&&!r.path.split('/').includes('..');
 check(safe,`${key} unsafe path`);if(!safe)continue;
 const file=resolve(dir,r.path);check(existsSync(file)&&statSync(file).isFile(),`${key} missing file: ${r.path}`);if(!existsSync(file)||!statSync(file).isFile())continue;
 check(statSync(file).size>0,`${key} empty file`);
 if(key==='slides'){try{const deck=JSON.parse(readFileSync(file,'utf8'));check(Array.isArray(deck)&&deck.length>0,'empty deck');for(const s of deck){check(['cover','content','code','split','case','summary'].includes(s.kind)&&typeof s.title==='string'&&typeof s.body==='string'&&typeof s.notes==='string','invalid slide');check(s.body.length<=1000&&(!s.code||s.code.split('\n').length<=18),'split long slide');}}catch{check(false,'invalid slides JSON');}}
 }
 check(Array.isArray(c.references)&&c.references.every(r=>r.title&&/^https:\/\//.test(r.url)),'invalid reference URLs');
 if(c.status==='published'){
 check(c.validation?.reviewed===true&&c.validation?.executionVerified===true,'publication needs review and execution evidence');
 check(c.resources?.reading?.status==='available','published chapter missing reading module');
 const readPath=c.resources?.reading?.path;
 if(readPath){
  const readContent=readFileSync(resolve(dir,readPath),'utf8');
  check(/latihan/i.test(readContent),'module must contain exercises section');
  check(/tugas|mini project/i.test(readContent),'module must contain assignment/project section');
  check(/rubrik/i.test(readContent)&&/100/i.test(readContent),'module must contain 100-point rubric');
 }
 if(c.resources?.practice?.status==='available'){
  const nbFile=resolve(dir,`notebooks/chapter_${String(c.id).padStart(2,'0')}_companion.ipynb`);
  check(existsSync(nbFile)&&statSync(nbFile).size>0,`missing companion notebook: notebooks/chapter_${String(c.id).padStart(2,'0')}_companion.ipynb`);
 }
 check(c.references?.length>=2&&c.references.every(r=>r.status==='verified'),'two verified references required');
 check(existsSync(resolve(dir,'audit.md')),'publication audit.md required');
 }
 return errors;
}

export function validateTrack(summary,chapters){
 const errors=[];const check=(ok,message)=>{if(!ok)errors.push(`track/${summary.id}: ${message}`);};
 const validIds=new Set(chapters.map(c=>c.data.id));
 check(typeof summary.id==='string'&&summary.id.length>0,'missing track id');
 check(['published','coming-soon'].includes(summary.status),'invalid track status');
 check(Array.isArray(summary.topics)&&summary.topics.length>0,'missing track topics');
 check(Array.isArray(summary.chapters)&&summary.chapters.length>0&&summary.chapters.every(id=>validIds.has(id)),'invalid chapter references in tracks.json');
 const trackDir=resolve(root,'content/tracks',summary.id);
 const metaPath=resolve(trackDir,'metadata.json');
 check(existsSync(metaPath),`missing content/tracks/${summary.id}/metadata.json`);
 if(!existsSync(metaPath))return errors;
 const meta=JSON.parse(readFileSync(metaPath,'utf8'));
 check(meta.id===summary.id&&meta.title===summary.title,'track metadata mismatch with tracks.json');
 check(typeof meta.shortDescription==='string'&&meta.shortDescription.trim().length>20,'missing shortDescription');
 check(Array.isArray(meta.learningOutcomes)&&meta.learningOutcomes.length>=3,'missing learningOutcomes');
 check(Array.isArray(meta.prerequisites)&&meta.prerequisites.every(id=>validIds.has(id)),'invalid track prerequisites');
 const fIds=meta.roadmap?.foundation||[];const cIds=meta.roadmap?.core||[];const aIds=meta.roadmap?.advanced||[];
 check(fIds.length>0&&cIds.length>0&&aIds.length>0,'track roadmap requires foundation, core, and advanced chapters');
 const allRoad=[...fIds,...cIds,...aIds];
 check(allRoad.every(id=>validIds.has(id)),'track roadmap contains invalid chapter ID');
 check(new Set(allRoad).size===allRoad.length,'duplicate chapter IDs inside track roadmap');
 check(Array.isArray(meta.labs)&&meta.labs.length>=5,'track requires at least 5 applied labs');
 for(const lab of meta.labs||[]){
  check(typeof lab.title==='string'&&lab.title.length>3,'lab missing title');
  check(validIds.has(lab.chapterId),`lab '${lab.title}' references invalid chapterId`);
  check(Array.isArray(lab.scriptPaths)&&lab.scriptPaths.length>0,`lab '${lab.title}' missing scriptPaths`);
  for(const sp of lab.scriptPaths||[]){
   check(existsSync(resolve(root,sp)),`lab '${lab.title}' missing script file: ${sp}`);
  }
 }
 check(meta.capstone&&typeof meta.capstone.title==='string'&&meta.capstone.rubricTotal===100,'track capstone with 100-point rubric required');
 check(Array.isArray(meta.references)&&meta.references.length>=2&&meta.references.every(r=>r.title&&/^https:\/\//.test(r.url)&&r.status==='verified'),'track requires verified HTTPS references');
 if(meta.status==='published'){
  for(const md of ['overview.md','practice.md','capstone.md']){
   const mp=resolve(trackDir,md);
   check(existsSync(mp)&&statSync(mp).size>50,`published track missing non-empty ${md}`);
  }
  const capText=existsSync(resolve(trackDir,'capstone.md'))?readFileSync(resolve(trackDir,'capstone.md'),'utf8'):'';
  check(/rubrik/i.test(capText)&&/100/i.test(capText),'track capstone.md must contain 100-point rubric');
 }
 return errors;
}

export function validateAll(){
 const files=walk(resolve(root,'content')).filter(f=>!f.replaceAll('\\','/').includes('/tracks/')&&f.endsWith('metadata.json'));
 const chapters=files.map(f=>({data:JSON.parse(readFileSync(f,'utf8')),dir:dirname(f)}));
 const errors=chapters.flatMap(({data,dir})=>validateChapter(data,dir));
 if(chapters.length!==40||new Set(chapters.map(c=>c.data.id)).size!==40)errors.push('Exactly 40 unique chapter IDs required');
 if(new Set(chapters.map(c=>c.data.slug)).size!==40)errors.push('Duplicate slugs');
 // Circular prerequisite check
 const prereqMap=new Map(chapters.map(c=>[c.data.id,c.data.prerequisites||[]]));
 const visiting=new Set();const visited=new Set();
 const hasCycle=id=>{if(visiting.has(id))return true;if(visited.has(id))return false;visiting.add(id);for(const p of prereqMap.get(id)||[]){if(hasCycle(p))return true;}visiting.delete(id);visited.add(id);return false;};
 for(const {data} of chapters){if(hasCycle(data.id))errors.push(`Circular prerequisite detected at chapter ${data.id}`);}
 const levels=JSON.parse(readFileSync(resolve(root,'content/levels.json'),'utf8'));
 if(levels.length!==4)errors.push('Exactly four levels required');
 for(const {data,dir} of chapters)if(relative(resolve(root,'content'),dir).replaceAll('\\','/')!==`${levels.find(l=>l.id===data.levelId)?.folder}/${data.slug}`)errors.push(`Chapter ${data.id} folder mismatch`);
 const tracks=JSON.parse(readFileSync(resolve(root,'content/specializations/tracks.json'),'utf8'));
 if(tracks.length!==5||new Set(tracks.map(t=>t.id)).size!==5)errors.push('Exactly five unique tracks required');
 for(const t of tracks)errors.push(...validateTrack(t,chapters));
 const forbidden=/Python Learning Hub|Pengantar Komputasi|Pengenalan Komputasi|\bAlpro\b|Program Studi Teknik Telekomunikasi|Fakultas Teknologi Industri|Pertemuan\s+\d/i;
 for(const f of [...walk(resolve(root,'src')),...files])if(forbidden.test(readFileSync(f,'utf8')))errors.push(`Old branding: ${relative(root,f)}`);
 for(const name of ['CURRICULUM_STANDARD','REFERENCE_MAPPING','LICENSE_REGISTER','CONTENT_AUDIT','CONTENT_MAINTENANCE_GUIDE','THEME_AND_ACCESSIBILITY_GUIDE','DEVELOPMENT_PLAN','CHAPTER_TEMPLATE','ADDING_CHAPTER','TECHNICAL_LANGUAGE_GUIDE'])if(!existsSync(resolve(root,`docs/${name}.md`)))errors.push(`Missing docs/${name}.md`);
 const template=JSON.parse(readFileSync(resolve(root,'content/templates/slides.json'),'utf8'));
 if(template.length<5||!template.some(s=>s.code)||!template.some(s=>s.tex)||!template.every(s=>s.notes))errors.push('Template capabilities incomplete');
 return errors;
}
if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url)){
 const errors=validateAll();if(errors.length){console.error(errors.join('\n'));process.exitCode=1;}else console.log('PASS: 40 chapters (concise overviews & highlights), four levels, five active specialization tracks, paths, branding, docs & presentation template.');
}
