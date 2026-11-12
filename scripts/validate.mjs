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
 check(c.references?.length>=2&&c.references.every(r=>r.status==='verified'),'two verified references required');
 check(existsSync(resolve(dir,'audit.md')),'publication audit.md required');
 }
 return errors;
}
export function validateAll(){
 const files=walk(resolve(root,'content')).filter(f=>f.endsWith('metadata.json'));
 const chapters=files.map(f=>({data:JSON.parse(readFileSync(f,'utf8')),dir:dirname(f)}));
 const errors=chapters.flatMap(({data,dir})=>validateChapter(data,dir));
 if(chapters.length!==40||new Set(chapters.map(c=>c.data.id)).size!==40)errors.push('Exactly 40 unique chapter IDs required');
 if(new Set(chapters.map(c=>c.data.slug)).size!==40)errors.push('Duplicate slugs');
 const levels=JSON.parse(readFileSync(resolve(root,'content/levels.json'),'utf8'));
 if(levels.length!==4)errors.push('Exactly four levels required');
 for(const {data,dir} of chapters)if(relative(resolve(root,'content'),dir).replaceAll('\\','/')!==`${levels.find(l=>l.id===data.levelId)?.folder}/${data.slug}`)errors.push(`Chapter ${data.id} folder mismatch`);
 const tracks=JSON.parse(readFileSync(resolve(root,'content/specializations/tracks.json'),'utf8'));
 if(tracks.length!==5||new Set(tracks.map(t=>t.id)).size!==5)errors.push('Exactly five unique tracks required');
 for(const t of tracks)if(!t.topics.length||!t.chapters.every(id=>chapters.some(c=>c.data.id===id)))errors.push(`Invalid track ${t.id}`);
 const forbidden=/Python Learning Hub|Pengantar Komputasi|Pengenalan Komputasi|\bAlpro\b|Program Studi Teknik Telekomunikasi|Fakultas Teknologi Industri|Pertemuan\s+\d/i;
 for(const f of [...walk(resolve(root,'src')),...files])if(forbidden.test(readFileSync(f,'utf8')))errors.push(`Old branding: ${relative(root,f)}`);
 for(const name of ['CURRICULUM_STANDARD','REFERENCE_MAPPING','LICENSE_REGISTER','CONTENT_AUDIT','DEVELOPMENT_PLAN','CHAPTER_TEMPLATE','ADDING_CHAPTER','TECHNICAL_LANGUAGE_GUIDE'])if(!existsSync(resolve(root,`docs/${name}.md`)))errors.push(`Missing docs/${name}.md`);
 const template=JSON.parse(readFileSync(resolve(root,'content/templates/slides.json'),'utf8'));
 if(template.length<5||!template.some(s=>s.code)||!template.some(s=>s.tex)||!template.every(s=>s.notes))errors.push('Template capabilities incomplete');
 return errors;
}
if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url)){
 const errors=validateAll();if(errors.length){console.error(errors.join('\n'));process.exitCode=1;}else console.log('PASS: 40 chapters, four levels, five tracks; metadata, paths, availability, branding, documentation and presentation template.');
}
