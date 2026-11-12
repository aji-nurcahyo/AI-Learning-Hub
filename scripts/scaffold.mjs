import {readFileSync,writeFileSync,existsSync,mkdirSync} from 'node:fs';
import {resolve} from 'node:path';
import {walk,root} from './validate.mjs';
const id=Number(process.argv[2]);
if(!Number.isInteger(id)||id<1||id>40)throw new Error('Usage: node scripts/scaffold.mjs <chapter-id 1..40>');
const metadata=walk(resolve(root,'content')).find(f=>f.endsWith('metadata.json')&&JSON.parse(readFileSync(f,'utf8')).id===id);
const dir=resolve(metadata,'..');
const files={
 'index.md':readFileSync(resolve(root,'docs/CHAPTER_TEMPLATE.md'),'utf8'),
 'slides/deck.json':readFileSync(resolve(root,'content/templates/slides.json'),'utf8'),
 'code/README.md':'# Eksekusi contoh\n\nDRAFT: catat versi Python dan dependensi teruji, lisensi, CPU/GPU, seed, perintah eksekusi dan hasil nyata. Tidak ada contoh yang telah tervalidasi. Jangan mengunduh model pada build website.\n',
 'notebooks/template.ipynb':JSON.stringify({nbformat:4,nbformat_minor:5,metadata:{kernelspec:{display_name:'Python 3',language:'python',name:'python3'}},cells:[{cell_type:'markdown',id:'overview',metadata:{},source:['# Draft notebook\n','Belum dieksekusi. Isi penjelasan, dataset/lisensi, baseline dan evaluasi sebelum publikasi.']},{cell_type:'code',id:'implementation',execution_count:null,metadata:{},outputs:[],source:['# Implementasi yang akan diuji pada tahap penulisan materi.']}]},null,2),
 'exercises/index.md':'# Draft latihan\n\nSusun 3–5 latihan yang dipetakan ke tujuan, termasuk kasus batas dan pertanyaan analisis.\n',
 'assignments/index.md':'# Draft tugas terapan\n\nDefinisikan masalah, deliverable, batasan, reproducibility, dan rubrik 100 poin. Gunakan docs/CHAPTER_TEMPLATE.md.\n',
 'references.md':'# Draft referensi\n\nTambahkan dua sumber kredibel, versi, tanggal akses, lisensi dan atribusi.\n',
 'audit.md':'# Audit chapter\n\nStatus: DRAFT. Review akademik: belum. Eksekusi: belum. Lisensi: belum. UI/slide: belum.\n'
};
for(const [name,body]of Object.entries(files)){const path=resolve(dir,name);if(existsSync(path)){console.log(`Preserved ${name}`);continue;}mkdirSync(resolve(path,'..'),{recursive:true});writeFileSync(path,body);console.log(`Created draft ${name}`);}
console.log('Metadata and availability unchanged. Review and validate before publishing.');
