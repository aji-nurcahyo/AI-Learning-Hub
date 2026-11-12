import {readFileSync,readdirSync,mkdirSync,copyFileSync,writeFileSync,existsSync} from 'node:fs';
import {resolve} from 'node:path';
const manifest=JSON.parse(readFileSync('package.json','utf8'));
const lock=JSON.parse(readFileSync('package-lock.json','utf8'));
const direct=new Set(Object.keys({...manifest.dependencies,...manifest.devDependencies}));
const records=[];
for(const path of Object.keys(lock.packages).filter(p=>p.startsWith('node_modules/'))){
 const dir=resolve(path);if(!existsSync(resolve(dir,'package.json')))continue;
 const pkg=JSON.parse(readFileSync(resolve(dir,'package.json'),'utf8'));
 const licenses=readdirSync(dir,{withFileTypes:true}).filter(f=>f.isFile()&&/^(license|licence|notice|copying)(\.|$)/i.test(f.name)).map(f=>f.name);
 const destName=path.replaceAll('/','__');const dest=resolve('public/licenses',destName);mkdirSync(dest,{recursive:true});
 for(const file of licenses)copyFileSync(resolve(dir,file),resolve(dest,file));
 records.push({name:pkg.name,version:pkg.version,direct:direct.has(pkg.name),license:pkg.license,repository:pkg.repository,notices:licenses.map(f=>`licenses/${destName}/${f}`)});
}
writeFileSync('docs/DEPENDENCY_LICENSES.json',JSON.stringify(records,null,2)+'\n');
console.log(`Recorded versions and notices for ${records.length} installed packages, including transitive packages.`);
