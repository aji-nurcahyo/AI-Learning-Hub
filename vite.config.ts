import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { cpSync } from 'node:fs';
export default defineConfig({base:process.env.VITE_BASE_PATH || '/ai-learning-hub/',plugins:[react(),{name:'static-content',closeBundle(){cpSync('content','dist/content',{recursive:true});cpSync('docs','dist/docs',{recursive:true});cpSync('LICENSE.upstream','dist/LICENSE.upstream');cpSync('dist/index.html','dist/404.html');}}],build:{target:'es2022'}});
