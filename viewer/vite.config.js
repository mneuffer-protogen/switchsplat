import {defineConfig} from 'vite';
import basicSsl from '@vitejs/plugin-basic-ssl';

// `npm run dev -- --host` serves HTTPS (a self-signed certificate) so a headset on the LAN gets the secure
// context WebXR needs. `npm run dev:http` (mode "http") is plain HTTP on the same fixed port for local
// browser testing on localhost.
export default defineConfig(({mode})=>({
 base:'./',
 plugins:mode==='http'?[]:[basicSsl()],
 server:{port:5198,strictPort:true},
 preview:{port:5198,strictPort:true},
 build:{outDir:'dist',emptyOutDir:true,target:'es2022',chunkSizeWarningLimit:4000},
}));
