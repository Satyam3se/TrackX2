import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Dev server reverse-proxies the Django backend so the SPA talks to a single
// origin in development exactly like it does in production (Nginx same-origin).
const BACKEND = process.env.VITE_DEV_BACKEND ?? 'http://localhost:8000';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    host: true,
    strictPort: true,
    watch: {
      ignored: ['**/public/sample_videos/**', '**/*.mp4', '**/*.mov'],
    },
    proxy: {
      '/api': { target: BACKEND, changeOrigin: true },
      '/ws': { target: BACKEND.replace(/^http/, 'ws'), ws: true },
      '/media': { target: BACKEND, changeOrigin: true },
      '/admin': { target: BACKEND, changeOrigin: true },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
  },
});
