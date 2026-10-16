import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  // Transformers.js utilise ONNX Runtime WASM et des ressources chargées à la demande.
  // Exclure le paquet du pre-bundling évite certains problèmes de workers/WASM en dev.
  optimizeDeps: {
    exclude: ['@huggingface/transformers'],
  },
  worker: {
    format: 'es',
  },
  assetsInclude: ['**/*.onnx'],
  build: {
    outDir: 'dist',
    target: 'esnext',
  },
});
