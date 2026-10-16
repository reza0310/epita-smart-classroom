import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { Readable } from 'node:stream';

const ALLOWED_REPOSITORIES = new Set([
  'Xenova/owlvit-base-patch32',
  'onnx-community/yolov10n',
]);

function validateModelPath(relativePath) {
  if (!relativePath || relativePath.includes('\\') || relativePath.split('/').some((part) => part === '.' || part === '..')) return false;
  const parts = relativePath.split('/');
  if (parts.length < 5 || parts[2] !== 'resolve' || parts[3] !== 'main') return false;
  if (!ALLOWED_REPOSITORIES.has(`${parts[0]}/${parts[1]}`)) return false;
  return /^[A-Za-z0-9._/-]+$/.test(parts.slice(4).join('/'));
}

// En développement, le relais reste sur la même origine que Vite et suit les redirections HF côté serveur.
// En production, l'équivalent est assuré par functions/api/hf/[[path]].js sur Cloudflare Pages.
function huggingFaceDevProxy() {
  return {
    name: 'atlas-huggingface-model-proxy',
    configureServer(server) {
      server.middlewares.use(async (req, res, next) => {
        const requestUrl = new URL(req.url || '/', 'http://localhost');
        if (!requestUrl.pathname.startsWith('/api/hf/')) return next();

        if (!['GET', 'HEAD'].includes((req.method || 'GET').toUpperCase())) {
          res.statusCode = 405;
          res.setHeader('Content-Type', 'application/json; charset=utf-8');
          res.end(JSON.stringify({ error: 'Méthode non autorisée.' }));
          return;
        }

        if (requestUrl.pathname === '/api/hf/_health') {
          try {
            const upstream = await fetch('https://huggingface.co/Xenova/owlvit-base-patch32/resolve/main/config.json', { method: 'HEAD', redirect: 'follow' });
            res.statusCode = upstream.ok ? 200 : 502;
            res.setHeader('Content-Type', 'application/json; charset=utf-8');
            res.end(JSON.stringify({ ok: upstream.ok, upstreamStatus: upstream.status }));
          } catch (error) {
            res.statusCode = 502;
            res.setHeader('Content-Type', 'application/json; charset=utf-8');
            res.end(JSON.stringify({ ok: false, detail: String(error?.message || error) }));
          }
          return;
        }

        const relativePath = requestUrl.pathname.slice('/api/hf/'.length);
        if (!validateModelPath(relativePath)) {
          res.statusCode = 403;
          res.setHeader('Content-Type', 'application/json; charset=utf-8');
          res.end(JSON.stringify({ error: 'Chemin de modèle non autorisé.' }));
          return;
        }

        try {
          const headers = {};
          if (req.headers.range) headers.Range = req.headers.range;
          if (req.headers['if-none-match']) headers['If-None-Match'] = req.headers['if-none-match'];
          if (req.headers['if-modified-since']) headers['If-Modified-Since'] = req.headers['if-modified-since'];
          const upstreamUrl = `https://huggingface.co/${relativePath}${requestUrl.search}`;
          const upstream = await fetch(upstreamUrl, { method: req.method, headers, redirect: 'follow' });
          res.statusCode = upstream.status;
          for (const name of ['content-type', 'content-range', 'accept-ranges', 'etag', 'last-modified', 'expires']) {
            const value = upstream.headers.get(name);
            if (value) res.setHeader(name, value);
          }
          res.setHeader('Cache-Control', upstream.ok ? 'public, max-age=3600' : 'no-store');
          res.setHeader('X-Atlas-Model-Proxy', 'vite-dev');
          if (req.method === 'HEAD' || !upstream.body) {
            res.end();
          } else {
            Readable.fromWeb(upstream.body).on('error', (error) => res.destroy(error)).pipe(res);
          }
        } catch (error) {
          res.statusCode = 502;
          res.setHeader('Content-Type', 'application/json; charset=utf-8');
          res.end(JSON.stringify({ error: 'Échec du relais vers Hugging Face.', detail: String(error?.message || error).slice(0, 240) }));
        }
      });
    },
  };
}

export default defineConfig({
  plugins: [react(), huggingFaceDevProxy()],
  // Transformers.js et ONNX Runtime sont chargés à la demande par les détecteurs secondaires.
  optimizeDeps: {
    exclude: ['@huggingface/transformers'],
  },
  worker: {
    format: 'es',
  },
  assetsInclude: ['**/*.onnx'],
  server: {
    host: '0.0.0.0',
  },
  build: {
    outDir: 'dist',
    target: 'esnext',
  },
});
