const ALLOWED_REPOSITORIES = new Set([
  'Xenova/owlvit-base-patch32',
  'onnx-community/yolov10n',
]);

function json(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      'Content-Type': 'application/json; charset=utf-8',
      'Cache-Control': 'no-store',
      'X-Content-Type-Options': 'nosniff',
    },
  });
}

function isAllowedModelPath(relativePath) {
  if (!relativePath || relativePath.includes('\\')) return false;
  const parts = relativePath.split('/');
  if (parts.some((part) => !part || part === '.' || part === '..')) return false;
  if (parts.length < 5 || parts[2] !== 'resolve' || parts[3] !== 'main') return false;
  if (!ALLOWED_REPOSITORIES.has(`${parts[0]}/${parts[1]}`)) return false;
  return /^[A-Za-z0-9._/-]+$/.test(parts.slice(4).join('/'));
}

async function handleModelProxy(request, url) {
  const method = request.method.toUpperCase();
  if (method !== 'GET' && method !== 'HEAD') {
    return json({ error: 'Méthode non autorisée.' }, 405);
  }

  const prefix = '/api/hf/';
  let relativePath;
  try {
    relativePath = decodeURIComponent(url.pathname.slice(prefix.length));
  } catch {
    return json({ error: 'Chemin encodé invalide.' }, 400);
  }

  if (relativePath === '_health') {
    try {
      const upstream = await fetch(
        'https://huggingface.co/Xenova/owlvit-base-patch32/resolve/main/config.json',
        { method: 'HEAD', redirect: 'follow' },
      );
      return json({
        ok: upstream.ok,
        upstreamStatus: upstream.status,
        message: upstream.ok
          ? 'Le relais de modèles est accessible.'
          : 'Hugging Face ne fournit pas le fichier demandé.',
      }, upstream.ok ? 200 : 502);
    } catch (error) {
      return json({
        ok: false,
        message: 'Le Worker ne peut pas joindre Hugging Face.',
        detail: String(error?.message || error).slice(0, 240),
      }, 502);
    }
  }

  if (!isAllowedModelPath(relativePath)) {
    return json({ error: 'Chemin de modèle non autorisé.' }, 403);
  }

  const upstreamUrl = `https://huggingface.co/${relativePath}${url.search}`;
  const forwardedHeaders = new Headers();
  for (const name of ['range', 'if-none-match', 'if-modified-since']) {
    const value = request.headers.get(name);
    if (value) forwardedHeaders.set(name, value);
  }

  try {
    const upstream = await fetch(upstreamUrl, {
      method,
      headers: forwardedHeaders,
      redirect: 'follow',
      cf: { cacheEverything: true, cacheTtl: 3600 },
    });

    const responseHeaders = new Headers();
    for (const name of [
      'content-type', 'content-range', 'accept-ranges', 'etag', 'last-modified', 'expires',
    ]) {
      const value = upstream.headers.get(name);
      if (value) responseHeaders.set(name, value);
    }
    responseHeaders.set('Cache-Control', upstream.ok ? 'public, max-age=3600' : 'no-store');
    responseHeaders.set('X-Content-Type-Options', 'nosniff');
    responseHeaders.set('X-Atlas-Model-Proxy', 'cloudflare-worker');

    return new Response(method === 'HEAD' ? null : upstream.body, {
      status: upstream.status,
      statusText: upstream.statusText,
      headers: responseHeaders,
    });
  } catch (error) {
    return json({
      error: 'Échec du téléchargement du modèle depuis Hugging Face.',
      detail: String(error?.message || error).slice(0, 240),
    }, 502);
  }
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname.startsWith('/api/hf/')) {
      return handleModelProxy(request, url);
    }
    return env.ASSETS.fetch(request);
  },
};
