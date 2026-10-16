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

function validateModelPath(relativePath) {
  // Expected shape: <org>/<repo>/resolve/main/<file path>
  if (!relativePath || relativePath.includes('\\') || relativePath.split('/').some((part) => part === '.' || part === '..')) {
    return false;
  }
  const parts = relativePath.split('/');
  if (parts.length < 5 || parts[2] !== 'resolve' || parts[3] !== 'main') return false;
  const repository = `${parts[0]}/${parts[1]}`;
  if (!ALLOWED_REPOSITORIES.has(repository)) return false;
  const filePath = parts.slice(4).join('/');
  return Boolean(filePath) && /^[A-Za-z0-9._/-]+$/.test(filePath);
}

export async function onRequest({ request }) {
  const requestUrl = new URL(request.url);
  const method = request.method.toUpperCase();
  if (method !== 'GET' && method !== 'HEAD') {
    return json({ error: 'Méthode non autorisée.' }, 405);
  }

  const prefix = '/api/hf/';
  if (!requestUrl.pathname.startsWith(prefix)) return json({ error: 'Chemin invalide.' }, 404);
  let relativePath;
  try {
    relativePath = decodeURIComponent(requestUrl.pathname.slice(prefix.length));
  } catch {
    return json({ error: 'Chemin encodé invalide.' }, 400);
  }

  if (relativePath === '_health') {
    try {
      const upstream = await fetch('https://huggingface.co/Xenova/owlvit-base-patch32/resolve/main/config.json', {
        method: 'HEAD',
        redirect: 'follow',
      });
      return json({
        ok: upstream.ok,
        upstreamStatus: upstream.status,
        message: upstream.ok ? 'Le relais de modèles est accessible.' : 'Hugging Face a refusé ou ne peut pas fournir le fichier demandé.',
      }, upstream.ok ? 200 : 502);
    } catch (error) {
      return json({ ok: false, message: 'Le relais Cloudflare ne peut pas joindre Hugging Face.', detail: String(error?.message || error).slice(0, 240) }, 502);
    }
  }

  if (!validateModelPath(relativePath)) {
    return json({ error: 'Le chemin ne correspond pas à un fichier autorisé des modèles Atlas.' }, 403);
  }

  // Le proxy est limité à deux dépôts publics connus : il ne peut pas servir de proxy arbitraire.
  const upstreamUrl = `https://huggingface.co/${relativePath}${requestUrl.search}`;
  const forwardedHeaders = new Headers();
  const range = request.headers.get('range');
  if (range) forwardedHeaders.set('Range', range);
  const ifNoneMatch = request.headers.get('if-none-match');
  if (ifNoneMatch) forwardedHeaders.set('If-None-Match', ifNoneMatch);
  const ifModifiedSince = request.headers.get('if-modified-since');
  if (ifModifiedSince) forwardedHeaders.set('If-Modified-Since', ifModifiedSince);

  try {
    const upstream = await fetch(upstreamUrl, {
      method,
      headers: forwardedHeaders,
      redirect: 'follow',
      // Les fichiers des modèles sont immuables pour une révision donnée dans la pratique ;
      // une durée courte évite de conserver trop longtemps un éventuel état transitoire.
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
    responseHeaders.set('X-Atlas-Model-Proxy', 'cloudflare-pages');

    return new Response(method === 'HEAD' ? null : upstream.body, {
      status: upstream.status,
      statusText: upstream.statusText,
      headers: responseHeaders,
    });
  } catch (error) {
    return json({
      error: 'Échec de téléchargement du fichier du modèle depuis Hugging Face.',
      detail: String(error?.message || error).slice(0, 240),
    }, 502);
  }
}
