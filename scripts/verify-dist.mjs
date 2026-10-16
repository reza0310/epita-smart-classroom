import { access, stat } from 'node:fs/promises';
import { resolve } from 'node:path';

const dist = resolve(process.cwd(), 'dist');
const indexHtml = resolve(dist, 'index.html');
try {
  const info = await stat(dist);
  if (!info.isDirectory()) throw new Error('dist n’est pas un dossier.');
  await access(indexHtml);
  console.log('Build vérifié : dist/index.html est présent.');
} catch {
  console.error('ERREUR : le build Vite n’a pas créé dist/index.html. Vérifiez que `npm run build` s’exécute depuis la racine du dépôt et se termine sans erreur. Wrangler ne sera pas lancé.');
  process.exit(1);
}
