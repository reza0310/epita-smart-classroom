import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(scriptDir, '..');
const destination = path.join(projectRoot, 'public', 'ort');

const candidates = [];
try {
  const entry = require.resolve('onnxruntime-web');
  candidates.push(path.dirname(entry));
} catch {
  // Le chemin direct est également utilisé pour les installations npm classiques.
}
candidates.push(
  path.join(projectRoot, 'node_modules', 'onnxruntime-web', 'dist'),
  path.join(projectRoot, 'node_modules', '@huggingface', 'transformers', 'node_modules', 'onnxruntime-web', 'dist'),
);

const source = candidates.find((candidate) => fs.existsSync(candidate) && fs.statSync(candidate).isDirectory());
if (!source) {
  console.error('[Atlas] Impossible de localiser onnxruntime-web/dist. Lancez npm install avant le démarrage ou le build.');
  process.exitCode = 1;
} else {
  fs.mkdirSync(destination, { recursive: true });
  // Copier les fichiers runtime nécessaires au backend WASM sur la même origine que l'application.
  // Cela évite une requête vers jsDelivr et les erreurs réseau/CORS associées.
  const files = fs.readdirSync(source).filter((name) => /^(ort-wasm.*\.(?:wasm|mjs|js)|ort\.wasm.*)$/i.test(name));
  if (files.length === 0) {
    console.error(`[Atlas] Aucun fichier WASM ONNX Runtime n'a été trouvé dans ${source}.`);
    process.exitCode = 1;
  } else {
    for (const file of files) fs.copyFileSync(path.join(source, file), path.join(destination, file));
    console.log(`[Atlas] ${files.length} fichiers ONNX Runtime copiés vers public/ort (origine locale).`);
  }
}
