import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import './styles.css';

const CATEGORIES = [
  {
    id: 'chair', label: 'Chaises', singular: 'Chaise', modelClass: 'chair', method: 'auto',
    expectedDefault: '4', color: '#4ee0a1', symbol: 'chair', family: 'Mobilier',
    detail: 'Détection COCO-SSD',
  },
  {
    id: 'table', label: 'Tables & bureaux', singular: 'Table / bureau', modelClass: 'dining table', method: 'auto',
    expectedDefault: '1', color: '#70b8ff', symbol: 'table', family: 'Mobilier',
    detail: 'COCO-SSD détecte surtout les tables de repas ; OWL-ViT peut proposer des bureaux sur image fixe.',
  },
  {
    id: 'laptop', label: 'Ordinateurs portables', singular: 'Ordinateur portable', modelClass: 'laptop', method: 'auto',
    expectedDefault: '1', color: '#c2a3ff', symbol: 'laptop', family: 'Informatique',
    detail: 'Détection COCO-SSD',
  },
  {
    id: 'screen', label: 'Écrans', singular: 'Écran', modelClass: 'tv', method: 'auto',
    expectedDefault: '1', color: '#ffbd72', symbol: 'screen', family: 'Informatique',
    detail: 'La classe « tv » peut aussi correspondre à un téléviseur.',
  },
  {
    id: 'board', label: 'Tableaux', singular: 'Tableau', method: 'manual',
    expectedDefault: '1', color: '#f7db65', symbol: 'board', family: 'Éléments de salle',
    detail: 'OWL-ViT peut proposer des cadres ; confirmation humaine recommandée.',
  },
  {
    id: 'window', label: 'Fenêtres', singular: 'Fenêtre', method: 'manual',
    expectedDefault: '2', color: '#79d9de', symbol: 'window', family: 'Éléments de salle',
    detail: 'OWL-ViT peut proposer des cadres ; confirmation humaine recommandée.',
  },
  {
    id: 'anomaly', label: 'Anomalies visuelles', singular: 'Anomalie', method: 'manual',
    expectedDefault: '0', color: '#ff8585', symbol: 'alert', family: 'Contrôle visuel',
    detail: 'Évaluation humaine requise ; aucune détection automatique d’anomalie.',
  },
];

const AUTO_IDS = CATEGORIES.filter((c) => c.method === 'auto').map((c) => c.id);
const OWL_IDS = ['table', 'board', 'window'];
// Les libellés décrivent uniquement du mobilier et des éléments de salle ; aucune classe personne n'est proposée.
const OWL_PROMPTS = [
  { prompt: 'a whiteboard', categoryId: 'board' },
  { prompt: 'a classroom whiteboard', categoryId: 'board' },
  { prompt: 'a chalkboard', categoryId: 'board' },
  { prompt: 'a blackboard mounted on a wall', categoryId: 'board' },
  { prompt: 'a window', categoryId: 'window' },
  { prompt: 'a window in a room', categoryId: 'window' },
  { prompt: 'an office desk', categoryId: 'table' },
  { prompt: 'a computer desk', categoryId: 'table' },
  { prompt: 'a work desk', categoryId: 'table' },
  { prompt: 'a table in a room', categoryId: 'table' },
];
const OWL_CATEGORY_IDS = new Set(OWL_IDS);
const OWL_MODEL_ID = 'Xenova/owlvit-base-patch32';
const DENSE_MODEL_ID = 'onnx-community/yolov10n';
const DENSE_CLASS_TO_CATEGORY = { chair: 'chair', 'dining table': 'table', laptop: 'laptop', tv: 'screen' };
const INITIAL_EXPECTED = Object.fromEntries(CATEGORIES.map((category) => [category.id, category.expectedDefault]));
const EMPTY_MANUAL = { board: '', window: '', anomaly: '' };
const EMPTY_COUNTS = Object.fromEntries(CATEGORIES.map((category) => [category.id, 0]));

function Icon({ name, size = 20, strokeWidth = 1.8 }) {
  const paths = {
    grid: <><rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/></>,
    camera: <><path d="M14 5 16 8h3a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-8a2 2 0 0 1 2-2h3l2-3z"/><circle cx="12" cy="13" r="3.2"/></>,
    stop: <><rect x="5" y="5" width="14" height="14" rx="2.3"/></>,
    report: <><path d="M7 3h7l4 4v14H7a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2Z"/><path d="M14 3v5h5M9 12h6M9 16h6"/></>,
    shield: <><path d="M12 22s8-4 8-11V5l-8-3-8 3v6c0 7 8 11 8 11Z"/><path d="m9 12 2 2 4-4"/></>,
    chair: <><path d="M7 11V5.5A2.5 2.5 0 0 1 9.5 3h5A2.5 2.5 0 0 1 17 5.5V11"/><path d="M5 11h14v4H5zM7 15v6M17 15v6"/></>,
    table: <><path d="M3 8h18v3H3zM5 11v10M19 11v10"/><path d="M8 8V5h8v3"/></>,
    laptop: <><rect x="5" y="3" width="14" height="12" rx="1.5"/><path d="M2.5 18h19l-1.5 3H4z"/></>,
    screen: <><rect x="3" y="3" width="18" height="13" rx="1.5"/><path d="M12 16v5M8 21h8"/></>,
    board: <><rect x="3" y="3" width="18" height="14" rx="1"/><path d="M7 17v4M17 17v4M7 7h5M7 11h9"/></>,
    window: <><rect x="3" y="3" width="18" height="18" rx="1"/><path d="M12 3v18M3 12h18"/></>,
    alert: <><path d="m12 3 9 16H3L12 3Z"/><path d="M12 9v4M12 16h.01"/></>,
    target: <><circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1"/></>,
    check: <><path d="m5 12 4 4L19 6"/></>,
    lock: <><rect x="4" y="10" width="16" height="11" rx="2"/><path d="M8 10V7a4 4 0 0 1 8 0v3"/></>,
    download: <><path d="M12 3v12M7 10l5 5 5-5M4 20h16"/></>,
    arrow: <><path d="M5 12h14M13 6l6 6-6 6"/></>,
    close: <><path d="m6 6 12 12M18 6 6 18"/></>,
    refresh: <><path d="M20 7v5h-5M4 17v-5h5"/><path d="M5.5 9A7 7 0 0 1 18 6l2 6M4 12l2 6a7 7 0 0 0 12.5-3"/></>,
    info: <><circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/></>,
  };
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name] || paths.info}</svg>;
}

function getValue(value) {
  if (value === '' || value === null || value === undefined) return null;
  const number = Number(value);
  return Number.isFinite(number) && number >= 0 ? Math.floor(number) : null;
}

function makeRows(expectedValues, autoCounts, manualCounts, observed, snapshotMode = false, owlObserved = false) {
  return CATEGORIES.map((category) => {
    const expected = getValue(expectedValues[category.id]);
    const owlUsed = Boolean(owlObserved && OWL_CATEGORY_IDS.has(category.id));
    const machineChecked = category.method === 'auto' || owlUsed;
    const detected = machineChecked
      ? (observed ? (autoCounts[category.id] ?? 0) : null)
      : getValue(manualCounts[category.id]);
    let status = 'En attente';
    let statusType = 'pending';
    let gap = null;
    let note = '';

    if (expected === null) {
      status = 'Objectif à définir';
      statusType = 'pending';
    } else if (machineChecked && detected === null) {
      status = snapshotMode ? 'Non relevé par le modèle' : 'En attente d’une analyse';
      statusType = 'pending';
    } else if (!machineChecked && detected === null) {
      status = 'Relevé manuel requis';
      statusType = 'pending';
    } else if (machineChecked && detected === 0) {
      status = expected > 0 ? '0 détecté · à confirmer' : '0 détecté · à vérifier';
      statusType = 'caution';
      note = 'Une absence de détection ne prouve pas l’absence de l’objet ; vérifier le cadrage, les occultations et les faux négatifs.';
    } else if (detected === expected) {
      status = 'Concordant au relevé';
      statusType = 'good';
    } else {
      gap = detected - expected;
      status = machineChecked ? 'Écart de comptage observé' : 'Écart relevé manuellement';
      statusType = 'bad';
      note = machineChecked
        ? 'Le nombre détecté diffère de l’attendu ; vérifier les cadres, les occultations et les éventuels doublons.'
        : 'Écart déclaré lors du contrôle manuel.';
    }

    return { ...category, expected, detected, gap, status, statusType, note, machineChecked, owlUsed };
  });
}

function summarize(rows) {
  const configured = rows.filter((row) => row.expected !== null).length;
  const checked = rows.filter((row) => row.expected !== null && row.detected !== null).length;
  const deviations = rows.filter((row) => row.statusType === 'bad').length;
  const unresolved = rows.filter((row) => row.statusType === 'pending' || row.statusType === 'caution').length;
  const total = rows.reduce((sum, row) => sum + (row.detected ?? 0), 0);
  const recognized = rows.filter((row) => (row.detected ?? 0) > 0).length;
  let status = 'Prêt à relever';
  let statusType = 'pending';
  if (deviations > 0) {
    status = 'Écarts observés';
    statusType = 'bad';
  } else if (checked > 0 && unresolved > 0) {
    status = 'Vérification partielle';
    statusType = 'caution';
  } else if (configured > 0 && checked === configured && unresolved === 0) {
    status = 'Concordant au relevé';
    statusType = 'good';
  }
  return { configured, checked, deviations, unresolved, total, recognized, status, statusType };
}

function formatDate(dateValue) {
  if (!dateValue) return 'Pas encore enregistré';
  return new Intl.DateTimeFormat('fr-FR', { dateStyle: 'long', timeStyle: 'short' }).format(new Date(dateValue));
}

function StatusPill({ type = 'pending', children }) {
  return <span className={`status-pill status-${type}`}><span className="status-dot" />{children}</span>;
}

function StatCard({ label, value, detail, icon, tone = '' }) {
  return <article className={`stat-card ${tone}`}>
    <div className="stat-top"><span className="stat-label">{label}</span><span className="stat-icon"><Icon name={icon} size={18} /></span></div>
    <div className="stat-value">{value}</div>
    <div className="stat-detail">{detail}</div>
  </article>;
}

function CategoryIcon({ category, size = 20 }) {
  return <span className="category-icon" style={{ '--category-color': category.color }}><Icon name={category.symbol} size={size} /></span>;
}

const DB_NAME = 'atlas-visual-audit';
const DB_STORE = 'training-samples';

function openTrainingDb() {
  return new Promise((resolve, reject) => {
    if (!('indexedDB' in window)) return reject(new Error('IndexedDB indisponible dans ce navigateur.'));
    const request = indexedDB.open(DB_NAME, 1);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(DB_STORE)) db.createObjectStore(DB_STORE, { keyPath: 'id' });
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error || new Error('Ouverture de la base locale impossible.'));
  });
}

async function readTrainingSamples() {
  const db = await openTrainingDb();
  return new Promise((resolve, reject) => {
    const request = db.transaction(DB_STORE, 'readonly').objectStore(DB_STORE).getAll();
    request.onsuccess = () => { db.close(); resolve((request.result || []).sort((a, b) => b.createdAt.localeCompare(a.createdAt))); };
    request.onerror = () => { db.close(); reject(request.error || new Error('Lecture des exemples impossible.')); };
  });
}

async function writeTrainingSample(sample) {
  const db = await openTrainingDb();
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(DB_STORE, 'readwrite');
    transaction.objectStore(DB_STORE).put(sample);
    transaction.oncomplete = () => { db.close(); resolve(sample); };
    transaction.onerror = () => { db.close(); reject(transaction.error || new Error('Enregistrement local impossible.')); };
  });
}

async function removeTrainingSample(id) {
  const db = await openTrainingDb();
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(DB_STORE, 'readwrite');
    transaction.objectStore(DB_STORE).delete(id);
    transaction.oncomplete = () => { db.close(); resolve(); };
    transaction.onerror = () => { db.close(); reject(transaction.error || new Error('Suppression impossible.')); };
  });
}

function downloadJsonFile(data, fileName) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = fileName;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1500);
}

function readFileAsDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(new Error('Lecture de l’image impossible.'));
    reader.readAsDataURL(file);
  });
}

function loadImageElement(src) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error('Cette image n’a pas pu être ouverte.'));
    image.src = src;
  });
}

// Réduit les très grandes images avant les modèles Transformers.js pour limiter
// les allocations de mémoire sur mobile. Les cadres sont ensuite reprojetés à l'échelle source.
function fitCanvasForInference(sourceCanvas, maxDimension = 1280) {
  const sourceWidth = sourceCanvas.width;
  const sourceHeight = sourceCanvas.height;
  const scale = Math.min(1, maxDimension / Math.max(sourceWidth, sourceHeight));
  if (scale >= 1) return { canvas: sourceCanvas, scaleX: 1, scaleY: 1 };
  const canvas = document.createElement('canvas');
  canvas.width = Math.max(1, Math.round(sourceWidth * scale));
  canvas.height = Math.max(1, Math.round(sourceHeight * scale));
  const context = canvas.getContext('2d');
  if (!context) throw new Error('Impossible de préparer une image réduite pour le modèle.');
  context.drawImage(sourceCanvas, 0, 0, canvas.width, canvas.height);
  return { canvas, scaleX: sourceWidth / canvas.width, scaleY: sourceHeight / canvas.height };
}

const LEARNABLE_CATEGORIES = CATEGORIES.map((category) => ({
  id: category.id,
  label: category.label,
  color: category.color,
}));

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const fixedImageRef = useRef(null);
  const fixedImageCanvasRef = useRef(null);
  const fileInputRef = useRef(null);
  const datasetInputRef = useRef(null);
  const drawStartRef = useRef(null);
  const modelRef = useRef(null);
  const owlModelRef = useRef(null);
  const owlLoadPromiseRef = useRef(null);
  const denseModelRef = useRef(null);
  const denseLoadPromiseRef = useRef(null);
  const transformersModuleRef = useRef(null);
  const owlVideoPredictionsRef = useRef([]);
  const denseVideoPredictionsRef = useRef([]);
  const lastOwlVideoRunAtRef = useRef(0);
  const lastDenseVideoRunAtRef = useRef(0);
  const owlThresholdRef = useRef(0.12);
  const busyRef = useRef(false);
  const enhancedBusyRef = useRef(false);
  const thresholdRef = useRef(0.55);
  const [stream, setStream] = useState(null);
  const [cameraOn, setCameraOn] = useState(false);
  const [model, setModel] = useState(null);
  const [modelStatus, setModelStatus] = useState('idle');
  const [modelError, setModelError] = useState('');
  const [owlModelStatus, setOwlModelStatus] = useState('idle');
  const [owlModelError, setOwlModelError] = useState('');
  const [owlProgress, setOwlProgress] = useState(0);
  const [owlEnabled, setOwlEnabled] = useState(false);
  const [owlVideoEnabled, setOwlVideoEnabled] = useState(false);
  const [denseSceneEnabled, setDenseSceneEnabled] = useState(false);
  const [denseVideoEnabled, setDenseVideoEnabled] = useState(false);
  const [denseModelStatus, setDenseModelStatus] = useState('idle');
  const [denseModelError, setDenseModelError] = useState('');
  const [denseProgress, setDenseProgress] = useState(0);
  const [owlThreshold, setOwlThreshold] = useState(0.12);
  const [owlObserved, setOwlObserved] = useState(false);
  const [imageOwlObserved, setImageOwlObserved] = useState(false);
  const [cameraError, setCameraError] = useState('');
  const [threshold, setThreshold] = useState(0.55);
  const [objects, setObjects] = useState([]);
  const [counts, setCounts] = useState({ ...EMPTY_COUNTS });
  const [hasObservedFrame, setHasObservedFrame] = useState(false);
  const [lastInferenceAt, setLastInferenceAt] = useState(null);
  const [expected, setExpected] = useState({ ...INITIAL_EXPECTED });
  const [manual, setManual] = useState({ ...EMPTY_MANUAL });
  const [auditSnapshot, setAuditSnapshot] = useState(null);
  const [view, setView] = useState('live');
  const [activeSection, setActiveSection] = useState('audit');
  const [notice, setNotice] = useState('');
  const [showMethodNotes, setShowMethodNotes] = useState(false);
  const [sourceMode, setSourceMode] = useState('camera');
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePredictions, setImagePredictions] = useState([]);
  const [imageAnalyzed, setImageAnalyzed] = useState(false);
  const [annotationMode, setAnnotationMode] = useState(false);
  const [annotationCategory, setAnnotationCategory] = useState('board');
  const [imageAnnotations, setImageAnnotations] = useState([]);
  const [pendingBox, setPendingBox] = useState(null);
  const [trainingSamples, setTrainingSamples] = useState([]);
  const [editingSampleId, setEditingSampleId] = useState(null);
  const [datasetBusy, setDatasetBusy] = useState(false);

  thresholdRef.current = threshold;
  owlThresholdRef.current = owlThreshold;

  useEffect(() => {
    let active = true;
    readTrainingSamples().then((samples) => { if (active) setTrainingSamples(samples); }).catch((error) => {
      console.warn('Training dataset initialization:', error);
      if (active) setNotice('La base locale d’exemples n’est pas disponible dans ce navigateur.');
    });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    return () => {
      if (stream) stream.getTracks().forEach((track) => track.stop());
    };
  }, [stream]);

  useEffect(() => {
    if (!stream || !videoRef.current) return;
    videoRef.current.srcObject = stream;
    videoRef.current.play().catch(() => {});
  }, [stream]);

  const stopCamera = useCallback(() => {
    if (stream) stream.getTracks().forEach((track) => track.stop());
    setStream(null);
    setCameraOn(false);
    setCameraError('');
    owlVideoPredictionsRef.current = [];
    denseVideoPredictionsRef.current = [];
    lastOwlVideoRunAtRef.current = 0;
    lastDenseVideoRunAtRef.current = 0;
    setNotice('Caméra arrêtée. Le dernier relevé reste disponible pour consultation.');
    if (videoRef.current) videoRef.current.srcObject = null;
  }, [stream]);

  const loadDetectionModel = async () => {
    setModelError('');
    setModelStatus('loading');
    try {
      await tf.ready();
      const loadedModel = await cocoSsd.load({ base: 'lite_mobilenet_v2' });
      modelRef.current = loadedModel;
      setModel(loadedModel);
      setModelStatus('ready');
      return true;
    } catch (error) {
      console.error('COCO-SSD load error', error);
      setModelStatus('error');
      setModelError('Le modèle n’a pas pu être chargé. Vérifiez la connexion au premier chargement, puis réessayez.');
      return false;
    }
  };

  const retryModelLoad = async () => {
    await loadDetectionModel();
  };

  const getTransformersModule = async () => {
    if (transformersModuleRef.current) return transformersModuleRef.current;
    const module = await import('@huggingface/transformers');
    // En application navigateur, les modèles sont récupérés depuis Hugging Face et
    // le cache navigateur est autorisé. Aucun serveur applicatif ne reçoit les images.
    module.env.allowLocalModels = false;
    module.env.allowRemoteModels = true;
    module.env.useBrowserCache = true;
    // Le mono-thread WASM évite des pannes de Worker / SharedArrayBuffer sur certains navigateurs mobiles.
    if (module.env.backends?.onnx?.wasm) module.env.backends.onnx.wasm.numThreads = 1;
    transformersModuleRef.current = module;
    return module;
  };

  const friendlyModelError = (error, modelName) => {
    const detail = String(error?.message || error || 'Erreur inconnue').replace(/\s+/g, ' ').slice(0, 360);
    return `${modelName} n’a pas pu être chargé ou exécuté. Vérifiez la connexion, l’espace mémoire disponible et le chargement des fichiers du modèle. Détail : ${detail}`;
  };

  const loadOwlModel = async () => {
    if (owlModelRef.current) return owlModelRef.current;
    if (owlLoadPromiseRef.current) return owlLoadPromiseRef.current;
    setOwlModelStatus('loading');
    setOwlModelError('');
    setOwlProgress(0);
    owlLoadPromiseRef.current = (async () => {
      try {
        const { pipeline } = await getTransformersModule();
        // Le modèle ONNX q8 est le compromis par défaut pour WASM / navigateur mobile.
        const detector = await pipeline('zero-shot-object-detection', OWL_MODEL_ID, {
          // Force le backend navigateur/WASM (pas de backend Node ni de GPU supposé).
          device: 'wasm',
          dtype: 'q8',
          progress_callback: (progress) => {
            if (progress?.status === 'progress' && Number.isFinite(progress.progress)) {
              setOwlProgress(Math.max(0, Math.min(100, Math.round(progress.progress))));
            } else if (progress?.status === 'initiate') {
              setOwlProgress(0);
            }
          },
        });
        owlModelRef.current = detector;
        setOwlModelStatus('ready');
        setOwlProgress(100);
        return detector;
      } catch (error) {
        console.error('OWL-ViT load error', error);
        setOwlModelStatus('error');
        setOwlModelError(friendlyModelError(error, 'OWL-ViT'));
        throw error;
      } finally {
        owlLoadPromiseRef.current = null;
      }
    })();
    return owlLoadPromiseRef.current;
  };

  const loadDenseModel = async () => {
    if (denseModelRef.current) return denseModelRef.current;
    if (denseLoadPromiseRef.current) return denseLoadPromiseRef.current;
    setDenseModelStatus('loading');
    setDenseModelError('');
    setDenseProgress(0);
    denseLoadPromiseRef.current = (async () => {
      try {
        const { pipeline } = await getTransformersModule();
        const detector = await pipeline('object-detection', DENSE_MODEL_ID, {
          device: 'wasm',
          dtype: 'q8',
          progress_callback: (progress) => {
            if (progress?.status === 'progress' && Number.isFinite(progress.progress)) {
              setDenseProgress(Math.max(0, Math.min(100, Math.round(progress.progress))));
            } else if (progress?.status === 'initiate') {
              setDenseProgress(0);
            }
          },
        });
        denseModelRef.current = detector;
        setDenseModelStatus('ready');
        setDenseProgress(100);
        return detector;
      } catch (error) {
        console.error('YOLOv10n load error', error);
        setDenseModelStatus('error');
        setDenseModelError(friendlyModelError(error, 'YOLOv10n'));
        throw error;
      } finally {
        denseLoadPromiseRef.current = null;
      }
    })();
    return denseLoadPromiseRef.current;
  };

  const detectWithOwl = async (canvas) => {
    const detector = await loadOwlModel();
    const fitted = fitCanvasForInference(canvas, 1280);
    const outputs = await detector(fitted.canvas, OWL_PROMPTS.map((item) => item.prompt), {
      threshold: owlThresholdRef.current,
      top_k: 80,
    });
    return outputs.map((prediction, index) => {
      const prompt = OWL_PROMPTS.find((item) => item.prompt === prediction.label);
      if (!prompt) return null;
      const category = CATEGORIES.find((item) => item.id === prompt.categoryId);
      const box = prediction.box || {};
      const xmin = Math.max(0, Math.min(fitted.canvas.width, Number(box.xmin) || 0));
      const ymin = Math.max(0, Math.min(fitted.canvas.height, Number(box.ymin) || 0));
      const xmax = Math.max(xmin, Math.min(fitted.canvas.width, Number(box.xmax) || 0));
      const ymax = Math.max(ymin, Math.min(fitted.canvas.height, Number(box.ymax) || 0));
      if (xmax - xmin < 4 || ymax - ymin < 4) return null;
      return {
        id: `owl-${prompt.categoryId}-${index}-${Date.now()}`,
        categoryId: prompt.categoryId,
        label: category.label,
        color: category.color,
        score: Number(prediction.score) || 0,
        bbox: [xmin * fitted.scaleX, ymin * fitted.scaleY, (xmax - xmin) * fitted.scaleX, (ymax - ymin) * fitted.scaleY],
        source: 'OWL-ViT',
      };
    }).filter(Boolean);
  };

  const detectWithDenseModel = async (canvas) => {
    const detector = await loadDenseModel();
    const fitted = fitCanvasForInference(canvas, 1280);
    // Même chemin d'entrée que l'API officielle : le pipeline accepte le canvas.
    const outputs = await detector(fitted.canvas, {
      threshold: Math.min(0.85, Math.max(0.12, thresholdRef.current - 0.08)),
      percentage: false,
    });
    return (Array.isArray(outputs) ? outputs : []).map((prediction, index) => {
      const categoryId = DENSE_CLASS_TO_CATEGORY[String(prediction.label || '').toLowerCase().trim()];
      if (!categoryId) return null;
      const category = CATEGORIES.find((item) => item.id === categoryId);
      const box = prediction.box || {};
      const xmin = Math.max(0, Math.min(fitted.canvas.width, Number(box.xmin) || 0));
      const ymin = Math.max(0, Math.min(fitted.canvas.height, Number(box.ymin) || 0));
      const xmax = Math.max(xmin, Math.min(fitted.canvas.width, Number(box.xmax) || 0));
      const ymax = Math.max(ymin, Math.min(fitted.canvas.height, Number(box.ymax) || 0));
      if (xmax - xmin < 3 || ymax - ymin < 3) return null;
      return {
        id: `yolo-${categoryId}-${index}-${Date.now()}`,
        categoryId,
        label: category.label,
        color: category.color,
        score: Number(prediction.score) || 0,
        bbox: [xmin * fitted.scaleX, ymin * fitted.scaleY, (xmax - xmin) * fitted.scaleX, (ymax - ymin) * fitted.scaleY],
        source: 'YOLOv10n',
      };
    }).filter((item) => item && item.score >= Math.min(0.85, Math.max(0.12, thresholdRef.current - 0.08)));
  };

  const detectCocoTiled = async (image) => {
    const width = image.naturalWidth || image.width;
    const height = image.naturalHeight || image.height;
    const tileWidth = Math.min(width, Math.max(320, Math.ceil(width * 0.62)));
    const tileHeight = Math.min(height, Math.max(320, Math.ceil(height * 0.62)));
    const startsX = [...new Set([0, Math.max(0, width - tileWidth)])];
    const startsY = [...new Set([0, Math.max(0, height - tileHeight)])];
    const detections = [];
    let tileIndex = 0;
    for (const top of startsY) {
      for (const left of startsX) {
        const tileCanvas = document.createElement('canvas');
        tileCanvas.width = tileWidth;
        tileCanvas.height = tileHeight;
        const context = tileCanvas.getContext('2d');
        if (!context) continue;
        context.drawImage(image, left, top, tileWidth, tileHeight, 0, 0, tileWidth, tileHeight);
        const predictions = await modelRef.current.detect(tileCanvas, 40, thresholdRef.current);
        predictions.forEach((prediction, index) => {
          const category = CATEGORIES.find((item) => item.modelClass === prediction.class && AUTO_IDS.includes(item.id));
          if (!category) return;
          detections.push({
            id: `coco-tile-${tileIndex}-${prediction.class}-${index}`,
            categoryId: category.id,
            label: category.label,
            color: category.color,
            score: prediction.score,
            bbox: [prediction.bbox[0] + left, prediction.bbox[1] + top, prediction.bbox[2], prediction.bbox[3]],
            source: 'COCO-SSD · zones',
          });
        });
        tileIndex += 1;
      }
    }
    return detections;
  };

  const boxIoU = (boxA, boxB) => {
    const left = Math.max(boxA[0], boxB[0]);
    const top = Math.max(boxA[1], boxB[1]);
    const right = Math.min(boxA[0] + boxA[2], boxB[0] + boxB[2]);
    const bottom = Math.min(boxA[1] + boxA[3], boxB[1] + boxB[3]);
    const intersection = Math.max(0, right - left) * Math.max(0, bottom - top);
    const union = boxA[2] * boxA[3] + boxB[2] * boxB[3] - intersection;
    return union > 0 ? intersection / union : 0;
  };

  const suppressDuplicateBoxes = (detections, iouThreshold = 0.52) => {
    const sorted = [...detections].sort((a, b) => b.score - a.score);
    const kept = [];
    sorted.forEach((candidate) => {
      const duplicate = kept.some((accepted) => accepted.categoryId === candidate.categoryId && boxIoU(accepted.bbox, candidate.bbox) >= iouThreshold);
      if (!duplicate) kept.push(candidate);
    });
    return kept;
  };

  const handleImageUpload = async (event) => {
    const file = event.target.files?.[0];
    event.target.value = '';
    if (!file) return;
    if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) {
      setNotice('Choisissez une image au format JPEG, PNG ou WebP.');
      return;
    }
    try {
      const sourceDataUrl = await readFileAsDataUrl(file);
      const image = await loadImageElement(sourceDataUrl);
      const sourceWidth = image.naturalWidth || image.width;
      const sourceHeight = image.naturalHeight || image.height;
      const maxSide = 2048;
      const scale = Math.min(1, maxSide / Math.max(sourceWidth, sourceHeight));
      const width = Math.max(1, Math.round(sourceWidth * scale));
      const height = Math.max(1, Math.round(sourceHeight * scale));
      const normalizedCanvas = document.createElement('canvas');
      normalizedCanvas.width = width;
      normalizedCanvas.height = height;
      const normalizedContext = normalizedCanvas.getContext('2d');
      if (!normalizedContext) throw new Error('Le navigateur ne peut pas préparer cette image.');
      if (file.type !== 'image/png') {
        normalizedContext.fillStyle = '#ffffff';
        normalizedContext.fillRect(0, 0, width, height);
      }
      normalizedContext.drawImage(image, 0, 0, width, height);
      const dataUrl = normalizedCanvas.toDataURL(file.type === 'image/png' ? 'image/png' : 'image/jpeg', 0.92);
      setSelectedImage({
        name: file.name || 'image-importee',
        dataUrl,
        width,
        height,
      });
      setSourceMode('image');
      setEditingSampleId(null);
      setImagePredictions([]);
      setObjects([]);
      setCounts({ ...EMPTY_COUNTS });
      setHasObservedFrame(false);
      setLastInferenceAt(null);
      setImageAnnotations([]);
      setImageAnalyzed(false);
      setOwlObserved(false);
      setImageOwlObserved(false);
      setPendingBox(null);
      setAnnotationMode(false);
      setNotice(`Image préparée localement (${width} × ${height} px${scale < 1 ? ', réduite pour les performances mobiles' : ''}). Analysez-la, puis dessinez des cadres pour corriger ou ajouter des annotations.`);
    } catch (error) {
      setNotice(error.message || 'Impossible de lire cette image.');
    }
  };

  const analyzeFixedImage = async () => {
    if (!selectedImage) {
      setNotice('Importez d’abord une image.');
      return;
    }
    if (!modelRef.current && !(await loadDetectionModel())) return;
    setDatasetBusy(true);
    setModelError('');
    setOwlModelError('');
    setDenseModelError('');
    try {
      const image = await loadImageElement(selectedImage.dataUrl);
      const predictions = await modelRef.current.detect(image, 80, thresholdRef.current);
      const cocoEquipment = predictions
        .filter((prediction) => AUTO_IDS.some((id) => CATEGORIES.find((category) => category.id === id)?.modelClass === prediction.class))
        .map((prediction, index) => {
          const category = CATEGORIES.find((item) => item.modelClass === prediction.class);
          return {
            id: `image-${prediction.class}-${index}`,
            categoryId: category.id,
            label: category.label,
            color: category.color,
            score: prediction.score,
            bbox: prediction.bbox,
            source: 'COCO-SSD',
          };
        });

      const canvas = document.createElement('canvas');
      canvas.width = selectedImage.width;
      canvas.height = selectedImage.height;
      const context = canvas.getContext('2d');
      if (!context) throw new Error('Le navigateur ne peut pas préparer l’image pour l’analyse complémentaire.');
      context.drawImage(image, 0, 0, canvas.width, canvas.height);

      let owlEquipment = [];
      let owlSucceeded = false;
      if (owlEnabled) {
        try {
          owlEquipment = await detectWithOwl(canvas);
          owlSucceeded = true;
        } catch (owlError) {
          console.error('OWL-ViT image detection error', owlError);
        }
      }

      let denseEquipment = [];
      let tileEquipment = [];
      let denseSucceeded = false;
      if (denseSceneEnabled) {
        try {
          denseEquipment = await detectWithDenseModel(canvas);
          denseSucceeded = true;
        } catch (denseError) {
          console.error('YOLOv10n image detection error', denseError);
        }
        try {
          // Pour les petits objets éloignés, une seconde passe COCO-SSD sur quatre
          // zones agrandit l’objet dans l’entrée du modèle. Les recouvrements seront dédoublonnés.
          tileEquipment = await detectCocoTiled(image);
          denseSucceeded = true;
        } catch (tileError) {
          console.error('COCO-SSD tiled inference error', tileError);
        }
      }

      const equipment = suppressDuplicateBoxes([...cocoEquipment, ...tileEquipment, ...denseEquipment, ...owlEquipment]);
      const nextCounts = { ...EMPTY_COUNTS };
      equipment.forEach((object) => { nextCounts[object.categoryId] = (nextCounts[object.categoryId] || 0) + 1; });
      setImagePredictions(equipment);
      setObjects(equipment);
      setCounts(nextCounts);
      setHasObservedFrame(true);
      setOwlObserved(owlSucceeded);
      setImageOwlObserved(owlSucceeded);
      setImageAnalyzed(true);
      setLastInferenceAt(new Date().toISOString());
      const modelNames = ['COCO-SSD', ...(denseSucceeded ? ['YOLOv10n + analyse par zones'] : []), ...(owlSucceeded ? ['OWL-ViT'] : [])];
      const notes = [];
      if (owlEnabled && !owlSucceeded) notes.push('OWL-ViT indisponible pour cette analyse ; tableaux et fenêtres à confirmer manuellement.');
      if (denseSceneEnabled && !denseSucceeded) notes.push('Le mode scènes denses n’a pas abouti ; les résultats COCO-SSD restent affichés.');
      setNotice(`${equipment.length} détection(s) sur l’image fixe · ${modelNames.join(' + ')}. ${notes.join(' ')} Vérifiez les cadres et les éventuels doublons avant validation.`);
    } catch (error) {
      console.error('Fixed-image detection error', error);
      setModelError(friendlyModelError(error, 'Analyse de l’image'));
    } finally {
      setDatasetBusy(false);
    }
  };

  const getImagePoint = (event) => {
    const canvas = fixedImageCanvasRef.current;
    if (!canvas || !selectedImage) return null;
    const rect = canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return null;
    return {
      x: Math.min(selectedImage.width, Math.max(0, ((event.clientX - rect.left) / rect.width) * selectedImage.width)),
      y: Math.min(selectedImage.height, Math.max(0, ((event.clientY - rect.top) / rect.height) * selectedImage.height)),
    };
  };

  const beginAnnotation = (event) => {
    if (!annotationMode || !selectedImage) return;
    const point = getImagePoint(event);
    if (!point) return;
    event.currentTarget.setPointerCapture?.(event.pointerId);
    drawStartRef.current = point;
    setPendingBox({ x: point.x, y: point.y, width: 0, height: 0 });
  };

  const moveAnnotation = (event) => {
    if (!annotationMode || !drawStartRef.current) return;
    const point = getImagePoint(event);
    if (!point) return;
    const start = drawStartRef.current;
    setPendingBox({ x: Math.min(start.x, point.x), y: Math.min(start.y, point.y), width: Math.abs(point.x - start.x), height: Math.abs(point.y - start.y) });
  };

  const finishAnnotation = (event) => {
    if (!annotationMode || !drawStartRef.current) return;
    const point = getImagePoint(event);
    const start = drawStartRef.current;
    drawStartRef.current = null;
    setPendingBox(null);
    if (!point) return;
    const x = Math.min(start.x, point.x);
    const y = Math.min(start.y, point.y);
    const width = Math.abs(point.x - start.x);
    const height = Math.abs(point.y - start.y);
    if (width < 5 || height < 5) {
      setNotice('Cadre trop petit. Faites glisser le doigt ou la souris autour de l’objet.');
      return;
    }
    const category = CATEGORIES.find((item) => item.id === annotationCategory);
    setImageAnnotations((previous) => [...previous, {
      id: `ann-${Date.now()}-${previous.length}`,
      categoryId: annotationCategory,
      bbox: [x, y, width, height],
      label: category?.label || annotationCategory,
      color: category?.color || '#f7db65',
    }]);
  };

  const saveCurrentSample = async () => {
    if (!selectedImage) {
      setNotice('Importez une image avant d’enregistrer un exemple.');
      return;
    }
    if (imageAnnotations.length === 0) {
      setNotice('Dessinez au moins un cadre et attribuez-lui une catégorie avant d’enregistrer cet exemple.');
      return;
    }
    setDatasetBusy(true);
    try {
      const sample = {
        id: editingSampleId || `sample-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
        name: selectedImage.name,
        dataUrl: selectedImage.dataUrl,
        width: selectedImage.width,
        height: selectedImage.height,
        annotations: imageAnnotations.map((annotation) => ({
          categoryId: annotation.categoryId,
          bbox: annotation.bbox,
        })),
        createdAt: new Date().toISOString(),
      };
      await writeTrainingSample(sample);
      setTrainingSamples((previous) => [sample, ...previous.filter((item) => item.id !== sample.id)]);
      const wasEditing = Boolean(editingSampleId);
      setEditingSampleId(null);
      setNotice(`${wasEditing ? 'Exemple annoté mis à jour' : 'Exemple annoté enregistré'} localement : ${sample.annotations.length} objet(s). Le jeu de données est actualisé, mais le modèle déjà chargé n'est pas modifié.`);
    } catch (error) {
      console.error('Saving training sample error', error);
      setNotice(error.message || 'Enregistrement de l’exemple impossible. Exportez ou réduisez la taille de l’image, puis réessayez.');
    } finally {
      setDatasetBusy(false);
    }
  };

  const exportTrainingDataset = () => {
    if (trainingSamples.length === 0) {
      setNotice('Aucun exemple à exporter. Annotez une image et enregistrez-la d’abord.');
      return;
    }
    const categoryIds = Object.fromEntries(CATEGORIES.map((category, index) => [category.id, index + 1]));
    let annotationId = 1;
    const cocoImages = trainingSamples.map((sample, index) => ({
      id: index + 1,
      file_name: sample.name || `image-${index + 1}.jpg`,
      width: sample.width,
      height: sample.height,
    }));
    const cocoAnnotations = trainingSamples.flatMap((sample, imageIndex) => sample.annotations.map((annotation) => ({
      id: annotationId++,
      image_id: imageIndex + 1,
      category_id: categoryIds[annotation.categoryId],
      bbox: annotation.bbox.map((value) => Math.round(value * 100) / 100),
      area: annotation.bbox[2] * annotation.bbox[3],
      iscrowd: 0,
    })));
    const payload = {
      format: 'atlas-labelled-dataset-v1',
      exportedAt: new Date().toISOString(),
      note: 'Le bloc samples embarque les images en dataUrl pour rendre l’export Atlas autonome ; le bloc coco respecte la structure d’annotations COCO. Les coordonnées bbox sont [x,y,largeur,hauteur] en pixels.',
      categories: CATEGORIES.map((category, index) => ({ id: index + 1, name: category.id, label: category.label })),
      samples: trainingSamples,
      coco: { images: cocoImages, annotations: cocoAnnotations, categories: CATEGORIES.map((category, index) => ({ id: index + 1, name: category.id, supercategory: category.family || 'equipment' })) },
    };
    downloadJsonFile(payload, `atlas-jeu-apprentissage-${new Date().toISOString().slice(0, 10)}.json`);
    setNotice('Jeu d’exemples exporté. Le JSON contient les images et les cadres annotés ; il pourra être préparé pour l’entraînement Faster R-CNN.');
  };

  const importTrainingDataset = async (event) => {
    const file = event.target.files?.[0];
    event.target.value = '';
    if (!file) return;
    setDatasetBusy(true);
    try {
      const parsed = JSON.parse(await file.text());
      const incoming = Array.isArray(parsed.samples) ? parsed.samples : null;
      if (!incoming) throw new Error('Ce fichier ne contient pas de liste « samples » reconnue. Importez un export Atlas.');
      let added = 0;
      const saved = [];
      for (const raw of incoming) {
        if (!raw.dataUrl || !Array.isArray(raw.annotations) || !raw.width || !raw.height) continue;
        const sample = {
          id: raw.id || `sample-${Date.now()}-${added}`,
          name: raw.name || raw.fileName || `image-importee-${added + 1}.jpg`,
          dataUrl: raw.dataUrl,
          width: Number(raw.width),
          height: Number(raw.height),
          annotations: raw.annotations.filter((annotation) => CATEGORIES.some((category) => category.id === annotation.categoryId) && Array.isArray(annotation.bbox) && annotation.bbox.length === 4),
          createdAt: raw.createdAt || new Date().toISOString(),
        };
        if (!sample.annotations.length) continue;
        await writeTrainingSample(sample);
        saved.push(sample);
        added += 1;
      }
      setTrainingSamples((previous) => [...saved, ...previous.filter((item) => !saved.some((sample) => sample.id === item.id))]);
      setNotice(`${added} exemple(s) importé(s) dans la base locale.`);
    } catch (error) {
      setNotice(error.message || 'Import de la base d’apprentissage impossible.');
    } finally {
      setDatasetBusy(false);
    }
  };

  const editSample = (sample) => {
    if (cameraOn) stopCamera();
    setSourceMode('image');
    setSelectedImage({ name: sample.name, dataUrl: sample.dataUrl, width: sample.width, height: sample.height });
    setImagePredictions([]);
    setObjects([]);
    setCounts({ ...EMPTY_COUNTS });
    setHasObservedFrame(false);
    setImageAnalyzed(false);
    setOwlObserved(false);
    setImageOwlObserved(false);
    setImageAnnotations(sample.annotations.map((annotation, index) => {
      const category = CATEGORIES.find((item) => item.id === annotation.categoryId);
      return {
        id: `edit-${sample.id}-${index}`,
        categoryId: annotation.categoryId,
        bbox: [...annotation.bbox],
        label: category?.label || annotation.categoryId,
        color: category?.color || '#f7db65',
      };
    }));
    setAnnotationMode(false);
    setPendingBox(null);
    setEditingSampleId(sample.id);
    setNotice(`Modification de « ${sample.name} ». Enregistrez l’exemple pour remplacer sa version locale.`);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const deleteSample = async (id) => {
    try {
      await removeTrainingSample(id);
      if (editingSampleId === id) setEditingSampleId(null);
      setTrainingSamples((previous) => previous.filter((sample) => sample.id !== id));
    } catch (error) {
      setNotice(error.message || 'Impossible de supprimer cet exemple local.');
    }
  };

  const startCamera = async () => {
    setSourceMode('camera');
    setObjects([]);
    setCounts({ ...EMPTY_COUNTS });
    setHasObservedFrame(false);
    setLastInferenceAt(null);
    setCameraError('');
    setModelError('');
    setNotice('');
    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraError('Ce navigateur ne donne pas accès à la caméra dans ce contexte. Ouvrez l’application en HTTPS ou sur localhost.');
      return;
    }
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } },
      });
      setStream(mediaStream);
      setCameraOn(true);
      setNotice('Caméra active. Orientez-la uniquement vers le mobilier et les équipements.');
      if (!modelRef.current) {
        await loadDetectionModel();
      } else {
        setModel(modelRef.current);
        setModelStatus('ready');
      }
    } catch (error) {
      console.error('Camera error', error);
      const messages = {
        NotAllowedError: 'Accès à la caméra refusé. Autorisez la caméra dans les réglages du navigateur.',
        NotFoundError: 'Aucune caméra compatible n’a été trouvée sur cet appareil.',
        NotReadableError: 'La caméra est déjà utilisée par une autre application ou est indisponible.',
      };
      setCameraError(messages[error.name] || 'Impossible de démarrer la caméra. Vérifiez les permissions du navigateur.');
      setCameraOn(false);
    }
  };

  useEffect(() => {
    if (!cameraOn || !model) return undefined;
    let cancelled = false;
    let timerId;
    if (!owlVideoEnabled) owlVideoPredictionsRef.current = [];
    if (!denseVideoEnabled) denseVideoPredictionsRef.current = [];
    const detectFrame = async () => {
      const video = videoRef.current;
      if (!cancelled && video && video.readyState >= 2 && !busyRef.current) {
        busyRef.current = true;
        try {
          const predictions = await model.detect(video, 80, thresholdRef.current);
          if (cancelled) return;
          const cocoEquipment = predictions
            .filter((prediction) => AUTO_IDS.some((id) => CATEGORIES.find((c) => c.id === id)?.modelClass === prediction.class))
            .map((prediction, index) => {
              const category = CATEGORIES.find((item) => item.modelClass === prediction.class);
              return {
                id: `${prediction.class}-${index}`,
                categoryId: category.id,
                label: category.label,
                color: category.color,
                score: prediction.score,
                bbox: prediction.bbox,
                source: 'COCO-SSD',
              };
            });

          const now = Date.now();
          const shouldRunOwl = owlVideoEnabled && owlModelRef.current && now - lastOwlVideoRunAtRef.current >= 6500;
          const shouldRunDense = denseVideoEnabled && denseModelRef.current && now - lastDenseVideoRunAtRef.current >= 4500;
          if (!enhancedBusyRef.current && (shouldRunOwl || shouldRunDense)) {
            const width = video.videoWidth || 1280;
            const height = video.videoHeight || 720;
            const frameCanvas = document.createElement('canvas');
            frameCanvas.width = width;
            frameCanvas.height = height;
            const context = frameCanvas.getContext('2d');
            if (context) {
              context.drawImage(video, 0, 0, width, height);
              enhancedBusyRef.current = true;
              if (shouldRunOwl) {
                lastOwlVideoRunAtRef.current = now;
                detectWithOwl(frameCanvas)
                  .then((items) => {
                    owlVideoPredictionsRef.current = items;
                    setOwlObserved(true);
                    setOwlModelStatus('ready');
                  })
                  .catch((owlError) => {
                    console.error('OWL-ViT live detection error', owlError);
                    setOwlModelError(friendlyModelError(owlError, 'OWL-ViT en vidéo'));
                    lastOwlVideoRunAtRef.current = Number.MAX_SAFE_INTEGER;
                  })
                  .finally(() => { enhancedBusyRef.current = false; });
              } else if (shouldRunDense) {
                lastDenseVideoRunAtRef.current = now;
                detectWithDenseModel(frameCanvas)
                  .then((items) => {
                    denseVideoPredictionsRef.current = items;
                    setDenseModelStatus('ready');
                  })
                  .catch((denseError) => {
                    console.error('YOLOv10n live detection error', denseError);
                    setDenseModelError(friendlyModelError(denseError, 'YOLOv10n en vidéo'));
                    lastDenseVideoRunAtRef.current = Number.MAX_SAFE_INTEGER;
                  })
                  .finally(() => { enhancedBusyRef.current = false; });
              } else {
                enhancedBusyRef.current = false;
              }
            }
          }

          const equipment = suppressDuplicateBoxes([
            ...cocoEquipment,
            ...(owlVideoEnabled ? owlVideoPredictionsRef.current : []),
            ...(denseVideoEnabled ? denseVideoPredictionsRef.current : []),
          ]);
          const nextCounts = { ...EMPTY_COUNTS };
          equipment.forEach((object) => { nextCounts[object.categoryId] += 1; });
          setObjects(equipment);
          setCounts(nextCounts);
          setHasObservedFrame(true);
          setLastInferenceAt(new Date().toISOString());
        } catch (error) {
          console.error('Detection error', error);
          setNotice('La détection a rencontré une erreur sur cette image ; le relevé précédent est conservé.');
        } finally {
          busyRef.current = false;
        }
      }
      if (!cancelled) timerId = window.setTimeout(detectFrame, 650);
    };
    detectFrame();
    return () => {
      cancelled = true;
      window.clearTimeout(timerId);
    };
  }, [cameraOn, model, owlVideoEnabled, denseVideoEnabled]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;
    const width = video.videoWidth || 1280;
    const height = video.videoHeight || 720;
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    const context = canvas.getContext('2d');
    if (!context) return;
    context.clearRect(0, 0, canvas.width, canvas.height);
    const scale = Math.max(1, width / 1280);
    objects.forEach((object) => {
      const [x, y, boxWidth, boxHeight] = object.bbox;
      context.lineWidth = 3 * scale;
      context.strokeStyle = object.color;
      context.fillStyle = `${object.color}26`;
      context.fillRect(x, y, boxWidth, boxHeight);
      context.strokeRect(x, y, boxWidth, boxHeight);
      const label = `${object.label} · ${Math.round(object.score * 100)}%`;
      context.font = `600 ${Math.max(13, 14 * scale)}px Inter, system-ui, sans-serif`;
      const textWidth = context.measureText(label).width;
      const tagHeight = 25 * scale;
      const tagY = Math.max(0, y - tagHeight);
      context.fillStyle = object.color;
      context.fillRect(x, tagY, textWidth + 16 * scale, tagHeight);
      context.fillStyle = '#12261f';
      context.fillText(label, x + 8 * scale, tagY + 17 * scale);
    });
  }, [objects, cameraOn, stream]);

  useEffect(() => {
    const canvas = fixedImageCanvasRef.current;
    if (!canvas || !selectedImage) return;
    canvas.width = selectedImage.width;
    canvas.height = selectedImage.height;
    const context = canvas.getContext('2d');
    if (!context) return;
    context.clearRect(0, 0, canvas.width, canvas.height);
    const drawBox = (bbox, color, label, dashed = false) => {
      const [x, y, width, height] = bbox;
      context.save();
      context.lineWidth = Math.max(3, selectedImage.width / 450);
      context.strokeStyle = color;
      context.fillStyle = `${color}20`;
      context.setLineDash(dashed ? [10, 7] : []);
      context.fillRect(x, y, width, height);
      context.strokeRect(x, y, width, height);
      context.setLineDash([]);
      const fontSize = Math.max(15, selectedImage.width / 70);
      context.font = `600 ${fontSize}px Inter, system-ui, sans-serif`;
      const textWidth = context.measureText(label).width;
      const tagHeight = fontSize + 12;
      const tagY = Math.max(0, y - tagHeight);
      context.fillStyle = color;
      context.fillRect(x, tagY, textWidth + 16, tagHeight);
      context.fillStyle = '#12261f';
      context.fillText(label, x + 8, tagY + fontSize + 1);
      context.restore();
    };
    imagePredictions.forEach((object) => drawBox(object.bbox, object.color, `${object.label} · ${Math.round(object.score * 100)}%`));
    imageAnnotations.forEach((annotation, index) => drawBox(annotation.bbox, annotation.color || CATEGORIES.find((item) => item.id === annotation.categoryId)?.color || '#f7db65', `Manuel ${index + 1} · ${annotation.label || CATEGORIES.find((item) => item.id === annotation.categoryId)?.label || annotation.categoryId}`, true));
    if (pendingBox) {
      drawBox([pendingBox.x, pendingBox.y, pendingBox.width, pendingBox.height], CATEGORIES.find((item) => item.id === annotationCategory)?.color || '#f7db65', 'Annotation en cours', true);
    }
  }, [selectedImage, imagePredictions, imageAnnotations, pendingBox, annotationCategory, sourceMode, activeSection]);

  const liveRows = useMemo(
    () => makeRows(expected, counts, manual, hasObservedFrame, false, owlObserved),
    [expected, counts, manual, hasObservedFrame, owlObserved],
  );
  const snapshotRows = useMemo(() => {
    if (!auditSnapshot) return null;
    return makeRows(auditSnapshot.expected, auditSnapshot.autoCounts, auditSnapshot.manualCounts, auditSnapshot.observed, true, Boolean(auditSnapshot.owlObserved));
  }, [auditSnapshot]);
  const liveSummary = useMemo(() => summarize(liveRows), [liveRows]);
  const reportSummary = useMemo(() => summarize(snapshotRows || liveRows), [snapshotRows, liveRows]);

  const saveAudit = () => {
    const snapshot = {
      createdAt: new Date().toISOString(),
      expected: { ...expected },
      autoCounts: { ...counts },
      manualCounts: { ...manual },
      observed: hasObservedFrame,
      owlObserved,
      owlThreshold,
      threshold,
      model: ['COCO-SSD · lite_mobilenet_v2', ...(objects.some((object) => object.source === 'YOLOv10n' || object.source === 'COCO-SSD · zones') ? ['YOLOv10n + COCO-SSD par zones'] : []), ...(owlObserved ? ['OWL-ViT · owlvit-base-patch32'] : [])].join(' + '),
      source: sourceMode === 'image' ? `Image fixe : ${selectedImage?.name || 'image importée'}` : 'Caméra / dernière image vidéo',
    };
    setAuditSnapshot(snapshot);
    setView('report');
    setNotice('Relevé figé. Le rapport indique explicitement les catégories encore à vérifier.');
  };

  const exportJson = () => {
    const rows = snapshotRows || liveRows;
    const summary = summarize(rows);
    const data = {
      application: 'Atlas — Audit visuel',
      generatedAt: new Date().toISOString(),
      auditCapturedAt: auditSnapshot?.createdAt || null,
      model: auditSnapshot?.model || ['COCO-SSD · lite_mobilenet_v2', ...(objects.some((object) => object.source === 'YOLOv10n' || object.source === 'COCO-SSD · zones') ? ['YOLOv10n + COCO-SSD par zones'] : []), ...(owlObserved ? ['OWL-ViT · owlvit-base-patch32'] : [])].join(' + '),
      confidenceThreshold: auditSnapshot?.threshold ?? threshold,
      owlConfidenceThreshold: auditSnapshot?.owlThreshold ?? (owlObserved ? owlThreshold : null),
      globalStatus: summary.status,
      summary: {
        totalObjectsReported: summary.total,
        recognizedCategories: summary.recognized,
        verifiableRules: summary.configured,
        verifiedRules: summary.checked,
        observedDeviations: summary.deviations,
      },
      rows: rows.map((row) => ({
        category: row.label,
        method: row.machineChecked ? (row.owlUsed ? (row.method === 'auto' ? 'COCO-SSD + OWL-ViT (automatique, à confirmer)' : 'OWL-ViT (automatique, à confirmer)') : 'COCO-SSD (automatique)') : 'Déclaration manuelle',
        expected: row.expected,
        detectedOrRecorded: row.detected,
        gapDetectedMinusExpected: row.gap,
        status: row.status,
        note: row.note || row.detail,
      })),
      privacyNote: 'Aucune image ni vidéo exportée. Les classes hors équipement, notamment person, ne sont pas conservées par l’interface.',
      limitationNote: 'Le modèle générique COCO-SSD peut effectuer des calculs internes multi-classes ; le filtrage des sorties ne constitue pas une garantie absolue de non-inférence sur une personne présente dans l’image.',
    };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `rapport-audit-visuel-${new Date().toISOString().slice(0, 10)}.json`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  };

  const updateExpected = (id, value) => setExpected((previous) => ({ ...previous, [id]: value }));
  const updateManual = (id, value) => setManual((previous) => ({ ...previous, [id]: value }));
  const currentTotal = hasObservedFrame ? Object.values(counts).reduce((sum, value) => sum + value, 0) : 0;
  const pageRows = view === 'report' ? (snapshotRows || liveRows) : liveRows;
  const pageSummary = view === 'report' ? reportSummary : liveSummary;
  const verifiedDescription = view === 'report' && auditSnapshot
    ? `Relevé du ${formatDate(auditSnapshot.createdAt)}`
    : hasObservedFrame ? (sourceMode === 'image' ? 'Dernière image fixe analysée' : 'Instantané courant · évolue avec la caméra') : 'Configurez les attentes puis démarrez la caméra';

  return <div className="app-shell">
    <header className="topbar">
      <a className="brand" href="#top" onClick={() => { setView('live'); setActiveSection('audit'); }} aria-label="Atlas, accueil">
        <span className="brand-mark"><span /><span /><span /><span /></span>
        <span className="brand-word">atlas<span>.</span><small>VISUAL AUDIT</small></span>
      </a>
      <nav className="top-nav" aria-label="Navigation principale">
        <button className={view === 'live' && activeSection === 'audit' ? 'nav-link active' : 'nav-link'} onClick={() => { setView('live'); setActiveSection('audit'); }}><Icon name="grid" size={16} /> Supervision</button>
        <button className={view === 'live' && activeSection === 'learning' ? 'nav-link active' : 'nav-link'} onClick={() => { if (cameraOn) stopCamera(); setSourceMode('image'); setAnnotationMode(false); setView('live'); setActiveSection('learning'); }}><Icon name="target" size={16} /> Apprentissage</button>
        <button className={view === 'report' ? 'nav-link active' : 'nav-link'} onClick={() => setView('report')}><Icon name="report" size={16} /> Rapport d’audit</button>
      </nav>
      <div className="topbar-right">
        <span className="privacy-chip"><Icon name="lock" size={14} /> Traitement local</span>
        <span className={`model-status model-${modelStatus}`}><span className="model-dot" />{modelStatus === 'ready' ? 'Modèle prêt' : modelStatus === 'loading' ? 'Chargement IA' : modelStatus === 'error' ? 'Modèle indisponible' : 'IA non initialisée'}</span>
      </div>
    </header>

    <main id="top" className="main-container">
      <section className="page-intro">
        <div className="intro-copy">
          <div className="eyebrow"><span className="eyebrow-line" /> CONTRÔLE DES ENVIRONNEMENTS</div>
          <h1>{view === 'report' ? <>Rapport <span>d’audit.</span></> : activeSection === 'learning' ? <>Apprentissage <span>manuel.</span></> : <>L’œil juste.<br /><span>Le relevé clair.</span></>}</h1>
          <p>{view === 'report'
            ? 'Confrontez le relevé visuel à la configuration attendue. Les détections non confirmées restent signalées comme telles.'
            : activeSection === 'learning' ? 'Corrigez les détections en dessinant des cadres et constituez un jeu d’exemples conservé sur cet appareil.' : 'Une supervision visuelle des équipements et du mobilier, directement dans votre navigateur. Les images restent sur votre appareil.'}</p>
          {view === 'live' && activeSection === 'audit' && <div className="intro-meta"><span><Icon name="target" size={16} /> 7 règles de contrôle</span><span><Icon name="shield" size={16} /> Aucune vidéo envoyée</span><span><Icon name="grid" size={16} /> Mobile & bureau</span></div>}
        </div>
        <div className="intro-aside">
          <span className="aside-index">{view === 'report' ? '02 / RÉSULTATS' : activeSection === 'learning' ? '03 / APPRENTISSAGE' : '01 / SUPERVISION'}</span>
          <span className="aside-rule" />
          <p>{view === 'report' ? 'Le relevé fait foi des observations disponibles, pas d’une preuve absolue d’absence.' : activeSection === 'learning' ? 'Les annotations enrichissent un jeu de données ; elles ne réentraînent pas immédiatement les modèles.' : 'Un outil d’aide au contrôle. Chaque écart appelle une vérification humaine.'}</p>
          {view === 'live' && activeSection === 'learning'
            ? <button className="button button-dark" onClick={() => setActiveSection('audit')}>Retour à la supervision <Icon name="arrow" size={17} /></button>
            : view === 'live'
              ? <button className="button button-dark" onClick={saveAudit}>Figer le relevé <Icon name="arrow" size={17} /></button>
              : <button className="button button-dark" onClick={() => { setView('live'); setActiveSection('audit'); }}>Retour à la caméra <Icon name="arrow" size={17} /></button>}
        </div>
      </section>

      {notice && <div className="notice-strip" role="status"><Icon name="info" size={17} /><span>{notice}</span><button onClick={() => setNotice('')} aria-label="Fermer le message"><Icon name="close" size={16} /></button></div>}

      {!(view === 'live' && activeSection === 'learning') && <section className="stats-grid" aria-label="Indicateurs de l’audit">
        <StatCard label="Objets relevés" value={pageSummary.total} detail={verifiedDescription} icon="target" />
        <StatCard label="Catégories reconnues" value={`${pageSummary.recognized} / 7`} detail="Catégories avec au moins une occurrence" icon="grid" />
        <StatCard label="Règles configurées" value={`${pageSummary.configured} / 7`} detail={`${pageSummary.checked} règle(s) déjà renseignée(s)`} icon="check" />
        <StatCard label="Écarts observés" value={pageSummary.deviations} detail={pageSummary.unresolved ? `${pageSummary.unresolved} contrôle(s) à confirmer` : 'Aucun contrôle en attente'} icon="alert" tone={pageSummary.deviations > 0 ? 'stat-card-alert' : ''} />
      </section>}

      {view === 'live' ? (activeSection === 'learning' ? <>
        <section className="panel learning-panel">
          <div className="panel-heading learning-heading">
            <div><div className="section-kicker">ANNOTATION & APPRENTISSAGE SUPERVISÉ</div><h2>Apprendre à partir de corrections humaines</h2></div>
            <span className="training-count">{trainingSamples.length} exemple(s) local(aux)</span>
          </div>
          <p className="panel-subtitle">Importez une image, choisissez une catégorie, puis dessinez un cadre autour de chaque objet. Annotez en priorité les fenêtres, tableaux et bureaux manqués, ainsi que les détections incorrectes. Les annotations constituent un jeu de données : elles ne réentraînent pas instantanément COCO-SSD.</p>
          <div className="annotation-toolbar">
            <input ref={fileInputRef} className="visually-hidden-input" type="file" accept="image/jpeg,image/png,image/webp" onChange={handleImageUpload} />
            <button className="button button-secondary" onClick={() => { setSourceMode('image'); fileInputRef.current?.click(); }}><Icon name="download" size={16} /> Importer une image</button>
            <label className="annotation-select-label" htmlFor="annotation-category">Catégorie à annoter</label>
            <select id="annotation-category" value={annotationCategory} onChange={(event) => setAnnotationCategory(event.target.value)}>
              {LEARNABLE_CATEGORIES.map((category) => <option key={category.id} value={category.id}>{category.label}</option>)}
            </select>
            <button className={annotationMode ? 'button button-primary' : 'button button-secondary'} disabled={!selectedImage || sourceMode !== 'image'} onClick={() => { setAnnotationMode((value) => !value); setPendingBox(null); }}><Icon name="target" size={16} />{annotationMode ? 'Terminer l’annotation' : 'Dessiner des cadres'}</button>
            <button className="button button-secondary" onClick={saveCurrentSample} disabled={!selectedImage || imageAnnotations.length === 0 || datasetBusy}><Icon name="check" size={16} />{editingSampleId ? 'Mettre à jour l’exemple' : 'Enregistrer l’exemple'}</button>
          </div>
          {sourceMode === 'image' && selectedImage && <div className="fixed-image-stage learning-image-stage" style={{ aspectRatio: `${selectedImage.width} / ${selectedImage.height}` }}>
            <img ref={fixedImageRef} src={selectedImage.dataUrl} alt="Image importée pour annotation manuelle" />
            <canvas ref={fixedImageCanvasRef} className={`fixed-image-canvas ${annotationMode ? 'is-annotating' : ''}`} onPointerDown={beginAnnotation} onPointerMove={moveAnnotation} onPointerUp={finishAnnotation} onPointerCancel={finishAnnotation} aria-label="Image avec cadres de détection et d’annotation" />
            <span className="image-detection-badge">{imageAnalyzed ? `${imagePredictions.length} détection(s) IA` : 'Annotation manuelle'}</span>
          </div>}
          {sourceMode !== 'image' || !selectedImage ? <div className="annotation-empty"><Icon name="info" size={18} /><span>Importez une photo avec le bouton ci-dessus pour commencer à dessiner des cadres.</span></div> : <>
            <div className={`annotation-hint ${annotationMode ? 'annotation-hint-active' : ''}`}><span className="hint-dot" />{annotationMode ? 'Mode dessin actif : faites glisser le doigt ou la souris sur l’image pour encadrer un objet.' : 'Choisissez une catégorie puis activez « Dessiner des cadres ». Les cadres manuels sont en pointillés.'}</div>
            {imageAnnotations.length > 0 ? <div className="annotation-list"><span className="annotation-list-title">Annotations non enregistrées ({imageAnnotations.length})</span>{imageAnnotations.map((annotation, index) => <div className="annotation-chip" key={annotation.id}><span className="annotation-color" style={{ background: annotation.color }} /><span>{index + 1}. {annotation.label}</span><small>{annotation.bbox.map((value) => Math.round(value)).join(' · ')}</small><button className="icon-button" onClick={() => setImageAnnotations((previous) => previous.filter((item) => item.id !== annotation.id))} aria-label={`Supprimer annotation ${index + 1}`}><Icon name="close" size={14} /></button></div>)}</div> : <p className="annotation-empty-text">Aucun cadre manuel pour cette image pour le moment.</p>}
          </>}
          <div className="dataset-controls">
            <div className="dataset-description"><strong>Jeu de données d’apprentissage</strong><span>Conservé dans IndexedDB sur cet appareil. Exportez régulièrement une copie de sauvegarde avant de changer de navigateur ou d’effacer ses données.</span></div>
            <div className="dataset-actions">
              <input ref={datasetInputRef} className="visually-hidden-input" type="file" accept="application/json,.json" onChange={importTrainingDataset} />
              <button className="button button-secondary" onClick={() => datasetInputRef.current?.click()} disabled={datasetBusy}>Importer un jeu</button>
              <button className="button button-primary" onClick={exportTrainingDataset} disabled={trainingSamples.length === 0 || datasetBusy}><Icon name="download" size={16} /> Exporter annotations + images</button>
            </div>
          </div>
          {trainingSamples.length > 0 && <div className="saved-samples-list">{trainingSamples.slice(0, 5).map((sample) => <div className="saved-sample-row" key={sample.id}><span className="saved-sample-thumb"><img src={sample.dataUrl} alt="" /></span><span className="saved-sample-name"><strong>{sample.name}</strong><small>{sample.annotations.length} annotation(s) · {new Date(sample.createdAt).toLocaleDateString('fr-FR')}</small></span><div className="saved-sample-actions"><button className="text-button" onClick={() => editSample(sample)}>Modifier</button><button className="text-button remove-sample" onClick={() => deleteSample(sample.id)}>Supprimer</button></div></div>)}{trainingSamples.length > 5 && <p className="dataset-more">+ {trainingSamples.length - 5} autre(s) exemple(s) enregistré(s)</p>}</div>}
          <div className="faster-rcnn-note"><span className="note-icon"><Icon name="target" size={17} /></span><p><strong>Étape d’entraînement recommandée : Faster R-CNN.</strong><br />Le ZIP inclut un pipeline Python pour extraire ces annotations au format COCO et préparer le jeu d’images. L’entraînement Faster R-CNN s’effectue sur une machine équipée de TensorFlow Object Detection API ; le modèle entraîné doit ensuite être converti et validé pour l’inférence web. Cette séparation évite de prétendre que COCO-SSD apprend dans la page.</p></div>
        </section>
      </> : <>
        <section className="workspace-grid">
          <article className="panel camera-panel">
            <div className="panel-heading">
              <div><div className="section-kicker">ESPACE DE CONTRÔLE</div><h2>{sourceMode === 'image' ? 'Analyse d’une image' : 'Flux de détection'}</h2></div>
              <div className={`live-indicator ${cameraOn && sourceMode === 'camera' ? 'is-live' : ''}`}><span />{sourceMode === 'image' ? 'IMAGE FIXE' : cameraOn ? 'EN DIRECT' : 'À L’ARRÊT'}</div>
            </div>
            <div className="source-tabs" role="tablist" aria-label="Source visuelle">
              <button className={sourceMode === 'camera' ? 'source-tab active' : 'source-tab'} onClick={() => { setSourceMode('camera'); setOwlObserved(false); setAnnotationMode(false); setObjects([]); setCounts({ ...EMPTY_COUNTS }); setHasObservedFrame(false); setLastInferenceAt(null); if (cameraOn) setNotice('La caméra est sélectionnée.'); }} role="tab" aria-selected={sourceMode === 'camera'}><Icon name="camera" size={16} /> Caméra / vidéo</button>
              <button className={sourceMode === 'image' ? 'source-tab active' : 'source-tab'} onClick={() => { if (cameraOn) stopCamera(); setSourceMode('image'); setOwlObserved(Boolean(imageOwlObserved)); setAnnotationMode(false); const restoredCounts = { ...EMPTY_COUNTS }; imagePredictions.forEach((object) => { restoredCounts[object.categoryId] += 1; }); setObjects(imagePredictions); setCounts(restoredCounts); setHasObservedFrame(Boolean(imageAnalyzed)); setLastInferenceAt(imageAnalyzed ? lastInferenceAt : null); }} role="tab" aria-selected={sourceMode === 'image'}><Icon name="report" size={16} /> Image téléchargée</button>
            </div>
            {sourceMode === 'camera' ? <>
            <div className={`video-stage ${cameraOn ? 'video-active' : ''}`}>
              <video ref={videoRef} autoPlay muted playsInline aria-label="Flux vidéo de la caméra" />
              <canvas ref={canvasRef} className="overlay-canvas" aria-label="Cadres de détection superposés" />
              {!cameraOn && <div className="camera-placeholder">
                <div className="placeholder-art"><div className="art-window"><span /><span /></div><div className="art-shelf" /><div className="art-chair" /><div className="art-desk" /><div className="art-laptop" /></div>
                <span className="placeholder-overline">ESPACE PRÊT POUR LE CONTRÔLE</span>
                <h3>Votre champ d’inspection<br />commence ici.</h3>
                <p>Orientez la caméra vers le mobilier et les équipements uniquement.</p>
                {modelStatus === 'loading' ? <span className="loading-inline"><span className="mini-spinner" /> Chargement du modèle IA…</span> : <button className="button button-primary" onClick={startCamera}><Icon name="camera" size={18} /> Démarrer la caméra</button>}
              </div>}
              {cameraOn && <div className="video-overlay-top"><span className="video-badge"><span className="record-dot" /> FLUX LOCAL</span><span className="frame-counter">{objects.length} objet(s) encadré(s)</span></div>}
              {cameraOn && <div className="video-overlay-bottom"><span><Icon name="lock" size={14} /> Aucune image enregistrée</span><span>{lastInferenceAt ? `Dernier calcul · ${new Date(lastInferenceAt).toLocaleTimeString('fr-FR')}` : 'Initialisation de la détection…'}</span></div>}
            </div>
            {cameraError && <div className="inline-error" role="alert"><Icon name="alert" size={17} />{cameraError}</div>}
            {modelError && <div className="inline-error" role="alert"><Icon name="alert" size={17} />{modelError}<button className="text-button" onClick={retryModelLoad}>Réessayer</button></div>}
            {sourceMode === 'camera' && <div className="camera-controls">
              {cameraOn ? <button className="button button-danger" onClick={stopCamera}><Icon name="stop" size={17} /> Arrêter la caméra</button> : <button className="button button-primary" onClick={startCamera} disabled={modelStatus === 'loading'}><Icon name="camera" size={17} />{modelStatus === 'loading' ? 'Chargement du modèle…' : 'Démarrer la caméra'}</button>}
              <div className="confidence-control">
                <div className="confidence-heading"><label htmlFor="confidence">Seuil de confiance</label><strong>{Math.round(threshold * 100)}%</strong></div>
                <input id="confidence" type="range" min="0.2" max="0.9" step="0.05" value={threshold} onChange={(event) => setThreshold(Number(event.target.value))} />
                <div className="range-scale"><span>Plus de détections</span><span>Plus strict</span></div>
              </div>
            </div>}
            <section className="owlvit-card live-model-options" aria-label="Modèles complémentaires du flux vidéo">
              <label className="owlvit-toggle">
                <input type="checkbox" checked={owlVideoEnabled} onChange={(event) => {
                  const enabled = event.target.checked;
                  setOwlVideoEnabled(enabled);
                  owlVideoPredictionsRef.current = [];
                  lastOwlVideoRunAtRef.current = 0;
                  if (enabled) loadOwlModel().catch(() => {});
                  else { setOwlObserved(false); setOwlModelError(''); }
                }} />
                <span className="owlvit-toggle-copy"><strong>OWL-ViT dans le flux vidéo</strong><small>Recherche tableaux, fenêtres et bureaux environ toutes les 6,5 secondes. Le flux sera plus lent pendant ce calcul.</small></span>
              </label>
              {owlVideoEnabled && <>
                <div className="confidence-control owlvit-threshold"><div className="confidence-heading"><label htmlFor="owl-video-threshold">Seuil OWL-ViT</label><strong>{Math.round(owlThreshold * 100)}%</strong></div><input id="owl-video-threshold" type="range" min="0.05" max="0.5" step="0.01" value={owlThreshold} onChange={(event) => setOwlThreshold(Number(event.target.value))} /><div className="range-scale"><span>Plus de propositions</span><span>Plus strict</span></div></div>
                <div className={`owlvit-status owlvit-${owlModelStatus}`} role="status">{owlModelStatus === 'loading' ? `Chargement OWL-ViT… ${owlProgress}%` : owlModelStatus === 'ready' ? 'OWL-ViT chargé ; prochaine détection vidéo planifiée' : owlModelStatus === 'error' ? 'OWL-ViT indisponible — consultez le détail ci-dessous' : 'OWL-ViT sera chargé à l’activation.'}</div>
                {owlModelStatus === 'loading' && <div className="owlvit-progress"><span style={{ width: `${owlProgress}%` }} /></div>}
                {owlModelError && <div className="inline-error model-detail-error" role="alert"><Icon name="alert" size={16} />{owlModelError}<button className="text-button" onClick={() => { setOwlModelStatus('idle'); setOwlModelError(''); lastOwlVideoRunAtRef.current = 0; loadOwlModel().catch(() => {}); }}>Réessayer</button></div>}
              </>}
              <div className="model-option-divider" />
              <label className="owlvit-toggle">
                <input type="checkbox" checked={denseVideoEnabled} onChange={(event) => {
                  const enabled = event.target.checked;
                  setDenseVideoEnabled(enabled);
                  denseVideoPredictionsRef.current = [];
                  lastDenseVideoRunAtRef.current = 0;
                  if (enabled) loadDenseModel().catch(() => {});
                }} />
                <span className="owlvit-toggle-copy"><strong>Mode vidéo « scène dense » (YOLOv10n)</strong><small>Détecteur complémentaire pour multiplier les propositions d’objets COCO, recalculé environ toutes les 4,5 secondes.</small></span>
              </label>
              {denseVideoEnabled && <>
                <div className={`owlvit-status owlvit-${denseModelStatus}`} role="status">{denseModelStatus === 'loading' ? `Chargement YOLOv10n… ${denseProgress}%` : denseModelStatus === 'ready' ? 'YOLOv10n chargé ; détections complémentaires actives' : denseModelStatus === 'error' ? 'YOLOv10n indisponible — consultez le détail ci-dessous' : 'YOLOv10n sera chargé à l’activation.'}</div>
                {denseModelStatus === 'loading' && <div className="owlvit-progress"><span style={{ width: `${denseProgress}%` }} /></div>}
                {denseModelError && <div className="inline-error model-detail-error" role="alert"><Icon name="alert" size={16} />{denseModelError}<button className="text-button" onClick={() => { setDenseModelStatus('idle'); setDenseModelError(''); lastDenseVideoRunAtRef.current = 0; loadDenseModel().catch(() => {}); }}>Réessayer</button></div>}
              </>}
            </section>
            <div className="camera-footnote"><Icon name="info" size={15} /><span>COCO-SSD tourne environ toutes les 0,65 seconde. Les modèles complémentaires s’exécutent à intervalle réduit pour préserver la réactivité sur mobile.</span></div>
            <div className="privacy-warning"><Icon name="shield" size={16} /><p><strong>Zone d’inspection réservée aux équipements.</strong> Si une personne entre dans le champ, arrêtez immédiatement la détection. Les modèles génériques restent multi-classes : filtrer les sorties ne garantit pas l’absence de calcul interne sur une personne.</p></div>
            </> : <>
              <div className="image-upload-row">
                <input ref={fileInputRef} className="visually-hidden-input" type="file" accept="image/jpeg,image/png,image/webp" onChange={handleImageUpload} />
                <button className="button button-secondary" onClick={() => fileInputRef.current?.click()}><Icon name="download" size={17} /> Choisir une image</button>
                <span className="image-file-name">{selectedImage ? `${selectedImage.name} · ${selectedImage.width} × ${selectedImage.height}` : 'JPEG, PNG ou WebP · image conservée sur cet appareil'}</span>
              </div>
              {selectedImage ? <>
                <div className="fixed-image-stage" style={{ aspectRatio: `${selectedImage.width} / ${selectedImage.height}` }}>
                  <img ref={fixedImageRef} src={selectedImage.dataUrl} alt="Image fixe importée pour audit" />
                  <canvas ref={fixedImageCanvasRef} className={`fixed-image-canvas ${annotationMode ? 'is-annotating' : ''}`} onPointerDown={beginAnnotation} onPointerMove={moveAnnotation} onPointerUp={finishAnnotation} onPointerCancel={finishAnnotation} aria-label="Image avec cadres de détection et d’annotation" />
                  <span className="image-detection-badge">{imageAnalyzed ? `${imagePredictions.length} détection(s) IA` : 'Pas encore analysée'}</span>
                </div>
                <div className="fixed-image-controls">
                  <button className="button button-primary" onClick={analyzeFixedImage} disabled={datasetBusy || modelStatus === 'loading' || owlModelStatus === 'loading' || denseModelStatus === 'loading'}><Icon name="target" size={17} />{datasetBusy ? 'Traitement…' : denseSceneEnabled && owlEnabled ? 'Analyser avec les 3 modèles' : denseSceneEnabled ? 'Analyser une scène dense' : owlEnabled ? 'Analyser avec COCO-SSD + OWL-ViT' : 'Analyser l’image'}</button>
                  <button className="button button-secondary" onClick={() => { setSelectedImage(null); setImagePredictions([]); setObjects([]); setCounts({ ...EMPTY_COUNTS }); setHasObservedFrame(false); setLastInferenceAt(null); setImageAnnotations([]); setImageAnalyzed(false); setOwlObserved(false); setImageOwlObserved(false); setAnnotationMode(false); setPendingBox(null); setEditingSampleId(null); }}><Icon name="close" size={16} /> Retirer l’image</button>
                </div>
                <section className="owlvit-card" aria-label="Détection complémentaire OWL-ViT">
                  <label className="owlvit-toggle">
                    <input type="checkbox" checked={owlEnabled} onChange={(event) => {
                      const enabled = event.target.checked;
                      setOwlEnabled(enabled);
                      if (enabled) loadOwlModel().catch(() => {});
                      else { setOwlModelError(''); }
                    }} />
                    <span className="owlvit-toggle-copy"><strong>Ajouter OWL-ViT pour tableaux, fenêtres et bureaux</strong><small>Détection zéro-shot complémentaire sur cette image fixe ; les résultats restent à confirmer.</small></span>
                  </label>
                  <p className="owlvit-disclosure">Le modèle commence à se charger dès que vous activez cette option. Le premier téléchargement est volumineux (environ 155 Mo pour les poids quantifiés) ; il peut être mis en cache par le navigateur. L’image elle-même est traitée localement.</p>
                  {owlEnabled && <>
                    <div className="confidence-control owlvit-threshold">
                      <div className="confidence-heading"><label htmlFor="owl-threshold">Seuil OWL-ViT</label><strong>{Math.round(owlThreshold * 100)}%</strong></div>
                      <input id="owl-threshold" type="range" min="0.05" max="0.5" step="0.01" value={owlThreshold} onChange={(event) => setOwlThreshold(Number(event.target.value))} />
                      <div className="range-scale"><span>Plus de propositions</span><span>Plus strict</span></div>
                    </div>
                    <div className={`owlvit-status owlvit-${owlModelStatus}`} role="status">
                      {owlModelStatus === 'loading' ? `Téléchargement / initialisation OWL-ViT… ${owlProgress}%` : owlModelStatus === 'ready' ? 'OWL-ViT chargé · prêt pour l’image fixe' : owlModelStatus === 'error' ? 'OWL-ViT indisponible — consultez le détail ci-dessous.' : 'OWL-ViT sera chargé au lancement de l’analyse.'}
                    </div>
                    {owlModelStatus === 'loading' && <div className="owlvit-progress"><span style={{ width: `${owlProgress}%` }} /></div>}
                    {owlModelError && <div className="inline-error" role="alert"><Icon name="alert" size={16} />{owlModelError}<button className="text-button" onClick={() => { setOwlModelStatus('idle'); setOwlModelError(''); loadOwlModel().then(() => analyzeFixedImage()).catch(() => {}); }}>Réessayer</button></div>}
                  </>}
                </section>
                <section className="owlvit-card" aria-label="Mode scènes denses YOLOv10n">
                  <label className="owlvit-toggle">
                    <input type="checkbox" checked={denseSceneEnabled} onChange={(event) => setDenseSceneEnabled(event.target.checked)} />
                    <span className="owlvit-toggle-copy"><strong>Scène dense : YOLOv10n + COCO-SSD par zones</strong><small>Ajoute un deuxième détecteur généraliste et quatre découpes avec chevauchement afin de mieux récupérer les petits objets ou les scènes chargées.</small></span>
                  </label>
                  <p className="owlvit-disclosure">Le modèle YOLOv10n est téléchargé à la demande et reste dans le navigateur. Cette passe est plus lente, peut produire des doublons ou faux positifs, et cible les mêmes classes COCO que COCO-SSD ; ce n’est pas un modèle spécialisé dans les fenêtres ou tableaux.</p>
                  {denseSceneEnabled && <>
                    <div className={`owlvit-status owlvit-${denseModelStatus}`} role="status">{denseModelStatus === 'loading' ? `Téléchargement / initialisation YOLOv10n… ${denseProgress}%` : denseModelStatus === 'ready' ? 'YOLOv10n chargé · prêt pour l’image fixe' : denseModelStatus === 'error' ? 'YOLOv10n indisponible — consultez le détail ci-dessous.' : 'YOLOv10n sera chargé au lancement de l’analyse.'}</div>
                    {denseModelStatus === 'loading' && <div className="owlvit-progress"><span style={{ width: `${denseProgress}%` }} /></div>}
                    {denseModelError && <div className="inline-error model-detail-error" role="alert"><Icon name="alert" size={16} />{denseModelError}<button className="text-button" onClick={() => { setDenseModelStatus('idle'); setDenseModelError(''); loadDenseModel().then(() => analyzeFixedImage()).catch(() => {}); }}>Réessayer</button></div>}
                  </>}
                </section>
              </> : <div className="image-empty-state"><Icon name="report" size={25} /><strong>Importez une image d’inspection</strong><span>La détection des objets s’effectue localement dans le navigateur, sans envoi du fichier.</span><button className="text-button" onClick={() => fileInputRef.current?.click()}>Parcourir les fichiers <Icon name="arrow" size={14} /></button></div>}
              {modelError && <div className="inline-error" role="alert"><Icon name="alert" size={17} />{modelError}<button className="text-button" onClick={retryModelLoad}>Réessayer</button></div>}
              <div className="confidence-control image-threshold">
                <div className="confidence-heading"><label htmlFor="confidence-image">Seuil de confiance</label><strong>{Math.round(threshold * 100)}%</strong></div>
                <input id="confidence-image" type="range" min="0.2" max="0.9" step="0.05" value={threshold} onChange={(event) => setThreshold(Number(event.target.value))} />
                <div className="range-scale"><span>Plus de détections</span><span>Plus strict</span></div>
              </div>
              <div className="privacy-warning"><Icon name="shield" size={16} /><p><strong>Importez une image cadrée sur les équipements uniquement.</strong> COCO-SSD, OWL-ViT et YOLOv10n peuvent calculer sur l’image entière, même si l’application n’affiche que des catégories autorisées. Pour respecter strictement l’exclusion des personnes, recadrez la photo avant analyse afin qu’aucune personne ne figure dans l’image.</p></div>
            </>}
          </article>

        <aside className="panel audit-overview-panel">
          <div className="panel-heading overview-heading">
            <div><div className="section-kicker">RÈGLES & RELEVÉ</div><h2>Configuration initiale & objets en vue</h2></div>
            <span className="count-orb overview-count">{hasObservedFrame ? currentTotal : '—'}</span>
          </div>
          <p className="panel-subtitle">Définissez les quantités attendues et comparez-les aux objets relevés dans la dernière image traitée.</p>
          <div className="compact-count-head"><span>Catégorie</span><span>Attendu</span><span>Vu / relevé</span></div>
          <div className="compact-count-list">
            {CATEGORIES.map((category) => {
              const owlCategoryChecked = owlObserved && OWL_CATEGORY_IDS.has(category.id);
              const machineCount = category.method === 'auto' || owlCategoryChecked;
              const count = machineCount ? (hasObservedFrame ? (counts[category.id] ?? 0) : null) : getValue(manual[category.id]);
              const sourceLabel = category.method === 'auto' ? (owlObserved && category.id === 'table' ? 'COCO + OWL-ViT' : 'COCO-SSD') : owlCategoryChecked ? 'OWL-ViT · à confirmer' : 'Manuel';
              return <div className="compact-count-row" key={category.id}>
                <div className="compact-category">
                  <CategoryIcon category={category} size={18} />
                  <div><strong>{category.label}</strong><small>{sourceLabel}</small></div>
                </div>
                <label className="compact-number-field">
                  <span className="sr-only">Nombre attendu : {category.label}</span>
                  <input type="number" inputMode="numeric" min="0" step="1" value={expected[category.id]} onChange={(event) => updateExpected(category.id, event.target.value)} aria-label={`Nombre attendu : ${category.label}`} />
                </label>
                {machineCount ? <div className={`compact-observed ${count > 0 ? 'has-count' : ''}`} aria-label={`Nombre détecté : ${category.label}`}><strong>{count === null ? '—' : count}</strong></div> : <label className="compact-number-field compact-manual-field">
                  <span className="sr-only">Relevé manuel : {category.label}</span>
                  <input type="number" inputMode="numeric" min="0" step="1" placeholder="—" value={manual[category.id]} onChange={(event) => updateManual(category.id, event.target.value)} aria-label={`Relevé manuel : ${category.label}`} />
                </label>}
              </div>;
            })}
          </div>
          <div className="compact-note">Les tirets indiquent qu’aucun relevé automatique n’est disponible. Une non-détection ne prouve pas l’absence d’un objet.</div>
          <div className="config-footer overview-footer">
            <button className="text-button reset-button" onClick={() => { setExpected({ ...INITIAL_EXPECTED }); setManual({ ...EMPTY_MANUAL }); setNotice('Configuration d’exemple restaurée. Vérifiez les quantités avant usage.'); }}><Icon name="refresh" size={15} /> Restaurer l’exemple</button>
            <button className="button button-primary" onClick={saveAudit}>Figer le relevé <Icon name="arrow" size={16} /></button>
          </div>
          <div className="detected-objects-block overview-detected-block">
            <div className="subsection-heading"><span>Détections individuelles</span><span>{objects.length} résultat(s)</span></div>
            <div className="object-list" aria-live="polite">
              {objects.length === 0 ? <div className="empty-objects compact-empty"><span className="empty-ring"><Icon name="target" size={19} /></span><strong>{hasObservedFrame ? 'Aucun équipement reconnu' : 'En attente d’une image'}</strong><span>{hasObservedFrame ? 'Cela ne prouve pas que les équipements sont absents.' : 'Démarrez la caméra ou importez une image.'}</span></div> : objects.map((object, index) => <div className="object-row" key={object.id}>
                <span className="object-index">{String(index + 1).padStart(2, '0')}</span>
                <span className="object-mark" style={{ background: object.color }} />
                <div className="object-info"><strong>{object.label}</strong><small>Objet {index + 1}{object.source ? ` · ${object.source}` : ''}</small></div>
                <span className="confidence-score">{Math.round(object.score * 100)}%</span>
              </div>)}
            </div>
          </div>
        </aside>
        </section>
        <section className="method-strip">
          <div className="method-strip-icon"><Icon name="info" size={20} /></div>
          <div><h3>À propos de la méthode</h3><p>COCO-SSD reste actif en caméra. Sur une image fixe, OWL-ViT peut proposer en zéro-shot des cadres pour tableaux, fenêtres et bureaux. Ce n’est pas un entraînement local : vérifiez les résultats et annotez les erreurs.</p></div>
          <button className="text-button" onClick={() => setShowMethodNotes((value) => !value)}>{showMethodNotes ? 'Réduire' : 'Limites & confidentialité'} <Icon name={showMethodNotes ? 'close' : 'arrow'} size={15} /></button>
        </section>
        {showMethodNotes && <section className="expanded-method panel">
          <h3>Limites à connaître avant l’audit</h3>
          <p><strong>Classes disponibles :</strong> la détection automatique exploite uniquement les classes <code>chair</code>, <code>dining table</code>, <code>laptop</code> et <code>tv</code>. Les bureaux, fenêtres, tableaux et anomalies visuelles ne sont pas couverts de façon native par ce modèle.</p>
          <p><strong>Personnes :</strong> l’interface ne présente, n’enregistre et n’exporte pas les sorties « person », et ne fait aucune identification. Cependant, COCO-SSD reste un modèle généraliste multi-classes : filtrer ses sorties n'est pas la preuve qu'aucun calcul interne relatif à une personne n'a lieu. Pour une exigence stricte, cadre la caméra uniquement sur les équipements, suspends la détection si quelqu’un entre dans le champ et remplace le modèle générique par un détecteur spécialisé limité aux classes d’équipement.</p>
          <p><strong>Annotations :</strong> les cadres manuels sont stockés en local et exportables avec leurs images pour entraîner un modèle spécialisé. Ils n’altèrent pas directement les poids de COCO-SSD.</p><p><strong>Faster R-CNN :</strong> le pipeline fourni prépare un jeu de données COCO pour un entraînement hors navigateur. Il faut exporter, convertir et valider un modèle spécialisé avant de le brancher comme moteur d’inférence web.</p>
        </section>}
      </>) : <>
        <section className="report-heading panel">
          <div className="report-heading-main"><div className="report-emblem"><Icon name="report" size={25} /></div><div><div className="section-kicker">SYNTHÈSE DE VÉRIFICATION</div><h2>Rapport d’audit visuel</h2><p>{auditSnapshot ? `Relevé enregistré le ${formatDate(auditSnapshot.createdAt)} · ${auditSnapshot.source || 'source non précisée'}.` : 'Aucun relevé figé. Ce rapport est une prévisualisation des données actuellement disponibles.'}</p></div></div>
          <div className="report-actions"><button className="button button-secondary" onClick={exportJson}><Icon name="download" size={17} /> Exporter JSON</button><button className="button button-primary" onClick={() => window.print()}><Icon name="report" size={17} /> Imprimer / PDF</button></div>
        </section>
        <section className="report-status-panel">
          <div className="report-status-left"><span className="status-overline">STATUT GLOBAL</span><h2>{pageSummary.status}</h2><p>Les résultats reflètent les observations renseignées et les sorties du modèle au seuil défini.</p></div>
          <StatusPill type={pageSummary.statusType}>{pageSummary.status}</StatusPill>
          <div className="report-status-pattern" aria-hidden="true"><span /><span /><span /><span /><span /><span /><span /><span /><span /></div>
        </section>
        <section className="report-table-section panel">
          <div className="panel-heading report-table-heading"><div><div className="section-kicker">DÉTAIL PAR RÈGLE</div><h2>Attendu vs relevé</h2></div><span className="report-threshold">Seuil COCO-SSD : <strong>{Math.round((auditSnapshot?.threshold ?? threshold) * 100)}%</strong>{(auditSnapshot?.owlObserved ?? owlObserved) && <> · OWL-ViT : <strong>{Math.round((auditSnapshot?.owlThreshold ?? owlThreshold) * 100)}%</strong></>}</span></div>
          <div className="report-table-wrap"><table className="report-table">
            <thead><tr><th>Catégorie</th><th>Mode</th><th>Attendu</th><th>Détecté / relevé</th><th>Écart*</th><th>Statut de vérification</th></tr></thead>
            <tbody>{pageRows.map((row) => <tr key={row.id}>
              <td><div className="table-category"><CategoryIcon category={row} size={18} /><div><strong>{row.label}</strong><small>{row.detail}</small></div></div></td>
              <td><span className={`method-tag ${row.machineChecked ? 'auto' : 'manual'}`}>{row.machineChecked ? (row.owlUsed ? (row.method === 'auto' ? 'COCO + OWL-ViT' : 'OWL-ViT') : 'Automatique') : 'Manuel'}</span></td>
              <td className="number-cell">{row.expected === null ? '—' : row.expected}</td>
              <td className="number-cell">{row.detected === null ? '—' : row.detected}</td>
              <td className={`gap-cell ${row.gap > 0 ? 'gap-positive' : row.gap < 0 ? 'gap-negative' : ''}`}>{row.gap === null ? '—' : row.gap > 0 ? `+${row.gap}` : row.gap}</td>
              <td><div className="report-status-cell"><StatusPill type={row.statusType}>{row.status}</StatusPill>{row.note && <small>{row.note}</small>}</div></td>
            </tr>)}</tbody>
          </table></div>
          <div className="table-footnote">* Écart = relevé − attendu. Lorsque 0 objet est détecté par l’IA alors qu’un objet est attendu, l’écart reste « à confirmer » : un objet masqué ou hors cadre n’est pas déclaré absent.</div>
        </section>
        <section className="report-notes-grid">
          <article className="panel report-note-card"><span className="report-note-icon"><Icon name="shield" size={19} /></span><div><h3>Lecture prudente des résultats</h3><p>Une détection ne vaut pas inventaire exhaustif. Vérifiez les écarts et les objets potentiellement masqués, petits, éloignés ou hors champ avant de conclure.</p></div></article>
          <article className="panel report-note-card"><span className="report-note-icon"><Icon name="lock" size={19} /></span><div><h3>Confidentialité du relevé</h3><p>Le rapport contient les nombres et statuts uniquement. Aucune photo ou vidéo de caméra n’y est intégrée ni envoyée au serveur.</p></div></article>
        </section>
        <div className="report-bottom-actions"><button className="button button-secondary" onClick={() => { setView('live'); setActiveSection('audit'); }}><Icon name="arrow" size={17} /> Retour à la supervision</button><button className="text-button" onClick={saveAudit}><Icon name="refresh" size={15} /> Figer un nouveau relevé</button></div>
      </>}

      <footer className="site-footer"><a className="footer-brand" href="#top" onClick={() => { setView('live'); setActiveSection('audit'); }}>atlas<span>.</span></a><span>Audit visuel d’équipements · Version de démonstration</span><span className="footer-local"><span /> Traitement dans le navigateur</span></footer>
    </main>
  </div>;
}

createRoot(document.getElementById('root')).render(<React.StrictMode><App /></React.StrictMode>);
