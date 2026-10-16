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
    detail: 'COCO-SSD détecte surtout les tables de repas ; un bureau peut être manqué.',
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
    detail: 'Non reconnu nativement par COCO-SSD — relevé manuel.',
  },
  {
    id: 'window', label: 'Fenêtres', singular: 'Fenêtre', method: 'manual',
    expectedDefault: '2', color: '#79d9de', symbol: 'window', family: 'Éléments de salle',
    detail: 'Non reconnu nativement par COCO-SSD — relevé manuel.',
  },
  {
    id: 'anomaly', label: 'Anomalies visuelles', singular: 'Anomalie', method: 'manual',
    expectedDefault: '0', color: '#ff8585', symbol: 'alert', family: 'Contrôle visuel',
    detail: 'Évaluation humaine requise ; aucune détection automatique d’anomalie.',
  },
];

const AUTO_IDS = CATEGORIES.filter((c) => c.method === 'auto').map((c) => c.id);
const INITIAL_EXPECTED = Object.fromEntries(CATEGORIES.map((category) => [category.id, category.expectedDefault]));
const EMPTY_MANUAL = { board: '', window: '', anomaly: '' };
const EMPTY_COUNTS = Object.fromEntries(AUTO_IDS.map((id) => [id, 0]));

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

function makeRows(expectedValues, autoCounts, manualCounts, observed, snapshotMode = false) {
  return CATEGORIES.map((category) => {
    const expected = getValue(expectedValues[category.id]);
    const detected = category.method === 'auto'
      ? (observed ? (autoCounts[category.id] ?? 0) : null)
      : getValue(manualCounts[category.id]);
    let status = 'En attente';
    let statusType = 'pending';
    let gap = null;
    let note = '';

    if (expected === null) {
      status = 'Objectif à définir';
      statusType = 'pending';
    } else if (category.method === 'auto' && detected === null) {
      status = snapshotMode ? 'Non relevé par la caméra' : 'En attente de la caméra';
      statusType = 'pending';
    } else if (category.method === 'manual' && detected === null) {
      status = 'Relevé manuel requis';
      statusType = 'pending';
    } else if (category.method === 'auto' && detected === 0) {
      status = expected > 0 ? '0 détecté · à confirmer' : '0 détecté · à vérifier';
      statusType = 'caution';
      note = 'Une absence de détection ne prouve pas l’absence de l’objet ; vérifier le cadrage et les éventuels masquages.';
    } else if (detected === expected) {
      status = 'Concordant au relevé';
      statusType = 'good';
    } else {
      gap = detected - expected;
      status = category.method === 'manual' ? 'Écart relevé manuellement' : 'Écart de comptage observé';
      statusType = 'bad';
      note = category.method === 'auto'
        ? 'Le nombre détecté diffère de l’attendu ; vérifier le cadrage et les objets masqués.'
        : 'Écart déclaré lors du contrôle manuel.';
    }

    return { ...category, expected, detected, gap, status, statusType, note };
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

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const modelRef = useRef(null);
  const busyRef = useRef(false);
  const thresholdRef = useRef(0.55);
  const [stream, setStream] = useState(null);
  const [cameraOn, setCameraOn] = useState(false);
  const [model, setModel] = useState(null);
  const [modelStatus, setModelStatus] = useState('idle');
  const [modelError, setModelError] = useState('');
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
  const [notice, setNotice] = useState('');
  const [showMethodNotes, setShowMethodNotes] = useState(false);

  thresholdRef.current = threshold;

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

  const startCamera = async () => {
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
    const detectFrame = async () => {
      const video = videoRef.current;
      if (!cancelled && video && video.readyState >= 2 && !busyRef.current) {
        busyRef.current = true;
        try {
          const predictions = await model.detect(video, 20, thresholdRef.current);
          if (cancelled) return;
          // Ne conserver que les catégories d'équipement autorisées. La sortie "person"
          // ainsi que toutes les autres classes sont ignorées et ne sont pas journalisées.
          const equipment = predictions
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
              };
            });
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
  }, [cameraOn, model]);

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

  const liveRows = useMemo(
    () => makeRows(expected, counts, manual, hasObservedFrame, false),
    [expected, counts, manual, hasObservedFrame],
  );
  const snapshotRows = useMemo(() => {
    if (!auditSnapshot) return null;
    return makeRows(auditSnapshot.expected, auditSnapshot.autoCounts, auditSnapshot.manualCounts, auditSnapshot.observed, true);
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
      threshold,
      model: 'COCO-SSD · lite_mobilenet_v2',
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
      model: auditSnapshot?.model || 'COCO-SSD · lite_mobilenet_v2',
      confidenceThreshold: auditSnapshot?.threshold ?? threshold,
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
        method: row.method === 'auto' ? 'COCO-SSD (automatique)' : 'Déclaration manuelle',
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
    : hasObservedFrame ? 'Instantané courant · évolue avec la caméra' : 'Configurez les attentes puis démarrez la caméra';

  return <div className="app-shell">
    <header className="topbar">
      <a className="brand" href="#top" onClick={() => setView('live')} aria-label="Atlas, accueil">
        <span className="brand-mark"><span /><span /><span /><span /></span>
        <span className="brand-word">atlas<span>.</span><small>VISUAL AUDIT</small></span>
      </a>
      <nav className="top-nav" aria-label="Navigation principale">
        <button className={view === 'live' ? 'nav-link active' : 'nav-link'} onClick={() => setView('live')}><Icon name="grid" size={16} /> Supervision</button>
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
          <h1>{view === 'report' ? <>Rapport <span>d’audit.</span></> : <>L’œil juste.<br /><span>Le relevé clair.</span></>}</h1>
          <p>{view === 'report'
            ? 'Confrontez le relevé visuel à la configuration attendue. Les détections non confirmées restent signalées comme telles.'
            : 'Une supervision visuelle des équipements et du mobilier, directement dans votre navigateur. Les images restent sur votre appareil.'}</p>
          {view === 'live' && <div className="intro-meta"><span><Icon name="target" size={16} /> 7 règles de contrôle</span><span><Icon name="shield" size={16} /> Aucune vidéo envoyée</span><span><Icon name="grid" size={16} /> Mobile & bureau</span></div>}
        </div>
        <div className="intro-aside">
          <span className="aside-index">{view === 'report' ? '02 / RÉSULTATS' : '01 / SUPERVISION'}</span>
          <span className="aside-rule" />
          <p>{view === 'report' ? 'Le relevé fait foi des observations disponibles, pas d’une preuve absolue d’absence.' : 'Un outil d’aide au contrôle. Chaque écart appelle une vérification humaine.'}</p>
          {view === 'live'
            ? <button className="button button-dark" onClick={saveAudit}>Figer le relevé <Icon name="arrow" size={17} /></button>
            : <button className="button button-dark" onClick={() => setView('live')}>Retour à la caméra <Icon name="arrow" size={17} /></button>}
        </div>
      </section>

      {notice && <div className="notice-strip" role="status"><Icon name="info" size={17} /><span>{notice}</span><button onClick={() => setNotice('')} aria-label="Fermer le message"><Icon name="close" size={16} /></button></div>}

      <section className="stats-grid" aria-label="Indicateurs de l’audit">
        <StatCard label="Objets relevés" value={pageSummary.total} detail={verifiedDescription} icon="target" />
        <StatCard label="Catégories reconnues" value={`${pageSummary.recognized} / 7`} detail="Catégories avec au moins une occurrence" icon="grid" />
        <StatCard label="Règles configurées" value={`${pageSummary.configured} / 7`} detail={`${pageSummary.checked} règle(s) déjà renseignée(s)`} icon="check" />
        <StatCard label="Écarts observés" value={pageSummary.deviations} detail={pageSummary.unresolved ? `${pageSummary.unresolved} contrôle(s) à confirmer` : 'Aucun contrôle en attente'} icon="alert" tone={pageSummary.deviations > 0 ? 'stat-card-alert' : ''} />
      </section>

      {view === 'live' ? <>
        <section className="workspace-grid">
          <article className="panel camera-panel">
            <div className="panel-heading">
              <div><div className="section-kicker">ESPACE DE CONTRÔLE</div><h2>Flux de détection</h2></div>
              <div className={`live-indicator ${cameraOn ? 'is-live' : ''}`}><span />{cameraOn ? 'EN DIRECT' : 'À L’ARRÊT'}</div>
            </div>
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
            <div className="camera-controls">
              {cameraOn ? <button className="button button-danger" onClick={stopCamera}><Icon name="stop" size={17} /> Arrêter la caméra</button> : <button className="button button-primary" onClick={startCamera} disabled={modelStatus === 'loading'}><Icon name="camera" size={17} />{modelStatus === 'loading' ? 'Chargement du modèle…' : 'Démarrer la caméra'}</button>}
              <div className="confidence-control">
                <div className="confidence-heading"><label htmlFor="confidence">Seuil de confiance</label><strong>{Math.round(threshold * 100)}%</strong></div>
                <input id="confidence" type="range" min="0.2" max="0.9" step="0.05" value={threshold} onChange={(event) => setThreshold(Number(event.target.value))} />
                <div className="range-scale"><span>Plus de détections</span><span>Plus strict</span></div>
              </div>
            </div>
            <div className="camera-footnote"><Icon name="info" size={15} /><span>Les cadres sont mis à jour environ toutes les 0,65 seconde. Les performances dépendent de l’appareil et de l’éclairage.</span></div>
            <div className="privacy-warning"><Icon name="shield" size={16} /><p><strong>Zone d’inspection réservée aux équipements.</strong> Si une personne entre dans le champ, arrêtez immédiatement la détection. L’interface ignore et ne conserve pas les sorties « person », mais COCO-SSD est un modèle multi-classes générique : ce filtrage seul ne garantit pas l’absence de calcul interne sur une personne.</p></div>
          </article>

          <aside className="panel detections-panel">
            <div className="panel-heading detections-heading">
              <div><div className="section-kicker">LECTURE DU MODÈLE</div><h2>Objets en vue</h2></div>
              <span className="count-orb">{hasObservedFrame ? currentTotal : '—'}</span>
            </div>
            <p className="panel-subtitle">Comptage des objets visibles dans la dernière image traitée, pas un inventaire garanti de toute la pièce.</p>
            <div className="category-count-list">
              {CATEGORIES.filter((category) => category.method === 'auto').map((category) => {
                const count = hasObservedFrame ? counts[category.id] : null;
                return <div className="category-count-row" key={category.id}>
                  <CategoryIcon category={category} />
                  <div className="category-count-info"><span>{category.label}</span><small>{category.method === 'auto' ? 'COCO-SSD' : 'Manuel'}</small></div>
                  <span className={`category-count-value ${count > 0 ? 'has-count' : ''}`}>{count === null ? '—' : count}</span>
                </div>;
              })}
            </div>
            <div className="detected-objects-block">
              <div className="subsection-heading"><span>Détections individuelles</span><span>{objects.length} résultat(s)</span></div>
              <div className="object-list" aria-live="polite">
                {objects.length === 0 ? <div className="empty-objects"><span className="empty-ring"><Icon name="target" size={20} /></span><strong>{hasObservedFrame ? 'Aucun équipement reconnu' : 'En attente d’une image'}</strong><span>{hasObservedFrame ? 'Cela ne prouve pas que les équipements sont absents.' : 'Démarrez la caméra pour lancer la détection.'}</span></div> : objects.map((object, index) => <div className="object-row" key={object.id}>
                  <span className="object-index">{String(index + 1).padStart(2, '0')}</span>
                  <span className="object-mark" style={{ background: object.color }} />
                  <div className="object-info"><strong>{object.label}</strong><small>Objet {index + 1}</small></div>
                  <span className="confidence-score">{Math.round(object.score * 100)}%</span>
                </div>)}
              </div>
            </div>
            <div className="model-note"><span className="note-icon"><Icon name="shield" size={17} /></span><p><strong>Sorties limitées aux équipements</strong><br />Les résultats hors des quatre classes suivies ne sont pas affichés ni conservés par l’application.</p></div>
          </aside>
        </section>

        <section className="panel audit-config-panel">
          <div className="panel-heading config-heading">
            <div><div className="section-kicker">RÈGLES & RELEVÉ HUMAIN</div><h2>Configuration initiale</h2></div>
            <span className="sample-config-label">Valeurs d’exemple · modifiables</span>
          </div>
          <p className="panel-subtitle config-intro">Renseignez les quantités attendues avant le contrôle. Pour les fenêtres, tableaux et anomalies, indiquez manuellement ce que vous observez ; une case vide signifie « non relevé », pas zéro.</p>
          <div className="config-table-wrap">
            <table className="config-table">
              <thead><tr><th>Catégorie</th><th>Mode de vérification</th><th>Nombre attendu</th><th>Relevé manuel</th><th>Observation</th></tr></thead>
              <tbody>{CATEGORIES.map((category) => <tr key={category.id}>
                <td><div className="table-category"><CategoryIcon category={category} size={18} /><span>{category.label}</span></div></td>
                <td><span className={`method-tag ${category.method}`}>{category.method === 'auto' ? 'IA · COCO-SSD' : 'Contrôle humain'}</span></td>
                <td><label className="number-input"><input type="number" inputMode="numeric" min="0" step="1" value={expected[category.id]} onChange={(event) => updateExpected(category.id, event.target.value)} aria-label={`Nombre attendu : ${category.label}`} /><span>obj.</span></label></td>
                <td>{category.method === 'manual' ? <label className="number-input manual-number"><input type="number" inputMode="numeric" min="0" step="1" placeholder="—" value={manual[category.id]} onChange={(event) => updateManual(category.id, event.target.value)} aria-label={`Relevé manuel : ${category.label}`} /><span>obj.</span></label> : <span className="not-applicable">Auto</span>}</td>
                <td><span className="table-detail">{category.detail}</span></td>
              </tr>)}</tbody>
            </table>
          </div>
          <div className="config-footer">
            <button className="text-button reset-button" onClick={() => { setExpected({ ...INITIAL_EXPECTED }); setManual({ ...EMPTY_MANUAL }); setNotice('Configuration d’exemple restaurée. Vérifiez les quantités avant usage.'); }}><Icon name="refresh" size={15} /> Restaurer l’exemple</button>
            <button className="button button-primary" onClick={saveAudit}>Figer le relevé & ouvrir le rapport <Icon name="arrow" size={17} /></button>
          </div>
        </section>

        <section className="method-strip">
          <div className="method-strip-icon"><Icon name="info" size={20} /></div>
          <div><h3>À propos de la méthode</h3><p>COCO-SSD est un détecteur pré-entraîné, pas un modèle réentraîné sur votre salle. Le seuil modifie les sorties conservées, mais ne crée pas de nouvelles capacités de détection.</p></div>
          <button className="text-button" onClick={() => setShowMethodNotes((value) => !value)}>{showMethodNotes ? 'Réduire' : 'Limites & confidentialité'} <Icon name={showMethodNotes ? 'close' : 'arrow'} size={15} /></button>
        </section>
        {showMethodNotes && <section className="expanded-method panel">
          <h3>Limites à connaître avant l’audit</h3>
          <p><strong>Classes disponibles :</strong> la détection automatique exploite uniquement les classes <code>chair</code>, <code>dining table</code>, <code>laptop</code> et <code>tv</code>. Les bureaux, fenêtres, tableaux et anomalies visuelles ne sont pas couverts de façon native par ce modèle.</p>
          <p><strong>Personnes :</strong> l’interface ne présente, n’enregistre et n’exporte pas les sorties « person », et ne fait aucune identification. Cependant, COCO-SSD reste un modèle généraliste multi-classes : filtrer ses sorties n'est pas la preuve qu'aucun calcul interne relatif à une personne n'a lieu. Pour une exigence stricte, cadre la caméra uniquement sur les équipements, suspends la détection si quelqu’un entre dans le champ et remplace le modèle générique par un détecteur spécialisé limité aux classes d’équipement.</p>
          <p><strong>Apprentissage :</strong> cette version n'effectue ni apprentissage supervisé sur les retours d'audit ni apprentissage par renforcement. Un tel apprentissage réel exige un jeu de données annoté et un pipeline d'entraînement dédié.</p>
        </section>}
      </> : <>
        <section className="report-heading panel">
          <div className="report-heading-main"><div className="report-emblem"><Icon name="report" size={25} /></div><div><div className="section-kicker">SYNTHÈSE DE VÉRIFICATION</div><h2>Rapport d’audit visuel</h2><p>{auditSnapshot ? `Relevé enregistré le ${formatDate(auditSnapshot.createdAt)}.` : 'Aucun relevé figé. Ce rapport est une prévisualisation des données actuellement disponibles.'}</p></div></div>
          <div className="report-actions"><button className="button button-secondary" onClick={exportJson}><Icon name="download" size={17} /> Exporter JSON</button><button className="button button-primary" onClick={() => window.print()}><Icon name="report" size={17} /> Imprimer / PDF</button></div>
        </section>
        <section className="report-status-panel">
          <div className="report-status-left"><span className="status-overline">STATUT GLOBAL</span><h2>{pageSummary.status}</h2><p>Les résultats reflètent les observations renseignées et les sorties du modèle au seuil défini.</p></div>
          <StatusPill type={pageSummary.statusType}>{pageSummary.status}</StatusPill>
          <div className="report-status-pattern" aria-hidden="true"><span /><span /><span /><span /><span /><span /><span /><span /><span /></div>
        </section>
        <section className="report-table-section panel">
          <div className="panel-heading report-table-heading"><div><div className="section-kicker">DÉTAIL PAR RÈGLE</div><h2>Attendu vs relevé</h2></div><span className="report-threshold">Seuil de confiance : <strong>{Math.round((auditSnapshot?.threshold ?? threshold) * 100)}%</strong></span></div>
          <div className="report-table-wrap"><table className="report-table">
            <thead><tr><th>Catégorie</th><th>Mode</th><th>Attendu</th><th>Détecté / relevé</th><th>Écart*</th><th>Statut de vérification</th></tr></thead>
            <tbody>{pageRows.map((row) => <tr key={row.id}>
              <td><div className="table-category"><CategoryIcon category={row} size={18} /><div><strong>{row.label}</strong><small>{row.detail}</small></div></div></td>
              <td><span className={`method-tag ${row.method}`}>{row.method === 'auto' ? 'Automatique' : 'Manuel'}</span></td>
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
        <div className="report-bottom-actions"><button className="button button-secondary" onClick={() => setView('live')}><Icon name="arrow" size={17} /> Retour à la supervision</button><button className="text-button" onClick={saveAudit}><Icon name="refresh" size={15} /> Figer un nouveau relevé</button></div>
      </>}

      <footer className="site-footer"><a className="footer-brand" href="#top" onClick={() => setView('live')}>atlas<span>.</span></a><span>Audit visuel d’équipements · Version de démonstration</span><span className="footer-local"><span /> Traitement dans le navigateur</span></footer>
    </main>
  </div>;
}

createRoot(document.getElementById('root')).render(<React.StrictMode><App /></React.StrictMode>);
