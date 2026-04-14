// ============================================================
//  MorelEye — Smart Scan Router
//  Automatically uses offline AI when no signal,
//  falls back to Claude Vision API when online.
//  Drop this in and replace all fetch('/api/scan') calls.
// ============================================================

// Lazy-load TF.js only when needed (heavy library)
let engineModule = null;
let engine = null;
let engineReady = false;
let engineLoading = false;

// Track owned packs (loaded from localStorage / native purchase verification)
let ownedPacks = loadOwnedPacks();

function loadOwnedPacks() {
  try {
    const stored = localStorage.getItem('moreleye_owned_packs');
    return stored ? JSON.parse(stored) : ['morel'];
  } catch { return ['morel']; }
}

export function setOwnedPacks(packs) {
  ownedPacks = packs;
  try { localStorage.setItem('moreleye_owned_packs', JSON.stringify(packs)); } catch {}
}

export function getOwnedPacks() { return ownedPacks; }

// ── Network detection ──────────────────────────────
async function isOnline() {
  if (!navigator.onLine) return false;
  try {
    // Quick ping to verify real connectivity
    const ctrl = new AbortController();
    setTimeout(() => ctrl.abort(), 3000);
    await fetch('/api/ping', { method: 'HEAD', signal: ctrl.signal });
    return true;
  } catch { return false; }
}

// ── Load offline engine (lazy) ─────────────────────
async function loadOfflineEngine(onProgress) {
  if (engineReady) return engine;
  if (engineLoading) {
    // Wait for existing load to finish
    while (engineLoading) await new Promise(r => setTimeout(r, 100));
    return engine;
  }

  engineLoading = true;

  try {
    // Dynamically import TF.js and engine (deferred for performance)
    engineModule = await import('./offline-engine.js');
    engine = engineModule.offlineEngine;

    engine.onProgress = onProgress || (() => {});
    await engine.init(ownedPacks);

    engineReady = true;
    console.log('MorelEye offline engine ready');
  } catch(e) {
    console.error('Failed to load offline engine:', e);
    engine = null;
  }

  engineLoading = false;
  return engine;
}

// ── Main scan function — replaces fetch('/api/scan') ──
export async function scanImage(imageElement, options = {}) {
  const {
    preferOffline = false,
    onEngineProgress = null,
    onModeChange = null,
  } = options;

  const online = !preferOffline && await isOnline();

  // ── ONLINE: Use Claude Vision API ─────────────
  if (online) {
    onModeChange?.('online');

    // Capture frame to base64
    const b64 = captureBase64(imageElement);
    if (!b64) throw new Error('Could not capture image');

    const res = await fetch('/api/scan', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        imageB64: b64,
        ownedPacks,
        species: ownedPacks[0] || 'morel'
      })
    });

    const result = await res.json();
    result.offline = false;
    return result;
  }

  // ── OFFLINE: Use on-device TF.js engine ───────
  onModeChange?.('offline');

  // Show loading UI if engine not ready
  const eng = await loadOfflineEngine(onEngineProgress);

  if (!eng || !eng.ready) {
    return {
      status: 'UNCLEAR',
      confidence: 0,
      primary: 'Offline AI not ready.',
      details: 'Download the offline model from the Field Store while you have signal. Once downloaded it works with no connection.',
      caution: null,
      habitat_note: 'Open the Store tab to download offline packs.',
      detections: [],
      offline: true,
      error: 'engine_not_ready'
    };
  }

  try {
    const result = await eng.detectInImage(imageElement, ownedPacks);
    result.offline = true;
    return result;
  } catch(e) {
    console.error('Offline detection error:', e);
    return {
      status: 'UNCLEAR',
      confidence: 0,
      primary: 'Detection error.',
      details: 'On-device AI encountered an error. Try again or switch to online mode.',
      caution: null, habitat_note: null, detections: [],
      offline: true, error: e.message
    };
  }
}

// ── Preload offline engine in background ──────────
// Call this when the app loads to warm up the engine
// so it's instant when the user goes offline
export async function preloadOfflineEngine(onProgress) {
  if (engineReady || engineLoading) return;
  // Check if models are cached in IndexedDB
  try {
    const dbs = await indexedDB.databases?.() || [];
    const hasModel = dbs.some(db => db.name?.startsWith('moreleye-model-'));
    if (hasModel) {
      // Silently warm up in background
      setTimeout(() => loadOfflineEngine(onProgress), 2000);
    }
  } catch { /* indexedDB.databases() not supported — skip preload */ }
}

// ── Download model pack to IndexedDB ──────────────
export async function downloadModelPack(packId, onProgress) {
  const MODEL_BASE_URL = '/models'; // Served from your CDN or app bundle

  try {
    // In production: fetch the quantized TF.js model shards
    // and store them in IndexedDB via tf.loadLayersModel + .save()

    // Simulate download with progress for now
    // Replace with actual model download:
    //   const model = await tf.loadLayersModel(`${MODEL_BASE_URL}/${packId}/model.json`);
    //   await model.save(`indexeddb://moreleye-model-${packId}`);

    onProgress?.(0.1);
    await simulateDownload(packId, onProgress);

    // Mark pack as available offline
    const offlinePacks = JSON.parse(localStorage.getItem('moreleye_offline_packs') || '[]');
    if (!offlinePacks.includes(packId)) {
      offlinePacks.push(packId);
      localStorage.setItem('moreleye_offline_packs', JSON.stringify(offlinePacks));
    }

    onProgress?.(1.0);
    return { success: true, packId };
  } catch(e) {
    console.error('Download failed:', e);
    return { success: false, packId, error: e.message };
  }
}

async function simulateDownload(packId, onProgress) {
  // Replace with real model download in production
  let p = 0.1;
  while (p < 1.0) {
    await new Promise(r => setTimeout(r, 150));
    p = Math.min(p + Math.random() * 0.08, 1.0);
    onProgress?.(p);
  }
}

// ── Get offline status ─────────────────────────────
export function getOfflineStatus() {
  const offlinePacks = JSON.parse(localStorage.getItem('moreleye_offline_packs') || '[]');
  return {
    engineReady,
    engineLoading,
    offlinePacks,
    isOnline: navigator.onLine,
    memoryInfo: engineReady ? engine?.memoryInfo() : null,
  };
}

// ── Capture base64 from video/image element ────────
function captureBase64(el) {
  try {
    const c = document.getElementById('cap') || document.createElement('canvas');
    if (el.tagName === 'VIDEO') {
      c.width = el.videoWidth; c.height = el.videoHeight;
      c.getContext('2d').drawImage(el, 0, 0);
    } else {
      c.width = el.naturalWidth || el.width;
      c.height = el.naturalHeight || el.height;
      c.getContext('2d').drawImage(el, 0, 0);
    }
    return c.toDataURL('image/jpeg', 0.82).split(',')[1];
  } catch { return null; }
}

// ── Listen for online/offline events ──────────────
window.addEventListener('online', () => {
  console.log('MorelEye: Connection restored — switching to cloud AI');
});
window.addEventListener('offline', () => {
  console.log('MorelEye: Offline — switching to on-device AI');
  // Pre-warm engine if models are available
  preloadOfflineEngine();
});
