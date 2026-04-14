// MorelEye v2 — Service Worker
// Offline-first caching + background sync for no-signal field use

const CACHE_NAME   = 'moreleye-v2';
const STATIC_CACHE = 'moreleye-static-v2';
const MODEL_CACHE  = 'moreleye-models-v2';

// Assets cached on install — app shell
const SHELL = [
  '/',
  '/index.html',
  '/store.html',
  '/manifest.json',
  'https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=Exo+2:wght@300;400;600;700&family=Share+Tech+Mono&display=swap',
];

// ── Install ────────────────────────────────────────
self.addEventListener('install', e => {
  e.waitUntil(
    caches.open(STATIC_CACHE)
      .then(c => c.addAll(SHELL))
      .then(() => self.skipWaiting())
  );
});

// ── Activate ───────────────────────────────────────
self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(
        keys
          .filter(k => ![STATIC_CACHE, MODEL_CACHE].includes(k))
          .map(k => caches.delete(k))
      )
    ).then(() => self.clients.claim())
  );
});

// ── Fetch strategy ─────────────────────────────────
self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);

  // 1. Anthropic API — network only (needs signal)
  //    If offline and user taps scan, return friendly error
  if (url.hostname === 'api.anthropic.com') {
    e.respondWith(
      fetch(e.request).catch(() =>
        new Response(JSON.stringify({
          status: 'OFFLINE',
          primary: 'No signal detected.',
          details: 'You are offline. For offline scanning, download the on-device AI model from the Field Store before heading out.',
          confidence: 0,
          caution: null,
          habitat_note: 'Download offline models from the Store tab before your next trip.',
          detections: []
        }), { headers: { 'Content-Type': 'application/json' } })
      )
    );
    return;
  }

  // 2. Internal API scan — network first, offline fallback
  if (url.pathname === '/api/scan') {
    e.respondWith(
      fetch(e.request).catch(() =>
        new Response(JSON.stringify({
          status: 'OFFLINE',
          primary: 'No signal. Offline AI model not yet loaded.',
          details: 'Open the Field Store and download the on-device model pack. Once downloaded, scanning works with no signal.',
          confidence: 0,
          caution: null,
          habitat_note: 'Tip: download offline packs at home before heading into the woods.',
          detections: []
        }), { headers: { 'Content-Type': 'application/json' } })
      )
    );
    return;
  }

  // 3. Static app shell — cache first
  if (url.pathname === '/' ||
      url.pathname.endsWith('.html') ||
      url.pathname.endsWith('.js') ||
      url.pathname.endsWith('.json') ||
      url.pathname.endsWith('.css')) {
    e.respondWith(
      caches.match(e.request).then(cached => {
        const network = fetch(e.request).then(res => {
          if (res.ok) {
            const clone = res.clone();
            caches.open(STATIC_CACHE).then(c => c.put(e.request, clone));
          }
          return res;
        }).catch(() => cached);
        return cached || network;
      })
    );
    return;
  }

  // 4. Model files — cache first (large files, don't re-fetch)
  if (url.pathname.startsWith('/models/')) {
    e.respondWith(
      caches.match(e.request).then(cached => {
        if (cached) return cached;
        return fetch(e.request).then(res => {
          if (res.ok) {
            const clone = res.clone();
            caches.open(MODEL_CACHE).then(c => c.put(e.request, clone));
          }
          return res;
        });
      })
    );
    return;
  }

  // 5. Everything else — network with cache fallback
  e.respondWith(
    fetch(e.request).catch(() => caches.match(e.request))
  );
});

// ── Background sync (scan retry) ──────────────────
// When signal returns, retry any failed scan attempts
self.addEventListener('sync', e => {
  if (e.tag === 'retry-scan') {
    e.waitUntil(retryScan());
  }
});

async function retryScan() {
  // In production: pull from IndexedDB queue and retry
  const clients = await self.clients.matchAll();
  clients.forEach(c => c.postMessage({ type: 'SYNC_COMPLETE' }));
}

// ── Push notifications (seasonal alerts) ──────────
self.addEventListener('push', e => {
  const data = e.data?.json() || {};
  e.waitUntil(
    self.registration.showNotification(data.title || 'MorelEye Alert', {
      body: data.body || 'Morel season conditions detected in your area!',
      icon: '/icons/icon-192.png',
      badge: '/icons/badge-72.png',
      tag: 'moreleye-alert',
      data: { url: data.url || '/' }
    })
  );
});

self.addEventListener('notificationclick', e => {
  e.notification.close();
  e.waitUntil(clients.openWindow(e.notification.data.url || '/'));
});

// ── Message handler ────────────────────────────────
self.addEventListener('message', e => {
  if (e.data.type === 'DOWNLOAD_MODEL') {
    // Triggered from store.html when user downloads a model pack
    const { modelId, modelUrl } = e.data;
    caches.open(MODEL_CACHE).then(cache => {
      fetch(modelUrl).then(res => {
        cache.put(modelUrl, res);
        e.source.postMessage({ type: 'MODEL_READY', modelId });
      }).catch(() => {
        e.source.postMessage({ type: 'MODEL_FAILED', modelId });
      });
    });
  }

  if (e.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }
});
