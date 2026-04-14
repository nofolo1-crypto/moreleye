// server.js — MorelEye v2
// Multi-species detection + offline model manifest + static serving

const http  = require('http');
const https = require('https');
const fs    = require('fs');
const path  = require('path');

const PORT    = process.env.PORT || 3000;
const API_KEY = process.env.ANTHROPIC_API_KEY;

// ── Species system prompts ─────────────────────────
// Each purchased pack unlocks a richer system prompt
const SPECIES_PROMPTS = {

  morel: `You are MorelEye, expert mycologist. Detect MOREL MUSHROOMS (Morchella spp.) and HOST TREES.
MORELS: honeycomb/pitted cap, hollow inside, cap fused to stem, tan/grey/black coloring.
HOST TREES: Dead/dying elm (rough furrowed bark), ash (EAB-killed, diamond ridges), apple (gnarly low branches), tulip poplar (straight tall trunk), sycamore (white patchy bark), oak (lobed leaves/acorn caps).`,

  chicken: `You are MorelEye. Detect CHICKEN OF THE WOODS (Laetiporus sulphureus/cincinnatus).
Look for: bright orange and yellow shelf/bracket fungus on oak, cherry, locust, or other hardwood.
Distinguish from: Jack-o-lantern (thinner, gilled, grows in clusters at base). Note age and freshness.`,

  chanterelle: `You are MorelEye. Detect CHANTERELLE mushrooms (Cantharellus cibarius and species).
Look for: golden-yellow funnel cap, forking false ridges (not true gills), fruity apricot aroma clues in scene.
Dangerous lookalike: Jack-o-lantern (true gills, grows in clusters at wood base, glows faintly at night).`,

  oyster: `You are MorelEye. Detect OYSTER MUSHROOMS (Pleurotus ostreatus and species).
Look for: fan/oyster-shaped caps on dead or dying hardwood, white to grey to yellow color, gills run down stem.
Dangerous lookalike: Angel wings (thin, white, grows on conifer — toxic). Note harvest cluster size.`,

  hen: `You are MorelEye. Detect HEN OF THE WOODS / MAITAKE (Grifola frondosa).
Look for: overlapping grey-brown fronds at base of oak trees in fall. Polypore structure.
Compare with: Berkeley's polypore (paler, spring/summer, on oaks and other trees). Assess harvest size.`,

  lions: `You are MorelEye. Detect LION'S MANE (Hericium erinaceus).
Look for: white cascading teeth/spines on dead or wounded oak, beech, maple. No cap or gills.
Related edible species: Bear's head, coral tooth. All Hericium are edible. Assess freshness (white=fresh, yellow=old).`,

  ramps: `You are MorelEye. Detect RAMPS / WILD LEEKS (Allium tricoccum).
Look for: broad smooth lance-shaped leaves, reddish stem, onion/garlic smell clues in habitat.
TOXIC LOOKALIKE: Lily-of-the-valley (similar leaves, NO onion smell, toxic) — always confirm garlic scent.`,

  berries: `You are MorelEye. Detect EDIBLE WILD BERRIES.
Target species: blackberry, raspberry, blueberry, serviceberry, elderberry, mulberry, autumn olive.
DANGEROUS LOOKALIKES: Pokeweed berries (deep purple clusters on thick red stems — toxic), nightshade (small clusters — toxic).
Assess ripeness and note any toxic plants in scene.`,

  adaptogens: `You are MorelEye. Detect MEDICINAL ADAPTOGEN FUNGI.
REISHI (Ganoderma lucidum): shiny reddish-brown kidney-shaped shelf, lacquered surface.
CHAGA (Inonotus obliquus): black charcoal-like mass on birch trees, orange interior.  
TURKEY TAIL (Trametes versicolor): thin multicolored concentric-banded shelves on logs.
Note growth stage, freshness, and host tree for each.`,

  herbs: `You are MorelEye. Detect MEDICINAL WILD HERBS.
Target: yarrow (feathery leaves, flat white flower clusters), St. John's wort (yellow star flowers, perforated leaves), goldenrod (tall yellow plumes), mullein (tall spike, fuzzy leaves), echinacea (purple coneflower).
Note bloom stage and which plant part to harvest.`,

  default: `You are MorelEye, expert field naturalist. Identify any edible or medicinal wild plants and fungi visible in this image. Note species name, identifying features, and any dangerous lookalikes.`
};

const BASE_INSTRUCTIONS = `
Respond ONLY as valid JSON — no markdown, no preamble:
{
  "status": "MOREL_FOUND" | "HOST_TREE" | "SPECIES_FOUND" | "FAVORABLE_HABITAT" | "CLEAR" | "UNCLEAR",
  "confidence": 0-100,
  "primary": "One sentence naming what you found.",
  "details": "2-3 sentences: species name, identifying features visible, why it matches or doesn't.",
  "caution": "Dangerous lookalike warning if present, else null",
  "habitat_note": "Short location-specific tip based on this scene, else null",
  "detections": [
    {
      "type": "morel" | "tree" | "mushroom" | "plant" | "habitat",
      "label": "Exact species e.g. 'Chanterelle' or 'Dead American Elm' or 'Ramps'",
      "confidence": 0-100,
      "x": 5-90, "y": 5-85, "w": 5-45, "h": 5-50
    }
  ]
}
Place bounding boxes accurately. Multiple detections allowed. Be conservative — only confirm with clear visual evidence.`;

// ── MIME types ─────────────────────────────────────
const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js':   'application/javascript',
  '.json': 'application/json',
  '.css':  'text/css',
  '.png':  'image/png',
  '.jpg':  'image/jpeg',
  '.ico':  'image/x-icon',
  '.woff2':'font/woff2',
};

// ── Offline model manifest ─────────────────────────
// In production, these would be actual on-device model files
// served from a CDN or bundled in the native app wrapper
const OFFLINE_MANIFEST = {
  version: '2.0.0',
  models: [
    { id: 'morel',    name: 'Morel Detector',       size_mb: 38,  included: true  },
    { id: 'trees',    name: 'Host Tree Identifier',  size_mb: 22,  included: true  },
    { id: 'chicken',  name: 'Chicken of the Woods',  size_mb: 18,  included: false },
    { id: 'chanterelle', name: 'Chanterelle',        size_mb: 18,  included: false },
    { id: 'oyster',   name: 'Oyster Mushroom',       size_mb: 16,  included: false },
    { id: 'ramps',    name: 'Ramps & Wild Onion',    size_mb: 14,  included: false },
    { id: 'berries',  name: 'Wild Berries',          size_mb: 24,  included: false },
  ],
  topo_maps: { available: true, size_per_region_mb: 80 }
};

// ── HTTP Server ────────────────────────────────────
const server = http.createServer(async (req, res) => {
  // CORS
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, X-Species-Pack, X-Owned-Packs');
  if (req.method === 'OPTIONS') { res.writeHead(204); res.end(); return; }

  const url = new URL(req.url, `http://localhost:${PORT}`);

  // ── POST /api/scan ─────────────────────────────
  if (req.method === 'POST' && url.pathname === '/api/scan') {
    let body = '';
    req.on('data', c => body += c);
    req.on('end', async () => {
      try {
        const { imageB64, species = 'morel', ownedPacks = ['morel'] } = JSON.parse(body);

        if (!imageB64) {
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'No image data.' })); return;
        }

        // Build system prompt based on owned packs
        let systemPrompt = '';
        const packs = Array.isArray(ownedPacks) ? ownedPacks : [species];

        if (packs.length === 1) {
          systemPrompt = (SPECIES_PROMPTS[packs[0]] || SPECIES_PROMPTS.default) + BASE_INSTRUCTIONS;
        } else {
          // Multi-pack: detect everything the user owns
          const combined = packs
            .map(p => SPECIES_PROMPTS[p])
            .filter(Boolean)
            .join('\n\n');
          systemPrompt = `You are MorelEye with MULTI-SPECIES detection enabled. The user has unlocked multiple packs — detect ALL of the following:\n\n${combined}\n\nIf multiple species are present, return multiple detections.${BASE_INSTRUCTIONS}`;
        }

        const payload = JSON.stringify({
          model: 'claude-sonnet-4-20250514',
          max_tokens: 1200,
          system: systemPrompt,
          messages: [{
            role: 'user',
            content: [
              { type: 'image', source: { type: 'base64', media_type: 'image/jpeg', data: imageB64 } },
              { type: 'text', text: 'Analyze this field image. Identify all target species with accurate bounding boxes.' }
            ]
          }]
        });

        callClaude(payload, res);

      } catch(e) {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Server error.' }));
      }
    });
    return;
  }

  // ── GET /api/offline-manifest ──────────────────
  if (req.method === 'GET' && url.pathname === '/api/offline-manifest') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(OFFLINE_MANIFEST));
    return;
  }

  // ── GET /api/packs ─────────────────────────────
  if (req.method === 'GET' && url.pathname === '/api/packs') {
    const packs = Object.keys(SPECIES_PROMPTS).filter(k => k !== 'default').map(id => ({
      id, available: true, offline_size_mb: Math.floor(Math.random() * 20) + 12
    }));
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ packs }));
    return;
  }

  // ── Static files ───────────────────────────────
  let filePath = url.pathname === '/' ? '/index.html' : url.pathname;
  filePath = path.join(__dirname, filePath);

  fs.readFile(filePath, (err, data) => {
    if (err) {
      // Try serving store.html for /store
      if (url.pathname === '/store') {
        fs.readFile(path.join(__dirname, 'store.html'), (e2, d2) => {
          if (e2) { res.writeHead(404); res.end('Not found'); return; }
          res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
          res.end(d2);
        }); return;
      }
      res.writeHead(404); res.end('Not found'); return;
    }
    const ext = path.extname(filePath);
    res.writeHead(200, { 'Content-Type': MIME[ext] || 'text/plain' });
    res.end(data);
  });
});

// ── Claude API call helper ─────────────────────────
function callClaude(payload, res) {
  const options = {
    hostname: 'api.anthropic.com',
    path: '/v1/messages',
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': API_KEY,
      'anthropic-version': '2023-06-01',
      'Content-Length': Buffer.byteLength(payload)
    }
  };

  const apiReq = https.request(options, apiRes => {
    let data = '';
    apiRes.on('data', c => data += c);
    apiRes.on('end', () => {
      try {
        const parsed = JSON.parse(data);
        const raw = parsed.content?.find(b => b.type === 'text')?.text || '';
        let result;
        try { result = JSON.parse(raw.replace(/```json|```/g, '').trim()); }
        catch { const m = raw.match(/\{[\s\S]*\}/); result = m ? JSON.parse(m[0]) : null; }

        if (result) {
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify(result));
        } else {
          res.writeHead(500, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'Could not parse AI response.' }));
        }
      } catch(e) {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Parse error.' }));
      }
    });
  });

  apiReq.on('error', () => {
    res.writeHead(502, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: 'API connection failed. Check signal.' }));
  });

  apiReq.write(payload);
  apiReq.end();
}

server.listen(PORT, () => console.log(`MorelEye v2 running on :${PORT}`));
