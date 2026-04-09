// server.js — Replit entry point for MorelEye
// Serves the app AND proxies Claude Vision API calls securely

const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = process.env.PORT || 3000;
const API_KEY = process.env.ANTHROPIC_API_KEY;

const SYSTEM = `You are MorelEye, an expert mycologist integrated into a mobile AR foraging app. Analyze field images and detect:

1. MOREL MUSHROOMS (Morchella spp.) — honeycomb/pitted cap, hollow, cap fused to stem, tan/grey/black.
2. HOST TREES — dead/dying elm, EAB-killed ash, apple/fruit trees, tulip poplar, sycamore, oak. Look at bark, form, branching, any remaining leaves.
3. HABITAT — south-facing slope, creek bottom, disturbed soil, leaf litter, moisture.

Respond ONLY as valid JSON — no markdown, no preamble:
{
  "status": "MOREL_FOUND" | "HOST_TREE" | "FAVORABLE_HABITAT" | "CLEAR" | "UNCLEAR",
  "confidence": 0-100,
  "primary": "One crisp sentence.",
  "details": "2-3 sentences of field-relevant detail.",
  "caution": "False morel or safety warning if applicable, else null",
  "detections": [
    {
      "type": "morel" | "tree" | "habitat",
      "label": "Short label",
      "confidence": 0-100,
      "x": 5-90, "y": 5-85, "w": 5-45, "h": 5-45
    }
  ]
}
Be conservative. MOREL_FOUND only with clear morel morphology.`;

const MIME = {
  '.html': 'text/html',
  '.js':   'application/javascript',
  '.json': 'application/json',
  '.png':  'image/png',
  '.ico':  'image/x-icon',
};

const server = http.createServer(async (req, res) => {
  // ── CORS ──────────────────────────────────────────
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  if (req.method === 'OPTIONS') { res.writeHead(204); res.end(); return; }

  // ── API proxy ─────────────────────────────────────
  if (req.method === 'POST' && req.url === '/api/scan') {
    let body = '';
    req.on('data', chunk => body += chunk);
    req.on('end', async () => {
      try {
        const { imageB64 } = JSON.parse(body);
        if (!imageB64) {
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'No image data.' }));
          return;
        }

        // Call Claude Vision
        const https = require('https');
        const payload = JSON.stringify({
          model: 'claude-sonnet-4-20250514',
          max_tokens: 1000,
          system: SYSTEM,
          messages: [{
            role: 'user',
            content: [
              { type: 'image', source: { type: 'base64', media_type: 'image/jpeg', data: imageB64 } },
              { type: 'text', text: 'Analyze this field image for morel mushrooms, host trees, and habitat.' }
            ]
          }]
        });

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
              try {
                result = JSON.parse(raw.replace(/```json|```/g, '').trim());
              } catch {
                const m = raw.match(/\{[\s\S]*\}/);
                result = m ? JSON.parse(m[0]) : null;
              }
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

        apiReq.on('error', e => {
          res.writeHead(502, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'API connection failed.' }));
        });

        apiReq.write(payload);
        apiReq.end();

      } catch(e) {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Server error.' }));
      }
    });
    return;
  }

  // ── Static file server ────────────────────────────
  let filePath = req.url === '/' ? '/index.html' : req.url;
  filePath = path.join(__dirname, filePath);

  fs.readFile(filePath, (err, data) => {
    if (err) {
      res.writeHead(404, { 'Content-Type': 'text/plain' });
      res.end('Not found');
      return;
    }
    const ext = path.extname(filePath);
    res.writeHead(200, { 'Content-Type': MIME[ext] || 'text/plain' });
    res.end(data);
  });
});

server.listen(PORT, () => console.log(`MorelEye running on port ${PORT}`));
