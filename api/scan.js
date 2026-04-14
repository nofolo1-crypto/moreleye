// api/scan.js — MorelEye v2 — Vercel Serverless Function

export default async function handler(req, res) {
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });
  res.setHeader('Access-Control-Allow-Origin', '*');

  const SYSTEM = `You are MorelEye v2, an expert mycologist and forest ecologist in a real-time AR foraging app. Analyze the field image and detect:

1. MOREL MUSHROOMS (Morchella spp.) — honeycomb/pitted cap, hollow, cap fused to stem, tan/grey/black.
2. HOST TREES — Identify the SPECIES NAME. Look for: dead/dying elm (rough furrowed bark, opposite branching), ash (diamond-ridged bark, compound leaves, often EAB-killed), apple/fruit trees (gnarly low branches, rough bark), tulip poplar (flat-topped leaves, straight tall trunk), sycamore (white/grey patchy peeling bark), cottonwood (near water, furrowed bark), oak (lobed leaves, acorn caps on ground). Even without leaves, use bark texture, branching pattern, and tree form.
3. HABITAT FEATURES — south-facing slopes, creek bottoms, leaf litter depth, disturbed soil, moisture.

Respond ONLY as valid JSON with NO markdown, NO preamble:
{
  "status": "MOREL_FOUND" | "HOST_TREE" | "FAVORABLE_HABITAT" | "CLEAR" | "UNCLEAR",
  "confidence": 0-100,
  "primary": "One crisp sentence naming what you found.",
  "details": "2-3 sentences of specific field detail. For trees: species name, identifying features you see, and why it hosts morels. For morels: cap texture, color, size estimate, hollow confirmation if visible. For habitat: what specific conditions make this favorable.",
  "caution": "False morel or safety note if applicable, else null",
  "habitat_note": "Short location-specific foraging tip based on what you see in this scene, else null",
  "detections": [
    {
      "type": "morel" | "tree" | "habitat",
      "label": "Exact species or type e.g. 'Dead American Elm' or 'Yellow Morel' or 'Ash — EAB Killed'",
      "confidence": 0-100,
      "x": 5-90,
      "y": 5-85,
      "w": 5-45,
      "h": 5-50
    }
  ]
}

Rules:
- MOREL_FOUND only with clear morel morphology visible
- For HOST_TREE always give the species common name in the label
- Coordinates are percentage of image: x/y = top-left corner of bounding box, w/h = width/height
- Place bounding boxes accurately around the actual detected object
- Multiple detections allowed — label every significant host tree you see
- Be specific — "Dead American Elm" beats "dead tree"`;

  try {
    const { imageB64 } = req.body;
    if (!imageB64) return res.status(400).json({ error: 'No image data.' });

    const https = require('https');
    const payload = JSON.stringify({
      model: 'claude-sonnet-4-20250514',
      max_tokens: 1200,
      system: SYSTEM,
      messages: [{
        role: 'user',
        content: [
          { type: 'image', source: { type: 'base64', media_type: 'image/jpeg', data: imageB64 } },
          { type: 'text', text: 'Analyze this field image. Identify all host trees with species names, any morel mushrooms, and habitat conditions. Place accurate bounding boxes around each detection.' }
        ]
      }]
    });

    const options = {
      hostname: 'api.anthropic.com',
      path: '/v1/messages',
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': process.env.ANTHROPIC_API_KEY,
        'anthropic-version': '2023-06-01',
        'Content-Length': Buffer.byteLength(payload)
      }
    };

    const apiReq = require('https').request(options, apiRes => {
      let data = '';
      apiRes.on('data', c => data += c);
      apiRes.on('end', () => {
        try {
          const parsed = JSON.parse(data);
          const raw = parsed.content?.find(b => b.type === 'text')?.text || '';
          let result;
          try { result = JSON.parse(raw.replace(/```json|```/g, '').trim()); }
          catch { const m = raw.match(/\{[\s\S]*\}/); result = m ? JSON.parse(m[0]) : null; }

          if (result) { res.writeHead ? res.status(200).json(result) : res.status(200).json(result); }
          else { res.status(500).json({ error: 'Could not parse AI response.' }); }
        } catch(e) { res.status(500).json({ error: 'Parse error.' }); }
      });
    });

    apiReq.on('error', () => res.status(502).json({ error: 'API connection failed.' }));
    apiReq.write(payload);
    apiReq.end();

  } catch(e) {
    res.status(500).json({ error: 'Server error.' });
  }
}
