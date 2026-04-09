// api/scan.js  —  Vercel Serverless Function
// Proxies Claude Vision requests without exposing your API key

export default async function handler(req, res) {
  // Only allow POST
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  // Basic rate limiting header (Vercel edge handles most abuse)
  res.setHeader('Access-Control-Allow-Origin', '*');

  try {
    const { imageB64 } = req.body;

    if (!imageB64) {
      return res.status(400).json({ error: 'No image data provided' });
    }

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
}`;

    const response = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': process.env.ANTHROPIC_API_KEY,   // ← set this in Vercel dashboard
        'anthropic-version': '2023-06-01'
      },
      body: JSON.stringify({
        model: 'claude-sonnet-4-20250514',
        max_tokens: 1000,
        system: SYSTEM,
        messages: [{
          role: 'user',
          content: [
            {
              type: 'image',
              source: { type: 'base64', media_type: 'image/jpeg', data: imageB64 }
            },
            {
              type: 'text',
              text: 'Analyze this field image for morel mushrooms, host trees, and habitat suitability.'
            }
          ]
        }]
      })
    });

    if (!response.ok) {
      const err = await response.text();
      console.error('Anthropic error:', err);
      return res.status(502).json({ error: 'AI service error. Try again.' });
    }

    const data = await response.json();
    const raw = data.content?.find(b => b.type === 'text')?.text || '';

    let result;
    try {
      result = JSON.parse(raw.replace(/```json|```/g, '').trim());
    } catch {
      const m = raw.match(/\{[\s\S]*\}/);
      result = m ? JSON.parse(m[0]) : null;
    }

    if (!result) {
      return res.status(500).json({ error: 'Could not parse AI response.' });
    }

    return res.status(200).json(result);

  } catch (err) {
    console.error('Scan error:', err);
    return res.status(500).json({ error: 'Server error. Try again.' });
  }
}
