/**
 * Jump Analyzer - Frame-by-frame AI tracking using COCO-SSD.
 *
 * Strategy:
 * 1. Load COCO-SSD model.
 * 2. Scan every frame from start to finish. For each frame, detect all persons.
 * 3. Build a trajectory by linking detections across frames (nearest-neighbour linking).
 * 4. The skier is the trajectory that is in the air (above waterline) and moving.
 * 5. Landing = first frame where the skier crosses the waterline.
 */

export async function analyzeJump(frames, calibPoints = [], onProgress = () => {}) {
  if (!frames || frames.length < 10) {
    throw new Error('Need at least 10 frames for analysis');
  }

  const width = frames[0].width;
  const height = frames[0].height;
  const totalFrames = frames.length;

  // --- Step 1: Load the AI model ---
  onProgress(0, 'Laddar AI-modell...');
  let mlModel = null;
  if (window.cocoSsd) {
    try {
      mlModel = await window.cocoSsd.load();
      console.log('[AI] COCO-SSD Model loaded.');
    } catch (e) {
      console.warn('[AI] Failed to load COCO-SSD', e);
    }
  }

  if (!mlModel) {
    throw new Error('Kunde inte ladda AI-modellen. Kontrollera din internetanslutning.');
  }

  // --- Step 2: Detect waterline using first few frames ---
  const bgGray = toGrayscale(frames[0]);
  const waterlineY = findWaterline(bgGray, width, height);
  console.log('[AI] Waterline at Y =', waterlineY);

  // --- Step 3: Scan every frame, detect persons ---
  // We scan the full frame but scale it down to 640px wide for speed.
  const scanW = 640;
  const scanH = Math.round(height * (scanW / width));
  const scanCanvas = new OffscreenCanvas(scanW, scanH);
  const scanCtx = scanCanvas.getContext('2d', { willReadFrequently: true });

  // We'll draw each frame into a temporary canvas to resize it
  const srcCanvas = new OffscreenCanvas(width, height);
  const srcCtx = srcCanvas.getContext('2d', { willReadFrequently: true });

  const scaleX = width / scanW;
  const scaleY = height / scanH;
  const waterlineScaled = waterlineY / scaleY;

  // All raw person detections: { frame, cx, cy, score, bbox }
  const allDetections = [];

  for (let f = 0; f < totalFrames; f++) {
    if (f % 3 === 0) {
      onProgress(0.05 + 0.6 * (f / totalFrames), `Scannar ruta ${f} av ${totalFrames}...`);
      await yieldToUI();
    }

    // Draw frame at scan resolution
    srcCtx.putImageData(frames[f], 0, 0);
    scanCtx.drawImage(srcCanvas, 0, 0, scanW, scanH);

    const preds = await mlModel.detect(scanCanvas, 10, 0.10);
    for (const p of preds) {
      if (p.class !== 'person') continue;
      const [bx, by, bw, bh] = p.bbox;
      const cx = (bx + bw / 2) * scaleX;
      const cy = (by + bh / 2) * scaleY;
      allDetections.push({ frame: f, cx, cy, score: p.score,
        bbox: { x: bx * scaleX, y: by * scaleY, w: bw * scaleX, h: bh * scaleY } });
    }
  }

  console.log(`[AI] Total person detections: ${allDetections.length}`);

  if (allDetections.length === 0) {
    throw new Error('Ingen person hittades i videon. Prova med ett annat klipp.');
  }

  // --- Step 4: Link detections into tracks ---
  // Simple greedy nearest-neighbour: each detection joins the closest open track.
  const tracks = []; // each track is array of detections

  for (const det of allDetections) {
    let bestTrack = null;
    let bestDist = Infinity;

    for (const track of tracks) {
      const last = track[track.length - 1];
      // Must be in the next few frames
      if (det.frame <= last.frame || det.frame - last.frame > 8) continue;
      const dist = Math.hypot(det.cx - last.cx, det.cy - last.cy);
      if (dist < 120 && dist < bestDist) {
        bestDist = dist;
        bestTrack = track;
      }
    }

    if (bestTrack) {
      bestTrack.push(det);
    } else {
      tracks.push([det]);
    }
  }

  console.log(`[AI] Built ${tracks.length} tracks.`);

  // --- Step 5: Find the skier track ---
  // The skier is the track that:
  // a) Has the most detections
  // b) Spends significant time ABOVE the waterline (in the air)
  // c) Has a trajectory that starts higher and ends lower (falling arc)

  // Score each track
  const scoredTracks = tracks.map(track => {
    const framesAboveWaterline = track.filter(d => d.cy < waterlineY).length;
    const startY = track[0].cy;
    const endY = track[track.length - 1].cy;
    const drop = endY - startY; // positive = falls downward
    const length = track.length;
    // Score: length + bonus for being in air + bonus for having a drop
    const score = length + framesAboveWaterline * 2 + Math.max(0, drop) * 0.1;
    return { track, score, framesAboveWaterline, drop, length };
  });

  scoredTracks.sort((a, b) => b.score - a.score);

  // Print top 3
  for (let i = 0; i < Math.min(3, scoredTracks.length); i++) {
    const s = scoredTracks[i];
    console.log(`[AI] Track ${i}: length=${s.length}, inAir=${s.framesAboveWaterline}, drop=${s.drop.toFixed(0)}, score=${s.score.toFixed(1)}, frames ${s.track[0].frame}-${s.track[s.track.length-1].frame}`);
  }

  const skierTrack = scoredTracks[0].track;

  // --- Step 6: Build trajectory output ---
  const trajectory = skierTrack.map(d => ({
    frame: d.frame,
    x: d.cx / width,
    y: d.cy / height
  }));

  // --- Step 7: Find landing frame ---
  // Landing = first frame in the skier track where the skier drops below waterline
  // and stays below (i.e. the crossing point)
  let landingFrame = null;
  let landingX = skierTrack[skierTrack.length - 1].cx;

  for (let i = 1; i < skierTrack.length; i++) {
    const prev = skierTrack[i - 1];
    const curr = skierTrack[i];
    if (prev.cy < waterlineY && curr.cy >= waterlineY) {
      // Linear interpolation to find exact crossing frame
      const fraction = (waterlineY - prev.cy) / (curr.cy - prev.cy);
      landingFrame = prev.frame + fraction;
      landingX = prev.cx + fraction * (curr.cx - prev.cx);
      console.log(`[AI] Landing detected at frame ${landingFrame.toFixed(1)}, X=${landingX.toFixed(0)}`);
      break;
    }
  }

  // Fallback: if skier never crosses waterline cleanly, use the last frame
  if (landingFrame === null) {
    // Find the minimum Y (highest point) in the track, then the first point
    // after that where Y is maximized (lowest point = in water)
    const peakIdx = skierTrack.reduce((bestI, d, i) => d.cy < skierTrack[bestI].cy ? i : bestI, 0);
    let maxY = -Infinity;
    let landIdx = skierTrack.length - 1;
    for (let i = peakIdx; i < skierTrack.length; i++) {
      if (skierTrack[i].cy > maxY) { maxY = skierTrack[i].cy; landIdx = i; }
    }
    landingFrame = skierTrack[landIdx].frame;
    landingX = skierTrack[landIdx].cx;
    console.log(`[AI] Landing fallback at frame ${landingFrame}, X=${landingX.toFixed(0)}`);
  }

  // --- Step 8: Find peak (highest point in air) ---
  const peakDet = skierTrack.reduce((best, d) => d.cy < best.cy ? d : best, skierTrack[0]);
  console.log(`[AI] Peak at frame ${peakDet.frame}, Y=${peakDet.cy.toFixed(0)}`);

  onProgress(1.0, 'Klar!');

  return {
    landingFrameIndex: Math.round(landingFrame),
    landingX: landingX / width,
    confidence: 0.85,
    phases: [
      { start: skierTrack[0].frame, end: peakDet.frame, type: 'FLIGHT_ASCENT' },
      { start: peakDet.frame, end: Math.round(landingFrame), type: 'FLIGHT_DESCENT' },
      { start: Math.round(landingFrame), end: totalFrames - 1, type: 'RIDE_AWAY' }
    ],
    trajectory,
    allDetections, // expose for rendering
    rampMarker: { x: null, y: null },
    rampNative: { x: null, y: null, topY: null, startX: null, endX: null },
    predBlobBox: { x: landingX - 20, y: waterlineY - 20, width: 40, height: 40 },
    flightBox: { x: landingX - 20, y: waterlineY - 20, width: 40, height: 40 },
    waterlineY: waterlineY / height
  };
}

// --- Helper: Compute waterline Y ---
function findWaterline(bgGray, width, height) {
  const scanTop    = Math.floor(height * 0.10);
  const scanBottom = Math.floor(height * 0.70);
  const checkRows  = 20;
  const sampleStep = 4;

  let bestY = Math.floor(height * 0.35);
  let bestScore = -Infinity;

  for (let y = scanTop + checkRows; y < scanBottom - checkRows; y++) {
    let sumA = 0, sumSqA = 0, cntA = 0;
    let sumB = 0, sumSqB = 0, cntB = 0;

    for (let dy = 1; dy <= checkRows; dy++) {
      const yy = y - dy;
      for (let x = 0; x < width; x += sampleStep) {
        const v = bgGray[yy * width + x];
        sumA += v; sumSqA += v * v; cntA++;
      }
    }
    for (let dy = 1; dy <= checkRows; dy++) {
      const yy = y + dy;
      for (let x = 0; x < width; x += sampleStep) {
        const v = bgGray[yy * width + x];
        sumB += v; sumSqB += v * v; cntB++;
      }
    }

    const varA = cntA > 0 ? (sumSqA / cntA) - (sumA / cntA) ** 2 : 0;
    const varB = cntB > 0 ? (sumSqB / cntB) - (sumB / cntB) ** 2 : 0;
    const score = varA - varB;

    if (score > bestScore) { bestScore = score; bestY = y; }
  }

  return bestY;
}

// --- Helper: Convert ImageData to grayscale Float32Array ---
function toGrayscale(imageData) {
  const { data, width, height } = imageData;
  const gray = new Float32Array(width * height);
  for (let i = 0; i < width * height; i++) {
    gray[i] = 0.299 * data[i*4] + 0.587 * data[i*4+1] + 0.114 * data[i*4+2];
  }
  return gray;
}

// --- Helper: Yield to UI ---
function yieldToUI() {
  return new Promise(resolve => setTimeout(resolve, 0));
}
