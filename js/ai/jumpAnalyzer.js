/**
 * AI Jump Analyzer — Detects jump phases and landing point.
 * 
 * Uses frame differencing (no ML model) to:
 * 1. Detect the jumper's position per frame via motion analysis
 * 2. Classify jump phases: APPROACH → FLIGHT → LANDING → RIDE_AWAY
 * 3. Find the landing point (where back of ski hits water)
 * 
 * The landing point is defined as the small water "mound" that forms
 * where the back end of the ski contacts the water. Per official rules,
 * +2.1m is added to this measurement for the final jump distance.
 * 
 * All processing runs locally in the browser on the user's device.
 */

/**
 * Analyze an array of ImageData frames to detect the landing point.
 * 
 * @param {ImageData[]} frames - Array of video frames as ImageData
 * @param {Function} onProgress - Callback(progress: 0-1, message: string)
 * @returns {Promise<{
 *   landingFrameIndex: number,
 *   landingX: number,          // normalized 0..1
 *   confidence: number,        // 0..1
 *   phases: {start: number, end: number, type: string}[],
 *   trajectory: {frame: number, x: number, y: number}[]
 * }>}
 */
export async function analyzeJump(frames, calibPoints = [], onProgress = () => {}) {
  if (!frames || frames.length < 10) {
    throw new Error('Need at least 10 frames for analysis');
  }

  const width = frames[0].width;
  const height = frames[0].height;
  const totalFrames = frames.length;

  onProgress(0, 'Building background model...');

  // Step 1: Build background model from first few frames
  const bgFrameCount = Math.min(5, Math.floor(totalFrames * 0.1));
  const bgGray = new Float32Array(width * height);
  
  for (let f = 0; f < bgFrameCount; f++) {
    const gray = toGrayscale(frames[f]);
    for (let i = 0; i < gray.length; i++) {
      bgGray[i] += gray[i] / bgFrameCount;
    }
  }

  onProgress(0.1, 'Detecting motion...');

  // Step 2: Frame differencing — track motion per frame
  // We analyze the upper portion of the image where the jumper flies
  const waterlineY = Math.floor(height * 0.55);  // Below this is mostly water
  const roiTop = 0;
  const roiBottom = waterlineY;

  const trackingData = [];
  
  for (let f = 0; f < totalFrames; f++) {
    if (f % 5 === 0) {
      onProgress(0.1 + 0.5 * (f / totalFrames), `Analyzing frame ${f + 1}/${totalFrames}...`);
      // Yield to UI thread periodically
      await yieldToUI();
    }

    const gray = toGrayscale(frames[f]);
    
    // Compute difference from background
    const diff = new Uint8Array(width * height);
    for (let i = 0; i < gray.length; i++) {
      diff[i] = Math.min(255, Math.abs(gray[i] - bgGray[i]));
    }

    // Find motion blobs in ROI (upper portion)
    const blob = findLargestBlob(diff, width, height, roiTop, roiBottom, 25);
    
    trackingData.push({
      frame: f,
      ...blob
    });
  }

  onProgress(0.6, 'Analyzing jump trajectory...');

  // Step 3: Identify the flight arc
  // During flight, the jumper's Y position reaches a minimum (highest in image)
  const detectedFrames = trackingData.filter(t => t.detected);
  
  if (detectedFrames.length < 5) {
    return {
      landingFrameIndex: null,
      landingX: null,
      confidence: 0,
      phases: [],
      trajectory: trackingData.filter(t => t.detected).map(t => ({
        frame: t.frame, x: t.cx / width, y: t.cy / height
      })),
      error: 'Not enough motion detected. Try ensuring the jump is visible in the recording.'
    };
  }

  // Find the peak of the flight (minimum Y = highest point)
  // Filter for frames where the blob is reasonably sized (not background noise)
  const significantFrames = detectedFrames.filter(t => t.area > 300);
  
  if (significantFrames.length < 3) {
    return {
      landingFrameIndex: null,
      landingX: null,
      confidence: 0,
      phases: [],
      trajectory: detectedFrames.map(t => ({
        frame: t.frame, x: t.cx / width, y: t.cy / height
      })),
      error: 'Motion detected but no clear jump pattern found.'
    };
  }

  // Find peak frame (highest Y = smallest value)
  const peakFrame = significantFrames.reduce((best, t) => 
    t.cy < best.cy ? t : best, significantFrames[0]);
  
  onProgress(0.7, 'Detecting landing point...');

  // Step 4: Determine jumper direction (left-to-right or right-to-left)
  // Compare X positions at start vs peak of flight
  const earlyFrames = significantFrames.filter(t => t.frame < peakFrame.frame);
  let jumpDirection = 0; // -1 = right-to-left, +1 = left-to-right
  if (earlyFrames.length > 2) {
    const startX = earlyFrames[0].cx;
    const peakX = peakFrame.cx;
    jumpDirection = peakX > startX ? 1 : -1;
  }

  // Step 5: Find initial water contact
  // After the peak, the jumper descends. Contact = first frame where
  // Y rises back significantly toward the waterline
  const postPeakFrames = significantFrames.filter(t => t.frame > peakFrame.frame);
  
  const yValues = significantFrames.map(t => t.cy);
  const yMin = Math.min(...yValues);
  const yMax = Math.max(...yValues);
  const yRange = yMax - yMin;

  let initialContactFrame = null;
  for (const t of postPeakFrames) {
    if (t.cy > yMin + yRange * 0.5) {
      initialContactFrame = t.frame;
      break;
    }
  }

  // Fallback: area spike (splash start)
  if (initialContactFrame === null && postPeakFrames.length > 0) {
    for (let i = 1; i < postPeakFrames.length; i++) {
      if (postPeakFrames[i].area > postPeakFrames[i-1].area * 1.3) {
        initialContactFrame = postPeakFrames[i].frame;
        break;
      }
    }
  }

  if (initialContactFrame === null) {
    initialContactFrame = postPeakFrames.length > 0 
      ? postPeakFrames[Math.floor(postPeakFrames.length * 0.5)].frame 
      : peakFrame.frame + 5;
  }

  onProgress(0.8, 'Locating water mound...');

  // Step 6: Find the water "mound" — the precise landing measurement point
  //
  // Domain knowledge:
  // - The mound appears ~0.3-0.7s AFTER initial ski-water contact
  // - It's at the position where the BACK END of the ski hit the water
  // - This is the FURTHEST point from the ramp in the splash zone
  // - We detect it using frame-to-frame brightness change onset
  //
  // Strategy: 
  // 1. For each frame in the search window, compute brightness change from PREVIOUS frame
  //    (not from background) — this detects the moment the mound APPEARS
  // 2. Find the first significant brightness spike (onset, not maximum)
  // 3. X position = the edge of the disturbed water closest to the camera/furthest from ramp

  // Get last known jumper X near contact for search region
  let lastKnownX = null;
  const contactIdx = initialContactFrame || 0;
  for (const t of trackingData.slice(Math.max(0, contactIdx - 10), contactIdx + 1)) {
    if (t.detected) lastKnownX = t.cx;
  }
  if (lastKnownX === null) lastKnownX = peakFrame.cx;

  // HYBRID APPROACH:
  // Frame: initialContactFrame + 7 (ski contact is ~7 frames after blob tracker's estimate)
  // X: Detect splash at a LATER frame where it's visible, then use that X position
  
  let landingFrame = null;
  let landingX = null;
  let rampIsRight = null;
  
  const safeContactFrame = initialContactFrame || Math.floor(totalFrames * 0.4);
  
  // Find landing frame: frame of MAXIMUM cy (lowest point in image) after peak = full water contact
  // Then go back 10 frames = rear ski tip first touching water
  const LANDING_OFFSET = -10;
  let fullLandingFrame = null;
  
  // Collect detected frames after peak, up to a reasonable window
  const postPeakDetected = trackingData.filter(t => 
    t.detected && t.frame > peakFrame.frame && t.frame <= safeContactFrame + 40
  );
  
  if (postPeakDetected.length > 0) {
    // Find frame with maximum cy (lowest point in image = full water contact)
    const maxCyFrame = postPeakDetected.reduce((best, t) => 
      t.cy > best.cy ? t : best
    );
    fullLandingFrame = maxCyFrame.frame;
  }
  
  // Apply offset: go back 10 frames = rear ski tip contact
  if (fullLandingFrame !== null) {
    landingFrame = Math.max(0, fullLandingFrame + LANDING_OFFSET);
  } else {
    // Fallback
    landingFrame = Math.min(safeContactFrame + 7, totalFrames - 1);
  }
  
  console.log('[AI] initialContactFrame:', initialContactFrame, 'peakFrame:', peakFrame.frame);
  console.log('[AI] Max cy frame (full landing):', fullLandingFrame, '→ measurement frame (offset', LANDING_OFFSET, '):', landingFrame);
  
  // STEP 1: LOCATE RAMP using vertical edge detection at horizon zone
  // The ramp has STRAIGHT EDGES (triangle sides) that create strong vertical edges.
  let rampMarkerX = 0, rampMarkerY = 0;
  let rampNativeX = null, rampNativeY = null, rampNativeTopY = null;
  let rampNativeStartX = null, rampNativeEndX = null;
  let rampFullWidth = 200; // default: exclude 200px from ramp edge
  
  // First determine ramp side from splash data
  const splashFrame = Math.min(safeContactFrame + 23, totalFrames - 1);
  const splashGray = toGrayscale(frames[splashFrame]);
  const preSplashGray = toGrayscale(frames[Math.max(0, splashFrame - 4)]);
  
  const splashRy1 = Math.floor(height * 0.35);
  const splashRy2 = Math.floor(height * 0.48);
  const colProfile = new Array(width).fill(0);
  
  for (let y = splashRy1; y < splashRy2; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      const vsPreSplash = splashGray[idx] - preSplashGray[idx];
      const vsBg = splashGray[idx] - bgGray[idx];
      if (vsPreSplash > 30 && vsBg > 30 && splashGray[idx] > 180) {
        colProfile[x]++;
      }
    }
  }
  
  // Find main wake cluster for ramp side detection
  const clusters = [];
  let cStart = -1;
  for (let x = 0; x < width; x++) {
    if (colProfile[x] > 0) {
      if (cStart === -1) cStart = x;
    } else {
      if (cStart !== -1) {
        let mass = 0;
        for (let cx = cStart; cx <= x - 1; cx++) mass += colProfile[cx];
        clusters.push({ start: cStart, end: x - 1, mass });
        cStart = -1;
      }
    }
  }
  if (cStart !== -1) {
    let mass = 0;
    for (let cx = cStart; cx < width; cx++) mass += colProfile[cx];
    clusters.push({ start: cStart, end: width - 1, mass });
  }
  
  if (clusters.length > 0) {
    const mainWake = clusters.reduce((a, b) => a.mass > b.mass ? a : b);
    const wakeMid = Math.floor((mainWake.start + mainWake.end) / 2);
    let leftMass = 0, rightMass = 0;
    for (let x = mainWake.start; x <= mainWake.end; x++) {
      if (x < wakeMid) leftMass += colProfile[x];
      else rightMass += colProfile[x];
    }
    rampIsRight = rightMass < leftMass;
    console.log('[AI] Ramp side: left wake:', leftMass, 'right wake:', rightMass,
      '→', rampIsRight ? 'RIGHT' : 'LEFT');
  }
  
  // Find ramp position using vertical edge detection
  if (rampIsRight !== null) {
    const searchX1 = rampIsRight ? Math.floor(width * 0.55) : 0;
    const searchX2 = rampIsRight ? width : Math.floor(width * 0.45);
    const horizonY1 = Math.floor(height * 0.15);
    const horizonY2 = Math.floor(height * 0.45);
    
    const edgeStrength = new Array(width).fill(0);
    for (let x = searchX1 + 2; x < searchX2 - 2; x++) {
      let totalEdge = 0;
      for (let y = horizonY1; y < horizonY2; y++) {
        const left = bgGray[y * width + (x - 2)];
        const right = bgGray[y * width + (x + 2)];
        totalEdge += Math.abs(right - left);
      }
      edgeStrength[x] = totalEdge;
    }
    
    const smoothEdge = new Array(width).fill(0);
    for (let x = searchX1 + 5; x < searchX2 - 5; x++) {
      let sum = 0;
      for (let dx = -3; dx <= 3; dx++) sum += edgeStrength[x + dx];
      smoothEdge[x] = sum / 7;
    }
    
    let maxEdge = 0;
    for (let x = searchX1; x < searchX2; x++) {
      if (smoothEdge[x] > maxEdge) maxEdge = smoothEdge[x];
    }
    
    const edgeThreshold = maxEdge * 0.3;
    const edgeClusters = [];
    cStart = -1;
    for (let x = searchX1; x < searchX2; x++) {
      if (smoothEdge[x] > edgeThreshold) {
        if (cStart === -1) cStart = x;
      } else {
        if (cStart !== -1) {
          const w = x - cStart;
          if (w >= 10 && w <= 300) {
            edgeClusters.push({ start: cStart, end: x - 1, width: w, cx: Math.round((cStart + x - 1) / 2) });
          }
          cStart = -1;
        }
      }
    }
    
    console.log('[AI] Ramp edge clusters:', edgeClusters.map(c => 
      `[${c.start}-${c.end}](w=${c.width},cx=${c.cx})`).join(' '));
    
    if (edgeClusters.length > 0) {
      const rampCluster = rampIsRight 
        ? edgeClusters.reduce((a, b) => a.cx > b.cx ? a : b)
        : edgeClusters.reduce((a, b) => a.cx < b.cx ? a : b);
      
      // Find ramp top Y and base Y
      let rampTopY = horizonY2;
      for (let y = horizonY1; y < horizonY2; y++) {
        let rowEdge = 0;
        for (let x = rampCluster.start; x <= rampCluster.end; x++) {
          const left = bgGray[y * width + (x - 2)];
          const right = bgGray[y * width + (x + 2)];
          rowEdge += Math.abs(right - left);
        }
        if (rowEdge > maxEdge * 0.1) {
          rampTopY = y;
          break;
        }
      }
      let rampBaseY = horizonY2;
      for (let y = horizonY2; y > horizonY1; y--) {
        let rowEdge = 0;
        for (let x = rampCluster.start; x <= rampCluster.end; x++) {
          const left = bgGray[y * width + (x - 2)];
          const right = bgGray[y * width + (x + 2)];
          rowEdge += Math.abs(right - left);
        }
        if (rowEdge > maxEdge * 0.1) {
          rampBaseY = y;
          break;
        }
      }
      
      rampNativeX = rampCluster.cx;
      rampNativeY = rampBaseY;
      rampMarkerX = rampCluster.cx / width;
      rampMarkerY = rampTopY / height;
      
      // Save ramp dimensions for skier search
      rampNativeTopY = rampTopY;
      rampNativeStartX = rampCluster.start;
      rampNativeEndX = rampCluster.end;
      
      // Full ramp extent: from outermost cluster to innermost
      // This gives us the total ramp width for exclusion zone
      const outerEdge = rampIsRight
        ? edgeClusters.reduce((a, b) => a.start < b.start ? a : b).start  // leftmost = outer edge for right ramp
        : edgeClusters.reduce((a, b) => a.end > b.end ? a : b).end;       // rightmost = outer edge for left ramp
      rampFullWidth = Math.abs(rampCluster.cx - outerEdge);
      
      console.log('[AI] Ramp located: x=', rampCluster.cx, 
        'topY=', rampTopY, 'baseY=', rampBaseY, 'height=', rampBaseY - rampTopY,
        'cluster:', rampCluster.start, '-', rampCluster.end,
        'fullWidth:', rampFullWidth);
    }
  }
  
  // STEP 2: FULL-WIDTH SILHOUETTE SCAN (RAW GRAYSCALE)
  // No motion tracking or trajectory prediction used.
  // We scan across the waterline at the identified landing frame,
  // looking for a dark, high-aspect-ratio (vertical) silhouette.
  // Search boundaries are restricted by user calibration points if available,
  // and we exclude the ramp area to avoid false positives.

  let blobBox = null;
  let predBlobBox = null;
  const allDetections = new Array(totalFrames).fill(null);

  // --- Waterline estimate ---
  // Ramp base is the best reference; else use max cy of tracked flight blobs.
  const skierCys = trackingData.filter(t => t.detected && t.frame <= safeContactFrame).map(t => t.cy);
  const estimatedWaterlineY = rampNativeY ||
    (skierCys.length > 0 ? Math.max(...skierCys) : Math.floor(height * 0.25));
  console.log('[AI] Estimated waterline Y:', estimatedWaterlineY);

  // --- Define X Search Bounds ---
  let searchStartX = 0;
  let searchEndX = width - 1;

  if (calibPoints && calibPoints.length >= 2) {
    // If calibration points exist, use them as strict boundaries
    // Note: calib points now store normX (0..1)
    const calibXs = calibPoints.filter(p => typeof p.normX !== 'undefined').map(p => p.normX * width);
    if (calibXs.length >= 2) {
      searchStartX = Math.max(0, Math.floor(Math.min(...calibXs)));
      searchEndX = Math.min(width - 1, Math.ceil(Math.max(...calibXs)));
      console.log(`[AI] Bounding search to calibration zone: ${searchStartX} - ${searchEndX}`);
    }
  }

  // --- Peak-based constraints removed per user request ---
  // We rely entirely on finding visual candidates and sorting by distance to ramp.


  // --- Exclude Ramp Region ---
  // If we found the ramp, avoid searching inside it
  if (typeof rampNativeStartX !== 'undefined' && typeof rampNativeEndX !== 'undefined') {
    if (rampIsRight) {
      // Ramp is on the right, skier lands to the left of it
      searchEndX = Math.min(searchEndX, rampNativeStartX - 50); // 50px safety margin
    } else {
      // Ramp is on the left, skier lands to the right of it
      searchStartX = Math.max(searchStartX, rampNativeEndX + 50);
    }
    console.log(`[AI] Adjusted for ramp exclusion: ${searchStartX} - ${searchEndX}`);
  }

  // --- Scan Backwards from Splash to Peak ---
  // The user identified that looking at a single frame (-10 offset) is fragile.
  // Instead, we know the biggest splash is at fullLandingFrame.
  // We scan backwards frame-by-frame from the splash towards the peak.
  // We score all valid skier silhouettes.
  // The frame with the highest silhouette score will naturally be the moment
  // the skier is most clearly visible at the waterline, right before the splash obscures them.

  const yTop = Math.max(0, estimatedWaterlineY - Math.floor(height * 0.22));
  const yBot = Math.min(height - 1, estimatedWaterlineY + 20);
  
  const scanStartFrame = fullLandingFrame !== null ? fullLandingFrame : Math.min(safeContactFrame + 15, totalFrames - 1);
  const scanEndFrame = peakFrame.frame;
  
  let bestGlobalScore = 0;
  let bestLandingX = null;
  let bestLandingFrame = null;
  let bestBlobBox = null;
  
  // Parameters for silhouette shape
  const DARK_THRESH = 110;   // Skier is dark
  const MIN_BLOB_AREA = 30;  // Must be somewhat visible
  
  // Determine the X-coordinate of the ramp
  let rampX = rampIsRight ? width : 0;
  if (typeof rampNativeStartX !== 'undefined' && typeof rampNativeEndX !== 'undefined') {
    rampX = rampIsRight ? rampNativeStartX : rampNativeEndX;
  }

  for (let f = scanStartFrame; f >= scanEndFrame; f--) {
    let candidates = [];
    const fGray = toGrayscale(frames[f]);
    const darkMask = new Uint8Array(width * height);
    
    // Create a mask where dark pixels get a high value (like a diff map) so we can reuse floodFill
    for (let i = 0; i < width * height; i++) {
      darkMask[i] = fGray[i] < DARK_THRESH ? 255 : 0;
    }
    
    const vis = new Uint8Array(width * height);
    
    for (let y = yTop; y < yBot; y++) {
      for (let x = searchStartX; x <= searchEndX; x++) {
        const idx = y * width + x;
        if (vis[idx] || darkMask[idx] === 0) continue;
        
        // Found a dark pixel, flood-fill it
        const comp = floodFill(darkMask, vis, width, height, x, y, 128, yTop, yBot, searchStartX, searchEndX);
        
        if (comp.area >= MIN_BLOB_AREA) {
          // We want a high aspect ratio (taller than it is wide)
          const compAspect = comp.h / Math.max(1, comp.w);
          // We want it to be anchored near the waterline
          const distToWaterline = Math.abs(comp.maxY - estimatedWaterlineY);
          
          if (compAspect > 1.2 && distToWaterline < 40) {
            candidates.push(comp);
          }
        }
      }
    }
    
    if (candidates.length > 0) {
      // Sort candidates by distance to the ramp (closest first)
      candidates.sort((a, b) => Math.abs(a.cx - rampX) - Math.abs(b.cx - rampX));
      
      const bestComp = candidates[0];
      // Score = Area * AspectRatio
      const score = bestComp.area * (bestComp.h / Math.max(1, bestComp.w));
      
      if (score > bestGlobalScore) {
        bestGlobalScore = score;
        bestLandingX = rampIsRight ? bestComp.maxX : bestComp.minX;
        bestLandingFrame = f;
        bestBlobBox = { x: bestComp.minX / width, y: bestComp.minY / height, w: bestComp.w / width, h: bestComp.h / height };
      }
    }
  }

  // --- Determine final landing position ---
  if (bestLandingFrame !== null) {
    landingFrame = bestLandingFrame;
    landingX = bestLandingX;
    blobBox = bestBlobBox;
    allDetections[landingFrame] = { score: bestGlobalScore, box: blobBox };
    console.log(`[AI] Best skier silhouette found at frame ${landingFrame}: score=${Math.round(bestGlobalScore)}, x=${Math.round(landingX)}`);
  } else {
    // Pure fallback if nothing was found
    landingX = (searchStartX + searchEndX) / 2;
    console.log(`[AI] No silhouette found in entire window — fallback x: ${Math.round(landingX)}`);
  }
  
  console.log('[AI] Final: frame', landingFrame, 'x:', Math.round(landingX));
  
  // Normalize landing X to 0..1
  const landingXNorm = landingX !== null ? landingX / width : null;

  // Calculate confidence based on multiple signals
  let confidence = 0;
  if (yRange > 20) confidence += 0.3;
  if (detectedFrames.length > 10) confidence += 0.2;
  if (initialContactFrame !== null) confidence += 0.2;
  if (bestGlobalScore > 0) confidence += 0.3;
  else if (landingFrame !== null) confidence += 0.1;

  confidence = Math.min(1, confidence);

  onProgress(0.9, 'Building result...');

  // Build phase timeline
  const phases = [];
  if (significantFrames.length > 0) {
    const firstMotion = significantFrames[0].frame;
    if (firstMotion < peakFrame.frame) {
      phases.push({ start: firstMotion, end: peakFrame.frame, type: 'APPROACH_AND_FLIGHT' });
    }
    phases.push({ start: peakFrame.frame, end: initialContactFrame, type: 'DESCENT' });
    if (landingFrame !== null) {
      phases.push({ start: initialContactFrame, end: Math.min(landingFrame + 15, totalFrames - 1), type: 'LANDING' });
    }
  }

  const trajectory = trackingData
    .filter(t => t.detected)
    .map(t => ({ frame: t.frame, x: t.cx / width, y: t.cy / height }));

  onProgress(1, 'Analysis complete!');


  return {
    landingFrameIndex: landingFrame,
    landingX: landingXNorm,
    confidence,
    phases,
    trajectory,
    peakFrame: peakFrame.frame,
    initialContact: initialContactFrame,
    rampMarker: rampIsRight !== null ? { x: rampMarkerX, y: rampMarkerY } : null,
    blobBox,
    predBlobBox,
    frameWidth: width,
    allDetections,
    waterlineY: estimatedWaterlineY / height,  // normalized 0..1
  };
}


// --- Helper functions ---

/**
 * Compute pixel variance in a rectangular region of a grayscale image.
 * Higher variance = more visual structure (ramp, objects vs plain water/sky).
 */
function computeEdgeVariance(gray, imgWidth, imgHeight, x1, x2, y1, y2) {
  let sum = 0, sumSq = 0, count = 0;
  for (let y = y1; y < y2; y++) {
    for (let x = x1; x < x2; x++) {
      const val = gray[y * imgWidth + x];
      sum += val;
      sumSq += val * val;
      count++;
    }
  }
  if (count === 0) return 0;
  const mean = sum / count;
  return (sumSq / count) - (mean * mean);
}

/**
 * Convert ImageData to grayscale Float32Array.
 */
function toGrayscale(imageData) {
  const { data, width, height } = imageData;
  const gray = new Float32Array(width * height);
  for (let i = 0; i < width * height; i++) {
    const r = data[i * 4];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];
    gray[i] = 0.299 * r + 0.587 * g + 0.114 * b;
  }
  return gray;
}

/**
 * Find the largest motion blob in a region of the difference image.
 * Uses a simple connected-component approach via flood fill.
 */
function findLargestBlob(diff, width, height, roiTop, roiBottom, threshold, roiLeft = 0, roiRight = null) {
  if (roiRight === null) roiRight = width;
  // Threshold the diff image in the ROI
  const visited = new Uint8Array(width * height);
  
  let bestBlob = { detected: false, cx: 0, cy: 0, area: 0, top: height, bbox: null };
  
  for (let y = roiTop; y < roiBottom; y++) {
    for (let x = roiLeft; x < roiRight; x++) {
      const idx = y * width + x;
      if (visited[idx] || diff[idx] < threshold) continue;
      
      // Flood fill to find connected component (constrained to ROI)
      const component = floodFill(diff, visited, width, height, x, y, threshold, roiTop, roiBottom, roiLeft, roiRight);
      
      if (component.area > bestBlob.area) {
        // Filter out very flat/wide shapes (waves, wake)
        const aspect = component.h / Math.max(component.w, 1);
        if (aspect < 0.08 && component.area < 2000) continue;  // Skip flat noise
        
        bestBlob = {
          detected: true,
          cx: component.cx,
          cy: component.cy,
          area: component.area,
          top: component.minY,
          w: component.w,
          h: component.h,
          bbox: { x: component.minX, y: component.minY, w: component.w, h: component.h }
        };
      }
    }
  }
  
  return bestBlob;
}

/**
 * Simple flood fill to find connected components.
 */
function floodFill(diff, visited, width, height, startX, startY, threshold, roiTop, roiBottom, roiLeft = 0, roiRight = null) {
  if (roiRight === null) roiRight = width;
  const stack = [[startX, startY]];
  let sumX = 0, sumY = 0, count = 0;
  let minX = startX, maxX = startX, minY = startY, maxY = startY;
  
  while (stack.length > 0) {
    const [x, y] = stack.pop();
    const idx = y * width + x;
    
    if (x < roiLeft || x >= roiRight || y < roiTop || y >= roiBottom) continue;
    if (visited[idx]) continue;
    if (diff[idx] < threshold) continue;
    
    visited[idx] = 1;
    sumX += x;
    sumY += y;
    count++;
    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x);
    minY = Math.min(minY, y);
    maxY = Math.max(maxY, y);
    
    // 4-connectivity (skip pixels for speed on large images)
    stack.push([x + 2, y], [x - 2, y], [x, y + 2], [x, y - 2]);
    // Also check immediate neighbors for better accuracy
    stack.push([x + 1, y], [x - 1, y], [x, y + 1], [x, y - 1]);
  }
  
  return {
    cx: count > 0 ? Math.round(sumX / count) : 0,
    cy: count > 0 ? Math.round(sumY / count) : 0,
    area: count,
    w: maxX - minX + 1,
    h: maxY - minY + 1,
    minX, maxX, minY, maxY
  };
}

/**
 * Find the waterline Y coordinate — the horizontal boundary between
 * the shoreline/trees (background) and open water.
 * Works by finding the row with the strongest horizontal edge across
 * the full image width. Scans the middle 60% of the frame vertically
 * to avoid sky at the top and foreground water at the bottom.
 * Robust to both dark-shore/light-water and light-shore/dark-water.
 */
function findWaterline(bgGray, width, height) {
  const scanTop    = Math.floor(height * 0.05);
  const scanBottom = Math.floor(height * 0.70);
  const checkRows  = 25; // rows to sample above and below
  const sampleStep = 4;  // sample every 4th pixel for speed
  
  // For each candidate row: score = variance_above - variance_below
  // Shoreline: trees above (high var) + water below (low var) → high score
  // Wake:      water above (low var) + water below (low var) → near zero
  // Sky/tree:  sky above (low var)  + trees below (high var) → negative
  
  let bestY = Math.floor(height * 0.22);
  let bestScore = -Infinity;
  
  for (let y = scanTop + checkRows; y < scanBottom - checkRows; y++) {
    let sumA = 0, sumSqA = 0, cntA = 0;
    let sumB = 0, sumSqB = 0, cntB = 0;
    
    // Sample rows ABOVE
    for (let dy = 1; dy <= checkRows; dy++) {
      const yy = y - dy;
      for (let x = 0; x < width; x += sampleStep) {
        const v = bgGray[yy * width + x];
        sumA += v; sumSqA += v * v; cntA++;
      }
    }
    // Sample rows BELOW
    for (let dy = 1; dy <= checkRows; dy++) {
      const yy = y + dy;
      for (let x = 0; x < width; x += sampleStep) {
        const v = bgGray[yy * width + x];
        sumB += v; sumSqB += v * v; cntB++;
      }
    }
    
    const varA = cntA > 0 ? (sumSqA / cntA) - (sumA / cntA) ** 2 : 0;
    const varB = cntB > 0 ? (sumSqB / cntB) - (sumB / cntB) ** 2 : 0;
    const score = varA - varB; // high = trees above, water below = shoreline
    
    if (score > bestScore) {
      bestScore = score;
      bestY = y;
    }
  }
  
  console.log('[AI] Waterline (var_above - var_below): Y=', bestY, 'score=', bestScore.toFixed(1));
  return bestY;
}

/**
 * Simple 1D linear regression: fit y = slope*x + intercept.
 */
function fitLinear1D(xs, ys) {
  const n = xs.length;
  if (n < 2) return { slope: 0, intercept: ys[0] || 0 };
  let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
  for (let i = 0; i < n; i++) {
    sumX += xs[i]; sumY += ys[i];
    sumXY += xs[i] * ys[i]; sumX2 += xs[i] * xs[i];
  }
  const denom = n * sumX2 - sumX * sumX;
  if (Math.abs(denom) < 1e-9) return { slope: 0, intercept: sumY / n };
  const slope = (n * sumXY - sumX * sumY) / denom;
  const intercept = (sumY - slope * sumX) / n;
  return { slope, intercept };
}

/**
 * Yield control to the UI thread to prevent freezing.
 */
function yieldToUI() {
  return new Promise(resolve => setTimeout(resolve, 0));
}
