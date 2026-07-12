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
export async function analyzeJump(frames, onProgress = () => {}) {
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
  
  // STEP 2: FIND SKIER USING SHAPE-SCORED MULTI-FRAME SCAN
  // Scan frames in a window around the estimated landing.
  // For each frame: background-subtract → find blobs → score by silhouette shape.
  // The skier has: correct size, tall aspect ratio, ski-signature (wider at bottom).
  // Store all detections; measurement frame = frame with highest score.
  
  let blobBox = null;
  const allDetections = new Array(totalFrames).fill(null);
  
  // Get last known flight blob for visualization
  let predX = lastKnownX || (peakFrame ? peakFrame.cx : width / 2);
  let predBlobFrame = -1;
  for (let f = Math.max(0, safeContactFrame - 15); f <= safeContactFrame; f++) {
    if (f < trackingData.length && trackingData[f].detected && trackingData[f].area < 8000) {
      predX = trackingData[f].cx;
      predBlobFrame = f;
    }
  }
  const lastFlightBlob = predBlobFrame >= 0 ? trackingData[predBlobFrame] : null;
  const predBlobBox = lastFlightBlob ? {
    x: lastFlightBlob.bbox.x / width,
    y: lastFlightBlob.bbox.y / height,
    w: lastFlightBlob.bbox.w / width,
    h: lastFlightBlob.bbox.h / height,
  } : null;
  console.log('[AI] Last flight blob: frame', predBlobFrame, 'cx=', predX);
  
  // Search zone X: from ramp exclusion to opposite side
  const rampEdge = rampIsRight
    ? (rampNativeStartX || Math.floor(width * 0.7))
    : (rampNativeEndX || Math.floor(width * 0.3));
  const searchXStart = rampIsRight ? 0 : rampEdge + rampFullWidth;
  const searchXEnd   = rampIsRight ? rampEdge - rampFullWidth : width;
  
  // Search zone Y: detect waterline from background frame
  // The waterline = strongest horizontal edge across full width (shoreline vs water).
  // Skier body is ABOVE the waterline (head at top, skis touch water at bottom).
  let detectedWaterlineY = findWaterline(bgGray, width, height);
  
  // Sanity check: if ramp was detected its base is AT the waterline.
  // If our detected waterline is far below rampNativeY, prefer rampNativeY.
  if (rampNativeY && Math.abs(detectedWaterlineY - rampNativeY) > height * 0.10) {
    console.log('[AI] Waterline sanity: detected', detectedWaterlineY, 'vs ramp', rampNativeY, '→ using ramp');
    detectedWaterlineY = rampNativeY;
  }
  
  const skierHeight = Math.floor(height * 0.22); // approx max skier height in frame
  const searchYTop = Math.max(0, detectedWaterlineY - skierHeight); // above waterline
  const searchYBot = Math.min(height - 1, detectedWaterlineY + Math.floor(height * 0.04)); // just below
  
  console.log('[AI] Waterline Y:', detectedWaterlineY, '→ search Y:', searchYTop, '-', searchYBot);
  
  // Skier size constraints (native pixels)
  const MIN_W = 15, MAX_W = 120;
  const MIN_H = 50, MAX_H = 250;
  const DIFF_THRESHOLD = 20;
  
  // Scan window: fullLandingFrame - 20 to fullLandingFrame + 5
  const scanFrom = Math.max(0, (fullLandingFrame || safeContactFrame) - 20);
  const scanTo   = Math.min(totalFrames - 1, (fullLandingFrame || safeContactFrame) + 5);
  
  let bestScore = -1;
  let bestFrame = landingFrame;
  let bestLandingX = null;
  
  for (let f = scanFrom; f <= scanTo; f++) {
    const fGray = toGrayscale(frames[f]);
    const fDiff = new Uint8Array(width * height);
    for (let i = 0; i < fGray.length; i++) {
      fDiff[i] = Math.min(255, Math.abs(fGray[i] - bgGray[i]));
    }
    
    // Find all blobs in search zone
    const visited = new Uint8Array(width * height);
    const frameCandidates = [];
    
    for (let y = searchYTop; y < searchYBot; y++) {
      for (let x = searchXStart; x < searchXEnd; x++) {
        const idx = y * width + x;
        if (visited[idx] || fDiff[idx] < DIFF_THRESHOLD) continue;
        const comp = floodFill(fDiff, visited, width, height, x, y, DIFF_THRESHOLD,
          searchYTop, searchYBot, searchXStart, searchXEnd);
        if (comp.area < 50) continue; // skip tiny noise
        frameCandidates.push(comp);
      }
    }
    
    // Score each blob by silhouette criteria
    let frameScore = -1;
    let frameBlob = null;
    
    for (const c of frameCandidates) {
      const bw = c.w, bh = c.h;
      if (bw < MIN_W || bw > MAX_W || bh < MIN_H || bh > MAX_H) continue;
      
      const aspectRatio = bh / bw; // should be ~1.5-5 for a standing person
      if (aspectRatio < 1.0 || aspectRatio > 6.0) continue;
      
      // Ski signature: measure diff-pixel width at bottom 25% vs middle 40-60%
      const midY    = Math.round(c.minY + bh * 0.5);
      const bottomY = Math.round(c.minY + bh * 0.85);
      let midW = 0, botW = 0;
      for (let x = c.minX; x <= c.maxX; x++) {
        if (fDiff[midY    * width + x] >= DIFF_THRESHOLD) midW++;
        if (fDiff[bottomY * width + x] >= DIFF_THRESHOLD) botW++;
      }
      const skiRatio = midW > 0 ? botW / midW : 0;
      // skiRatio > 1 means bottom wider than middle = ski signature
      
      // Roundedness at top: top 10% width should be < body width
      const topY = Math.round(c.minY + bh * 0.1);
      let topW = 0;
      for (let x = c.minX; x <= c.maxX; x++) {
        if (fDiff[topY * width + x] >= DIFF_THRESHOLD) topW++;
      }
      const topRatio = bw > 0 ? topW / bw : 1; // <1 means tapered top = helmet
      
      // Composite score (higher = more skier-like)
      let score = 0;
      score += Math.min(aspectRatio / 2.5, 1.0) * 30;       // aspect ratio (tall)
      score += Math.min(skiRatio, 2.0) / 2.0 * 25;          // ski signature
      score += (1 - Math.min(topRatio, 1)) * 20;            // rounded top
      score += Math.min(c.area / 500, 1.0) * 15;            // sufficient size
      score += (bw >= 25 && bw <= 80) ? 10 : 0;             // right width range
      
      if (score > frameScore) {
        frameScore = score;
        frameBlob = c;
      }
    }
    
    if (frameBlob && frameScore > 30) {
      const detection = {
        score: frameScore,
        box: {
          x: frameBlob.minX / width,
          y: frameBlob.minY / height,
          w: frameBlob.w / width,
          h: frameBlob.h / height,
        },
        nativeX: rampIsRight ? frameBlob.maxX : frameBlob.minX, // ramp-side edge
        nativeBox: frameBlob,
      };
      allDetections[f] = detection;
      
      if (frameScore > bestScore) {
        bestScore = frameScore;
        bestFrame = f;
        bestLandingX = detection.nativeX;
        blobBox = detection.box;
      }
    }
  }
  
  // Update landing frame and X from best detection
  if (bestLandingX !== null) {
    landingFrame = bestFrame;
    landingX = bestLandingX;
    console.log('[AI] Best skier frame:', bestFrame, 'score:', bestScore.toFixed(1), 'x:', bestLandingX);
  } else {
    // Fallback
    landingX = lastKnownX || width / 2;
    console.log('[AI] No skier found by shape — fallback x:', Math.round(landingX));
  }
  
  console.log('[AI] Final: frame', landingFrame, 'x:', Math.round(landingX));

  // Normalize landing X to 0..1
  const landingXNorm = landingX !== null ? landingX / width : null;

  // Calculate confidence based on multiple signals
  let confidence = 0;
  if (yRange > 20) confidence += 0.3;
  if (detectedFrames.length > 10) confidence += 0.2;
  if (initialContactFrame !== null) confidence += 0.2;
  if (bestScore > 50) confidence += 0.3;
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
  const scanBottom = Math.floor(height * 0.60);
  const edgeThreshold = 8;
  
  // Step 1: per-row edge score (how many columns have a significant edge)
  const rowScore = new Float32Array(height).fill(0);
  for (let y = scanTop; y < scanBottom; y++) {
    let count = 0;
    for (let x = 0; x < width; x++) {
      if (Math.abs(bgGray[y * width + x] - bgGray[(y + 1) * width + x]) > edgeThreshold) {
        count++;
      }
    }
    rowScore[y] = count;
  }
  
  // Step 2: smooth over ±6 rows — thick edge zones (shoreline/trees) get high score,
  // thin streaks (wake = 1-2 rows) get diluted.
  const smoothed = new Float32Array(height).fill(0);
  const radius = 6;
  for (let y = scanTop; y < scanBottom; y++) {
    let sum = 0, cnt = 0;
    for (let dy = -radius; dy <= radius; dy++) {
      const yy = y + dy;
      if (yy >= scanTop && yy < scanBottom) { sum += rowScore[yy]; cnt++; }
    }
    smoothed[y] = sum / cnt;
  }
  
  // Step 3: find row with highest smoothed score = thickest edge zone = shoreline
  let maxVal = -1;
  let waterlineY = Math.floor(height * 0.22);
  for (let y = scanTop; y < scanBottom; y++) {
    if (smoothed[y] > maxVal) { maxVal = smoothed[y]; waterlineY = y; }
  }
  
  console.log('[AI] Waterline (thick edge): Y=', waterlineY, 'smoothed score=', maxVal.toFixed(0));
  return waterlineY;
}

/**
 * Yield control to the UI thread to prevent freezing.
 */
function yieldToUI() {
  return new Promise(resolve => setTimeout(resolve, 0));
}
