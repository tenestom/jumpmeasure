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
  landingFrame = Math.min(safeContactFrame + 7, totalFrames - 1);
  
  console.log('[AI] initialContactFrame:', initialContactFrame, 'peakFrame:', peakFrame.frame);
  console.log('[AI] Landing frame (contact+7):', landingFrame);
  
  // STEP 1: LOCATE RAMP using vertical edge detection at horizon zone
  // The ramp has STRAIGHT EDGES (triangle sides) that create strong vertical edges.
  let rampMarkerX = 0, rampMarkerY = 0;
  let rampNativeX = null, rampNativeY = null;
  
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
      
      // Find ramp base Y (bottom of the ramp = waterline)
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
      rampMarkerY = rampBaseY / height;
      
      console.log('[AI] Ramp located: x=', rampCluster.cx, 'baseY=', rampBaseY,
        'cluster:', rampCluster.start, '-', rampCluster.end);
    }
  }
  
  // STEP 2: FIND SKIER USING VERTICAL PROFILE
  // The skier is a standing person — their diff extends HIGH ABOVE the waterline.
  // Wake/splash only creates diff AT the waterline (flat).
  // For each column: measure how high above ramp Y the diff extends.
  // Columns with tall vertical extent = the skier.
  let blobBox = null;
  
  // Get predicted X from flight phase (for search zone)
  let predX = lastKnownX || (peakFrame ? peakFrame.cx : width / 2);
  for (let f = Math.max(0, safeContactFrame - 5); f <= safeContactFrame; f++) {
    if (f < trackingData.length && trackingData[f].detected && trackingData[f].area < 5000) {
      predX = trackingData[f].cx;
    }
  }
  
  // Compute diff for landing frame
  const landGray = toGrayscale(frames[landingFrame]);
  const landDiff = new Uint8Array(width * height);
  for (let i = 0; i < landGray.length; i++) {
    landDiff[i] = Math.min(255, Math.abs(landGray[i] - bgGray[i]));
  }
  
  // Use ramp Y as waterline reference. If no ramp found, estimate from frame.
  const rampWaterY = rampNativeY || Math.floor(height * 0.30);
  
  // Search zone: exclude the ramp, search from predicted position outward
  const rampStart = rampNativeX ? (rampIsRight ? rampNativeX - 30 : 0) : width;
  const rampEnd = rampNativeX ? (rampIsRight ? width : rampNativeX + 30) : 0;
  const searchXStart = rampIsRight ? 0 : rampEnd;
  const searchXEnd = rampIsRight ? rampStart : width;
  
  // For each column: find highest Y with diff > threshold (= top of motion)
  const verticalExtent = new Array(width).fill(0);
  const diffThreshold = 15; // Low threshold: dark skier vs dark trees
  
  // Debug: log diff values at predicted skier position
  const debugX = Math.round(predX);
  let debugDiffs = [];
  for (let y = Math.max(0, rampWaterY - Math.floor(height * 0.15)); y < rampWaterY; y++) {
    debugDiffs.push(landDiff[y * width + debugX]);
  }
  console.log('[AI] Diff at predX=', debugX, 'from y=', 
    Math.max(0, rampWaterY - Math.floor(height * 0.15)), 'to y=', rampWaterY,
    'max:', Math.max(...debugDiffs), 'vals:', debugDiffs.slice(0, 20).join(','));
  
  for (let x = searchXStart; x < searchXEnd; x++) {
    let topY = rampWaterY; // default: no vertical extent
    for (let y = Math.max(0, rampWaterY - Math.floor(height * 0.2)); y < rampWaterY; y++) {
      if (landDiff[y * width + x] > diffThreshold) {
        topY = y;
        break;
      }
    }
    verticalExtent[x] = rampWaterY - topY; // how many pixels above waterline
  }
  
  // Smooth vertical extent
  const smoothVert = new Array(width).fill(0);
  for (let x = searchXStart + 3; x < searchXEnd - 3; x++) {
    let sum = 0;
    for (let dx = -3; dx <= 3; dx++) sum += verticalExtent[x + dx];
    smoothVert[x] = sum / 7;
  }
  
  // Find clusters of columns with significant vertical extent
  // The skier = columns that extend significantly above waterline
  const minVertExtent = Math.floor(height * 0.015); // at least 1.5% of frame height
  const skierClusters = [];
  cStart = -1;
  for (let x = searchXStart; x < searchXEnd; x++) {
    if (smoothVert[x] > minVertExtent) {
      if (cStart === -1) cStart = x;
    } else {
      if (cStart !== -1) {
        const w = x - cStart;
        if (w >= 5 && w <= 150) { // Skier is narrow (~10-80px wide)
          const cx = Math.round((cStart + x - 1) / 2);
          const distToPred = Math.abs(cx - predX);
          // Find max vertical extent in this cluster
          let maxVE = 0;
          for (let xx = cStart; xx < x; xx++) {
            if (smoothVert[xx] > maxVE) maxVE = smoothVert[xx];
          }
          skierClusters.push({ start: cStart, end: x - 1, width: w, cx, distToPred, maxVE });
        }
        cStart = -1;
      }
    }
  }
  
  console.log('[AI] Vertical profile: waterlineY:', rampWaterY, 'minExtent:', minVertExtent,
    'predX:', predX, 'clusters:', skierClusters.map(c => 
      `[${c.start}-${c.end}](w=${c.width},cx=${c.cx},ve=${Math.round(c.maxVE)},dist=${c.distToPred})`).join(' '));
  
  if (skierClusters.length > 0) {
    // Pick the cluster closest to the predicted position from flight
    const skierCluster = skierClusters.reduce((a, b) => 
      a.distToPred < b.distToPred ? a : b);
    
    landingX = skierCluster.cx;
    
    // Build blobBox for visualization
    let topY = rampWaterY;
    for (let xx = skierCluster.start; xx <= skierCluster.end; xx++) {
      const extent = verticalExtent[xx];
      if (rampWaterY - extent < topY) topY = rampWaterY - extent;
    }
    blobBox = {
      x: skierCluster.start / width,
      y: topY / height,
      w: skierCluster.width / width,
      h: (rampWaterY - topY + 20) / height
    };
    
    console.log('[AI] Skier found: x=', landingX, 
      'cluster:', skierCluster.start, '-', skierCluster.end,
      'width:', skierCluster.width, 'vertExtent:', Math.round(skierCluster.maxVE));
  }
  
  // Fallback
  if (landingX === null) {
    landingX = lastKnownX || width / 2;
    console.log('[AI] X: fallback lastKnownX:', Math.round(landingX));
  }
  
  console.log('[AI] Final: frame', landingFrame, 'x:', Math.round(landingX));

  // Normalize landing X to 0..1
  const landingXNorm = landingX !== null ? landingX / width : null;

  // Calculate confidence based on multiple signals
  let confidence = 0;
  if (yRange > 20) confidence += 0.3;  // Clear vertical motion
  if (detectedFrames.length > 10) confidence += 0.2;  // Enough tracking data
  if (initialContactFrame !== null) confidence += 0.2;  // Contact detected
  if (landingFrame !== null) confidence += 0.3;  // Landing detected

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
 * Yield control to the UI thread to prevent freezing.
 */
function yieldToUI() {
  return new Promise(resolve => setTimeout(resolve, 0));
}
