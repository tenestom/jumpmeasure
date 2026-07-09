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
  
  // Ramp side will be determined from splash data (Step 2)
  // STEP 2: SPLASH-BASED X DETECTION
  // Analyze a later frame where the splash/wake is fully visible
  const splashFrame = Math.min(safeContactFrame + 23, totalFrames - 1);
  const splashGray = toGrayscale(frames[splashFrame]);
  const preSplashGray = toGrayscale(frames[Math.max(0, splashFrame - 4)]);
  
  // Waterline zone
  const splashRy1 = Math.floor(height * 0.35);
  const splashRy2 = Math.floor(height * 0.48);
  
  // Column profile: bright new pixels at waterline
  const colProfile = new Array(width).fill(0);
  let totalNewPixels = 0;
  
  for (let y = splashRy1; y < splashRy2; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      const vsPreSplash = splashGray[idx] - preSplashGray[idx];
      const vsBg = splashGray[idx] - bgGray[idx];
      if (vsPreSplash > 30 && vsBg > 30 && splashGray[idx] > 180) {
        colProfile[x]++;
        totalNewPixels++;
      }
    }
  }
  
  console.log('[AI] X: splashFrame:', splashFrame, 'totalNewPixels:', totalNewPixels);
  
  if (totalNewPixels > 5) {
    // Find clusters
    const clusters = [];
    let cStart = -1;
    for (let x = 0; x < width; x++) {
      if (colProfile[x] > 0) {
        if (cStart === -1) cStart = x;
      } else {
        if (cStart !== -1) {
          let mass = 0;
          for (let cx = cStart; cx <= x - 1; cx++) mass += colProfile[cx];
          clusters.push({ start: cStart, end: x - 1, width: x - cStart, mass });
          cStart = -1;
        }
      }
    }
    if (cStart !== -1) {
      let mass = 0;
      for (let cx = cStart; cx < width; cx++) mass += colProfile[cx];
      clusters.push({ start: cStart, end: width - 1, width: width - cStart, mass });
    }
    
    console.log('[AI] X: clusters:', clusters.map(c => 
      `[${c.start}-${c.end}](w=${c.width},m=${c.mass})`).join(' '));
    
    // Main wake = largest cluster by mass
    const mainWake = clusters.reduce((a, b) => a.mass > b.mass ? a : b);
    
    console.log('[AI] X: mainWake [', mainWake.start, '-', mainWake.end, '] mass:', mainWake.mass);
    
    // Smooth the profile within the main wake
    const smoothed = new Array(width).fill(0);
    for (let x = mainWake.start + 5; x <= mainWake.end - 5; x++) {
      let sum = 0;
      for (let dx = -5; dx <= 5; dx++) sum += colProfile[x + dx];
      smoothed[x] = sum;
    }
    
    // Find peak value for threshold
    let peakVal = 0;
    for (let x = mainWake.start; x <= mainWake.end; x++) {
      if (smoothed[x] > peakVal) peakVal = smoothed[x];
    }
    
    // RAMP SIDE DETECTION from splash data:
    // The skier lands (small splash) on the ramp side and skis away (bigger splash).
    // So the side of mainWake with LESS mass = ramp side.
    const wakeMid = Math.floor((mainWake.start + mainWake.end) / 2);
    let leftMass = 0, rightMass = 0;
    for (let x = mainWake.start; x <= mainWake.end; x++) {
      if (x < wakeMid) leftMass += colProfile[x];
      else rightMass += colProfile[x];
    }
    
    rampIsRight = rightMass < leftMass; // Less splash = ramp side
    const threshold = peakVal * 0.3;
    
    console.log('[AI] X: wakeMass left:', leftMass, 'right:', rightMass,
      '→ rampSide:', rampIsRight ? 'right' : 'left',
      'peak:', peakVal, 'threshold:', Math.round(threshold));
    
    // Scan from ramp side inward — first column above threshold = landing contact
    if (rampIsRight) {
      for (let x = mainWake.end; x >= mainWake.start; x--) {
        if (smoothed[x] > threshold) {
          landingX = x;
          break;
        }
      }
    } else {
      for (let x = mainWake.start; x <= mainWake.end; x++) {
        if (smoothed[x] > threshold) {
          landingX = x;
          break;
        }
      }
    }
    
    if (landingX !== null) {
      console.log('[AI] X: landing at native', landingX, 
        'normalized:', (landingX / width).toFixed(4));
    }
  }
  
  // LOCATE RAMP precisely for debug marker
  // Scan from BOTTOM up on the ramp side. Water is uniform (low gradient).
  // The ramp is the first STRUCTURE (high gradient) above the water.
  // It extends ABOVE the normal waterline as a triangle.
  let rampMarkerX = 0, rampMarkerY = 0;
  if (rampIsRight !== null) {
    const searchX1 = rampIsRight ? Math.floor(width * 0.7) : 0;
    const searchX2 = rampIsRight ? width : Math.floor(width * 0.3);
    
    // For each column, scan from bottom up and find first Y with strong local gradient
    const firstStructY = new Array(width).fill(0);
    for (let x = searchX1; x < searchX2; x++) {
      let foundY = 0; // default: top of frame
      for (let y = height - 10; y > 10; y--) {
        // Wider gradient window (±8px) to ignore water ripples
        const above = bgGray[(y - 8) * width + x];
        const below = bgGray[(y + 8) * width + x];
        const gradient = Math.abs(above - below);
        if (gradient > 100) { // Very high threshold: solid structures only
          foundY = y;
          break;
        }
      }
      firstStructY[x] = foundY;
    }
    
    // Find the median "first structure Y" — this is the normal waterline/shore
    const sortedY = firstStructY.slice(searchX1, searchX2).filter(y => y > 0).sort((a, b) => a - b);
    const medianY = sortedY[Math.floor(sortedY.length / 2)] || Math.floor(height * 0.5);
    
    // Ramp columns: first structure Y is significantly ABOVE the median (lower Y value)
    // The ramp sticks up above the normal waterline
    const rampThreshold = medianY - Math.floor(height * 0.03); // at least 3% above waterline
    
    let rampSumX = 0, rampSumY = 0, rampCount = 0;
    let rampMinY = height;
    for (let x = searchX1; x < searchX2; x++) {
      if (firstStructY[x] > 0 && firstStructY[x] < rampThreshold) {
        rampSumX += x;
        rampSumY += firstStructY[x];
        rampCount++;
        if (firstStructY[x] < rampMinY) rampMinY = firstStructY[x];
      }
    }
    
    if (rampCount > 3) {
      rampMarkerX = (rampSumX / rampCount) / width;
      rampMarkerY = rampMinY / height; // top of the ramp
      console.log('[AI] Ramp located: x=', Math.round(rampSumX / rampCount), 
        'topY=', rampMinY, 'medianWaterline:', medianY,
        'rampColumns:', rampCount);
    } else {
      // Fallback: center of ramp side
      rampMarkerX = rampIsRight ? 0.85 : 0.15;
      rampMarkerY = 0.4;
      console.log('[AI] Ramp: not enough columns found, using fallback');
    }
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
function findLargestBlob(diff, width, height, roiTop, roiBottom, threshold) {
  // Threshold the diff image in the ROI
  const visited = new Uint8Array(width * height);
  
  let bestBlob = { detected: false, cx: 0, cy: 0, area: 0, top: height, bbox: null };
  
  for (let y = roiTop; y < roiBottom; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      if (visited[idx] || diff[idx] < threshold) continue;
      
      // Flood fill to find connected component
      const component = floodFill(diff, visited, width, height, x, y, threshold, roiTop, roiBottom);
      
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
function floodFill(diff, visited, width, height, startX, startY, threshold, roiTop, roiBottom) {
  const stack = [[startX, startY]];
  let sumX = 0, sumY = 0, count = 0;
  let minX = startX, maxX = startX, minY = startY, maxY = startY;
  
  while (stack.length > 0) {
    const [x, y] = stack.pop();
    const idx = y * width + x;
    
    if (x < 0 || x >= width || y < roiTop || y >= roiBottom) continue;
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
