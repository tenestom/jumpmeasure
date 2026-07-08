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
  for (const t of trackingData.slice(Math.max(0, initialContactFrame - 10), initialContactFrame + 1)) {
    if (t.detected) lastKnownX = t.cx;
  }
  if (lastKnownX === null) lastKnownX = peakFrame.cx;

  // Search window: from initial contact to ~1s after
  const fps = totalFrames / (totalFrames / 30); // Approximate
  const searchStart = Math.max(0, initialContactFrame);
  const searchEnd = Math.min(totalFrames - 1, initialContactFrame + 30);
  
  // Compute frame-to-frame change scores
  const frameScores = [];
  let prevGray = toGrayscale(frames[Math.max(0, searchStart - 1)]);
  
  // Widen search region
  const searchRegionW = Math.floor(width * 0.2);
  const rx1 = Math.max(0, lastKnownX - searchRegionW);
  const rx2 = Math.min(width, lastKnownX + searchRegionW);
  const ry1 = Math.max(0, Math.floor(height * 0.25));
  const ry2 = Math.min(height, Math.floor(height * 0.50));

  for (let f = searchStart; f <= searchEnd; f++) {
    const gray = toGrayscale(frames[f]);
    
    let changeCount = 0;
    let brightNewCount = 0;
    
    for (let y = ry1; y < ry2; y++) {
      for (let x = rx1; x < rx2; x++) {
        const idx = y * width + x;
        const frameDiff = gray[idx] - prevGray[idx];
        if (frameDiff > 20) changeCount++;
        if (gray[idx] > 170 && Math.abs(gray[idx] - bgGray[idx]) > 25) brightNewCount++;
      }
    }
    
    frameScores.push({ frame: f, change: changeCount, bright: brightNewCount });
    prevGray = gray;
  }

  // Log diagnostics for debugging
  console.log('[AI] initialContactFrame:', initialContactFrame, 'peakFrame:', peakFrame.frame);
  console.log('[AI] frameScores:', frameScores.map(s => `f${s.frame}:ch=${s.change},br=${s.bright}`).join(' | '));

  let landingFrame = null;
  let landingX = null;

  // Strategy: find the frame with PEAK brightness change in a LATE window
  // The mound appears well after initial contact — skip the first splash frames
  // and look for the peak in the mound-formation window
  const lateWindowStart = Math.min(initialContactFrame + 8, searchEnd);
  const lateScores = frameScores.filter(s => s.frame >= lateWindowStart);
  
  if (lateScores.length > 0) {
    // Find peak change rate in the late window
    const peakScore = lateScores.reduce((a, b) => 
      (a.change + a.bright) > (b.change + b.bright) ? a : b
    );
    landingFrame = peakScore.frame;
    console.log('[AI] Landing frame (peak in late window):', landingFrame, 'score:', peakScore.change + peakScore.bright);
  } else if (frameScores.length > 0) {
    // Fallback: overall peak
    const best = frameScores.reduce((a, b) => a.change > b.change ? a : b);
    landingFrame = best.frame;
    console.log('[AI] Landing frame (fallback overall peak):', landingFrame);
  }

  // Step 7: Find the precise X position of the mound
  // Use weighted centroid of the brightest NEW pixels (frame vs background)
  // weighted by brightness difference — this finds the mound center,
  // not the edge of spray/wake noise
  if (landingFrame !== null) {
    const landingGray = toGrayscale(frames[landingFrame]);
    // Also get the previous frame for comparison
    const prevIdx = Math.max(0, landingFrame - 1);
    const prevGrayLanding = toGrayscale(frames[prevIdx]);
    
    let sumXWeighted = 0;
    let sumWeight = 0;
    
    for (let y = ry1; y < ry2; y++) {
      for (let x = rx1; x < rx2; x++) {
        const idx = y * width + x;
        const bgDiff = Math.abs(landingGray[idx] - bgGray[idx]);
        const frameDiffVal = landingGray[idx] - prevGrayLanding[idx];
        
        // Pixel must be:
        // 1. Significantly different from background (not static water)
        // 2. Bright (water mound reflects light)
        // 3. Getting brighter vs previous frame (new disturbance)
        if (bgDiff > 30 && landingGray[idx] > 160 && frameDiffVal > 5) {
          const weight = bgDiff * (landingGray[idx] / 255);
          sumXWeighted += x * weight;
          sumWeight += weight;
        }
      }
    }

    if (sumWeight > 0) {
      landingX = sumXWeighted / sumWeight;
    } else {
      // Fallback: centroid of all disturbed pixels
      let sumX = 0, countX = 0;
      for (let y = ry1; y < ry2; y++) {
        for (let x = rx1; x < rx2; x++) {
          const idx = y * width + x;
          if (Math.abs(landingGray[idx] - bgGray[idx]) > 25 && landingGray[idx] > 150) {
            sumX += x; countX++;
          }
        }
      }
      landingX = countX > 0 ? sumX / countX : lastKnownX;
    }
  }

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
  };
}


// --- Helper functions ---

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
