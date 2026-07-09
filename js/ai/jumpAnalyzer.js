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
  
  // Use FULL frame width for search (blob tracker lastKnownX is unreliable)
  const rx1 = 0;
  const rx2 = width;
  const ry1 = Math.max(0, Math.floor(height * 0.35));
  const ry2 = Math.min(height, Math.floor(height * 0.48));

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

  // Two-phase landing frame detection:
  // Phase 1: Find the elevated brightness region (where mound exists)
  // Phase 2: Within that region, find the frame with peak combined score
  
  const baselineFrames = frameScores.slice(0, Math.min(8, frameScores.length));
  const baselineBright = baselineFrames.reduce((s, f) => s + f.bright, 0) / baselineFrames.length;
  const brightnessThreshold = baselineBright * 1.03; // 3% to find the elevated region
  
  console.log('[AI] baselineBright:', Math.round(baselineBright), 'threshold:', Math.round(brightnessThreshold));

  // Phase 1: Find where the elevated region starts (5-frame moving avg)
  let elevatedStart = null;
  for (let i = 4; i < frameScores.length; i++) {
    const window5 = frameScores.slice(i - 4, i + 1);
    const avgBright = window5.reduce((s, f) => s + f.bright, 0) / 5;
    
    if (avgBright > brightnessThreshold) {
      elevatedStart = window5[0].frame; // Start of the elevated region
      break;
    }
  }

  // Phase 2: The mound is most distinct when it FIRST appears
  // Use the start of the elevated region directly
  if (elevatedStart !== null) {
    landingFrame = elevatedStart;
    console.log('[AI] Landing frame (elevated region start):', landingFrame);
  }

  // Fallback: frame with highest bright score in late window
  if (landingFrame === null) {
    const lateScores = frameScores.filter(s => s.frame >= initialContactFrame + 15);
    if (lateScores.length > 0) {
      const best = lateScores.reduce((a, b) => a.bright > b.bright ? a : b);
      landingFrame = best.frame;
      console.log('[AI] Landing frame (fallback late peak):', landingFrame);
    } else if (frameScores.length > 0) {
      const best = frameScores.reduce((a, b) => a.bright > b.bright ? a : b);
      landingFrame = best.frame;
      console.log('[AI] Landing frame (fallback overall peak):', landingFrame);
    }
  }

  // Step 7: Find the precise X position of the mound
  // The mound is a small ISOLATED bright feature. The wake trail is a broad band.
  // Strategy: 
  // 1. Triple filter: pixel must be brighter vs BOTH pre-mound AND background (strict)
  // 2. Build column profile of passing pixels
  // 3. Find the most ISOLATED bright cluster (farthest from the main wake mass)
  if (landingFrame !== null) {
    const preMoundIdx = Math.max(0, landingFrame - 4);
    const landingGray = toGrayscale(frames[landingFrame]);
    const preMoundGray = toGrayscale(frames[preMoundIdx]);
    
    // Build per-column count of truly NEW bright pixels (triple filter)
    const colProfile = new Array(width).fill(0);
    let totalNewPixels = 0;
    
    for (let y = ry1; y < ry2; y++) {
      for (let x = rx1; x < rx2; x++) {
        const idx = y * width + x;
        const vsPreMound = landingGray[idx] - preMoundGray[idx];
        const vsBg = landingGray[idx] - bgGray[idx];
        // Must be brighter than BOTH pre-mound AND background, and bright overall
        if (vsPreMound > 30 && vsBg > 30 && landingGray[idx] > 180) {
          colProfile[x]++;
          totalNewPixels++;
        }
      }
    }
    
    console.log('[AI] X: totalNewPixels (triple filter):', totalNewPixels);
    
    if (totalNewPixels > 5) {
      // Find all clusters of bright columns (groups of consecutive cols with count > 0)
      const clusters = [];
      let cStart = -1;
      for (let x = 0; x < width; x++) {
        if (colProfile[x] > 0) {
          if (cStart === -1) cStart = x;
        } else {
          if (cStart !== -1) {
            const cEnd = x - 1;
            let mass = 0;
            let sumCx = 0;
            for (let cx = cStart; cx <= cEnd; cx++) {
              mass += colProfile[cx];
              sumCx += cx * colProfile[cx];
            }
            clusters.push({ start: cStart, end: cEnd, width: cEnd - cStart + 1, mass, centroid: sumCx / mass });
            cStart = -1;
          }
        }
      }
      // Handle cluster at edge
      if (cStart !== -1) {
        const cEnd = width - 1;
        let mass = 0, sumCx = 0;
        for (let cx = cStart; cx <= cEnd; cx++) { mass += colProfile[cx]; sumCx += cx * colProfile[cx]; }
        clusters.push({ start: cStart, end: cEnd, width: cEnd - cStart + 1, mass, centroid: sumCx / mass });
      }
      
      console.log('[AI] X: found', clusters.length, 'clusters:', 
        clusters.map(c => `[${c.start}-${c.end}](w=${c.width},m=${c.mass})`).join(' '));
      
      if (clusters.length === 1) {
        landingX = clusters[0].centroid;
      } else if (clusters.length > 1) {
        // Find the LARGEST cluster = main wake trail
        const mainWake = clusters.reduce((a, b) => a.mass > b.mass ? a : b);
        
        // Find clusters ADJACENT to the main wake (within 20px gap)
        const adjacent = clusters.filter(c => 
          c !== mainWake && 
          c.mass >= 10 && // Significant (not edge noise)
          (Math.abs(c.end - mainWake.start) < 20 || Math.abs(c.start - mainWake.end) < 20)
        );
        
        console.log('[AI] X: mainWake:', mainWake.start, '-', mainWake.end, 
          'adjacent:', adjacent.map(c => `[${c.start}-${c.end}]`).join(' '));
        
        let moundCluster = null;
        if (adjacent.length === 1) {
          moundCluster = adjacent[0];
        } else if (adjacent.length > 1) {
          // Two adjacent clusters = one on each side of the wake.
          // The mound is on the RAMP SIDE. The ramp side has more frame space
          // (camera shows the approach/ramp with more room).
          const leftSpace = mainWake.start;          // pixels from left edge to wake
          const rightSpace = width - mainWake.end;   // pixels from wake to right edge
          
          // Pick the adjacent cluster on the side with MORE space = ramp side
          const rampSide = leftSpace >= rightSpace ? 'left' : 'right';
          const rampCluster = adjacent.find(c => 
            rampSide === 'left' ? c.end < mainWake.start : c.start > mainWake.end
          );
          moundCluster = rampCluster || adjacent[0];
          console.log('[AI] X: leftSpace:', leftSpace, 'rightSpace:', rightSpace, 
            'rampSide:', rampSide, 'picked:', moundCluster.start, '-', moundCluster.end);
        }
        
        if (moundCluster) {
          // The mound is at the edge of this cluster CLOSEST to the main wake
          // (not the peak density, which might be a boat or other object)
          // Use centroid of the 30% of the cluster nearest the main wake
          const isLeftOfWake = moundCluster.end < mainWake.start;
          let nearStart, nearEnd;
          if (isLeftOfWake) {
            // Cluster is LEFT of wake — mound is at the RIGHT end
            nearStart = Math.round(moundCluster.end - moundCluster.width * 0.3);
            nearEnd = moundCluster.end;
          } else {
            // Cluster is RIGHT of wake — mound is at the LEFT end
            nearStart = moundCluster.start;
            nearEnd = Math.round(moundCluster.start + moundCluster.width * 0.3);
          }
          nearStart = Math.max(nearStart, moundCluster.start);
          nearEnd = Math.min(nearEnd, moundCluster.end);
          
          let sumX = 0, sumW = 0;
          for (let x = nearStart; x <= nearEnd; x++) {
            sumX += x * colProfile[x];
            sumW += colProfile[x];
          }
          landingX = sumW > 0 ? sumX / sumW : (nearStart + nearEnd) / 2;
          console.log('[AI] X: mound cluster [', moundCluster.start, '-', moundCluster.end, 
            '] wake-facing range:', nearStart, '-', nearEnd, '→ landingX:', Math.round(landingX));
        } else {
          // No adjacent cluster found — use edge of main wake closest to frame edge
          // (The mound is at the beginning of the trail)
          const distToLeft = mainWake.start;
          const distToRight = width - mainWake.end;
          if (distToLeft < distToRight) {
            // Wake starts from left — mound at left edge
            landingX = mainWake.start;
          } else {
            // Wake starts from right — mound at right edge
            landingX = mainWake.end;
          }
          console.log('[AI] X: no adjacent cluster, using wake edge:', Math.round(landingX));
        }
      }
    }
    
    if (landingX === null) {
      // Fallback: weighted centroid of bg-diff
      let sx = 0, sw = 0;
      const landingGrayFb = toGrayscale(frames[landingFrame]);
      for (let y = ry1; y < ry2; y++) {
        for (let x = rx1; x < rx2; x++) {
          const idx = y * width + x;
          const bgDiff = Math.abs(landingGrayFb[idx] - bgGray[idx]);
          if (bgDiff > 40 && landingGrayFb[idx] > 170) { sx += x * bgDiff; sw += bgDiff; }
        }
      }
      landingX = sw > 0 ? sx / sw : width / 2;
      console.log('[AI] X: fallback bg-diff centroid:', Math.round(landingX));
    }
    
    console.log('[AI] X: final landingX:', Math.round(landingX));
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
