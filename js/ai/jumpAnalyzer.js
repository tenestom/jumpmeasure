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

  onProgress(0, 'Laddar AI-modell...');
  let mlModel = null;
  if (window.cocoSsd) {
    try {
      mlModel = await window.cocoSsd.load();
      console.log('[AI] COCO-SSD Model loaded.');
    } catch (e) {
      console.warn('[AI] Failed to load COCO-SSD', e);
    }
  } else {
    console.warn('[AI] window.cocoSsd is not available.');
  }

  onProgress(0.05, 'Building background model...');

  const bgFrameCount = Math.min(5, Math.floor(totalFrames * 0.1));
  const bgGray = new Float32Array(width * height);
  for (let f = 0; f < bgFrameCount; f++) {
    const gray = toGrayscale(frames[f]);
    for (let i = 0; i < gray.length; i++) {
      bgGray[i] += gray[i] / bgFrameCount;
    }
  }

  onProgress(0.1, 'Detecting motion...');
  const waterlineY = Math.floor(height * 0.55);
  const trackingData = [];
  
  for (let f = 0; f < totalFrames; f++) {
    if (f % 5 === 0) {
      onProgress(0.1 + 0.3 * (f / totalFrames), `Analyzing motion frame ${f + 1}/${totalFrames}...`);
      await yieldToUI();
    }
    const gray = toGrayscale(frames[f]);
    const diff = new Uint8Array(width * height);
    for (let i = 0; i < gray.length; i++) {
      diff[i] = Math.min(255, Math.abs(gray[i] - bgGray[i]));
    }
    const blob = findLargestBlob(diff, width, height, 0, waterlineY, 25);
    trackingData.push({ frame: f, ...blob });
  }

  onProgress(0.4, 'Analyzing jump trajectory...');
  const detectedFrames = trackingData.filter(t => t.detected);
  const significantFrames = detectedFrames.filter(t => t.area > 300);
  
  if (significantFrames.length < 3) {
    return {
      landingFrameIndex: null, landingX: null, confidence: 0, phases: [],
      trajectory: detectedFrames.map(t => ({ frame: t.frame, x: t.cx / width, y: t.cy / height })),
      error: 'Motion detected but no clear jump pattern found.'
    };
  }

  const cropCanvas = new OffscreenCanvas(300, 300);
  const cropCtx = cropCanvas.getContext('2d', { willReadFrequently: true });
  
  let peakFrame = significantFrames[0];
  if (mlModel) {
    onProgress(0.4, 'Verifying peak candidate with ML...');
    const candidatePeaks = [...significantFrames].sort((a, b) => a.cy - b.cy);
    for (const candidate of candidatePeaks) {
      const cropData = cropImageData(frames[candidate.frame], candidate.cx, candidate.cy, 300, 300);
      cropCtx.putImageData(cropData, 0, 0);
      const predictions = await mlModel.detect(cropCanvas, 10, 0.15);
      const person = predictions.find(p => p.class === 'person' || p.class === 'skis' || p.class === 'surfboard');
      
      if (person) {
         peakFrame = candidate;
         console.log(\[AI] ML Confirmed true peak at frame \ (Y=\)\);
         break;
      } else {
         console.log(\[AI] Rejected candidate peak at frame \ (No person found)\);
      }
    }
  } else {
    peakFrame = significantFrames.reduce((best, t) => t.cy < best.cy ? t : best, significantFrames[0]);
  }
  console.log('[AI] peakFrame:', peakFrame.frame);

  let splashY = null;
  let landingX = null;
  let mlTrack = [];
  
  if (mlModel) {
    onProgress(0.5, 'ML Tracking skier...');
    let currX = peakFrame.cx;
    let currY = peakFrame.cy;
    const cropW = 300, cropH = 300;
    
    const cropCanvas = new OffscreenCanvas(cropW, cropH);
    const cropCtx = cropCanvas.getContext('2d', { willReadFrequently: true });
    
    let velX = 0, velY = 5;
    const prePeak = trackingData.filter(t => t.frame <= peakFrame.frame && t.detected);
    if (prePeak.length >= 2) {
      const p1 = prePeak[0];
      const p2 = prePeak[prePeak.length - 1];
      if (p2.frame > p1.frame) {
        velX = (p2.cx - p1.cx) / (p2.frame - p1.frame);
        velY = (p2.cy - p1.cy) / (p2.frame - p1.frame);
      }
    }
    if (velY < 2) velY = 2; // ensure it drops downwards

    for (let f = peakFrame.frame; f < totalFrames; f++) {
      if (f % 2 === 0) {
        onProgress(0.5 + 0.3 * ((f - peakFrame.frame) / (totalFrames - peakFrame.frame)), `ML Tracking frame ${f}...`);
        await yieldToUI();
      }
      
      const cropX = currX - cropW / 2;
      const cropY = currY - cropH / 2;
      const cropData = cropImageData(frames[f], cropX, cropY, cropW, cropH);
      cropCtx.putImageData(cropData, 0, 0);
      
      const predictions = await mlModel.detect(cropCanvas, 10, 0.15);
      const person = predictions.find(p => p.class === 'person' || p.class === 'surfboard' || p.class === 'skis' || p.class === 'boat');
      
      if (person) {
        currX = cropX + person.bbox[0] + person.bbox[2]/2;
        currY = cropY + person.bbox[1] + person.bbox[3]/2;
        
        if (mlTrack.length >= 1) {
            velX = currX - mlTrack[mlTrack.length-1].cx;
            velY = currY - mlTrack[mlTrack.length-1].cy;
        }
        
        mlTrack.push({ frame: f, cx: currX, cy: currY, detected: true });
        splashY = currY; 
        landingX = currX;
        console.log(`[AI] ML Frame ${f}: Detected ${person.class} at ${currX.toFixed(1)}, ${currY.toFixed(1)}`);
      } else {
        currX += velX;
        currY += velY;
        
        mlTrack.push({ frame: f, cx: currX, cy: currY, detected: false });
        splashY = currY;
        landingX = currX;
        console.log(`[AI] ML Frame ${f}: Extrapolated to ${currX.toFixed(1)}, ${currY.toFixed(1)}`);
      }
    }
  } else {
    splashY = Math.floor(height * 0.85);
    landingX = width / 2;
  }

  onProgress(0.8, 'Locating water mound...');

  let landingFrame = null;
  const safeContactFrame = peakFrame.frame + 10;
  
  let rampMarkerX = 0, rampMarkerY = 0;
  let rampNativeX = null, rampNativeY = null, rampNativeTopY = null;
  let rampNativeStartX = null, rampNativeEndX = null;
  let rampFullWidth = 200;
  
  const splashFrame = Math.min(safeContactFrame + 23, totalFrames - 1);
  const splashGray = toGrayscale(frames[splashFrame]);
  const preSplashGray = toGrayscale(frames[Math.max(0, splashFrame - 4)]);
  
  const splashRy1 = Math.floor(height * 0.35);
  const splashRy2 = Math.floor(height * 0.48);
  const colProfile = new Array(width).fill(0);
  for (let y = splashRy1; y < splashRy2; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      if (splashGray[idx] - preSplashGray[idx] > 30 && splashGray[idx] - bgGray[idx] > 30 && splashGray[idx] > 180) {
        colProfile[x]++;
      }
    }
  }
  
  const windowSize = 20;
  let maxClusterVal = 0;
  let maxClusterCol = 0;
  for (let x = windowSize; x < width - windowSize; x++) {
    let sum = 0;
    for (let dx = -windowSize; dx <= windowSize; dx++) sum += colProfile[x+dx];
    if (sum > maxClusterVal) { maxClusterVal = sum; maxClusterCol = x; }
  }
  
  let leftSum = 0, rightSum = 0;
  for (let x = 0; x < maxClusterCol - 50; x++) leftSum += colProfile[x];
  for (let x = maxClusterCol + 50; x < width; x++) rightSum += colProfile[x];
  
  let rampIsRight = rightSum < leftSum;
  console.log('[AI] Ramp side:', rampIsRight ? 'RIGHT' : 'LEFT');

  let rampXStart = rampIsRight ? width - 400 : 0;
  let rampXEnd = rampIsRight ? width : 400;
  let rampZoneY1 = Math.floor(height * 0.2);
  let rampZoneY2 = Math.floor(height * 0.35);

  let clusterStarts = [];
  let currentStart = null;
  for(let x = rampXStart; x < rampXEnd; x++) {
      let isStructure = false;
      let varSum = 0;
      for(let y = rampZoneY1; y < rampZoneY2; y+=4) {
          varSum += Math.abs(bgGray[y*width + x] - bgGray[(y-4)*width + x]);
      }
      if(varSum > 60) isStructure = true;
      if(isStructure && currentStart === null) currentStart = x;
      if(!isStructure && currentStart !== null) {
          clusterStarts.push({start: currentStart, end: x-1, width: x-currentStart});
          currentStart = null;
      }
  }

  const validClusters = clusterStarts.filter(c => c.width >= 10 && c.width <= 150);
  if (validClusters.length > 0) {
      let targetCluster;
      if (rampIsRight) {
          targetCluster = validClusters.reduce((a, b) => b.end > a.end ? b : a);
      } else {
          targetCluster = validClusters.reduce((a, b) => a.start < b.start ? a : b);
      }
      rampNativeX = rampIsRight ? targetCluster.end : targetCluster.start;
      rampNativeStartX = targetCluster.start;
      rampNativeEndX = targetCluster.end;
      rampNativeTopY = rampZoneY1;
      rampNativeY = rampZoneY2;
      rampMarkerX = rampNativeX;
      rampMarkerY = rampNativeY;
      rampFullWidth = targetCluster.width * 2.5;
  }

  if (splashY === null) splashY = Math.floor(height * 0.85);

  let searchStartX = 0;
  let searchEndX = width;
  if (rampNativeStartX !== null && rampNativeEndX !== null) {
      if (rampIsRight) searchEndX = Math.max(0, rampNativeStartX - rampFullWidth);
      else searchStartX = Math.min(width, rampNativeEndX + rampFullWidth);
  }

  let splashScores = new Array(totalFrames).fill(0);
  let splashScanY1 = Math.max(0, Math.floor(splashY) - 20);
  let splashScanY2 = Math.min(height, Math.floor(splashY) + 20);
  
  let splashStartFrame = null;
  let baselineScore = 0;
  let peakScore = 0;
  
  const peakGrayForBaseline = toGrayscale(frames[peakFrame.frame]);
  for(let x=searchStartX; x<searchEndX; x++) {
      let score = 0;
      for(let y=splashScanY1; y<splashScanY2; y++) {
          let val = peakGrayForBaseline[y*width+x];
          if (val > 180) score += (val - 180);
      }
      baselineScore += score;
  }
  
  for (let f = totalFrames - 1; f >= peakFrame.frame; f--) {
      const g = toGrayscale(frames[f]);
      let frameScore = 0;
      for(let x=searchStartX; x<searchEndX; x++) {
          let score = 0;
          for(let y=splashScanY1; y<splashScanY2; y++) {
              let val = g[y*width+x];
              if (val > 180) score += (val - 180);
          }
          frameScore += score;
      }
      splashScores[f] = frameScore;
      if (frameScore > peakScore) peakScore = frameScore;
  }
  
  const threshold = baselineScore + (peakScore - baselineScore) * 0.15;
  console.log(`[AI] Splash scan Y: ${splashScanY1}-${splashScanY2}. Baseline: ${baselineScore}, Peak: ${peakScore}`);
  
  for (let f = totalFrames - 1; f >= peakFrame.frame; f--) {
      if (splashScores[f] < threshold && splashScores[Math.min(totalFrames-1, f+1)] >= threshold) {
          splashStartFrame = f;
          break;
      }
  }

  if (splashStartFrame !== null) {
      landingFrame = splashStartFrame;
      console.log(`[AI] Landing exactly at frame ${landingFrame}`);
  } else {
      landingFrame = peakFrame.frame + 60;
      console.log('[AI] Splash transition not found, using fallback.');
  }

  const boxSize = 30;
  const bestBlobBox = { x: landingX - boxSize/2, y: splashY - boxSize/2, width: boxSize, height: boxSize };
  onProgress(1.0, 'Analysis complete');

  const finalTrajectory = [];
  for (let t of trackingData) {
      if (t.frame < peakFrame.frame && t.detected) finalTrajectory.push({frame: t.frame, x: t.cx / width, y: t.cy / height});
  }
  for (let t of mlTrack) {
      finalTrajectory.push({frame: t.frame, x: t.cx / width, y: t.cy / height});
  }

  return {
    landingFrameIndex: landingFrame,
    landingX: landingX / width,
    confidence: mlModel ? 0.9 : 0.4,
    phases: [
      { start: 0, end: peakFrame.frame, type: 'FLIGHT_ASCENT' },
      { start: peakFrame.frame, end: landingFrame, type: 'FLIGHT_DESCENT' },
      { start: landingFrame, end: totalFrames - 1, type: 'RIDE_AWAY' }
    ],
    trajectory: finalTrajectory,
    rampMarker: { x: rampMarkerX, y: rampMarkerY },
    rampNative: { x: rampNativeX, y: rampNativeY, topY: rampNativeTopY, startX: rampNativeStartX, endX: rampNativeEndX },
    predBlobBox: bestBlobBox,
    flightBox: bestBlobBox,
    waterlineY: splashY / height
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

/**
 * Extract a cropped rectangular region from an ImageData object.
 */
function cropImageData(srcImageData, x, y, width, height) {
  const crop = new ImageData(width, height);
  const src = srcImageData.data;
  const dst = crop.data;
  const srcW = srcImageData.width;
  const srcH = srcImageData.height;
  
  for (let cy = 0; cy < height; cy++) {
    for (let cx = 0; cx < width; cx++) {
      const srcX = Math.max(0, Math.min(srcW - 1, Math.floor(x) + cx));
      const srcY = Math.max(0, Math.min(srcH - 1, Math.floor(y) + cy));
      
      const srcIdx = (srcY * srcW + srcX) * 4;
      const dstIdx = (cy * width + cx) * 4;
      
      dst[dstIdx] = src[srcIdx];
      dst[dstIdx+1] = src[srcIdx+1];
      dst[dstIdx+2] = src[srcIdx+2];
      dst[dstIdx+3] = src[srcIdx+3];
    }
  }
  return crop;
}
