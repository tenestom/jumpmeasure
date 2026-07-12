// Measurement logic: interpolation + drawing measurement line + calibration points
// NOTE: No import from rendering.js to avoid circular dependency.
//       Layout info (ox, oy, vw, vh) is passed in by the caller.

import { state } from './state.js';
import { ctx, measurementValue } from './dom.js';

export function getInterpolatedValue(pixelX, points) {
  if (!points || points.length === 0) return null;

  // Sort points by pixelX without modifying the original array
  const sortedPoints = [...points].sort((a, b) => a.pixelX - b.pixelX);

  // Handle cases where pixelX is outside the range (clamping)
  if (pixelX <= sortedPoints[0].pixelX) return sortedPoints[0].meters;
  if (pixelX >= sortedPoints[sortedPoints.length - 1].pixelX) {
    return sortedPoints[sortedPoints.length - 1].meters;
  }

  // Find the segment and interpolate
  for (let i = 0; i < sortedPoints.length - 1; i++) {
    const p1 = sortedPoints[i];
    const p2 = sortedPoints[i + 1];

    if (pixelX >= p1.pixelX && pixelX <= p2.pixelX) {
      const gap = p2.pixelX - p1.pixelX;
      if (gap === 0) return p1.meters;
      const t = (pixelX - p1.pixelX) / gap;
      return p1.meters + t * (p2.meters - p1.meters);
    }
  }

  return null;
}

export function drawLine(ox, oy, vw, vh) {
  // Draw Vertical Line (X-axis)
  if (state.lineX !== null) {
    const px = ox + state.lineX * vw;
    ctx.beginPath();
    ctx.moveTo(px, oy);
    ctx.lineTo(px, oy + vh);
    ctx.strokeStyle = 'red';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Measurement display
    // Show native pixel X (same as algorithm logs)
    const nativePixelX = Math.round(state.lineX * (state.aiFrameWidth || vw));
    const pixelXDisplay = nativePixelX;
    const meters = getInterpolatedValue(state.lineX * vw, state.calibPoints);
    if (meters !== null) {
      const finalMeters = meters;
      const adjustedMeters = finalMeters + 2.1;

      const displayText = `${adjustedMeters.toFixed(2)} m (${finalMeters.toFixed(2)} m)`;
      const debugText = `px: X=${pixelXDisplay}`;
      measurementValue.textContent = `${displayText}  [${debugText}]`;
      
      ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
      ctx.font = '14px sans-serif';
      ctx.fillText(displayText, px + 10, oy + 25);
      ctx.fillStyle = 'rgba(255, 255, 100, 0.8)';
      ctx.font = '11px monospace';
      ctx.fillText(debugText, px + 10, oy + 42);
    } else {
      const debugText = `px: X=${pixelXDisplay}`;
      measurementValue.textContent = `--.-- m  [${debugText}]`;
      ctx.fillStyle = 'rgba(255, 255, 100, 0.8)';
      ctx.font = '11px monospace';
      ctx.fillText(debugText, px + 10, oy + 25);
    }
  } else {
    measurementValue.textContent = '--.-- m';
  }
  
  // Debug: draw detected waterline as cyan horizontal line
  if (state.aiWaterlineY !== null) {
    const wy = oy + state.aiWaterlineY * vh;
    ctx.beginPath();
    ctx.moveTo(ox, wy);
    ctx.lineTo(ox + vw, wy);
    ctx.strokeStyle = 'rgba(0, 255, 255, 0.8)';
    ctx.lineWidth = 1;
    ctx.setLineDash([8, 4]);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = 'rgba(0, 255, 255, 0.9)';
    ctx.font = '11px monospace';
    ctx.fillText('WATERLINE', ox + 4, wy - 4);
  }

  if (state.rampMarker) {
    const rpx = ox + state.rampMarker.x * vw;
    const rpy = oy + state.rampMarker.y * vh;
    ctx.beginPath();
    ctx.arc(rpx, rpy, 10, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(255, 0, 0, 0.8)';
    ctx.fill();
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.fillStyle = 'white';
    ctx.font = '11px sans-serif';
    ctx.fillText('RAMP', rpx + 14, rpy + 4);
  }
  
  // Draw per-frame skier detection from allDetections[]
  const frameDetection = state.allDetections && state.allDetections[state.replayIndex];
  const isMeasurementFrame = state.replayIndex === Math.round((state.lineX || 0) * (state.aiFrameWidth || vw) / (state.aiFrameWidth || vw) * (state.frames ? state.frames.length - 1 : 0));
  
  // Also draw the stored blobBox (measurement frame) in solid red
  if (state.blobBox) {
    const bx = ox + state.blobBox.x * vw;
    const by = oy + state.blobBox.y * vh;
    const bw = state.blobBox.w * vw;
    const bh = state.blobBox.h * vh;
    ctx.beginPath();
    ctx.rect(bx, by, bw, bh);
    ctx.strokeStyle = 'rgba(255, 0, 0, 0.9)';
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
    ctx.font = '11px sans-serif';
    ctx.fillText('BLOB', bx, by - 4);
  }
  
  // Draw current frame detection in orange (if different from measurement frame)
  if (frameDetection) {
    const dbx = ox + frameDetection.box.x * vw;
    const dby = oy + frameDetection.box.y * vh;
    const dbw = frameDetection.box.w * vw;
    const dbh = frameDetection.box.h * vh;
    ctx.beginPath();
    ctx.rect(dbx, dby, dbw, dbh);
    ctx.strokeStyle = 'rgba(255, 140, 0, 0.9)';
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.fillStyle = 'rgba(255, 140, 0, 0.9)';
    ctx.font = '11px sans-serif';
    ctx.fillText(`score:${frameDetection.score.toFixed(0)}`, dbx, dby - 4);
  }
  
  // Draw last flight blob (yellow) - where skier was just before landing
  if (state.predBlobBox) {
    const pbx = ox + state.predBlobBox.x * vw;
    const pby = oy + state.predBlobBox.y * vh;
    const pbw = state.predBlobBox.w * vw;
    const pbh = state.predBlobBox.h * vh;
    ctx.beginPath();
    ctx.rect(pbx, pby, pbw, pbh);
    ctx.strokeStyle = 'rgba(255, 220, 0, 0.9)';
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.fillStyle = 'rgba(255, 220, 0, 0.9)';
    ctx.font = '11px sans-serif';
    ctx.fillText('FLIGHT', pbx, pby - 4);
  }
}

export function drawCalibrationPoints(ox, oy) {
  state.calibPoints.forEach(p => {
    const px = ox + p.pixelX;
    const py = oy + p.pixelY;
    
    // Draw small circle at actual position
    ctx.beginPath();
    ctx.arc(px, py, 6, 0, Math.PI * 2);
    ctx.fillStyle = '#2ecc71';
    ctx.fill();
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Draw meter value above circle
    ctx.fillStyle = 'white';
    ctx.font = '12px sans-serif';
    ctx.shadowBlur = 4;
    ctx.shadowColor = 'black';
    ctx.textAlign = 'center';
    ctx.fillText(p.meters + ' m', px, py - 10);
    ctx.textAlign = 'start';
    ctx.shadowBlur = 0;
  });
}
