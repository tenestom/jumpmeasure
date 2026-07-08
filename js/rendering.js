// Canvas rendering utilities: resize, layout, zoom/pan, frame drawing

import { state, ZOOM_MIN, ZOOM_MAX } from './state.js';
import { video, canvas, ctx, offscreen, offCtx, scrubber, frameInfo } from './dom.js';
import { drawLine, drawCalibrationPoints } from './measurement.js';

export function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

// Returns the unzoomed image layout (before zoom transform is applied).
// ox/oy/vw/vh are in CSS pixels referring to where the image sits inside
// the canvas element at zoom=1.
export function getBaseLayout() {
  const rect = canvas.getBoundingClientRect();
  const fw = video.videoWidth  || (state.replayMode && state.frames.length > 0 ? state.frames[state.replayIndex].width  : 0);
  const fh = video.videoHeight || (state.replayMode && state.frames.length > 0 ? state.frames[state.replayIndex].height : 0);

  if (!fw || !fh) {
    return { ox: 0, oy: 0, vw: rect.width, vh: rect.height, cw: rect.width, ch: rect.height };
  }
  const videoAspect = fw / fh;
  const canvasAspect = rect.width / rect.height;
  let drawW, drawH, offsetX, offsetY;
  if (videoAspect > canvasAspect) {
    drawW = rect.width;  drawH = rect.width / videoAspect;
    offsetX = 0;         offsetY = (rect.height - drawH) / 2;
  } else {
    drawH = rect.height; drawW = rect.height * videoAspect;
    offsetX = (rect.width - drawW) / 2; offsetY = 0;
  }
  return { ox: offsetX, oy: offsetY, vw: drawW, vh: drawH, cw: rect.width, ch: rect.height };
}

// Clamp panX/panY so we can't pan past the image edges
export function clampPan() {
  const rect = canvas.getBoundingClientRect();
  const { ox, oy, vw, vh } = getBaseLayout();
  // Zoomed image dimensions
  const zw = vw * state.zoomLevel;
  const zh = vh * state.zoomLevel;
  // Maximum allowed pan so the image never shows blank outside its bounds
  const maxPanX = Math.max(0, (zw - rect.width)  / 2 + ox * state.zoomLevel);
  const maxPanY = Math.max(0, (zh - rect.height) / 2 + oy * state.zoomLevel);
  state.panX = Math.max(-maxPanX, Math.min(maxPanX, state.panX));
  state.panY = Math.max(-maxPanY, Math.min(maxPanY, state.panY));
}

// Apply zoom around a focal point given in client coordinates.
// localX/Y must be relative to the canvas CENTER to match the transform.
export function applyZoom(newZoom, focalClientX, focalClientY) {
  const rect = canvas.getBoundingClientRect();
  // Focal point relative to canvas CENTER (not top-left)
  const localX = focalClientX - rect.left - rect.width  / 2;
  const localY = focalClientY - rect.top  - rect.height / 2;
  // Keep the pixel under the focal point stationary
  const ratio = newZoom / state.zoomLevel;
  state.panX = localX - ratio * (localX - state.panX);
  state.panY = localY - ratio * (localY - state.panY);
  state.zoomLevel = newZoom;
  clampPan();
}

// Convert a client-space point to image-relative (0..1) coordinates,
// accounting for the current zoom/pan transform.
export function getCanvasPoint(clientX, clientY) {
  const rect  = canvas.getBoundingClientRect();
  const { ox, oy, vw, vh } = getBaseLayout();
  // Canvas-local coords (CSS pixels)
  const localX = clientX - rect.left;
  const localY = clientY - rect.top;
  // Invert the zoom/pan transform: canvas origin = rect centre + panOffset
  const cx = rect.width  / 2;
  const cy = rect.height / 2;
  // Point in zoomed space relative to centre
  const relX = localX - cx - state.panX + cx;
  const relY = localY - cy - state.panY + cy;
  // Unzoom back to base layout
  const baseX = (relX - cx) / state.zoomLevel + cx;
  const baseY = (relY - cy) / state.zoomLevel + cy;
  // Subtract image offset so (0,0) = top-left of image
  return {
    x:  baseX - ox,
    y:  baseY - oy,
    vw, vh,
    ox, oy
  };
}

export function getCanvasX(clientX) {
  const pt = getCanvasPoint(clientX, 0);
  return pt.x / pt.vw;
}

export function getCanvasY(clientY) {
  const pt = getCanvasPoint(0, clientY);
  return pt.y / pt.vh;
}

// Wraps drawing with zoom/pan transform applied around the canvas centre
export function withZoomTransform(fn) {
  const rect = canvas.getBoundingClientRect();
  const cx = rect.width  / 2;
  const cy = rect.height / 2;
  ctx.save();
  ctx.translate(cx + state.panX, cy + state.panY);
  ctx.scale(state.zoomLevel, state.zoomLevel);
  ctx.translate(-cx, -cy);
  fn();
  ctx.restore();
}

export function drawFrame() {
  const rect = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);

  if (!state.replayMode && video.readyState >= 2) {
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    if (!vw || !vh) {
      state.rafId = requestAnimationFrame(drawFrame);
      return;
    }

    const { ox: offsetX, oy: offsetY, vw: drawW, vh: drawH } = getBaseLayout();

    withZoomTransform(() => {
      ctx.drawImage(video, 0, 0, vw, vh, offsetX, offsetY, drawW, drawH);
    });

    // Capture frame during recording (always at native resolution)
    if (state.recording) {
      offscreen.width = vw;
      offscreen.height = vh;
      offCtx.drawImage(video, 0, 0);
      state.frames.push(offCtx.getImageData(0, 0, offscreen.width, offscreen.height));
    }

    withZoomTransform(() => { drawLine(offsetX, offsetY, drawW, drawH); drawCalibrationPoints(offsetX, offsetY); });

  } else if (state.replayMode && state.frames.length > 0) {
    const frame = state.frames[state.replayIndex];
    offscreen.width  = frame.width;
    offscreen.height = frame.height;
    offCtx.putImageData(frame, 0, 0);

    const { ox: offsetX, oy: offsetY, vw: drawW, vh: drawH } = getBaseLayout();

    withZoomTransform(() => {
      ctx.drawImage(offscreen, 0, 0, frame.width, frame.height, offsetX, offsetY, drawW, drawH);
    });
    withZoomTransform(() => { drawLine(offsetX, offsetY, drawW, drawH); drawCalibrationPoints(offsetX, offsetY); });
  }
  state.rafId = requestAnimationFrame(drawFrame);
}

export function showReplayFrame() {
  const rect = canvas.getBoundingClientRect();
  const frame = state.frames[state.replayIndex];
  offscreen.width  = frame.width;
  offscreen.height = frame.height;
  offCtx.putImageData(frame, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);

  const { ox: offsetX, oy: offsetY, vw: drawW, vh: drawH } = getBaseLayout();
  withZoomTransform(() => {
    ctx.drawImage(offscreen, 0, 0, frame.width, frame.height, offsetX, offsetY, drawW, drawH);
  });
  withZoomTransform(() => { drawLine(offsetX, offsetY, drawW, drawH); drawCalibrationPoints(offsetX, offsetY); });

  // Sync scrubber thumb position
  scrubber.value = state.replayIndex;
  frameInfo.textContent = `Frame ${state.replayIndex + 1} / ${state.frames.length}`;
}
