/**
 * JumpMeasure — Main application entry point.
 * 
 * Imports all modules and sets up event listeners.
 * No application logic here — just wiring.
 */

import { state, ZOOM_MIN, ZOOM_MAX } from './state.js';
import {
  video, canvas, ctx, startBtn, uploadBtn, uploadInput,
  errorEl, cameraSelect, recordControls, recordBtn, stopBtn,
  playPauseBtn, liveBtn, scrubberBar, scrubber, frameInfo,
  homeScreen, appScreen, homeStartBtn, aiDetectBtn, aiStatus,
  offscreen, offCtx
} from './dom.js';
import { resizeCanvas, drawFrame, showReplayFrame, getCanvasX, applyZoom } from './rendering.js';
import { handleInteraction, setupCalibrationListeners } from './calibration.js';
import { startCamera, populateCameras } from './video.js';
import { analyzeJump } from './ai/jumpAnalyzer.js';
import { extractFrames } from './frameExtractor.js';

// --- Window setup ---
window.addEventListener('resize', resizeCanvas);
window.addEventListener('load', resizeCanvas);

// --- Home screen ---
homeStartBtn.addEventListener('click', () => {
  homeScreen.style.display = 'none';
  appScreen.style.display = 'block';
  resizeCanvas();
});

// --- Camera start ---
startBtn.addEventListener('click', async () => {
  errorEl.textContent = '';
  await startCamera(null);
  await populateCameras();
  startBtn.style.display = 'none';
  uploadBtn.style.display = 'none';
  recordControls.style.display = 'flex';
});

// --- Upload ---
uploadBtn.addEventListener('click', () => {
  uploadInput.click();
});

uploadInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) return;

  if (video.srcObject) {
    video.srcObject.getTracks().forEach(t => t.stop());
    video.srcObject = null;
  }
  
  video.src = URL.createObjectURL(file);
  state.uploadedVideoMode = true;
  state.replayMode = false;
  state.recording = false;
  
  startBtn.style.display = 'none';
  uploadBtn.style.display = 'none';
  cameraSelect.style.display = 'none';
  
  recordBtn.style.display = 'none';
  stopBtn.style.display = 'none';
  playPauseBtn.style.display = 'inline-block';
  liveBtn.textContent = '✖ Close';
  liveBtn.disabled = false;
  
  recordControls.style.display = 'flex';
  state.zoomLevel = 1; state.panX = 0; state.panY = 0;

  video.onloadedmetadata = async () => {
    // Enforce 15-second max duration
    if (video.duration > 15) {
      if (aiStatus) {
        aiStatus.style.display = 'block';
        aiStatus.textContent = `Video is ${video.duration.toFixed(1)}s — max 15s allowed. Trim the clip and try again.`;
        aiStatus.style.borderColor = 'rgba(231, 76, 60, 0.5)';
      }
      video.src = '';
      state.uploadedVideoMode = false;
      startBtn.style.display = 'block';
      uploadBtn.style.display = 'block';
      recordControls.style.display = 'none';
      return;
    }

    // Show video controls first
    scrubber.max = video.duration;
    scrubber.step = 0.01;
    scrubber.value = 0;
    scrubberBar.style.display = 'flex';
    if (!state.rafId) drawFrame();

    // Extract frames for AI analysis in the background
    if (aiStatus) {
      aiStatus.style.display = 'block';
      aiStatus.textContent = 'Extracting frames for AI...';
      aiStatus.style.borderColor = 'rgba(155, 89, 182, 0.3)';
    }

    try {
      const extractedFrames = await extractFrames(video, offscreen, offCtx, (progress, msg) => {
        if (aiStatus) {
          aiStatus.textContent = msg;
        }
      });

      state.frames = extractedFrames;
      state.replayMode = true;
      state.replayIndex = 0;
      state.uploadedVideoMode = false;

      // Switch scrubber to frame mode
      scrubber.max = state.frames.length - 1;
      scrubber.step = 1;
      scrubber.value = 0;
      playPauseBtn.style.display = 'none';

      showReplayFrame();

      if (aiStatus) {
        aiStatus.textContent = `${state.frames.length} frames extracted. Ready for AI analysis!`;
        aiStatus.style.borderColor = 'rgba(46, 204, 113, 0.5)';
        setTimeout(() => { aiStatus.style.display = 'none'; }, 3000);
      }
    } catch (err) {
      console.error('Frame extraction failed:', err);
      // Fallback: keep video mode, AI won't work but manual measurement still does
      video.currentTime = 0;
      video.play();
      playPauseBtn.textContent = '⏸ Pause';
      if (aiStatus) {
        aiStatus.textContent = 'Frame extraction failed. Manual measurement still works.';
        aiStatus.style.borderColor = 'rgba(231, 76, 60, 0.5)';
        setTimeout(() => { aiStatus.style.display = 'none'; }, 4000);
      }
    }
  };
});

// --- Play/Pause ---
playPauseBtn.addEventListener('click', () => {
  if (video.paused) {
    video.play();
    playPauseBtn.textContent = '⏸ Pause';
  } else {
    video.pause();
    playPauseBtn.textContent = '▶ Play';
  }
});

// --- Video time update ---
video.addEventListener('timeupdate', () => {
  if (state.uploadedVideoMode && !video.paused) {
    scrubber.value = video.currentTime;
    frameInfo.textContent = `Time: ${video.currentTime.toFixed(2)}s / ${video.duration.toFixed(2)}s`;
  }
});

// --- Camera select ---
cameraSelect.addEventListener('change', () => {
  startCamera(cameraSelect.value);
});

// --- Recording ---
recordBtn.addEventListener('click', () => {
  state.frames = [];
  state.recording = true;
  state.replayMode = false;
  recordBtn.disabled = true;
  stopBtn.disabled = false;
  liveBtn.disabled = true;
  scrubberBar.style.display = 'none';
  frameInfo.textContent = 'Recording...';
});

stopBtn.addEventListener('click', () => {
  state.recording = false;
  state.replayMode = true;
  state.replayIndex = state.frames.length - 1;
  recordBtn.disabled = false;
  stopBtn.disabled = true;
  liveBtn.disabled = false;
  if (state.frames.length > 0) {
    scrubber.max = state.frames.length - 1;
    scrubber.value = state.replayIndex;
    scrubberBar.style.display = 'flex';
    showReplayFrame();
  } else {
    frameInfo.textContent = 'No frames captured.';
  }
});

// --- Scrubber ---
scrubber.addEventListener('input', () => {
  if (state.uploadedVideoMode) {
    video.pause();
    playPauseBtn.textContent = '▶ Play';
    video.currentTime = parseFloat(scrubber.value);
    frameInfo.textContent = `Time: ${video.currentTime.toFixed(2)}s / ${video.duration.toFixed(2)}s`;
  } else {
    state.replayIndex = parseInt(scrubber.value, 10);
    showReplayFrame();
  }
});

// --- Live button ---
liveBtn.addEventListener('click', () => {
  if (state.uploadedVideoMode) {
    video.pause();
    video.src = '';
    state.uploadedVideoMode = false;
    
    recordControls.style.display = 'none';
    scrubberBar.style.display = 'none';
    startBtn.style.display = 'block';
    uploadBtn.style.display = 'block';
    frameInfo.textContent = '';
    
    liveBtn.textContent = '▶ Live';
    recordBtn.style.display = 'inline-block';
    stopBtn.style.display = 'inline-block';
    playPauseBtn.style.display = 'none';
    
    const rect = canvas.getBoundingClientRect();
    ctx.clearRect(0, 0, rect.width, rect.height);
    state.zoomLevel = 1; state.panX = 0; state.panY = 0;
  } else {
    state.replayMode = false;
    state.recording = false;
    scrubberBar.style.display = 'none';
    liveBtn.disabled = true;
    recordBtn.disabled = false;
    stopBtn.disabled = true;
    frameInfo.textContent = '';
  }
});

// --- Calibration ---
setupCalibrationListeners();

// --- AI Detect ---
if (aiDetectBtn) {
  aiDetectBtn.addEventListener('click', async () => {
    if (state.frames.length < 10) {
      if (aiStatus) {
        aiStatus.style.display = 'block';
        aiStatus.textContent = 'Upload or record a video first (need at least 10 frames)';
        setTimeout(() => { aiStatus.style.display = 'none'; }, 3000);
      }
      return;
    }

    aiDetectBtn.disabled = true;
    aiDetectBtn.textContent = '⏳ Analyzing...';
    
    if (aiStatus) {
      aiStatus.style.display = 'block';
      aiStatus.textContent = 'Starting AI analysis...';
    }

    try {
      const result = await analyzeJump(state.frames, (progress, message) => {
        if (aiStatus) {
          aiStatus.textContent = `${message} (${Math.round(progress * 100)}%)`;
        }
      });

      if (result.landingX !== null && result.confidence > 0.3) {
        // Set the measurement line to the detected landing point
        state.lineX = result.landingX;
        
        // Navigate to the landing frame
        if (result.landingFrameIndex !== null) {
          state.replayIndex = Math.min(result.landingFrameIndex, state.frames.length - 1);
        }
        
        showReplayFrame();

        if (aiStatus) {
          const confidencePercent = Math.round(result.confidence * 100);
          aiStatus.textContent = `Landing detected at frame ${result.landingFrameIndex + 1} (confidence: ${confidencePercent}%). Line placed — adjust if needed.`;
          aiStatus.style.borderColor = result.confidence > 0.6 
            ? 'rgba(46, 204, 113, 0.5)' 
            : 'rgba(241, 196, 15, 0.5)';
        }
      } else {
        if (aiStatus) {
          aiStatus.textContent = result.error || 'Could not detect landing. Try manual placement.';
          aiStatus.style.borderColor = 'rgba(231, 76, 60, 0.5)';
        }
      }
    } catch (err) {
      console.error('AI analysis failed:', err);
      if (aiStatus) {
        aiStatus.textContent = `Error: ${err.message}`;
        aiStatus.style.borderColor = 'rgba(231, 76, 60, 0.5)';
      }
    } finally {
      aiDetectBtn.disabled = false;
      aiDetectBtn.textContent = '🤖 AI Detect';
      
      // Hide status after 8 seconds
      setTimeout(() => {
        if (aiStatus) aiStatus.style.display = 'none';
      }, 8000);
    }
  });
}

// --- Mouse events ---
canvas.addEventListener('mousedown', e => { handleInteraction(e.clientX, e.clientY); });
canvas.addEventListener('mousemove', e => { 
  if (state.draggingX && !state.calibrateMode) {
    state.lineX = getCanvasX(e.clientX);
  }
});
canvas.addEventListener('mouseup', () => { state.draggingX = false; });

// --- Touch events — single finger drags the line, two fingers pinch-to-zoom ---
canvas.addEventListener('touchstart', e => {
  if (e.touches.length === 2) {
    // Begin pinch: record initial distance and mid-point
    const dx = e.touches[1].clientX - e.touches[0].clientX;
    const dy = e.touches[1].clientY - e.touches[0].clientY;
    state.lastPinchDist = Math.hypot(dx, dy);
    state.lastPinchMidX = (e.touches[0].clientX + e.touches[1].clientX) / 2;
    state.lastPinchMidY = (e.touches[0].clientY + e.touches[1].clientY) / 2;
    state.draggingX = false;  // cancel any single-finger drag
  } else if (e.touches.length === 1) {
    state.lastPinchDist = null;
    handleInteraction(e.touches[0].clientX, e.touches[0].clientY);
  }
}, { passive: false });

canvas.addEventListener('touchmove', e => {
  e.preventDefault();
  if (e.touches.length === 2) {
    // Pinch zoom
    const dx = e.touches[1].clientX - e.touches[0].clientX;
    const dy = e.touches[1].clientY - e.touches[0].clientY;
    const dist = Math.hypot(dx, dy);
    const midX = (e.touches[0].clientX + e.touches[1].clientX) / 2;
    const midY = (e.touches[0].clientY + e.touches[1].clientY) / 2;
    if (state.lastPinchDist !== null) {
      const pinchRatio = dist / state.lastPinchDist;
      const newZoom = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, state.zoomLevel * pinchRatio));
      applyZoom(newZoom, midX, midY);
    }
    state.lastPinchDist = dist;
    state.lastPinchMidX = midX;
    state.lastPinchMidY = midY;
  } else if (e.touches.length === 1 && state.draggingX && !state.calibrateMode) {
    state.lineX = getCanvasX(e.touches[0].clientX);
  }
}, { passive: false });

canvas.addEventListener('touchend', e => {
  if (e.touches.length < 2) state.lastPinchDist = null;
  if (e.touches.length === 0) state.draggingX = false;
});

// --- Desktop: Ctrl + scroll wheel to zoom ---
canvas.addEventListener('wheel', e => {
  if (!e.ctrlKey && !e.metaKey) return;
  e.preventDefault();
  const delta = e.deltaY > 0 ? 0.9 : 1.1;  // scroll down = zoom out
  const newZoom = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, state.zoomLevel * delta));
  applyZoom(newZoom, e.clientX, e.clientY);
}, { passive: false });
