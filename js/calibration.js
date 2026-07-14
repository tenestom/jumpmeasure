// Calibration logic: mode toggle, point placement overlay, interaction handler

import { state } from './state.js';
import {
  calibOverlay, calibInput, calibOk, calibCancel,
  calibBtn, frameInfo
} from './dom.js';
import { getCanvasPoint, getCanvasX } from './rendering.js';

export function handleInteraction(clientX, clientY) {
  const pt = getCanvasPoint(clientX, clientY);
  const cx = pt.x;
  const cy = pt.y;

  if (state.calibrateMode) {
    // Position overlay near click
    calibOverlay.style.display = 'block';
    calibOverlay.style.left = Math.min(window.innerWidth - 180, Math.max(10, clientX - 80)) + 'px';
    calibOverlay.style.top = Math.min(window.innerHeight - 150, Math.max(10, clientY - 120)) + 'px';
    
    calibInput.value = '';
    setTimeout(() => {
      calibInput.focus();
      calibInput.select();
    }, 0);

    const handleOk = () => {
      let m = calibInput.value;
      if (m !== '' && !isNaN(parseFloat(m))) {
        state.calibPoints.push({ pixelX: cx, pixelY: cy, normX: cx / pt.vw, meters: parseFloat(m) });
        state.calibPoints.sort((a, b) => a.pixelX - b.pixelX);
      }
      cleanup();
    };

    const cleanup = () => {
      calibOverlay.style.display = 'none';
      calibOk.removeEventListener('click', handleOk);
      calibCancel.removeEventListener('click', cleanup);
      window.removeEventListener('keydown', handleKey);
    };

    const handleKey = (e) => {
      if (e.key === 'Enter') handleOk();
      if (e.key === 'Escape') cleanup();
    };

    calibOk.addEventListener('click', handleOk);
    calibCancel.addEventListener('click', cleanup);
    window.addEventListener('keydown', handleKey);
  } else {
    state.draggingX = true;
    state.lineX = cx / pt.vw;
  }
}

export function setupCalibrationListeners() {
  calibBtn.addEventListener('click', () => {
    state.calibrateMode = !state.calibrateMode;
    calibBtn.style.background = state.calibrateMode ? '#2ecc71' : '#e67e22';
    frameInfo.textContent = state.calibrateMode ? 'Calibration ON. Click to add points.' : '';
  });

  const clearCalibBtn = document.getElementById('clearCalibBtn');
  clearCalibBtn.addEventListener('click', () => {
    state.calibPoints = [];
  });
}
