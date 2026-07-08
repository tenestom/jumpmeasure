// Video and camera handling: camera start, upload, recording

import { state } from './state.js';
import { video, errorEl } from './dom.js';
import { drawFrame } from './rendering.js';

export async function startCamera(deviceId) {
  const constraints = { 
    video: deviceId ? { deviceId: { exact: deviceId } } : true, 
    audio: false 
  };
  try {
    if (video.srcObject) {
      video.srcObject.getTracks().forEach(t => t.stop());
    }
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = stream;
    
    await new Promise(resolve => {
      video.onloadedmetadata = () => resolve();
    });
    
    await video.play();

    if (!state.rafId) drawFrame();
  } catch (err) {
    errorEl.textContent = 'Camera error: ' + err.message;
  }
}

export async function populateCameras() {
  const cameraSelect = document.getElementById('cameraSelect');
  const devices = await navigator.mediaDevices.enumerateDevices();
  const cameras = devices.filter(d => d.kind === 'videoinput');
  cameraSelect.innerHTML = '';
  cameras.forEach((cam, i) => {
    const opt = document.createElement('option');
    opt.value = cam.deviceId;
    opt.textContent = cam.label || 'Camera ' + (i + 1);
    cameraSelect.appendChild(opt);
  });
  cameraSelect.style.display = cameras.length > 1 ? 'block' : 'none';
}
