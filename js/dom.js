// DOM element references — cached once at module load time.

export const video = document.getElementById('video');
export const canvas = document.getElementById('canvas');
export const ctx = canvas.getContext('2d');
export const startBtn = document.getElementById('startBtn');
export const uploadBtn = document.getElementById('uploadBtn');
export const uploadInput = document.getElementById('uploadInput');
export const errorEl = document.getElementById('error');
export const cameraSelect = document.getElementById('cameraSelect');
export const recordControls = document.getElementById('recordControls');
export const recordBtn = document.getElementById('recordBtn');
export const stopBtn = document.getElementById('stopBtn');
export const playPauseBtn = document.getElementById('playPauseBtn');
export const liveBtn = document.getElementById('liveBtn');
export const calibBtn = document.getElementById('calibBtn');
export const clearCalibBtn = document.getElementById('clearCalibBtn');
export const aiDetectBtn = document.getElementById('aiDetectBtn');
export const scrubberBar = document.getElementById('scrubberBar');
export const scrubber = document.getElementById('scrubber');
export const frameInfo = document.getElementById('frameInfo');
export const measurementValue = document.getElementById('measurementValue');
export const calibOverlay = document.getElementById('calibOverlay');
export const calibInput = document.getElementById('calibInput');
export const calibOk = document.getElementById('calibOk');
export const calibCancel = document.getElementById('calibCancel');
export const homeScreen = document.getElementById('homeScreen');
export const appScreen = document.getElementById('appScreen');
export const homeStartBtn = document.getElementById('homeStartBtn');
export const aiStatus = document.getElementById('aiStatus');

// Offscreen canvas for frame capture
export const offscreen = document.createElement('canvas');
export const offCtx = offscreen.getContext('2d');
