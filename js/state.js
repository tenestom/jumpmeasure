// Shared application state — all mutable state lives here
// so every module imports from one source of truth.

export const state = {
  lineX: null,
  draggingX: false,
  recording: false,
  uploadedVideoMode: false,
  frames: [],       // array of ImageData
  replayMode: false,
  replayIndex: 0,
  calibrateMode: false,
  calibPoints: [],   // [{pixelX, pixelY, meters}]
  zoomLevel: 1,
  panX: 0,
  panY: 0,
  lastPinchDist: null,
  lastPinchMidX: 0,
  lastPinchMidY: 0,
  rafId: null,
  rampMarker: null,
  blobBox: null,
  predBlobBox: null,
  aiFrameWidth: null,
  allDetections: [],  // array indexed by frame, each {box, score} or null
};

export const ZOOM_MIN = 1;
export const ZOOM_MAX = 8;
