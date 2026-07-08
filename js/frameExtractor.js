/**
 * Frame extraction from uploaded videos.
 * 
 * Steps through a <video> element frame-by-frame using seeking,
 * capturing each frame as ImageData for AI analysis.
 */

/**
 * Extract all frames from a video element into ImageData array.
 * 
 * @param {HTMLVideoElement} videoEl - The video element with loaded source
 * @param {HTMLCanvasElement} offscreen - Offscreen canvas for capture
 * @param {CanvasRenderingContext2D} offCtx - Offscreen canvas context
 * @param {Function} onProgress - Callback(progress: 0-1, message: string)
 * @returns {Promise<ImageData[]>} Array of captured frames
 */
export async function extractFrames(videoEl, offscreen, offCtx, onProgress = () => {}) {
  const duration = videoEl.duration;
  const fps = 30;  // Assume 30fps — standard for most phone videos
  const totalFrames = Math.floor(duration * fps);
  
  // Cap at reasonable frame count to avoid memory issues
  // 1920x1080 × 4 bytes × 300 frames = ~2.5GB
  const maxFrames = 400;
  const frameStep = totalFrames > maxFrames ? totalFrames / maxFrames : 1;
  const framesToExtract = Math.min(totalFrames, maxFrames);

  const vw = videoEl.videoWidth;
  const vh = videoEl.videoHeight;

  if (!vw || !vh) {
    throw new Error('Video dimensions not available');
  }

  offscreen.width = vw;
  offscreen.height = vh;

  const frames = [];
  
  onProgress(0, `Extracting ${framesToExtract} frames...`);

  // Pause the video for seeking
  videoEl.pause();

  for (let i = 0; i < framesToExtract; i++) {
    const frameIndex = Math.floor(i * frameStep);
    const time = frameIndex / fps;

    // Seek to the target time
    videoEl.currentTime = time;

    // Wait for the seek to complete
    await new Promise((resolve) => {
      const onSeeked = () => {
        videoEl.removeEventListener('seeked', onSeeked);
        resolve();
      };
      videoEl.addEventListener('seeked', onSeeked);
    });

    // Capture the frame
    offCtx.drawImage(videoEl, 0, 0, vw, vh);
    const imageData = offCtx.getImageData(0, 0, vw, vh);
    frames.push(imageData);

    // Report progress every 5 frames
    if (i % 5 === 0 || i === framesToExtract - 1) {
      onProgress((i + 1) / framesToExtract, 
        `Extracting frame ${i + 1}/${framesToExtract}...`);
      // Yield to UI thread
      await new Promise(resolve => setTimeout(resolve, 0));
    }
  }

  onProgress(1, `Extracted ${frames.length} frames`);
  return frames;
}
