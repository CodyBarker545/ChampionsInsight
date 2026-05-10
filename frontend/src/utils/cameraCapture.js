const NORMALIZED_UPLOAD_WIDTH = 1215;
const NORMALIZED_UPLOAD_HEIGHT = 2160;

export const GUIDED_CAMERA_UPLOAD_SIZE = {
  width: NORMALIZED_UPLOAD_WIDTH,
  height: NORMALIZED_UPLOAD_HEIGHT,
};

/**
 * CameraCapturePage.jsx uses this inside:
 *
 * navigator.mediaDevices.getUserMedia({
 *   audio: false,
 *   video: guidedCameraVideoConstraints,
 * })
 *
 * So this must be ONLY the video constraint object.
 */
export const guidedCameraVideoConstraints = {
  facingMode: { exact: "environment" },
  width: { ideal: 1920 },
  height: { ideal: 1080 },
};

export const fallbackGuidedCameraVideoConstraints = {
  facingMode: { ideal: "environment" },
  width: { ideal: 1920 },
  height: { ideal: 1080 },
};

/**
 * Converts one guide box from the displayed preview coordinate system
 * into the actual camera video frame coordinate system.
 *
 * This matches CSS:
 *   object-fit: cover;
 */
export function getVideoCropFromPreviewBox(
  videoElement,
  previewElement,
  guideBoxPercent,
) {
  const videoWidth = videoElement.videoWidth;
  const videoHeight = videoElement.videoHeight;

  const previewWidth = previewElement.clientWidth;
  const previewHeight = previewElement.clientHeight;

  if (!videoWidth || !videoHeight) {
    throw new Error("Video dimensions are not ready.");
  }

  if (!previewWidth || !previewHeight) {
    throw new Error("Preview dimensions are not ready.");
  }

  const videoAspect = videoWidth / videoHeight;
  const previewAspect = previewWidth / previewHeight;

  let visibleVideoWidth = videoWidth;
  let visibleVideoHeight = videoHeight;
  let visibleVideoX = 0;
  let visibleVideoY = 0;

  if (videoAspect > previewAspect) {
    // Raw video is wider than the preview.
    // object-fit: cover crops left and right.
    visibleVideoHeight = videoHeight;
    visibleVideoWidth = videoHeight * previewAspect;
    visibleVideoX = (videoWidth - visibleVideoWidth) / 2;
  } else {
    // Raw video is taller/narrower than the preview.
    // object-fit: cover crops top and bottom.
    visibleVideoWidth = videoWidth;
    visibleVideoHeight = videoWidth / previewAspect;
    visibleVideoY = (videoHeight - visibleVideoHeight) / 2;
  }

  const boxLeft = parseFloat(guideBoxPercent.left) / 100;
  const boxTop = parseFloat(guideBoxPercent.top) / 100;
  const boxWidth = parseFloat(guideBoxPercent.width) / 100;
  const boxHeight = parseFloat(guideBoxPercent.height) / 100;

  const sx = visibleVideoX + boxLeft * visibleVideoWidth;
  const sy = visibleVideoY + boxTop * visibleVideoHeight;
  const sw = boxWidth * visibleVideoWidth;
  const sh = boxHeight * visibleVideoHeight;

  return {
    sx,
    sy,
    sw,
    sh,
    debug: {
      videoWidth,
      videoHeight,
      previewWidth,
      previewHeight,
      videoAspect,
      previewAspect,
      visibleVideoX,
      visibleVideoY,
      visibleVideoWidth,
      visibleVideoHeight,
      guideBoxPercent,
    },
  };
}

/**
 * Captures ONLY the big guided team box.
 * Then resizes that crop to 1215x2160.
 *
 * Final backend upload is always:
 *   1215x2160
 */
export async function captureGuidedCameraBlob(
  videoElement,
  canvasElement,
  previewElement,
  guideBoxPercent,
  options = {},
) {
  const {
    outputWidth = NORMALIZED_UPLOAD_WIDTH,
    outputHeight = NORMALIZED_UPLOAD_HEIGHT,
    mimeType = "image/jpeg",
    quality = 0.95,
    debug = true,
  } = options;

  if (!videoElement) {
    throw new Error("Cannot capture because the video element is missing.");
  }

  if (!previewElement) {
    throw new Error("Cannot capture because the preview element is missing.");
  }

  if (!guideBoxPercent) {
    throw new Error("Cannot capture because the guide box is missing.");
  }

  const crop = getVideoCropFromPreviewBox(
    videoElement,
    previewElement,
    guideBoxPercent,
  );

  const canvas = canvasElement || document.createElement("canvas");
  canvas.width = outputWidth;
  canvas.height = outputHeight;

  const context = canvas.getContext("2d");

  if (!context) {
    throw new Error("Could not create canvas context.");
  }

  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";

  context.drawImage(
    videoElement,
    crop.sx,
    crop.sy,
    crop.sw,
    crop.sh,
    0,
    0,
    outputWidth,
    outputHeight,
  );

  const metadata = {
    inputWidth: videoElement.videoWidth,
    inputHeight: videoElement.videoHeight,
    previewWidth: crop.debug.previewWidth,
    previewHeight: crop.debug.previewHeight,

    visibleVideoX: Math.round(crop.debug.visibleVideoX),
    visibleVideoY: Math.round(crop.debug.visibleVideoY),
    visibleVideoWidth: Math.round(crop.debug.visibleVideoWidth),
    visibleVideoHeight: Math.round(crop.debug.visibleVideoHeight),

    cropX: Math.round(crop.sx),
    cropY: Math.round(crop.sy),
    cropWidth: Math.round(crop.sw),
    cropHeight: Math.round(crop.sh),

    outputWidth,
    outputHeight,
  };

  if (debug) {
    console.log("[GuidedCameraCapture] original video size:", {
      width: metadata.inputWidth,
      height: metadata.inputHeight,
    });

    console.log("[GuidedCameraCapture] preview size:", {
      width: metadata.previewWidth,
      height: metadata.previewHeight,
    });

    console.log("[GuidedCameraCapture] visible video box:", {
      x: metadata.visibleVideoX,
      y: metadata.visibleVideoY,
      width: metadata.visibleVideoWidth,
      height: metadata.visibleVideoHeight,
    });

    console.log("[GuidedCameraCapture] crop box:", {
      x: metadata.cropX,
      y: metadata.cropY,
      width: metadata.cropWidth,
      height: metadata.cropHeight,
    });

    console.log("[GuidedCameraCapture] final upload size:", {
      width: metadata.outputWidth,
      height: metadata.outputHeight,
    });
  }

  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => {
        if (!blob) {
          reject(new Error("Failed to create image blob."));
          return;
        }

        resolve({
          blob,
          metadata,
        });
      },
      mimeType,
      quality,
    );
  });
}

/**
 * Optional camera improvements.
 * Phones/browsers that do not support these settings will ignore them.
 */
export async function applyBestCameraTrackSettings(stream) {
  if (!stream) return;

  const videoTracks = stream.getVideoTracks();

  if (!videoTracks || videoTracks.length === 0) return;

  const track = videoTracks[0];

  if (!track || typeof track.applyConstraints !== "function") return;

  const capabilities =
    typeof track.getCapabilities === "function"
      ? track.getCapabilities()
      : {};

  const advancedSettings = {};

  if (capabilities.focusMode?.includes("continuous")) {
    advancedSettings.focusMode = "continuous";
  }

  if (capabilities.exposureMode?.includes("continuous")) {
    advancedSettings.exposureMode = "continuous";
  }

  if (capabilities.whiteBalanceMode?.includes("continuous")) {
    advancedSettings.whiteBalanceMode = "continuous";
  }

  if ("torch" in capabilities) {
    advancedSettings.torch = false;
  }

  try {
    if (Object.keys(advancedSettings).length > 0) {
      await track.applyConstraints({
        advanced: [advancedSettings],
      });
    }

    console.log(
      "[GuidedCameraCapture] camera settings:",
      track.getSettings?.(),
    );
  } catch (error) {
    console.warn("[GuidedCameraCapture] advanced settings ignored:", error);
  }
}