// API helpers for opponent image upload and detection.
import { readJsonResponse, resolveApiUrl } from "./http.js";
import { prepareOpponentPhotoForUpload } from "../utils/opponentPhoto.js";


export async function uploadOpponentImageFile(imageFile, options = {}) {
  const formData = new FormData();
  formData.append("image", await prepareOpponentPhotoForUpload(imageFile));

  if (options.skipDetection) {
    formData.append("skipDetection", "1");
  }

  if (options.backgroundDetection) {
    formData.append("backgroundDetection", "1");
  }

  const response = await fetch(resolveApiUrl("/api/opponent/image"), {
    method: "POST",
    body: formData,
  });

  return readJsonResponse(response, "Image upload failed.");
}


export async function fetchLatestOpponentUpload(signal = AbortSignal.timeout(8000)) {
  const response = await fetch(resolveApiUrl("/api/opponent/latest"), { signal });

  return readJsonResponse(response, "Could not check latest opponent upload.");
}


export async function fetchLatestOpponentPrediction(signal = AbortSignal.timeout(8000)) {
  const response = await fetch(resolveApiUrl("/api/opponent/prediction/latest"), {
    signal,
  });

  return readJsonResponse(response, "Could not load latest opponent prediction.");
}


export async function detectOpponentTeam(filename = null) {
  const body = filename ? { filename } : {};
  const response = await fetch(resolveApiUrl("/api/opponent/detect-mobilenetv3"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    signal: AbortSignal.timeout(180000),
    body: JSON.stringify(body),
  });

  return readJsonResponse(response, "MobileNetV3 opponent detection failed.");
}


export async function detectOpponentTeamWithEmbedding(filename = null) {
  // Temporarily disabled: MobileNetV3 is the only active opponent-image
  // predictor while we validate the supervised classifier. Keep this helper as
  // a reminder of the old fallback path without allowing the UI to call it.
  throw new Error(
    "Embedding opponent detection is disabled. Use MobileNetV3 detection.",
  );
}
