// Defines the mobile camera capture page for opponent team uploads.
// Flow:
// 1. User taps "Open Camera & Take Picture"
// 2. Phone opens native back camera
// 3. User takes photo
// 4. User taps "Start Detection System"
// 5. Photo uploads to backend, then backend crops/predicts

import React, { useRef, useState } from "react";
import { Camera, PlayCircle, ShieldCheck, X } from "lucide-react";
import { uploadOpponentImageFile } from "../api/championsInsightApi.js";
import "./CameraCapturePage.css";

function CameraCapturePage() {
  const fileInputRef = useRef(null);

  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [uploading, setUploading] = useState(false);
  const [status, setStatus] = useState("Take a clear close photo of all six red team slots.");
  const [error, setError] = useState("");

  function openNativeCamera() {
    setError("");
    setStatus("Opening phone camera...");

    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  }

  function handlePhotoSelected(event) {
    const file = event.target.files?.[0];

    if (!file) {
      setStatus("No photo selected.");
      return;
    }

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    const nextPreviewUrl = URL.createObjectURL(file);

    setSelectedFile(file);
    setPreviewUrl(nextPreviewUrl);
    setError("");
    setStatus("Photo ready. Tap Start Detection System.");
  }

  async function startDetectionSystem() {
    if (!selectedFile) {
      setError("Take a photo first.");
      setStatus("");
      return;
    }

    setUploading(true);
    setError("");
    setStatus("Uploading photo and starting backend detection...");

    try {
      const result = await uploadOpponentImageFile(selectedFile);

      console.log("[CameraCapturePage] upload/detection result:", result);

      if (result.status === "needs_retake") {
        console.warn(
          "[CameraCapturePage] Backend requested retake; staying on camera page.",
          result,
        );
        setError(result.detectionError || "Photo needs to be retaken.");
        setStatus(result.message || "Photo needs to be retaken.");
        return;
      }

      setStatus("Photo uploaded. Detection started. You can return to the battle page.");
    } catch (err) {
      console.error("[CameraCapturePage] Upload/detection failed:", err);
      setError(err.message || "Upload failed.");
      setStatus("");
    } finally {
      setUploading(false);
    }
  }

  function closeCameraPage() {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    window.location.href = "/";
  }

  return (
    <main className="ci-camera-page">
      <section className="ci-camera-shell">
        <header className="ci-camera-header">
          <div>
            <p className="ci-camera-kicker">Champions Insight</p>
            <h1>Opponent Team Capture</h1>
          </div>

          <button
            type="button"
            className="ci-camera-close"
            onClick={closeCameraPage}
            aria-label="Close camera page"
          >
            <X size={22} />
          </button>
        </header>

        <input
          ref={fileInputRef}
          className="ci-native-camera-input"
          type="file"
          accept="image/*"
          capture="environment"
          onChange={handlePhotoSelected}
        />

        <div className="ci-native-photo-stage">
          {previewUrl ? (
            <img
              className="ci-native-photo-preview"
              src={previewUrl}
              alt="Selected opponent team"
            />
          ) : (
            <div className="ci-native-photo-placeholder">
              <Camera size={42} />
              <p>Take a close, clear photo of the six red opponent team slots.</p>
            </div>
          )}
        </div>

        <div className="ci-camera-instructions">
          <div className="ci-camera-instruction-card">
            <Camera size={20} />
            <span>Use the phone back camera and fill the photo with the six red team slots.</span>
          </div>

          <div className="ci-camera-instruction-card">
            <ShieldCheck size={20} />
            <span>Avoid glare. Keep the Pokémon and type icons sharp.</span>
          </div>
        </div>

        {status && <p className="ci-camera-status">{status}</p>}
        {error && <p className="ci-camera-error">{error}</p>}

        <div className="ci-two-button-row">
          <button
            type="button"
            className="ci-camera-capture-button"
            onClick={openNativeCamera}
            disabled={uploading}
          >
            <Camera size={20} />
            <span>Open Camera & Take Picture</span>
          </button>

          <button
            type="button"
            className="ci-camera-start-button"
            onClick={startDetectionSystem}
            disabled={!selectedFile || uploading}
          >
            <PlayCircle size={20} />
            <span>{uploading ? "Running..." : "Start Detection System"}</span>
          </button>
        </div>
      </section>
    </main>
  );
}

export default CameraCapturePage;
