// Defines the mobile camera capture page for opponent team uploads.
// Flow:
// 1. User taps "Open Camera & Take Picture"
// 2. Phone opens native back camera
// 3. User takes photo
// 4. User taps "Start Detection System"
// 5. Photo uploads to backend, then backend crops/predicts

import React, { useEffect, useRef, useState } from "react";
import { Camera, PlayCircle, ShieldCheck, X } from "lucide-react";
import { uploadOpponentImageFile } from "../api/championsInsightApi.js";
import "./CameraCapturePage.css";

const CAMERA_DRAFT_STORAGE_KEY = "champions-insight-camera-photo-draft";
const CAMERA_DRAFT_DB_NAME = "champions-insight-camera";
const CAMERA_DRAFT_STORE_NAME = "photo-drafts";
const CAMERA_DRAFT_ID = "latest";

function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = () => reject(reader.error || new Error("Could not read photo."));
    reader.readAsDataURL(file);
  });
}

function openCameraDraftDatabase() {
  if (!window.indexedDB) {
    return Promise.resolve(null);
  }

  return new Promise((resolve, reject) => {
    const request = window.indexedDB.open(CAMERA_DRAFT_DB_NAME, 1);

    request.onupgradeneeded = () => {
      request.result.createObjectStore(CAMERA_DRAFT_STORE_NAME);
    };

    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error || new Error("Could not open photo storage."));
  });
}

async function saveCameraDraft(file) {
  const database = await openCameraDraftDatabase();

  if (!database) {
    const dataUrl = await fileToDataUrl(file);
    window.sessionStorage.setItem(
      CAMERA_DRAFT_STORAGE_KEY,
      JSON.stringify({
        dataUrl,
        name: file.name,
        type: file.type,
        lastModified: file.lastModified,
      }),
    );
    return;
  }

  await new Promise((resolve, reject) => {
    const transaction = database.transaction(CAMERA_DRAFT_STORE_NAME, "readwrite");
    const store = transaction.objectStore(CAMERA_DRAFT_STORE_NAME);
    store.put(
      {
        file,
        name: file.name,
        type: file.type,
        lastModified: file.lastModified,
        savedAt: Date.now(),
      },
      CAMERA_DRAFT_ID,
    );
    transaction.oncomplete = resolve;
    transaction.onerror = () =>
      reject(transaction.error || new Error("Could not save photo draft."));
  });

  database.close();
}

async function loadCameraDraft() {
  const database = await openCameraDraftDatabase();

  if (database) {
    const draft = await new Promise((resolve, reject) => {
      const transaction = database.transaction(CAMERA_DRAFT_STORE_NAME, "readonly");
      const store = transaction.objectStore(CAMERA_DRAFT_STORE_NAME);
      const request = store.get(CAMERA_DRAFT_ID);

      request.onsuccess = () => resolve(request.result || null);
      request.onerror = () =>
        reject(request.error || new Error("Could not load photo draft."));
    });

    database.close();

    if (draft?.file) {
      return draft.file;
    }
  }

  const savedDraft = window.sessionStorage.getItem(CAMERA_DRAFT_STORAGE_KEY);

  if (!savedDraft) {
    return null;
  }

  const draft = JSON.parse(savedDraft);

  if (!draft?.dataUrl) {
    return null;
  }

  return dataUrlToFile(
    draft.dataUrl,
    draft.name,
    draft.type,
    draft.lastModified,
  );
}

async function clearCameraDraft() {
  window.sessionStorage.removeItem(CAMERA_DRAFT_STORAGE_KEY);

  try {
    const database = await openCameraDraftDatabase();

    if (!database) {
      return;
    }

    await new Promise((resolve, reject) => {
      const transaction = database.transaction(CAMERA_DRAFT_STORE_NAME, "readwrite");
      transaction.objectStore(CAMERA_DRAFT_STORE_NAME).delete(CAMERA_DRAFT_ID);
      transaction.oncomplete = resolve;
      transaction.onerror = () =>
        reject(transaction.error || new Error("Could not clear photo draft."));
    });

    database.close();
  } catch {
    // Clearing cached camera data is best-effort.
  }
}

function dataUrlToFile(dataUrl, fileName, fileType, lastModified) {
  const [metadata, encodedData = ""] = dataUrl.split(",");
  const mimeMatch = metadata.match(/data:([^;]+);base64/);
  const mimeType = fileType || mimeMatch?.[1] || "image/jpeg";
  const binaryString = window.atob(encodedData);
  const bytes = new Uint8Array(binaryString.length);

  for (let index = 0; index < binaryString.length; index += 1) {
    bytes[index] = binaryString.charCodeAt(index);
  }

  return new File([bytes], fileName || "opponent-team.jpg", {
    type: mimeType,
    lastModified: lastModified || Date.now(),
  });
}

function CameraCapturePage({ onClose }) {
  const fileInputRef = useRef(null);

  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [uploading, setUploading] = useState(false);
  const [status, setStatus] = useState("Take a clear close photo of all six red team slots.");
  const [error, setError] = useState("");

  useEffect(() => {
    let isCurrent = true;

    loadCameraDraft()
      .then((restoredFile) => {
        if (!isCurrent || !restoredFile) {
          return;
        }

      const restoredPreviewUrl = URL.createObjectURL(restoredFile);

      setSelectedFile(restoredFile);
      setPreviewUrl(restoredPreviewUrl);
      setError("");
      setStatus("Photo restored. Tap Start Detection System.");
      })
      .catch(() => {
        void clearCameraDraft();
      });

    return () => {
      isCurrent = false;
    };
  }, []);

  function openNativeCamera() {
    setError("");
    setStatus("Opening phone camera...");

    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  }

  async function handlePhotoSelected(event) {
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

    try {
      await saveCameraDraft(file);
    } catch {
      // If storage fails, keep the in-memory preview and let the user continue.
    }
  }

  async function startDetectionSystem() {
    if (!selectedFile) {
      setError("Take a photo first.");
      setStatus("");
      return;
    }

    setUploading(true);
    setError("");
    setStatus("Uploading photo...");

    try {
      const result = await uploadOpponentImageFile(selectedFile, {
        backgroundDetection: true,
      });

      console.log("[CameraCapturePage] upload/detection result:", result);

      if (result.status === "needs_retake") {
        console.warn(
          "[CameraCapturePage] Backend returned a photo quality warning; continuing anyway.",
          result,
        );
      }

      setStatus("Photo uploaded. Battle Prep will update when detection is ready.");
      void clearCameraDraft();
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

    void clearCameraDraft();
    onClose?.();
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
