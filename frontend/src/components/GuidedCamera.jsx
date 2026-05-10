// Defines the guided camera preview used for opponent image capture.
import React from "react";
import { Camera, Crosshair, ShieldCheck, X } from "lucide-react";

/**
 * One big crop box only.
 * Give the user a little room around the red team column.
 */
const opponentTeamCaptureBox = {
  left: "18%",
  top: "5%",
  width: "64%",
  height: "88%",
};

// Displays a live camera preview with one large crop guide.
function GuidedCamera({
  cameraError,
  isCameraOpen,
  onCapture,
  onClose,
  onOpen,
  isUploading = false,
  videoRef,
  canvasRef,
}) {
  return (
    <>
      <button
        className="secondary-button"
        type="button"
        onClick={onOpen}
        disabled={isUploading}
      >
        <Camera size={18} aria-hidden="true" />
        <span>Open guided camera</span>
      </button>

      {cameraError && <p className="helper-text warning">{cameraError}</p>}

      {isCameraOpen && (
        <div className="camera-panel">
          <div className="camera-frame">
            <video ref={videoRef} autoPlay playsInline muted />

            <div className="alignment-overlay" aria-hidden="true">
              <div
                className="alignment-team-crop-box"
                style={opponentTeamCaptureBox}
              >
                <span className="alignment-team-crop-label">
                  Fit full team column here
                </span>
              </div>

              <div className="alignment-bottom-hint">
                <span>
                  Only what is inside the yellow box will be cropped and uploaded.
                </span>
              </div>
            </div>
          </div>

          <div className="camera-guide-notes">
            <div className="camera-guide-note">
              <Crosshair size={16} aria-hidden="true" />
              <span>Fit the whole red opponent team column inside the yellow box.</span>
            </div>

            <div className="camera-guide-note">
              <ShieldCheck size={16} aria-hidden="true" />
              <span>Leave a little space around the edges and avoid glare.</span>
            </div>
          </div>

          <div className="camera-actions">
            <button
              className="secondary-button compact"
              type="button"
              onClick={onClose}
            >
              <X size={18} aria-hidden="true" />
              <span>Close</span>
            </button>

            <button
              className="primary-button compact"
              type="button"
              onClick={onCapture}
              disabled={isUploading}
            >
              <Camera size={18} aria-hidden="true" />
              <span>{isUploading ? "Processing" : "Capture"}</span>
            </button>
          </div>

          <canvas ref={canvasRef} className="capture-canvas" />
        </div>
      )}
    </>
  );
}

export default GuidedCamera;