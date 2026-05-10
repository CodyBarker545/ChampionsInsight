import React, { useState } from "react";
import { QRCodeCanvas } from "qrcode.react";

function PhoneCameraQr() {
  const [isOpen, setIsOpen] = useState(false);
  const cameraUrl = `${window.location.origin}/camera`;;

  return (
    <div className="phone-camera-qr-wrapper">
      <button
        type="button"
        className="phone-camera-qr-button"
        onClick={() => setIsOpen(true)}
      >
        Phone QR Camera
      </button>

      {isOpen && (
        <div className="phone-camera-qr-backdrop" onClick={() => setIsOpen(false)}>
          <section
            className="phone-camera-qr-popup"
            onClick={(event) => event.stopPropagation()}
          >
            <button
              type="button"
              className="phone-camera-qr-close"
              onClick={() => setIsOpen(false)}
            >
              ×
            </button>

            <h2>Open Camera on Phone</h2>

            <p>
              Scan this QR code with your phone to open the guided camera page.
            </p>

            <QRCodeCanvas
              value={cameraUrl}
              size={180}
              includeMargin
            />

            <small>{cameraUrl}</small>
          </section>
        </div>
      )}
    </div>
  );
}

export default PhoneCameraQr;