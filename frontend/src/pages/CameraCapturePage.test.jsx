import "@testing-library/jest-dom/vitest";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import CameraCapturePage from "./CameraCapturePage.jsx";
import { uploadOpponentImageFile } from "../api/championsInsightApi.js";

vi.mock("../api/championsInsightApi.js", () => ({
  uploadOpponentImageFile: vi.fn(),
}));

describe("CameraCapturePage", () => {
  const originalLocation = window.location;

  afterEach(() => {
    cleanup();
    window.sessionStorage.clear();
    vi.restoreAllMocks();
    Object.defineProperty(window, "location", {
      configurable: true,
      value: originalLocation,
    });
  });

  beforeEach(() => {
    if (!URL.revokeObjectURL) {
      URL.revokeObjectURL = vi.fn();
    }

    vi.spyOn(URL, "createObjectURL").mockReturnValue("blob:opponent-photo");
    vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {});
    uploadOpponentImageFile.mockResolvedValue({
      status: "received",
      filename: "opponent-team.jpg",
    });
  });

  it("uploads a selected photo and stays ready for the next capture", async () => {
    const user = userEvent.setup();

    render(<CameraCapturePage />);

    const image = new File(["photo"], "opponent.jpg", { type: "image/jpeg" });
    await user.upload(document.querySelector("input[type='file']"), image);
    await user.click(screen.getByRole("button", { name: /Start Detection System/i }));

    await waitFor(() => {
      expect(uploadOpponentImageFile).toHaveBeenCalledWith(image, {
        backgroundDetection: true,
      });
    });

    expect(
      await screen.findByText(/Battle Prep will update when detection is ready/i),
    ).toBeInTheDocument();
  });

  it("continues after a photo quality warning from upload", async () => {
    const user = userEvent.setup();
    uploadOpponentImageFile.mockResolvedValueOnce({
      status: "needs_retake",
      filename: "opponent-team.jpg",
      message: "Photo quality warning.",
    });

    render(<CameraCapturePage />);

    const image = new File(["photo"], "opponent.jpg", { type: "image/jpeg" });
    await user.upload(document.querySelector("input[type='file']"), image);
    await user.click(screen.getByRole("button", { name: /Start Detection System/i }));

    expect(
      await screen.findByText(/Battle Prep will update when detection is ready/i),
    ).toBeInTheDocument();
    expect(screen.queryByText(/Photo quality warning/i)).not.toBeInTheDocument();
  });

  it("restores the selected photo after the camera page remounts", async () => {
    const user = userEvent.setup();

    const { unmount } = render(<CameraCapturePage />);

    const image = new File(["photo"], "opponent.jpg", { type: "image/jpeg" });
    await user.upload(document.querySelector("input[type='file']"), image);

    await waitFor(() => {
      expect(window.sessionStorage.getItem("champions-insight-camera-photo-draft")).toContain(
        "opponent.jpg",
      );
    });

    unmount();
    render(<CameraCapturePage />);

    expect(await screen.findByAltText("Selected opponent team")).toBeInTheDocument();
    expect(
      await screen.findByText("Photo restored. Tap Start Detection System."),
    ).toBeInTheDocument();
  });
});
