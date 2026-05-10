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

  it("uploads a selected photo and opens the battle page after detection is ready", async () => {
    const user = userEvent.setup();
    const assignedUrls = [];
    Object.defineProperty(window, "location", {
      configurable: true,
      value: {
        ...originalLocation,
        set href(value) {
          assignedUrls.push(value);
        },
        get href() {
          return "http://localhost:3000/camera";
        },
      },
    });

    render(<CameraCapturePage />);

    const image = new File(["photo"], "opponent.jpg", { type: "image/jpeg" });
    await user.upload(document.querySelector("input[type='file']"), image);
    await user.click(screen.getByRole("button", { name: /Start Detection System/i }));

    await waitFor(() => {
      expect(uploadOpponentImageFile).toHaveBeenCalledWith(image);
    });

    expect(assignedUrls).toEqual(["/?opponentPrediction=1"]);
  });
});
