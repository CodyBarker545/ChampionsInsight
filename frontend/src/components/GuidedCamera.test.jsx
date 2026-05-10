// Tests guided camera controls and alignment boxes.
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createRef } from "react";
import { describe, expect, it, vi } from "vitest";
import GuidedCamera from "./GuidedCamera.jsx";


describe("GuidedCamera", () => {
  const baseProps = {
    cameraError: "",
    isCameraOpen: false,
    onCapture: vi.fn(),
    onClose: vi.fn(),
    onOpen: vi.fn(),
    videoRef: createRef(),
    canvasRef: createRef(),
  };

  it("opens the guided camera when requested", async () => {
    const user = userEvent.setup();
    const onOpen = vi.fn();
    render(<GuidedCamera {...baseProps} onOpen={onOpen} />);

    await user.click(screen.getByRole("button", { name: "Open guided camera" }));

    expect(onOpen).toHaveBeenCalledTimes(1);
  });

  it("renders six alignment boxes when camera is open", () => {
    render(<GuidedCamera {...baseProps} isCameraOpen />);

    expect(screen.getByText("1")).toBeInTheDocument();
    expect(screen.getByText("6")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Capture" })).toBeInTheDocument();
  });

  it("shows camera errors", () => {
    render(<GuidedCamera {...baseProps} cameraError="Camera blocked" />);

    expect(screen.getByText("Camera blocked")).toBeInTheDocument();
  });
});
