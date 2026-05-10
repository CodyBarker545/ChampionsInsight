import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import OpponentPanel from "./OpponentPanel.jsx";

vi.mock("../api/championsInsightApi.js", () => ({
  resolveApiUrl: vi.fn((pathOrUrl = "") => pathOrUrl),
}));

const defaultProps = {
  cameraError: "",
  imagePreview: "",
  imageQuality: null,
  imageStatus: "",
  isUploadingImage: false,
  onSelectOpponent: vi.fn(),
  onSelectOpponentImage: vi.fn(),
  onUploadOpponentImage: vi.fn(),
  opponentTeam: [
    {
      id: "opponent-1",
      name: "Charizard",
      spriteUrl: "/api/pokemon/sprite/normal/charizard.png",
    },
  ],
  selectedImage: null,
  selectedImageDetails: "",
  selectedOpponentIndex: 0,
};

describe("OpponentPanel", () => {
  it("renders opponent names and sprites", () => {
    render(<OpponentPanel {...defaultProps} />);

    expect(screen.getByRole("button", { name: /Charizard/ })).toBeInTheDocument();
    expect(screen.getByAltText("Charizard sprite")).toHaveAttribute(
      "src",
      "/api/pokemon/sprite/normal/charizard.png"
    );
  });
});
