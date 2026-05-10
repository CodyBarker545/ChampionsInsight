// Tests the API status indicator text.
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import StatusPill from "./StatusPill.jsx";


describe("StatusPill", () => {
  it("renders the API status", () => {
    render(<StatusPill status="online" />);

    expect(screen.getByText("API online")).toBeInTheDocument();
  });
});
