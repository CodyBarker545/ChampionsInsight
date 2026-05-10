// Tests damage, speed, and recommendation result cards.
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import ResultsGrid from "./ResultsGrid.jsx";


describe("ResultsGrid", () => {
  it("shows empty result prompts before analysis", () => {
    render(<ResultsGrid analysis={null} error="" />);

    expect(screen.getByText("Run an analysis to compare speed.")).toBeInTheDocument();
    expect(screen.getByText("Damage ranges will appear here.")).toBeInTheDocument();
  });

  it("shows analysis values", () => {
    render(
      <ResultsGrid
        error=""
        analysis={{
          summary: "Aegis has a favorable opening into Volt.",
          speed: { result: "Your lead is faster", yourSpeed: 152, opponentSpeed: 128 },
          damage: { range: "42% - 51%", koChance: "Guaranteed 3HKO" },
          recommendations: ["Lead with your fastest attacker."],
        }}
      />
    );

    expect(screen.getByText("Your lead is faster")).toBeInTheDocument();
    expect(screen.getByText("42% - 51%")).toBeInTheDocument();
    expect(screen.getByText("Aegis has a favorable opening into Volt.")).toBeInTheDocument();
  });

  it("shows errors", () => {
    render(<ResultsGrid analysis={null} error="Something failed" />);

    expect(screen.getByText("Something failed")).toBeInTheDocument();
  });
});
