import "@testing-library/jest-dom/vitest";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import ArchetypeDashboardPage from "./ArchetypeDashboardPage.jsx";
import {
  fetchTournamentArchetypeClusters,
  fetchTournamentArchetypeRules,
} from "../api/championsInsightApi.js";

vi.mock("../api/championsInsightApi.js", () => ({
  fetchTournamentArchetypeClusters: vi.fn(),
  fetchTournamentArchetypeRules: vi.fn(),
}));

describe("ArchetypeDashboardPage", () => {
  afterEach(() => {
    cleanup();
  });

  beforeEach(() => {
    vi.clearAllMocks();

    fetchTournamentArchetypeRules.mockResolvedValue({
      totalTeamsAnalyzed: 100,
      archetypeSummaries: [
        {
          archetype: "Rain",
          teamCount: 25,
          percentageOfTeams: 25,
          mostCommonPokemon: [{ name: "Pelipper", count: 20 }],
          mostCommonMoves: [{ name: "Hurricane", count: 15 }],
          mostCommonItems: [{ name: "Mystic Water", count: 10 }],
          averageWinRate: 0.52,
        },
      ],
      dataQualityNotes: ["missing items"],
    });

    fetchTournamentArchetypeClusters.mockResolvedValue({
      totalTeamsAnalyzed: 100,
      totalClusters: 2,
      clustersByParentArchetype: {
        Rain: [
          {
            clusterId: "rain-1",
            suggestedLabel: "Pelipper Rain",
            teamCount: 12,
            commonPokemon: [{ name: "Pelipper", count: 12 }],
            commonMoves: [{ name: "Hurricane", count: 10 }],
            commonItems: [{ name: "Mystic Water", count: 8 }],
            averageTeamStats: {
              averageSpeed: 88.2,
              averageAttack: 92.4,
              averageSpecialAttack: 101.8,
              averageBulk: 250.3,
            },
            averageWinRate: 0.58,
          },
        ],
      },
      dataQualityNotes: ["missing moves"],
    });
  });

  it("renders tournament summaries, notes, and cluster details", async () => {
    render(<ArchetypeDashboardPage onBack={() => {}} />);

    expect(screen.getByText("Loading archetype trends...")).toBeInTheDocument();

    await screen.findByRole("heading", { name: "Archetype Distribution" });

    expect(screen.getByText("100")).toBeInTheDocument();
    expect(screen.getAllByText("Rain").length).toBeGreaterThan(0);
    expect(screen.getByText("A weather team that uses rain to boost Water pressure and enable Swift Swim attackers.")).toBeInTheDocument();
    expect(screen.getAllByText("Pelipper").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Hurricane").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Mystic Water").length).toBeGreaterThan(0);
    expect(screen.getByText("Pelipper Rain")).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText("Missing Items")).toBeInTheDocument();
      expect(screen.getByText("Missing Moves")).toBeInTheDocument();
    });
  });
});
