import "@testing-library/jest-dom/vitest";
import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import PokedexPage from "./PokedexPage.jsx";
import {
  fetchPokedexEntries,
  fetchPokedexEntry,
} from "../api/championsInsightApi.js";

vi.mock("../api/championsInsightApi.js", () => ({
  fetchPokedexEntries: vi.fn(),
  fetchPokedexEntry: vi.fn(),
  resolveApiUrl: vi.fn((path = "") => path),
}));

describe("PokedexPage", () => {
  afterEach(() => {
    cleanup();
  });

  beforeEach(() => {
    vi.clearAllMocks();
    fetchPokedexEntries.mockResolvedValue({
      pokemon: [
        {
          id: 3,
          name: "Venusaur",
          speciesName: "Venusaur",
          types: ["grass", "poison"],
          baseStats: {},
          spriteUrl: "/venusaur.png",
        },
      ],
    });
    fetchPokedexEntry.mockResolvedValue({
      id: 3,
      name: "Venusaur",
      speciesName: "Venusaur",
      types: ["grass", "poison"],
      baseStats: {},
      spriteUrl: "/venusaur.png",
      moves: [
        {
          name: "leaf-storm",
          displayName: "Leaf Storm",
          type: "grass",
          category: "special",
          effect: "130 power, 90% accuracy, 5 PP",
        },
        {
          name: "tackle",
          displayName: "Tackle",
          type: "normal",
          category: "physical",
          effect: "40 power, 100% accuracy, 35 PP",
        },
      ],
      abilities: [],
      usage: {},
    });
  });

  it("shows searchable detailed moves in the Pokemon details modal", async () => {
    const user = userEvent.setup();

    render(<PokedexPage onBack={() => {}} />);

    await user.click(await screen.findByRole("button", { name: /Venusaur/i }));
    await user.click(await screen.findByRole("button", { name: "moves" }));

    const leafStormRow = screen.getByText("Leaf Storm").closest("article");
    expect(leafStormRow).toBeInTheDocument();
    expect(within(leafStormRow).getByText("Grass")).toBeInTheDocument();
    expect(within(leafStormRow).getByText("Special")).toBeInTheDocument();
    expect(within(leafStormRow).getByText("130 power, 90% accuracy, 5 PP")).toBeInTheDocument();

    await user.type(screen.getByLabelText("Search Moves"), "normal");

    await waitFor(() => {
      const moveList = screen.getByText("Tackle").closest(".pokedex-move-list");
      expect(within(moveList).queryByText("Leaf Storm")).not.toBeInTheDocument();
    });
    expect(screen.getByText("Tackle")).toBeInTheDocument();
  });

  it("cycles through forms with one form button", async () => {
    const user = userEvent.setup();
    fetchPokedexEntries.mockResolvedValue({
      pokemon: [
        {
          id: 6,
          name: "Charizard",
          speciesName: "Charizard",
          formApiName: "charizard",
          types: ["fire", "flying"],
          baseStats: {},
          spriteUrl: "/charizard.png",
        },
        {
          id: 6,
          name: "Mega Charizard X",
          speciesName: "Charizard",
          formApiName: "charizard-mega-x",
          types: ["fire", "dragon"],
          baseStats: {},
          spriteUrl: "/charizard-x.png",
        },
        {
          id: 6,
          name: "Mega Charizard Y",
          speciesName: "Charizard",
          formApiName: "charizard-mega-y",
          types: ["fire", "flying"],
          baseStats: {},
          spriteUrl: "/charizard-y.png",
        },
      ],
    });
    fetchPokedexEntry.mockImplementation((name) => Promise.resolve({
      id: 6,
      name,
      speciesName: "Charizard",
      formApiName: name.toLowerCase().replaceAll(" ", "-"),
      types: ["fire", "flying"],
      baseStats: {},
      spriteUrl: `/${name.toLowerCase().replaceAll(" ", "-")}.png`,
      moves: [],
      abilities: [],
      usage: {},
    }));

    render(<PokedexPage onBack={() => {}} />);

    await user.click(await screen.findByRole("button", { name: /Charizard/i }));

    const formButton = await screen.findByRole("button", {
      name: /Current form Base/i,
    });
    expect(screen.queryByRole("button", { name: "Mega X" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Mega Y" })).not.toBeInTheDocument();

    await user.click(formButton);

    await waitFor(() => {
      expect(fetchPokedexEntry).toHaveBeenLastCalledWith("Mega Charizard X");
    });
  });
});
