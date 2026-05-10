// Tests the main battle preparation workflow.
import "@testing-library/jest-dom/vitest";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import BattlePrepPage from "./BattlePrepPage.jsx";
import {
  analyzeTeamMatchup,
  askRagQuestion,
  calculateDamage,
  detectOpponentTeam,
  fetchHealthStatus,
  fetchLatestOpponentPrediction,
  fetchLatestOpponentUpload,
  fetchPokemonDetails,
  fetchPokemonLevel50Stats,
  fetchPokemonTopMoves,
  fetchUserTeam,
  searchPokemon,
  uploadOpponentImageFile,
} from "../api/championsInsightApi.js";

vi.mock("../api/championsInsightApi.js", () => ({
  analyzeTeamMatchup: vi.fn(),
  askRagQuestion: vi.fn(),
  calculateDamage: vi.fn(),
  detectOpponentTeam: vi.fn(),
  fetchHealthStatus: vi.fn(),
  fetchLatestOpponentPrediction: vi.fn(),
  fetchLatestOpponentUpload: vi.fn(),
  fetchPokemonDetails: vi.fn(),
  fetchPokemonLevel50Stats: vi.fn(),
  fetchPokemonTopMoves: vi.fn(),
  fetchUserTeam: vi.fn(),
  searchPokemon: vi.fn(),
  uploadOpponentImageFile: vi.fn(),
  resolveApiUrl: vi.fn((pathOrUrl = "") => pathOrUrl),
}));


describe("BattlePrepPage", () => {
  afterEach(() => {
    cleanup();
  });

  beforeEach(() => {
    fetchHealthStatus.mockResolvedValue({ status: "ok" });
    fetchLatestOpponentPrediction.mockResolvedValue({
      hasPrediction: false,
      detectedTeam: [],
    });
    fetchLatestOpponentUpload.mockResolvedValue({
      filename: "latest-opponent.jpg",
      modifiedTime: 1,
      sizeBytes: 2048,
    });
    searchPokemon.mockResolvedValue({ query: "", results: [] });
    uploadOpponentImageFile.mockResolvedValue({
      status: "received",
      filename: "new-opponent.jpg",
      originalFilename: "new-opponent.jpg",
      contentType: "image/jpeg",
      sizeBytes: 2048,
      message: "Image received.",
      quality: { canAnalyze: true },
      detectedTeam: [],
      detectionError: "",
    });
    fetchUserTeam.mockResolvedValue({
      userId: "local-demo-user",
      team: [
        {
          id: "aegis",
          name: "Aegis",
          image: "",
          ability: "Battle Armor",
          item: "Focus Sash",
          stats: { hp: 161, attack: 177, defense: 121, specialAttack: 72, specialDefense: 95, speed: 152 },
          moves: ["Power Strike", "Iron Guard"],
        },
      ],
    });
    fetchPokemonDetails.mockResolvedValue({
      name: "Charizard",
      image: "/api/pokemon/sprite/normal/charizard.png",
      types: ["Fire", "Flying"],
      baseStats: { hp: 78, attack: 84, defense: 78, specialAttack: 109, specialDefense: 85, speed: 100 },
      abilities: ["Blaze", "Drought", "Solar Power", "Tough Claws"],
      moves: ["Flamethrower"],
    });
    fetchPokemonLevel50Stats.mockResolvedValue({
      name: "Charizard",
      level: 50,
      nature: "hardy",
      baseStats: { hp: 78, attack: 84, defense: 78, special_attack: 109, special_defense: 85, speed: 100 },
      finalStats: { hp: 153, attack: 104, defense: 98, special_attack: 129, special_defense: 105, speed: 120 },
    });
    fetchPokemonTopMoves.mockResolvedValue({
      name: "Charizard",
      matchedName: "Charizard",
      moves: [
        { move: "Protect", count: 88 },
        { move: "Heat Wave", count: 72 },
        { move: "Air Slash", count: 41 },
        { move: "Tailwind", count: 39 },
      ],
    });
    analyzeTeamMatchup.mockResolvedValue({
      summary: "Aegis has a favorable opening into Volt.",
      speed: { result: "Your lead is faster", yourSpeed: 152, opponentSpeed: 128 },
      damage: { range: "42% - 51%", koChance: "Guaranteed 3HKO" },
      recommendations: ["Lead with your fastest attacker."],
    });
    calculateDamage.mockResolvedValue({
      attacker: "Aegis",
      defender: "Opponent lead",
      speed: { result: "Aegis moves first." },
      damage: {
        percentRange: "42% - 51%",
        notes: [],
      },
    });
    detectOpponentTeam.mockResolvedValue({
      detectedTeam: [
        {
          name: "Charizard",
          pokemonName: "Charizard",
          image: "/api/pokemon/sprite/normal/charizard.png",
          abilities: ["Blaze", "Drought"],
          baseStats: { hp: 78, attack: 84, defense: 78, specialAttack: 109, specialDefense: 85, speed: 100 },
          finalStats: { hp: 153, attack: 104, defense: 98, specialAttack: 129, specialDefense: 105, speed: 120 },
        },
      ],
    });
    askRagQuestion.mockResolvedValue({
      question: "What does this ability do?",
      answer: "The RAG system retrieves relevant battle-mechanics context.",
      source: "Local RAG Knowledge Base",
    });
  });

  it("shows API online after health check", async () => {
    render(<BattlePrepPage />);

    expect(await screen.findByText("API online")).toBeInTheDocument();
  });

  it("does not show the old analyze matchup button", async () => {
    render(<BattlePrepPage />);

    await screen.findByRole("button", { name: /Aegis/ });
    expect(screen.queryByRole("button", { name: "Analyze matchup" })).not.toBeInTheDocument();
  });

  it("loads backend opponent abilities after typing a Pokemon name", async () => {
    const user = userEvent.setup();
    render(<BattlePrepPage />);

    await screen.findByRole("button", { name: /Aegis/ });
    expect(screen.queryByRole("button", { name: "Load Pokemon Data" })).not.toBeInTheDocument();
    await user.clear(screen.getByLabelText("Opponent name"));
    await user.type(screen.getByLabelText("Opponent name"), "Charizard");

    await waitFor(() => {
      expect(fetchPokemonDetails).toHaveBeenCalledWith("Charizard");
    });
    await waitFor(() => {
      expect(fetchPokemonTopMoves).toHaveBeenCalledWith("Charizard", 4);
    });

    let abilitySelect;
    await waitFor(() => {
      abilitySelect = screen.getByLabelText("Opponent ability");
      expect(abilitySelect.tagName).toBe("SELECT");
    });
    expect(Array.from(abilitySelect.options).map((option) => option.value)).toEqual(["Blaze", "Drought", "Solar Power", "Tough Claws"]);
    expect(await screen.findByRole("button", { name: "Protect" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Heat Wave" })).toBeInTheDocument();
  });

  it("sends the displayed opponent SpD to the damage calculator for special moves", async () => {
    const user = userEvent.setup();
    fetchPokemonDetails.mockResolvedValue({
      name: "Toxapex",
      image: "/api/pokemon/sprite/normal/toxapex.png",
      types: ["Poison", "Water"],
      baseStats: {
        hp: 50,
        attack: 63,
        defense: 152,
        specialAttack: 53,
        specialDefense: 142,
        speed: 35,
      },
      abilities: ["Merciless", "Limber", "Regenerator"],
      moves: ["Recover", "Haze", "Wide Guard", "Baneful Bunker"],
    });
    fetchPokemonLevel50Stats.mockResolvedValue({
      name: "Toxapex",
      level: 50,
      nature: "hardy",
      baseStats: {
        hp: 50,
        attack: 63,
        defense: 152,
        special_attack: 53,
        special_defense: 142,
        speed: 35,
      },
      finalStats: {
        hp: 125,
        attack: 83,
        defense: 172,
        special_attack: 73,
        special_defense: 162,
        speed: 55,
      },
    });
    fetchPokemonTopMoves.mockResolvedValue({
      name: "Toxapex",
      moves: [
        { move: "Recover", count: 10 },
        { move: "Haze", count: 8 },
      ],
    });

    render(
      <BattlePrepPage
        initialTeam={[
          {
            id: "dragonite-mega",
            name: "Dragonite Mega",
            ability: "Inner Focus",
            item: "",
            types: ["Dragon", "Flying"],
            stats: {
              hp: 198,
              attack: 138,
              defense: 119,
              specialAttack: 158,
              specialDefense: 121,
              speed: 105,
            },
            moves: ["Hurricane"],
          },
        ]}
      />
    );

    await screen.findByRole("button", { name: "Hurricane" });
    await user.clear(screen.getByLabelText("Opponent name"));
    await user.type(screen.getByLabelText("Opponent name"), "Toxapex");

    await waitFor(() => {
      const toxapexCall = calculateDamage.mock.calls.find(([payload]) =>
        payload.defender.name === "Toxapex" &&
        payload.move.name === "Hurricane"
      );

      expect(toxapexCall?.[0].attacker.stats).toEqual({
        hp: 198,
        attack: 138,
        defense: 119,
        special_attack: 158,
        special_defense: 121,
        speed: 105,
      });
      expect(toxapexCall?.[0].defender.stats).toEqual({
        hp: 125,
        attack: 83,
        defense: 172,
        special_attack: 73,
        special_defense: 162,
        speed: 55,
      });
      expect(toxapexCall?.[0].defender.maxHp).toBe(125);
    });
  });

  it("detects the existing uploaded opponent image", async () => {
    const user = userEvent.setup();
    render(<BattlePrepPage />);

    await user.click(screen.getByRole("button", { name: "Detect Existing Uploaded Opponent" }));

    await waitFor(() => {
      expect(detectOpponentTeam).toHaveBeenCalled();
    });
    expect(await screen.findByText("Charizard")).toBeInTheDocument();
  });

  it("restores the latest stored opponent prediction on refresh", async () => {
    fetchLatestOpponentPrediction.mockResolvedValue({
      hasPrediction: true,
      filename: "stored-opponent.jpg",
      quality: { canAnalyze: true },
      detectedTeam: [
        {
          name: "Charizard",
          pokemonName: "Charizard",
          image: "/api/pokemon/sprite/normal/charizard.png",
          abilities: ["Blaze", "Drought"],
          baseStats: { hp: 78, attack: 84, defense: 78, specialAttack: 109, specialDefense: 85, speed: 100 },
          finalStats: { hp: 153, attack: 104, defense: 98, specialAttack: 129, specialDefense: 105, speed: 120 },
        },
      ],
    });

    render(<BattlePrepPage />);

    expect(await screen.findByText("Charizard")).toBeInTheDocument();
    expect(await screen.findByText("Restored latest opponent prediction from stored-opponent.jpg.")).toBeInTheDocument();
  });

  it("loads detected opponent sprites without clicking each Pokemon", async () => {
    const user = userEvent.setup();
    detectOpponentTeam.mockResolvedValue({
      detectedTeam: [
        {
          name: "Charizard",
          pokemonName: "Charizard",
          image: "/api/pokemon/sprite/normal/charizard.png",
        },
        {
          name: "Venusaur",
          pokemonName: "Venusaur",
        },
      ],
    });
    fetchPokemonDetails.mockImplementation((name) =>
      Promise.resolve({
        name,
        image: `/api/pokemon/sprite/normal/${name.toLowerCase()}.png`,
        types: name === "Venusaur" ? ["Grass", "Poison"] : ["Fire", "Flying"],
        abilities: [],
        moves: [],
      })
    );

    render(<BattlePrepPage />);

    await user.click(screen.getByRole("button", { name: "Detect Existing Uploaded Opponent" }));

    expect(await screen.findByAltText("Venusaur sprite")).toHaveAttribute(
      "src",
      "/api/pokemon/sprite/normal/venusaur.png"
    );
  });

  it("shows a message when guided camera is unavailable", async () => {
    const user = userEvent.setup();
    const originalMediaDevices = navigator.mediaDevices;
    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: undefined,
    });

    render(<BattlePrepPage />);
    await user.click(screen.getByRole("button", { name: "Open guided camera" }));

    expect(screen.getByText("Guided camera is not available in this browser. Use the file picker instead.")).toBeInTheDocument();

    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: originalMediaDevices,
    });
  });

  it("asks a RAG question and displays the popup answer", async () => {
    const user = userEvent.setup();
    render(<BattlePrepPage />);

    await user.type(screen.getByLabelText("Ask Questions to RAG"), "What does this ability do?{enter}");

    await waitFor(() => {
      expect(askRagQuestion).toHaveBeenCalledWith("What does this ability do?");
    });
    expect(await screen.findByRole("dialog", { name: "RAG Response" })).toBeInTheDocument();
    expect(await screen.findByText("The RAG system retrieves relevant battle-mechanics context.")).toBeInTheDocument();
  });
});
