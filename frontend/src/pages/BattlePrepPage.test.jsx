// Tests the main battle preparation workflow.
import "@testing-library/jest-dom/vitest";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import BattlePrepPage, { buildDetectedOpponentMembers } from "./BattlePrepPage.jsx";
import {
  analyzeTeamMatchup,
  calculateDamage,
  detectOpponentTeam,
  fetchBattleAdviceChat,
  fetchHealthStatus,
  fetchLatestOpponentPrediction,
  fetchLatestOpponentUpload,
  fetchPokemonDetails,
  fetchPokemonLevel50Stats,
  fetchPokemonTopMoves,
  fetchTeamSetProbabilities,
  fetchUserTeam,
  saveUserTeam,
  searchPokemon,
  uploadOpponentImageFile,
} from "../api/championsInsightApi.js";

vi.mock("../api/championsInsightApi.js", () => ({
  analyzeTeamMatchup: vi.fn(),
  calculateDamage: vi.fn(),
  detectOpponentTeam: vi.fn(),
  fetchBattleAdviceChat: vi.fn(),
  fetchHealthStatus: vi.fn(),
  fetchLatestOpponentPrediction: vi.fn(),
  fetchLatestOpponentUpload: vi.fn(),
  fetchPokemonDetails: vi.fn(),
  fetchPokemonLevel50Stats: vi.fn(),
  fetchPokemonTopMoves: vi.fn(),
  fetchTeamSetProbabilities: vi.fn(),
  fetchUserTeam: vi.fn(),
  saveUserTeam: vi.fn(),
  searchPokemon: vi.fn(),
  uploadOpponentImageFile: vi.fn(),
  resolveApiUrl: vi.fn((pathOrUrl = "") => pathOrUrl),
}));


describe("partial opponent detection slots", () => {
  it("keeps missing positions editable and preserves later positions", () => {
    const team = buildDetectedOpponentMembers([
      {position: 1, name: "Pikachu"}, {position: 3, name: "Raichu"},
    ]);
    expect(team).toHaveLength(6);
    expect(team.map((member) => member.name)).toEqual(["Pikachu", "", "Raichu", "", "", ""]);
    expect(team[2].id).toBe("opponent-3");
  });
  it("retains all six manual correction slots when no cards are validated", () => {
    const team = buildDetectedOpponentMembers([]);
    expect(team).toHaveLength(6);
    expect(team.every((member) => member.name === "")).toBe(true);
  });
});

describe("BattlePrepPage", () => {
  afterEach(() => {
    cleanup();
  });

  beforeEach(() => {
    vi.clearAllMocks();
    window.history.pushState({}, "", "/");
    fetchHealthStatus.mockResolvedValue({ status: "ok" });
    fetchBattleAdviceChat.mockResolvedValue({
      answer: "Lead Milotic and respect Sneasler pressure.",
      keyPoints: ["Milotic checks sand pressure."],
      recommendedActions: ["Keep Dragonite healthy for speed control turns."],
      uncertaintyNotes: ["Sample size is still limited."],
      source: "rule_based",
      questionType: ["counterplay"],
    });
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
    saveUserTeam.mockImplementation((team) =>
      Promise.resolve({ userId: "local-demo-user", team })
    );
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
    fetchTeamSetProbabilities.mockResolvedValue({
      team: ["Charizard"],
      predictions: {
        Charizard: {
          likelyMoves: [],
          likelyItems: [],
          likelyAbilities: [],
          likelyNature: [],
          predictedSet: {
            item: "Safety Goggles",
            ability: "Blaze",
            nature: "",
            moves: ["Protect", "Heat Wave", "Air Slash", "Tailwind"],
          },
          confidence: 0.8,
          confidenceLabel: "High",
          explanations: [],
          dataQualityNotes: [],
        },
      },
      teamExplanations: [],
      dataQualityNotes: [],
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
  });

  it("checks backend health on load", async () => {
    render(<BattlePrepPage />);

    await waitFor(() => {
      expect(fetchHealthStatus).toHaveBeenCalled();
    });
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

    await user.click(screen.getByRole("button", { name: "Retry Last Uploaded Image" }));

    await waitFor(() => {
      expect(detectOpponentTeam).toHaveBeenCalled();
    });
    expect(await screen.findByText("Charizard")).toBeInTheDocument();
  });

  it("does not poll latest uploads on load", async () => {
    render(<BattlePrepPage />);

    await screen.findByRole("button", { name: /Aegis/ });
    expect(fetchLatestOpponentUpload).not.toHaveBeenCalled();
    expect(detectOpponentTeam).not.toHaveBeenCalled();
  });

  it("restores the saved phone camera prediction from the URL flag", async () => {
    window.history.pushState({}, "", "/?opponentPrediction=1");
    fetchLatestOpponentPrediction.mockResolvedValue({
      hasPrediction: true,
      filename: "phone-opponent.jpg",
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
    expect(await screen.findByText("Phone camera opponent team loaded.")).toBeInTheDocument();
    expect(fetchLatestOpponentPrediction).toHaveBeenCalled();
    expect(detectOpponentTeam).not.toHaveBeenCalled();
    expect(window.location.search).toBe("");
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

    await user.click(screen.getByRole("button", { name: "Retry Last Uploaded Image" }));

    expect(await screen.findByAltText("Venusaur sprite")).toHaveAttribute(
      "src",
      "/api/pokemon/sprite/normal/venusaur.png"
    );
  });

  it("renders an available roster sprite while another details request is still pending", async () => {
    window.history.pushState({}, "", "/?opponentPrediction=1");
    fetchLatestOpponentPrediction.mockResolvedValue({
      hasPrediction: true,
      detectedTeam: [{ name: "Charizard" }, { name: "Venusaur" }],
    });
    fetchPokemonDetails.mockImplementation((name) => name === "Charizard"
      ? new Promise(() => {})
      : Promise.resolve({ name, image: "/venusaur.png", types: ["Grass"] }));
    render(<BattlePrepPage />);
    expect(await screen.findByAltText("Venusaur sprite")).toHaveAttribute("src", "/venusaur.png");
  });

  it("retries missing roster sprites after a details timeout without slot clicks", async () => {
    window.history.pushState({}, "", "/?opponentPrediction=1");
    fetchLatestOpponentPrediction.mockResolvedValue({
      hasPrediction: true,
      detectedTeam: [{ name: "Charizard", image: "/charizard.png" }, { name: "Venusaur" }],
    });
    let venusaurRequests = 0;
    fetchPokemonDetails.mockImplementation((name) => {
      if (name === "Venusaur" && venusaurRequests++ === 0) return Promise.reject(new Error("timeout"));
      return Promise.resolve({ name, image: `/${name.toLowerCase()}.png`, abilities: [], moves: [] });
    });
    render(<BattlePrepPage />);
    expect(await screen.findByAltText("Venusaur sprite")).toHaveAttribute("src", "/venusaur.png");
    expect(venusaurRequests).toBe(2);
  });

  it("keeps opponent details and sprites when tournament moves fail", async () => {
    window.history.pushState({}, "", "/?opponentPrediction=1");
    fetchLatestOpponentPrediction.mockResolvedValue({
      hasPrediction: true, detectedTeam: [{ name: "Charizard", image: "/charizard.png" }],
    });
    fetchPokemonTopMoves.mockRejectedValue(new Error("moves timeout"));
    render(<BattlePrepPage />);
    await waitFor(() => expect(screen.getByLabelText("Opponent ability").tagName).toBe("SELECT"));
    expect(screen.getByRole("button", { name: "Flamethrower" })).toBeInTheDocument();
  });

  it("shows a likely sets trigger for detected opponents", async () => {
    const user = userEvent.setup();
    render(<BattlePrepPage />);

    await user.click(screen.getByRole("button", { name: "Retry Last Uploaded Image" }));

    expect(await screen.findByRole("button", { name: "Likely Sets" })).toBeInTheDocument();
  });

  it("uses the phone camera flow instead of the old guided camera control", () => {
    render(<BattlePrepPage />);

    expect(screen.queryByRole("button", { name: "Open guided camera" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Phone QR Camera" })).toBeInTheDocument();
  });

  it("asks a question from the top bar and shows it in the shared chat modal", async () => {
    const user = userEvent.setup();
    render(<BattlePrepPage />);

    let resolveAdvice;
    fetchBattleAdviceChat.mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveAdvice = resolve;
        }),
    );

    await user.type(screen.getByLabelText("Ask Questions"), "What does this ability do?{enter}");

    await waitFor(() => {
      expect(fetchBattleAdviceChat).toHaveBeenCalled();
    });
    expect(await screen.findByRole("dialog", { name: "Ask Questions" })).toBeInTheDocument();
    expect(screen.getByText("Checking mechanics notes and battle rules...")).toBeInTheDocument();

    resolveAdvice({
      answer: "Lead Milotic and respect Sneasler pressure.",
      keyPoints: ["Milotic checks sand pressure."],
      recommendedActions: ["Keep Dragonite healthy for speed control turns."],
      uncertaintyNotes: ["Sample size is still limited."],
      source: "rule_based",
      questionType: ["counterplay"],
    });

    expect(await screen.findByText("Lead Milotic and respect Sneasler pressure.")).toBeInTheDocument();
  });

  it("opens the ask questions details from the info button", async () => {
    const user = userEvent.setup();
    render(<BattlePrepPage />);

    await user.click(screen.getByRole("button", { name: "Open Ask Questions details" }));

    expect(await screen.findByRole("dialog", { name: "Ask Questions" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Explain Trick Room" })).toBeInTheDocument();
  });

  it("labels factual lookups as Battle Info and shows their source", async () => {
    const user = userEvent.setup();
    fetchBattleAdviceChat.mockResolvedValueOnce({
      intent: "battle_info",
      answer: "Sitrus Berry restores 25% of maximum HP at half HP or less.",
      mechanicsUsed: ["berry.txt"],
      source: "rule_based",
    });
    render(<BattlePrepPage />);
    await user.type(screen.getByLabelText("Ask Questions"), "sitrus berry{enter}");
    expect(await screen.findByText("Battle Info")).toBeInTheDocument();
    expect(screen.getByText("berry.txt")).toBeInTheDocument();
  });

  it("submits suggestion buttons directly into the chat thread", async () => {
    const user = userEvent.setup();
    let resolveAdvice;
    fetchBattleAdviceChat.mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveAdvice = resolve;
        }),
    );

    render(<BattlePrepPage />);

    await user.click(screen.getByRole("button", { name: "Open Ask Questions details" }));
    await user.click(screen.getByRole("button", { name: "Explain Trick Room" }));

    await waitFor(() => {
      expect(fetchBattleAdviceChat).toHaveBeenCalled();
    });
    expect(screen.getByText("Checking mechanics notes and battle rules...")).toBeInTheDocument();

    resolveAdvice({
      answer: "Trick Room reverses normal move order for five turns.",
      keyPoints: [],
      recommendedActions: [],
      uncertaintyNotes: [],
      source: "rule_based",
      questionType: ["gameplay_mechanic"],
    });

    expect(await screen.findByText("Trick Room reverses normal move order for five turns.")).toBeInTheDocument();
  });

  it("lets the user type and submit from inside the chat modal", async () => {
    const user = userEvent.setup();
    let resolveAdvice;
    fetchBattleAdviceChat.mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveAdvice = resolve;
        }),
    );

    render(<BattlePrepPage />);

    await user.click(screen.getByRole("button", { name: "Open Ask Questions details" }));
    await user.type(screen.getByLabelText("Ask Questions in chat"), "What should I lead?");
    await user.click(screen.getByRole("button", { name: "Send battle question" }));

    await waitFor(() => {
      expect(fetchBattleAdviceChat).toHaveBeenCalled();
    });
    expect(screen.getByText("Reviewing matchup pressure and counterplay ideas...")).toBeInTheDocument();

    resolveAdvice({
      answer: "Lead Milotic into their sand core.",
      keyPoints: [],
      recommendedActions: [],
      uncertaintyNotes: [],
      source: "rule_based",
      questionType: ["team_recommendation"],
    });

    expect(await screen.findByText("Lead Milotic into their sand core.")).toBeInTheDocument();
  });

  it("clears the chat question input even when battle advice fails", async () => {
    const user = userEvent.setup();
    fetchBattleAdviceChat.mockRejectedValueOnce(new Error("RAG timeout"));

    render(<BattlePrepPage />);

    await user.click(screen.getByRole("button", { name: "Open Ask Questions details" }));
    const input = screen.getByLabelText("Ask Questions in chat");
    await user.type(input, "what does sleep do");
    await user.click(screen.getByRole("button", { name: "Send battle question" }));

    await waitFor(() => {
      expect(fetchBattleAdviceChat).toHaveBeenCalled();
    });

    expect(input).toHaveValue("");
    expect(await screen.findByText("I couldn't finish that answer right now.")).toBeInTheDocument();
  });
});
