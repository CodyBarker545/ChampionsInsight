// Tests the frontend API helper functions.
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  analyzeTeamMatchup,
  askRagQuestion,
  fetchPokemonLevel50Stats,
  fetchPokemonTopMoves,
  fetchHealthStatus,
  fetchUserTeam,
  resolveApiUrl,
  saveUserTeam,
  searchPokemon,
  uploadOpponentImageFile,
} from "./championsInsightApi.js";


describe("championsInsightApi", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("fetches backend health status", async () => {
    vi.spyOn(global, "fetch").mockResolvedValue({
      json: vi.fn().mockResolvedValue({ status: "ok", app: "Champions Insight" }),
    });

    await expect(fetchHealthStatus()).resolves.toEqual({ status: "ok", app: "Champions Insight" });
    expect(fetch).toHaveBeenCalledWith("/api/health", expect.objectContaining({ signal: expect.any(AbortSignal) }));
  });

  it("fetches the local user team", async () => {
    vi.spyOn(global, "fetch").mockResolvedValue({
      ok: true,
      json: vi.fn().mockResolvedValue({ userId: "local-demo-user", team: [{ name: "Aegis" }] }),
    });

    await expect(fetchUserTeam()).resolves.toEqual({ userId: "local-demo-user", team: [{ name: "Aegis" }] });
    expect(fetch).toHaveBeenCalledWith("/api/user/team", expect.objectContaining({ signal: expect.any(AbortSignal) }));
  });

  it("saves the user team", async () => {
    vi.spyOn(global, "fetch").mockResolvedValue({
      ok: true,
      json: vi.fn().mockResolvedValue({
        userId: "local-demo-user",
        teamName: "Saved Battle Team",
        team: [{ name: "Dragonite" }],
      }),
    });

    await expect(saveUserTeam([{ name: "Dragonite" }])).resolves.toEqual({
      userId: "local-demo-user",
      teamName: "Saved Battle Team",
      team: [{ name: "Dragonite" }],
    });
    expect(fetch).toHaveBeenCalledWith(
      "/api/user/team",
      expect.objectContaining({ method: "PUT" })
    );
  });

  it("returns matchup analysis when the backend succeeds", async () => {
    vi.spyOn(global, "fetch").mockResolvedValue({
      ok: true,
      json: vi.fn().mockResolvedValue({ summary: "Aegis has a favorable opening into Volt." }),
    });

    await expect(analyzeTeamMatchup([{ name: "Aegis" }], "Volt")).resolves.toEqual({
      summary: "Aegis has a favorable opening into Volt.",
    });
  });

  it("asks the backend RAG system a question", async () => {
    vi.spyOn(global, "fetch").mockResolvedValue({
      ok: true,
      json: vi.fn().mockResolvedValue({ question: "What does burn do?", answer: "Burn reduces damage." }),
    });

    await expect(askRagQuestion("What does burn do?")).resolves.toEqual({
      question: "What does burn do?",
      answer: "Burn reduces damage.",
    });
    expect(fetch).toHaveBeenCalledWith("/api/rag/ask", expect.objectContaining({ method: "POST" }));
  });

  it("fetches level 50 Pokemon stats", async () => {
    vi.spyOn(global, "fetch").mockResolvedValue({
      ok: true,
      json: vi.fn().mockResolvedValue({
        name: "Charizard",
        level: 50,
        finalStats: { hp: 153, speed: 120 },
      }),
    });

    await expect(fetchPokemonLevel50Stats("Charizard")).resolves.toEqual({
      name: "Charizard",
      level: 50,
      finalStats: { hp: 153, speed: 120 },
    });
    expect(fetch).toHaveBeenCalledWith(
      "/api/pokemon/Charizard/stats?nature=hardy",
      expect.objectContaining({ signal: expect.any(AbortSignal) })
    );
  });

  it("fetches top tournament moves", async () => {
    vi.spyOn(global, "fetch").mockResolvedValue({
      ok: true,
      json: vi.fn().mockResolvedValue({
        name: "Aerodactyl",
        matchedName: "Aerodactyl",
        moves: [{ move: "Rock Slide", count: 12 }],
      }),
    });

    await expect(fetchPokemonTopMoves("Aerodactyl")).resolves.toEqual({
      name: "Aerodactyl",
      matchedName: "Aerodactyl",
      moves: [{ move: "Rock Slide", count: 12 }],
    });
    expect(fetch).toHaveBeenCalledWith(
      "/api/pokemon/Aerodactyl/moves/top?limit=4",
      expect.objectContaining({ signal: expect.any(AbortSignal) })
    );
  });

  it("searches Pokemon by name", async () => {
    vi.spyOn(global, "fetch").mockResolvedValue({
      ok: true,
      json: vi.fn().mockResolvedValue({
        query: "Aero",
        results: [{ name: "Aerodactyl" }],
      }),
    });

    await expect(searchPokemon("Aero")).resolves.toEqual({
      query: "Aero",
      results: [{ name: "Aerodactyl" }],
    });
    expect(fetch).toHaveBeenCalledWith(
      "/api/pokemon/search?q=Aero&limit=12",
      expect.objectContaining({ signal: expect.any(AbortSignal) })
    );
  });

  it("throws matchup errors from the backend", async () => {
    vi.spyOn(global, "fetch").mockResolvedValue({
      ok: false,
      json: vi.fn().mockResolvedValue({ error: "At least one team member is required." }),
    });

    await expect(analyzeTeamMatchup([], "Volt")).rejects.toThrow("At least one team member is required.");
  });

  it("uploads an opponent image file", async () => {
    vi.spyOn(global, "fetch").mockResolvedValue({
      ok: true,
      json: vi.fn().mockResolvedValue({ status: "received", filename: "team-1234.jpg" }),
    });

    const image = new File(["image"], "team.jpg", { type: "image/jpeg" });

    await expect(uploadOpponentImageFile(image)).resolves.toEqual({
      status: "received",
      filename: "team-1234.jpg",
    });
    expect(fetch).toHaveBeenCalledWith("/api/opponent/image", expect.objectContaining({ method: "POST" }));
  });
});
  it("resolves backend-relative media URLs", () => {
    expect(resolveApiUrl("/api/pokemon/sprite/normal/charizard.png")).toBe(
      "/api/pokemon/sprite/normal/charizard.png"
    );
    expect(resolveApiUrl("https://example.com/sprite.png")).toBe("https://example.com/sprite.png");
  });
