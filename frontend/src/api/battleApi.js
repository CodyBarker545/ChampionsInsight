// API helpers for matchup and damage workflows.
import { readJsonResponse, resolveApiUrl } from "./http.js";


export async function calculateDamage(payload, signal = AbortSignal.timeout(12000)) {
  const response = await fetch(resolveApiUrl("/api/damage/calculate"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    signal,
    body: JSON.stringify(payload),
  });

  return readJsonResponse(response, "Could not calculate damage.");
}


export async function analyzeTeamMatchup(team, opponent) {
  const response = await fetch(resolveApiUrl("/api/team/analyze"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    signal: AbortSignal.timeout(15000),
    body: JSON.stringify({ team, opponent }),
  });

  return readJsonResponse(response, "Analysis failed.");
}
