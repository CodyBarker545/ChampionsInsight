import { resolveApiUrl, readJsonResponse } from "./http.js";

export async function fetchTeamSetProbabilities(opponentTeam, signal) {
  const response = await fetch(resolveApiUrl("/api/tournament/set-probabilities/team"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ opponentTeam }),
    signal,
  });

  return readJsonResponse(
    response,
    "Could not load likely tournament-backed sets right now.",
  );
}
