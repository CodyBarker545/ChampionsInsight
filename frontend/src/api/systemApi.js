// API helpers for health checks and saved user teams.
import { readJsonResponse, resolveApiUrl } from "./http.js";


export async function fetchHealthStatus() {
  const response = await fetch(resolveApiUrl("/api/health"), {
    signal: AbortSignal.timeout(8000),
  });

  return response.json();
}


export async function fetchUserTeam() {
  const response = await fetch(resolveApiUrl("/api/user/team"), {
    signal: AbortSignal.timeout(8000),
  });

  return readJsonResponse(response, "Could not load user team.");
}


export async function saveUserTeam(team, teamName = "Saved Battle Team") {
  const response = await fetch(resolveApiUrl("/api/user/team"), {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    signal: AbortSignal.timeout(12000),
    body: JSON.stringify({
      userId: "local-demo-user",
      teamName,
      team,
    }),
  });

  return readJsonResponse(response, "Could not save user team.");
}
