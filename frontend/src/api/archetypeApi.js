// API helpers for tournament archetype classification.
import { readJsonResponse, resolveApiUrl } from "./http.js";


export async function classifyTournamentArchetype(payload, signal = AbortSignal.timeout(12000)) {
  const response = await fetch(resolveApiUrl("/api/tournament/archetypes/classify"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    signal,
    body: JSON.stringify(payload),
  });

  return readJsonResponse(response, "Could not classify tournament archetype.");
}

export async function fetchTournamentArchetypeRules(
  signal = AbortSignal.timeout(12000),
) {
  const response = await fetch(
    resolveApiUrl("/api/tournament/archetypes/rules"),
    { signal },
  );

  return readJsonResponse(
    response,
    "Could not load tournament archetype summaries.",
  );
}

export async function fetchTournamentArchetypeClusters(
  signal = AbortSignal.timeout(12000),
) {
  const response = await fetch(
    resolveApiUrl("/api/tournament/archetypes/clusters"),
    { signal },
  );

  return readJsonResponse(
    response,
    "Could not load tournament archetype clusters.",
  );
}
