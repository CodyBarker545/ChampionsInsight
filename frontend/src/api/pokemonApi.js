// API helpers for Pokemon lookup, stats, moves, and Pokedex data.
import { readJsonResponse, resolveApiUrl } from "./http.js";


export async function fetchPokemonDetails(name) {
  const response = await fetch(
    resolveApiUrl(`/api/pokemon/${encodeURIComponent(name)}`),
    { signal: AbortSignal.timeout(8000) }
  );

  return readJsonResponse(response, "Could not load Pokemon details.");
}


export async function searchPokemon(query, limit = 12) {
  const params = new URLSearchParams({
    q: query,
    limit: String(limit),
  });

  const response = await fetch(
    resolveApiUrl(`/api/pokemon/search?${params.toString()}`),
    { signal: AbortSignal.timeout(8000) }
  );

  return readJsonResponse(response, "Could not search Pokemon.");
}


export async function fetchPokemonLevel50Stats(name, nature = "hardy") {
  const params = new URLSearchParams({ nature });
  const response = await fetch(
    resolveApiUrl(`/api/pokemon/${encodeURIComponent(name)}/stats?${params.toString()}`),
    { signal: AbortSignal.timeout(8000) }
  );

  return readJsonResponse(response, "Could not load level 50 Pokemon stats.");
}


export async function fetchPokemonTopMoves(name, limit = 4) {
  const params = new URLSearchParams({ limit: String(limit) });
  const response = await fetch(
    resolveApiUrl(`/api/pokemon/${encodeURIComponent(name)}/moves/top?${params.toString()}`),
    { signal: AbortSignal.timeout(8000) }
  );

  return readJsonResponse(response, "Could not load tournament moves.");
}


export async function fetchPokedexEntries() {
  const response = await fetch(resolveApiUrl("/api/pokedex"), {
    signal: AbortSignal.timeout(10000),
  });

  return readJsonResponse(response, "Failed to fetch Pokedex entries.");
}


export async function fetchPokedexEntry(name) {
  const encodedName = encodeURIComponent(name);
  const response = await fetch(resolveApiUrl(`/api/pokedex/${encodedName}`), {
    signal: AbortSignal.timeout(10000),
  });

  return readJsonResponse(response, `Failed to fetch Pokedex entry: ${name}`);
}
