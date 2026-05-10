// Public API surface kept for existing imports.
export { resolveApiUrl } from "./http.js";
export {
  fetchHealthStatus,
  fetchUserTeam,
  saveUserTeam,
} from "./systemApi.js";
export {
  analyzeTeamMatchup,
  calculateDamage,
} from "./battleApi.js";
export {
  askRagQuestion,
} from "./ragApi.js";
export {
  detectOpponentTeam,
  fetchLatestOpponentPrediction,
  fetchLatestOpponentUpload,
  uploadOpponentImageFile,
} from "./opponentApi.js";
export {
  fetchPokedexEntries,
  fetchPokedexEntry,
  fetchPokemonDetails,
  fetchPokemonLevel50Stats,
  fetchPokemonTopMoves,
  searchPokemon,
} from "./pokemonApi.js";
