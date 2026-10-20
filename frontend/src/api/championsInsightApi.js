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
  fetchBattleAdviceChat,
} from "./battleApi.js";
export {
  classifyTournamentArchetype,
  fetchTournamentArchetypeClusters,
  fetchTournamentArchetypeRules,
} from "./archetypeApi.js";
export {
  fetchTeamSetProbabilities,
} from "./setProbabilityApi.js";
export {
  askRagQuestion,
} from "./ragApi.js";
export {
  detectOpponentTeam,
  detectOpponentTeamWithEmbedding,
  fetchLatestOpponentPrediction,
  fetchLatestOpponentUpload,
  uploadOpponentImageFile,
} from "./opponentApi.js";
export {
  fetchPokedexEntries,
  fetchPokedexEntry,
  fetchPokemonRecommendedTeams,
  fetchPokemonMetaIntelligence,
  fetchPokemonDetails,
  fetchPokemonLevel50Stats,
  fetchPokemonTopMoves,
  searchPokemon,
} from "./pokemonApi.js";
