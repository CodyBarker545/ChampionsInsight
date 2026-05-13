// Defines the main battle preparation page and workflow state.
import React, { useEffect, useRef, useState } from "react";
import { Moon, Sun } from "lucide-react";
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
  resolveApiUrl,
  saveUserTeam,
  searchPokemon,
  uploadOpponentImageFile,
} from "../api/championsInsightApi.js";
import MatchupStage from "../components/MatchupStage.jsx";
import OpponentPanel from "../components/OpponentPanel.jsx";
import RagPopup from "../components/RagPopup.jsx";
import ResultsGrid from "../components/ResultsGrid.jsx";
import StatusPill from "../components/StatusPill.jsx";
import TeamForm from "../components/TeamForm.jsx";
import { formatBytes } from "../utils/formatters.js";
import {
  applyBestCameraTrackSettings,
  captureGuidedCameraBlob,
  guidedCameraVideoConstraints,
} from "../utils/cameraCapture.js";
import PhoneCameraQr from "../components/PhoneCameraQr.jsx";

const emptyMember = {
  id: "",
  name: "",
  species: "",
  form: "",
  formOptions: [],
  image: "",
  spriteUrl: "",
  types: [],
  item: "",
  ability: "",
  abilities: [],
  nature: "",
  stats: {
    hp: 100,
    attack: 100,
    defense: 100,
    specialAttack: 100,
    specialDefense: 100,
    speed: 100,
  },
  baseStats: {},
  finalStats: {},
  statPoints: {},
  moves: [],
  moveOptions: [],
};

const statKeys = [
  "hp",
  "attack",
  "defense",
  "specialAttack",
  "specialDefense",
  "speed",
];

const statLabels = {
  hp: "HP",
  attack: "Atk",
  defense: "Def",
  specialAttack: "SpA",
  specialDefense: "SpD",
  speed: "Spe",
};

const statColors = {
  hp: "#ff4d5a",
  attack: "#ff7a22",
  defense: "#facc15",
  specialAttack: "#4f8fff",
  specialDefense: "#22c55e",
  speed: "#e879f9",
};

const statBarRanges = {
  hp: { min: 110, max: 267 },
  attack: { min: 61, max: 260 },
  defense: { min: 54, max: 310 },
  specialAttack: { min: 31, max: 249 },
  specialDefense: { min: 50, max: 226 },
  speed: { min: 36, max: 222 },
};

const MAX_STAT_POINTS = 32;
const MAX_TOTAL_STAT_POINTS = 66;
const MINIMUM_VISIBLE_STAT_BAR = 10;

const natures = [
  "hardy",
  "lonely",
  "brave",
  "adamant",
  "naughty",
  "bold",
  "docile",
  "relaxed",
  "impish",
  "lax",
  "timid",
  "hasty",
  "serious",
  "jolly",
  "naive",
  "modest",
  "mild",
  "quiet",
  "bashful",
  "rash",
  "calm",
  "gentle",
  "sassy",
  "careful",
  "quirky",
];

const natureEffects = {
  lonely: ["attack", "defense"],
  brave: ["attack", "speed"],
  adamant: ["attack", "specialAttack"],
  naughty: ["attack", "specialDefense"],
  bold: ["defense", "attack"],
  relaxed: ["defense", "speed"],
  impish: ["defense", "specialAttack"],
  lax: ["defense", "specialDefense"],
  timid: ["speed", "attack"],
  hasty: ["speed", "defense"],
  jolly: ["speed", "specialAttack"],
  naive: ["speed", "specialDefense"],
  modest: ["specialAttack", "attack"],
  mild: ["specialAttack", "defense"],
  quiet: ["specialAttack", "speed"],
  rash: ["specialAttack", "specialDefense"],
  calm: ["specialDefense", "attack"],
  gentle: ["specialDefense", "defense"],
  sassy: ["specialDefense", "speed"],
  careful: ["specialDefense", "specialAttack"],
};

function normalizeLookupName(value = "") {
  return String(value)
    .toLowerCase()
    .replace(/[^a-z0-9]/g, "");
}

function normalizeStats(stats = {}) {
  return {
    hp: Number(stats.hp ?? 0),
    attack: Number(stats.attack ?? 0),
    defense: Number(stats.defense ?? 0),
    specialAttack: Number(stats.specialAttack ?? stats.special_attack ?? 0),
    specialDefense: Number(stats.specialDefense ?? stats.special_defense ?? 0),
    speed: Number(stats.speed ?? 0),
  };
}

function ensureFourMoves(moves = []) {
  return [...moves, "", "", "", ""].slice(0, 4);
}

function uniqueOptions(primaryOptions = [], fallbackOptions = []) {
  return Array.from(
    new Set([...(primaryOptions ?? []), ...(fallbackOptions ?? [])]),
  ).filter(Boolean);
}

function formatNature(nature) {
  const displayName = nature.charAt(0).toUpperCase() + nature.slice(1);
  const effect = natureEffects[nature];

  if (!effect) {
    return `${displayName} (Neutral)`;
  }

  return `${displayName} (+${statLabels[effect[0]]} / -${statLabels[effect[1]]})`;
}

function clampStatPointInvestment(
  currentPoints = {},
  statName,
  requestedValue,
) {
  const requestedPoints = Math.max(0, Number(requestedValue) || 0);
  const otherPointsTotal = statKeys.reduce((total, currentStatName) => {
    if (currentStatName === statName) {
      return total;
    }

    return total + Number(currentPoints[currentStatName] ?? 0);
  }, 0);
  const remainingTotalPoints = Math.max(
    0,
    MAX_TOTAL_STAT_POINTS - otherPointsTotal,
  );

  return Math.min(requestedPoints, MAX_STAT_POINTS, remainingTotalPoints);
}

function emptyStatPoints() {
  return Object.fromEntries(statKeys.map((statName) => [statName, 0]));
}

function natureMultiplier(statName, nature) {
  const effect = natureEffects[nature];

  if (!effect) {
    return 1;
  }

  if (effect[0] === statName) {
    return 1.1;
  }

  if (effect[1] === statName) {
    return 0.9;
  }

  return 1;
}

function calculateInvestedStat(member, statName, points = 0) {
  const baseStat = Number(member.baseStats?.[statName] ?? 0);

  if (!baseStat) {
    return (
      Number(member.finalStats?.[statName] ?? member.stats?.[statName] ?? 0) +
      points
    );
  }

  if (statName === "hp") {
    return Math.floor(((2 * baseStat + 31) * 50) / 100) + 60 + points;
  }

  const neutralLevel50Stat = Math.floor(((2 * baseStat + 31) * 50) / 100) + 5;
  return Math.floor(
    (neutralLevel50Stat + points) *
      natureMultiplier(statName, member.nature || "hardy"),
  );
}

function calculateStatBarWidth(statName, statValue) {
  const safeValue = Number(statValue ?? 0);
  const range = statBarRanges[statName];

  if (!safeValue || !range) {
    return "0%";
  }

  const rangeSize = Math.max(1, range.max - range.min);
  const scaledValue = (safeValue - range.min) / rangeSize;
  const width =
    MINIMUM_VISIBLE_STAT_BAR +
    Math.max(0, Math.min(1, scaledValue)) *
      (100 - MINIMUM_VISIBLE_STAT_BAR);

  return `${width}%`;
}

function buildInvestedStats(member, statPoints = member.statPoints ?? {}) {
  return Object.fromEntries(
    statKeys.map((statName) => [
      statName,
      calculateInvestedStat(
        member,
        statName,
        Number(statPoints?.[statName] ?? 0),
      ),
    ]),
  );
}

function applyInvestedStats(member) {
  const statPoints = {
    ...emptyStatPoints(),
    ...(member.statPoints ?? {}),
  };

  return {
    ...member,
    statPoints,
    stats: buildInvestedStats(member, statPoints),
  };
}

// Builds a frontend opponent slot from backend detection data.
function buildOpponentMember(slot, index) {
  const pokemon = slot.pokemon ?? {};

  const baseStats = normalizeStats(
    slot.baseStats ?? slot.stats?.baseStats ?? pokemon.baseStats ?? {},
  );

  const finalStats = normalizeStats(
    slot.finalStats ?? slot.stats?.finalStats ?? {},
  );

  const displayName = slot.name ?? slot.pokemonName ?? pokemon.name ?? "";

  return applyInvestedStats({
    ...emptyMember,
    id: `opponent-${slot.position ?? index + 1}`,
    name: displayName === "unknown" ? "" : displayName,

    image: slot.image ?? pokemon.image ?? "",
    spriteUrl: slot.spriteUrl ?? pokemon.spriteUrl ?? "",

    types: slot.types ?? slot.detectedTypes ?? pokemon.types ?? [],

    ability: slot.abilities?.[0] ?? pokemon.abilities?.[0] ?? "",
    abilities: slot.abilities ?? pokemon.abilities ?? [],

    item: "",
    nature: slot.stats?.nature ?? "hardy",

    baseStats,
    finalStats,

    statPoints: emptyStatPoints(),

    moves: slot.moves?.slice(0, 4) ?? pokemon.moves?.slice(0, 4) ?? [],
    moveOptions: pokemon.moves ?? slot.moveOptions ?? slot.moves ?? [],

    detection: {
      confidence: slot.confidence ?? 0,
      matchReason: slot.matchReason ?? "",
      detectedTypes: slot.detectedTypes ?? slot.types ?? [],
      referenceTypes: slot.referenceTypes ?? slot.types ?? [],
      typeEvidence: slot.typeEvidence ?? null,
      typeMismatchWarning: slot.typeMismatchWarning ?? false,
      pokemonTopCandidates: slot.pokemonTopCandidates ?? [],
      referenceImage: slot.referenceImage ?? "",
    },
  });
}

// Converts tournament move usage records into display names.
function formatTournamentMoves(topMovesResult) {
  return (topMovesResult?.moves ?? [])
    .map((moveRecord) => moveRecord.move)
    .filter(Boolean)
    .slice(0, 4);
}

// Applies local Pokemon details to a battle slot.
function mergePokemonDetails(
  member,
  details,
  level50Stats = null,
  topMovesResult = null,
  options = {},
) {
  const baseStats = normalizeStats(
    level50Stats?.baseStats ?? details.baseStats ?? member.baseStats ?? {},
  );
  const finalStats = normalizeStats(
    level50Stats?.finalStats ?? member.finalStats ?? {},
  );
  const tournamentMoves = formatTournamentMoves(topMovesResult);
  const learnedMoves = details.moves?.length
    ? details.moves
    : (member.moveOptions ?? member.moves ?? []);
  const nextMoves =
    options.preserveMoves && member.moves?.some(Boolean)
      ? member.moves
      : tournamentMoves.length > 0
        ? tournamentMoves
        : learnedMoves.slice(0, 4);

  return applyInvestedStats({
    ...member,
    name: level50Stats?.name ?? details.name ?? member.name,
    species: details.species ?? member.species ?? "",
    form: details.form ?? level50Stats?.form ?? member.form ?? "",
    formOptions: details.formOptions ?? member.formOptions ?? [],
    image: details.image || member.image || level50Stats?.image || "",
    spriteUrl:
      details.spriteUrl || member.spriteUrl || level50Stats?.spriteUrl || "",
    types: details.types ?? member.types,
    ability: details.abilities?.[0] ?? member.ability,
    abilities: details.abilities ?? member.abilities ?? [],
    nature: level50Stats?.nature ?? member.nature ?? "hardy",
    baseStats,
    finalStats,
    statPoints: {
      ...emptyStatPoints(),
      ...(member.statPoints ?? {}),
    },
    moves: nextMoves,
    moveOptions: learnedMoves,
  });
}

// Renders the battle preparation workflow.
function BattlePrepPage({
  initialTeam = null,
  onOpenPokedex,
  onTeamSaved,
  theme = "dark",
  onToggleTheme,
}) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const cameraStreamRef = useRef(null);
  const opponentDetailsRequestRef = useRef(0);
  const opponentSpriteHydrationRequestRef = useRef(0);
  const latestOpponentUploadRef = useRef("");
  const latestOpponentPredictionRef = useRef("");
  const latestOpponentPredictionSavedAtRef = useRef(0);

  const [apiStatus, setApiStatus] = useState("checking");

  const [team, setTeam] = useState(
    Array.from({ length: 6 }, (_, index) => ({
      ...emptyMember,
      id: `loading-${index + 1}`,
      name: `Pokemon ${index + 1}`,
    })),
  );

  const [opponentTeam, setOpponentTeam] = useState(
    Array.from({ length: 6 }, (_, index) => ({
      ...emptyMember,
      id: `opponent-${index + 1}`,
      name: index === 0 ? "Opponent lead" : "",
    })),
  );

  const [selectedUserIndex, setSelectedUserIndex] = useState(0);
  const [selectedOpponentIndex, setSelectedOpponentIndex] = useState(0);
  const [selectedMove, setSelectedMove] = useState("");
  const [selectedMoveSource, setSelectedMoveSource] = useState("user");
  const [opponentSearchQuery, setOpponentSearchQuery] = useState("");
  const [opponentSearchResults, setOpponentSearchResults] = useState([]);
  const [isSearchingOpponent, setIsSearchingOpponent] = useState(false);
  const [isLoadingOpponentPokemon, setIsLoadingOpponentPokemon] =
    useState(false);
  const [isUserTeamDialogOpen, setIsUserTeamDialogOpen] = useState(false);
  const [userSearchQuery, setUserSearchQuery] = useState("");
  const [userSearchResults, setUserSearchResults] = useState([]);
  const [isSearchingUserPokemon, setIsSearchingUserPokemon] = useState(false);
  const [isLoadingUserPokemon, setIsLoadingUserPokemon] = useState(false);
  const [isSavingUserTeam, setIsSavingUserTeam] = useState(false);
  const [focusedUserMoveIndex, setFocusedUserMoveIndex] = useState(null);

  const [isRagOpen, setIsRagOpen] = useState(false);
  const [ragQuestion, setRagQuestion] = useState("");
  const [ragSubmittedQuestion, setRagSubmittedQuestion] = useState("");
  const [ragAnswer, setRagAnswer] = useState("");
  const [ragSource, setRagSource] = useState("");
  const [ragError, setRagError] = useState("");
  const [isAskingRag, setIsAskingRag] = useState(false);

  const [analysis, setAnalysis] = useState(null);
  const [error, setError] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const [imagePreview, setImagePreview] = useState("");
  const [imageStatus, setImageStatus] = useState("");
  const [selectedImageDetails, setSelectedImageDetails] = useState("");
  const [selectedImage, setSelectedImage] = useState(null);
  const [isUploadingImage, setIsUploadingImage] = useState(false);

  const [isCameraOpen, setIsCameraOpen] = useState(false);
  const [cameraError, setCameraError] = useState("");

  useEffect(
    () => () => {
      releaseGuidedCameraStream();
    },
    [],
  );

  useEffect(() => {
    fetchHealthStatus()
      .then(() => setApiStatus("online"))
      .catch(() => setApiStatus("offline"));

    if (initialTeam?.length) {
      const hydratedTeam = initialTeam.map(applyInvestedStats);
      setTeam(hydratedTeam);
      setSelectedUserIndex(0);
      setSelectedMove(hydratedTeam[0]?.moves?.[0] ?? "");
      setSelectedMoveSource("user");
      return;
    }

    fetchUserTeam()
      .then((result) => {
        const hydratedTeam = result.team.map(applyInvestedStats);
        setTeam(hydratedTeam);
        setSelectedMove(hydratedTeam[0]?.moves?.[0] ?? "");
        setSelectedMoveSource("user");
      })
      .catch(() => setError("Could not load the locally stored user team."));
  }, [initialTeam]);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const shouldRestorePrediction = params.get("opponentPrediction") === "1";
    const shouldAutoDetect = params.get("autoDetect") === "1";
    const uploadedFilename = params.get("opponentUpload");

    if (!shouldRestorePrediction && (!shouldAutoDetect || !uploadedFilename)) {
      return;
    }

    if (shouldRestorePrediction) {
      void restoreLatestOpponentPrediction({
        loadingMessage: "Loading phone camera opponent team...",
        successMessage: "Phone camera opponent team loaded.",
      });
    } else {
      void loadDetectedOpponentTeam(uploadedFilename, {
        loadingMessage: "Loading opponent team from guided camera...",
        successMessage: "Guided camera opponent team loaded.",
      });
    }

    params.delete("opponentPrediction");
    params.delete("autoDetect");
    params.delete("opponentUpload");
    const nextQuery = params.toString();
    const nextUrl = `${window.location.pathname}${nextQuery ? `?${nextQuery}` : ""}`;
    window.history.replaceState({}, "", nextUrl);
  }, []);

  useEffect(() => {
    const intervalId = window.setInterval(() => {
      void refreshLatestOpponentPrediction();
    }, 3000);

    return () => window.clearInterval(intervalId);
  }, []);

  useEffect(() => {
    const selectedOpponent = opponentTeam[selectedOpponentIndex];
    const selectedName = selectedOpponent?.name?.trim();

    if (
      !selectedName ||
      selectedName.toLowerCase().startsWith("opponent") ||
      (selectedOpponent?.abilities?.length &&
        selectedOpponent?.moveOptions?.length)
    ) {
      return undefined;
    }

    const timeoutId = window.setTimeout(() => {
      loadOpponentPokemonDetails(selectedOpponentIndex);
    }, 350);

    return () => window.clearTimeout(timeoutId);
  }, [
    opponentTeam[selectedOpponentIndex]?.name,
    opponentTeam[selectedOpponentIndex]?.abilities?.length,
    opponentTeam[selectedOpponentIndex]?.moveOptions?.length,
    selectedOpponentIndex,
  ]);

  useEffect(() => {
    const trimmedQuery = opponentSearchQuery.trim();
    if (trimmedQuery.length < 2) {
      setOpponentSearchResults([]);
      return undefined;
    }

    const timeoutId = window.setTimeout(async () => {
      setIsSearchingOpponent(true);
      setError("");

      try {
        const searchResult = await searchPokemon(trimmedQuery, 8);
        setOpponentSearchResults(searchResult.results ?? []);
      } catch (requestError) {
        setError(requestError.message);
      } finally {
        setIsSearchingOpponent(false);
      }
    }, 250);

    return () => window.clearTimeout(timeoutId);
  }, [opponentSearchQuery]);

  useEffect(() => {
    const trimmedQuery = userSearchQuery.trim();
    if (trimmedQuery.length < 2) {
      setUserSearchResults([]);
      return undefined;
    }

    const timeoutId = window.setTimeout(async () => {
      setIsSearchingUserPokemon(true);
      setError("");

      try {
        const searchResult = await searchPokemon(trimmedQuery, 10);
        setUserSearchResults(searchResult.results ?? []);
      } catch (requestError) {
        setError(requestError.message);
      } finally {
        setIsSearchingUserPokemon(false);
      }
    }, 250);

    return () => window.clearTimeout(timeoutId);
  }, [userSearchQuery]);

  useEffect(() => {
    const selectedMember = team[selectedUserIndex];
    const selectedName = selectedMember?.name?.trim();

    if (
      !selectedName ||
      selectedName.toLowerCase().startsWith("pokemon")
    ) {
      return undefined;
    }

    const hasFullEditorOptions =
      selectedMember?.editorOptionsHydrated &&
      selectedMember?.moveOptions?.length;

    if (hasFullEditorOptions) {
      return undefined;
    }

    let isCurrent = true;
    fetchPokemonDetails(selectedName)
      .then((details) => {
        if (!isCurrent) {
          return;
        }

        const selectedKey = normalizeLookupName(selectedName);
        const detailKeys = [details.name, details.species, details.form].map(
          normalizeLookupName,
        );
        if (!detailKeys.includes(selectedKey)) {
          return;
        }

        setTeam((currentTeam) =>
          currentTeam.map((member, memberIndex) =>
            memberIndex === selectedUserIndex
              ? {
                  ...member,
                  species: details.species ?? member.species ?? "",
                  form: details.form ?? member.form ?? "",
                  formOptions: details.formOptions ?? [],
                  image: member.image || details.image || "",
                  spriteUrl: member.spriteUrl || details.spriteUrl || "",
                  abilities: details.abilities?.length
                    ? details.abilities
                    : member.abilities,
                  ability:
                    member.ability || details.abilities?.[0] || "",
                  moveOptions: details.moves?.length
                    ? uniqueOptions(details.moves, member.moves)
                    : uniqueOptions(member.moveOptions, member.moves),
                  editorOptionsHydrated: true,
                }
              : member,
          ),
        );
      })
      .catch(() => {});

    return () => {
      isCurrent = false;
    };
  }, [
    selectedUserIndex,
    team[selectedUserIndex]?.name,
    team[selectedUserIndex]?.abilities?.length,
    team[selectedUserIndex]?.formOptions?.length,
    team[selectedUserIndex]?.moveOptions?.length,
  ]);

  function openUserTeamDialog(index) {
    setSelectedUserIndex(index);
    setSelectedMove(team[index]?.moves?.[0] ?? "");
    setSelectedMoveSource("user");
    setUserSearchQuery("");
    setUserSearchResults([]);
    setFocusedUserMoveIndex(null);
    setIsUserTeamDialogOpen(true);
  }

  function updateSelectedUser(field, value) {
    if (field.startsWith("moves.")) {
      const moveIndex = Number(field.replace("moves.", ""));
      const replacedMove = team[selectedUserIndex]?.moves?.[moveIndex];

      if (selectedMoveSource === "user" && selectedMove === replacedMove) {
        setSelectedMove(value);
      }
    }

    if (field === "nature") {
      setTeam((currentTeam) =>
        currentTeam.map((member, memberIndex) =>
          memberIndex === selectedUserIndex
            ? { ...member, nature: value || "hardy" }
            : member,
        ),
      );

      void updateUserNatureStats(selectedUserIndex, value || "hardy");
      return;
    }

    if (field === "form") {
      void loadUserPokemonForm(value);
      return;
    }

    setTeam((currentTeam) =>
      currentTeam.map((member, memberIndex) => {
        if (memberIndex !== selectedUserIndex) {
          return member;
        }

        if (field.startsWith("statPoints.")) {
          const statName = field.replace("statPoints.", "");
          const clampedValue = clampStatPointInvestment(
            member.statPoints,
            statName,
            value,
          );
          const statPoints = {
            ...emptyStatPoints(),
            ...(member.statPoints ?? {}),
            [statName]: clampedValue,
          };

          return {
            ...member,
            statPoints,
            stats: buildInvestedStats(member, statPoints),
          };
        }

        if (field.startsWith("moves.")) {
          const moveIndex = Number(field.replace("moves.", ""));
          const moves = ensureFourMoves(member.moves);
          moves[moveIndex] = value;

          return {
            ...member,
            moves,
            moveOptions: member.moveOptions ?? [],
          };
        }

        return { ...member, [field]: value };
      }),
    );
  }

  async function updateUserNatureStats(index, nature) {
    const selectedMember = team[index];
    const selectedName = selectedMember?.name?.trim();

    if (!selectedName || selectedName.toLowerCase().startsWith("pokemon")) {
      return;
    }

    setError("");

    try {
      const level50Stats = await fetchPokemonLevel50Stats(selectedName, nature);

      setTeam((currentTeam) =>
        currentTeam.map((member, memberIndex) => {
          if (memberIndex !== index) {
            return member;
          }

          const baseStats = normalizeStats(
            level50Stats.baseStats ?? member.baseStats ?? {},
          );
          const finalStats = normalizeStats(level50Stats.finalStats ?? {});
          const nextMember = {
            ...member,
            nature: level50Stats.nature ?? nature,
            baseStats,
            finalStats,
          };

          return {
            ...nextMember,
            stats: buildInvestedStats(nextMember),
          };
        }),
      );
    } catch (requestError) {
      setError(requestError.message);
    }
  }

  async function setUserPokemonFromSearch(index, pokemonName) {
    setSelectedUserIndex(index);
    setIsLoadingUserPokemon(true);
    setError("");

    try {
      const [details, level50Stats, topMoves] = await Promise.all([
        fetchPokemonDetails(pokemonName),
        fetchPokemonLevel50Stats(pokemonName, "hardy"),
        fetchPokemonTopMoves(pokemonName, 4),
      ]);

      const hydratedMember = mergePokemonDetails(
        {
          ...emptyMember,
          id: `user-${index + 1}-${pokemonName}`,
          nature: "hardy",
          statPoints: emptyStatPoints(),
          moves: [],
          moveOptions: [],
        },
        details,
        level50Stats,
        topMoves,
      );
      hydratedMember.editorOptionsHydrated = true;

      setTeam((currentTeam) =>
        currentTeam.map((member, memberIndex) =>
          memberIndex === index ? hydratedMember : member,
        ),
      );
      setSelectedMove(hydratedMember.moves?.[0] ?? "");
      setSelectedMoveSource("user");
      setUserSearchQuery("");
      setUserSearchResults([]);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setIsLoadingUserPokemon(false);
    }
  }

  function clearUserSlot(index) {
    const clearedMoveWasSelected =
      selectedMoveSource === "user" && team[index]?.moves?.includes(selectedMove);

    setTeam((currentTeam) =>
      currentTeam.map((member, memberIndex) =>
        memberIndex === index
          ? {
              ...emptyMember,
              id: `user-${index + 1}`,
              name: "",
              nature: "hardy",
              statPoints: emptyStatPoints(),
              moves: [],
              moveOptions: [],
            }
          : member,
      ),
    );
    setSelectedUserIndex(index);
    setUserSearchQuery("");
    setUserSearchResults([]);

    if (clearedMoveWasSelected) {
      setSelectedMove("");
    }
  }

  async function saveCurrentUserTeam() {
    setIsSavingUserTeam(true);
    setError("");

    try {
      const savedTeam = await saveUserTeam(
        team.map((member, index) => ({
          ...member,
          id: member.id || `user-${index + 1}`,
          moves: ensureFourMoves(member.moves).filter(Boolean),
        })),
      );
      const hydratedTeam = savedTeam.team.map(applyInvestedStats);
      setTeam(hydratedTeam);
      onTeamSaved?.(hydratedTeam);
      setIsUserTeamDialogOpen(false);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setIsSavingUserTeam(false);
    }
  }

  function filteredUserMoveOptions(moveValue) {
    const selectedMember = team[selectedUserIndex] ?? emptyMember;
    const normalizedValue = String(moveValue ?? "").trim().toLowerCase();
    const moveOptions = uniqueOptions(
      selectedMember.moveOptions,
      selectedMember.moves,
    );

    return moveOptions
      .filter((moveOption) =>
        moveOption.toLowerCase().includes(normalizedValue),
      )
      .slice(0, 8);
  }

  // Updates the selected opponent details from manual inputs.
  function updateSelectedOpponent(field, value) {
    if (field.startsWith("moves.")) {
      const moveIndex = Number(field.replace("moves.", ""));
      const removedMove =
        opponentTeam[selectedOpponentIndex]?.moves?.[moveIndex];

      if (
        !value &&
        selectedMoveSource === "opponent" &&
        selectedMove === removedMove
      ) {
        setSelectedMove("");
      }
    }

    if (field === "nature") {
      setOpponentTeam((currentTeam) =>
        currentTeam.map((member, memberIndex) =>
          memberIndex === selectedOpponentIndex
            ? { ...member, nature: value }
            : member,
        ),
      );

      void updateOpponentNatureStats(selectedOpponentIndex, value);
      return;
    }

    if (field === "form") {
      void loadOpponentPokemonForm(value);
      return;
    }

    setOpponentTeam((currentTeam) =>
      currentTeam.map((member, memberIndex) => {
        if (memberIndex !== selectedOpponentIndex) {
          return member;
        }

        if (field.startsWith("stats.")) {
          const statName = field.replace("stats.", "");

          return {
            ...member,
            stats: {
              ...member.stats,
              [statName]: value,
            },
          };
        }

        if (field.startsWith("statPoints.")) {
          const statName = field.replace("statPoints.", "");
          const clampedValue = clampStatPointInvestment(
            member.statPoints,
            statName,
            value,
          );

          const statPoints = {
            ...member.statPoints,
            [statName]: clampedValue,
          };

          return {
            ...member,
            statPoints,
            stats: buildInvestedStats(member, statPoints),
          };
        }

        if (field.startsWith("moves.")) {
          const moveIndex = Number(field.replace("moves.", ""));
          const moves = [...(member.moves ?? [])].slice(0, 4);
          moves[moveIndex] = value;

          return {
            ...member,
            moves,
            moveOptions: Array.from(
              new Set([...(member.moveOptions ?? []), value]),
            ).filter(Boolean),
          };
        }

        return { ...member, [field]: value };
      }),
    );
  }

  async function updateOpponentNatureStats(index, nature) {
    const selectedOpponent = opponentTeam[index];
    const selectedName = selectedOpponent?.name?.trim();

    if (!selectedName || selectedName.toLowerCase().startsWith("opponent")) {
      return;
    }

    setError("");
    const requestId = opponentDetailsRequestRef.current + 1;
    opponentDetailsRequestRef.current = requestId;

    try {
      const level50Stats = await fetchPokemonLevel50Stats(selectedName, nature);

      if (opponentDetailsRequestRef.current !== requestId) {
        return;
      }

      setOpponentTeam((currentTeam) =>
        currentTeam.map((member, memberIndex) => {
          if (memberIndex !== index) {
            return member;
          }

          const baseStats = normalizeStats(
            level50Stats.baseStats ?? member.baseStats ?? {},
          );
          const finalStats = normalizeStats(level50Stats.finalStats ?? {});
          const nextMember = {
            ...member,
            nature: level50Stats.nature ?? nature,
            baseStats,
            finalStats,
          };

          return {
            ...nextMember,
            stats: buildInvestedStats(nextMember),
          };
        }),
      );
    } catch (requestError) {
      setError(requestError.message);
    }
  }

  async function loadUserPokemonForm(formName) {
    const selectedMember = team[selectedUserIndex];

    if (!formName || !selectedMember?.name?.trim()) {
      return;
    }

    setError("");

    try {
      const [details, level50Stats, topMoves] = await Promise.all([
        fetchPokemonDetails(formName),
        fetchPokemonLevel50Stats(formName, selectedMember.nature || "hardy"),
        fetchPokemonTopMoves(formName, 4),
      ]);

      setTeam((currentTeam) =>
        currentTeam.map((member, memberIndex) =>
          memberIndex === selectedUserIndex
            ? {
                ...mergePokemonDetails(member, details, level50Stats, topMoves, {
                  preserveMoves: true,
                }),
                editorOptionsHydrated: true,
              }
            : member,
        ),
      );
    } catch (requestError) {
      setError(requestError.message);
    }
  }

  async function loadOpponentPokemonForm(
    formName,
    index = selectedOpponentIndex,
  ) {
    const selectedOpponent = opponentTeam[index];

    if (!formName || !selectedOpponent?.name?.trim()) {
      return;
    }

    setError("");
    const requestId = opponentDetailsRequestRef.current + 1;
    opponentDetailsRequestRef.current = requestId;

    try {
      const [details, level50Stats, topMoves] = await Promise.all([
        fetchPokemonDetails(formName),
        fetchPokemonLevel50Stats(formName, selectedOpponent.nature || "hardy"),
        fetchPokemonTopMoves(formName, 4),
      ]);

      if (opponentDetailsRequestRef.current !== requestId) {
        return;
      }

      setOpponentTeam((currentTeam) =>
        currentTeam.map((member, memberIndex) =>
          memberIndex === index
            ? mergePokemonDetails(member, details, level50Stats, topMoves, {
                preserveMoves: true,
              })
            : member,
        ),
      );
    } catch (requestError) {
      setError(requestError.message);
    }
  }

  async function loadOpponentPokemonDetails(index = selectedOpponentIndex) {
    const selectedOpponent = opponentTeam[index];
    const selectedName = selectedOpponent?.name?.trim();

    if (!selectedName) {
      setError("Select or enter an opponent Pokemon first.");
      return;
    }

    setError("");
    const requestId = opponentDetailsRequestRef.current + 1;
    opponentDetailsRequestRef.current = requestId;

    try {
      const [details, level50Stats, topMoves] = await Promise.all([
        fetchPokemonDetails(selectedName),
        fetchPokemonLevel50Stats(
          selectedName,
          selectedOpponent?.nature || "hardy",
        ),
        fetchPokemonTopMoves(selectedName, 4),
      ]);

      if (opponentDetailsRequestRef.current !== requestId) {
        return;
      }

      setOpponentTeam((currentTeam) =>
        currentTeam.map((member, memberIndex) => {
          if (memberIndex !== index) {
            return member;
          }

          return mergePokemonDetails(member, details, level50Stats, topMoves);
        }),
      );
    } catch (requestError) {
      setError(requestError.message);
    }
  }

  async function hydrateDetectedOpponentSprites(detectedMembers) {
    const missingSpriteMembers = detectedMembers
      .map((member, index) => ({ member, index }))
      .filter(({ member }) => {
        const memberName = member.name?.trim();
        return (
          memberName &&
          !memberName.toLowerCase().startsWith("opponent") &&
          !member.spriteUrl &&
          !member.image
        );
      });

    if (!missingSpriteMembers.length) {
      return;
    }

    const requestId = opponentSpriteHydrationRequestRef.current + 1;
    opponentSpriteHydrationRequestRef.current = requestId;

    const hydratedSprites = await Promise.all(
      missingSpriteMembers.map(async ({ member, index }) => {
        try {
          const details = await fetchPokemonDetails(member.name);
          return { details, id: member.id, index, name: member.name };
        } catch {
          return null;
        }
      }),
    );

    if (opponentSpriteHydrationRequestRef.current !== requestId) {
      return;
    }

    setOpponentTeam((currentTeam) =>
      currentTeam.map((member, index) => {
        const hydratedSprite = hydratedSprites.find(
          (result) =>
            result &&
            result.index === index &&
            result.id === member.id &&
            normalizeLookupName(result.name) ===
              normalizeLookupName(member.name),
        );

        if (!hydratedSprite || member.spriteUrl || member.image) {
          return member;
        }

        const details = hydratedSprite.details;
        return {
          ...member,
          species: details.species ?? member.species ?? "",
          form: details.form ?? member.form ?? "",
          formOptions: details.formOptions ?? member.formOptions ?? [],
          image: details.image || member.image || "",
          spriteUrl: details.spriteUrl || member.spriteUrl || "",
          types: member.types?.length ? member.types : (details.types ?? []),
        };
      }),
    );
  }

  async function setOpponentPokemonFromSearch(index, pokemonName) {
    setSelectedOpponentIndex(index);
    setIsLoadingOpponentPokemon(true);
    setError("");

    try {
      const [details, level50Stats, topMoves] = await Promise.all([
        fetchPokemonDetails(pokemonName),
        fetchPokemonLevel50Stats(pokemonName, "hardy"),
        fetchPokemonTopMoves(pokemonName, 4),
      ]);

      const hydratedMember = mergePokemonDetails(
        {
          ...emptyMember,
          id: `opponent-${index + 1}-${pokemonName}`,
          nature: "hardy",
          statPoints: emptyStatPoints(),
          moves: [],
          moveOptions: [],
        },
        details,
        level50Stats,
        topMoves,
      );

      setOpponentTeam((currentTeam) =>
        currentTeam.map((member, memberIndex) =>
          memberIndex === index ? hydratedMember : member,
        ),
      );
      setOpponentSearchQuery("");
      setOpponentSearchResults([]);
      setSelectedMove("");
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setIsLoadingOpponentPokemon(false);
    }
  }

  function clearOpponentSlot(index) {
    const clearedMoveWasSelected =
      selectedMoveSource === "opponent" &&
      opponentTeam[index]?.moves?.includes(selectedMove);

    setOpponentTeam((currentTeam) =>
      currentTeam.map((member, memberIndex) =>
        memberIndex === index
          ? {
              ...emptyMember,
              id: `opponent-${index + 1}`,
              nature: "hardy",
              statPoints: emptyStatPoints(),
              moves: [],
              moveOptions: [],
            }
          : member,
      ),
    );
    setSelectedOpponentIndex(index);
    setOpponentSearchQuery("");
    setOpponentSearchResults([]);

    if (clearedMoveWasSelected) {
      setSelectedMove("");
    }
  }

  // Selects an opponent and loads backend details needed by the center panel.
  async function selectOpponentFromPanel(index) {
    setSelectedOpponentIndex(index);
    setOpponentSearchQuery("");
    setOpponentSearchResults([]);

    const selectedOpponent = opponentTeam[index];
    if (!selectedOpponent?.name?.trim()) {
      return;
    }

    if (
      selectedOpponent.abilities?.length &&
      selectedOpponent.moveOptions?.length &&
      statKeys.every(
        (statName) => Number(selectedOpponent.stats?.[statName] ?? 0) > 0,
      ) &&
      (selectedOpponent.spriteUrl || selectedOpponent.image)
    ) {
      return;
    }

    await loadOpponentPokemonDetails(index);
  }
  async function calculateMoveDamage(move, source = "user") {
    const moveName = typeof move === "string" ? move : move?.name;

    if (!moveName?.trim()) {
      return;
    }

    setSelectedMove(moveName);
    setSelectedMoveSource(source);
    setIsAnalyzing(true);
    setError("");
    setAnalysis(null);

    try {
      const attacker =
        source === "opponent"
          ? opponentTeam[selectedOpponentIndex]
          : team[selectedUserIndex];

      const defender =
        source === "opponent"
          ? team[selectedUserIndex]
          : opponentTeam[selectedOpponentIndex];

      if (!attacker?.name?.trim()) {
        throw new Error("Select an attacking Pokémon.");
      }

      if (!defender?.name?.trim()) {
        throw new Error("Select a defending Pokémon.");
      }

      const payload = {
        attacker: {
          name: attacker.name,
          level: 50,
          types: attacker.types ?? [],
          ability: attacker.ability ?? "",
          item: attacker.item ?? "",
          status: "",
          isGrounded: true,
          stats: {
            hp: Number(attacker.stats?.hp ?? 0),
            attack: Number(attacker.stats?.attack ?? 0),
            defense: Number(attacker.stats?.defense ?? 0),
            special_attack: Number(attacker.stats?.specialAttack ?? 0),
            special_defense: Number(attacker.stats?.specialDefense ?? 0),
            speed: Number(attacker.stats?.speed ?? 0),
          },
          maxHp: Number(attacker.stats?.hp ?? 0),
          currentHp: Number(attacker.stats?.hp ?? 0),
          boosts: {},
        },

        defender: {
          name: defender.name,
          level: 50,
          types: defender.types ?? [],
          ability: defender.ability ?? "",
          item: defender.item ?? "",
          status: "",
          isGrounded: true,
          stats: {
            hp: Number(defender.stats?.hp ?? 0),
            attack: Number(defender.stats?.attack ?? 0),
            defense: Number(defender.stats?.defense ?? 0),
            special_attack: Number(defender.stats?.specialAttack ?? 0),
            special_defense: Number(defender.stats?.specialDefense ?? 0),
            speed: Number(defender.stats?.speed ?? 0),
          },
          maxHp: Number(defender.stats?.hp ?? 0),
          currentHp: Number(defender.stats?.hp ?? 0),
          boosts: {},
        },

        move: {
          name: moveName,
        },

        field: {
          weather: "",
          terrain: "",
          trickRoom: false,
        },
      };

      const result = await calculateDamage(payload);

      console.log("DAMAGE RESULT:", result);
      setAnalysis(result);
    } catch (requestError) {
      setError(requestError.message || "Damage calculation failed.");
    } finally {
      setIsAnalyzing(false);
    }
  }
  // Sends the current team to the backend for matchup analysis.
  async function analyzeTeam(event) {
    event.preventDefault();

    setIsAnalyzing(true);
    setError("");
    setAnalysis(null);

    try {
      const attacker =
        selectedMoveSource === "opponent"
          ? opponentTeam[selectedOpponentIndex]
          : team[selectedUserIndex];

      const defender =
        selectedMoveSource === "opponent"
          ? team[selectedUserIndex]
          : opponentTeam[selectedOpponentIndex];

      const moveName = selectedMove;

      if (!attacker?.name?.trim()) {
        throw new Error("Select an attacking Pokémon.");
      }

      if (!defender?.name?.trim()) {
        throw new Error("Select a defending Pokémon.");
      }

      if (!moveName?.trim()) {
        throw new Error("Select a move before calculating damage.");
      }

      const result = await calculateDamage({
        attacker: {
          name: attacker.name,
          level: 50,
          types: attacker.types ?? [],
          ability: attacker.ability ?? "",
          item: attacker.item ?? "",
          status: "",
          isGrounded: true,
          stats: {
            hp: Number(attacker.stats?.hp ?? 0),
            attack: Number(attacker.stats?.attack ?? 0),
            defense: Number(attacker.stats?.defense ?? 0),
            special_attack: Number(attacker.stats?.specialAttack ?? 0),
            special_defense: Number(attacker.stats?.specialDefense ?? 0),
            speed: Number(attacker.stats?.speed ?? 0),
          },
          maxHp: Number(attacker.stats?.hp ?? 0),
          currentHp: Number(attacker.stats?.hp ?? 0),
          boosts: {},
        },

        defender: {
          name: defender.name,
          level: 50,
          types: defender.types ?? [],
          ability: defender.ability ?? "",
          item: defender.item ?? "",
          status: "",
          isGrounded: true,
          stats: {
            hp: Number(defender.stats?.hp ?? 0),
            attack: Number(defender.stats?.attack ?? 0),
            defense: Number(defender.stats?.defense ?? 0),
            special_attack: Number(defender.stats?.specialAttack ?? 0),
            special_defense: Number(defender.stats?.specialDefense ?? 0),
            speed: Number(defender.stats?.speed ?? 0),
          },
          maxHp: Number(defender.stats?.hp ?? 0),
          currentHp: Number(defender.stats?.hp ?? 0),
          boosts: {},
        },

        move: {
          name: moveName,
        },

        field: {
          weather: "",
          terrain: "",
          trickRoom: false,
        },
      });

      console.log("DAMAGE RESULT:", result);
      setAnalysis(result);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setIsAnalyzing(false);
    }
  }

  async function detectExistingUploadedOpponentImage() {
    setError("");
    setImageStatus("Finding last uploaded opponent image...");
    setIsUploadingImage(true);

    try {
      const latestUpload = await fetchLatestOpponentUpload();
      const filename =
        latestUpload.filename ||
        latestOpponentUploadRef.current ||
        latestOpponentPredictionRef.current;

      if (!filename) {
        const message = "No previous opponent image is available to retry.";
        setError(message);
        setImageStatus(message);
        return;
      }

      latestOpponentUploadRef.current = filename;

      await loadDetectedOpponentTeam(filename, {
        loadingMessage: `Retrying detection for ${filename}...`,
        successMessage: "Detection complete. Opponent team loaded.",
        emptyMessage: "Detection finished, but no Pokemon were found.",
      });
    } catch (requestError) {
      const message =
        requestError.message || "Could not retry the last uploaded image.";
      setError(message);
      setImageStatus(message);
    } finally {
      setIsUploadingImage(false);
    }
  }

  async function restoreLatestOpponentPrediction(messages = {}) {
    setError("");
    setImageStatus(messages.loadingMessage ?? "Loading latest opponent prediction...");
    setIsUploadingImage(true);

    try {
      const prediction = await fetchLatestOpponentPrediction();

      if (prediction.detectedTeam?.length) {
        const detectedMembers = prediction.detectedTeam.map(buildOpponentMember);
        latestOpponentPredictionSavedAtRef.current = Number(prediction.savedAt ?? 0);
        latestOpponentPredictionRef.current =
          prediction.filename ?? latestOpponentPredictionRef.current;
        latestOpponentUploadRef.current =
          prediction.filename ?? latestOpponentUploadRef.current;
        setOpponentTeam(detectedMembers);
        setSelectedOpponentIndex(0);
        void hydrateDetectedOpponentSprites(detectedMembers);
        setImageStatus(messages.successMessage ?? "Latest opponent prediction loaded.");
      } else {
        setImageStatus("No latest opponent prediction is ready yet.");
      }
    } catch (requestError) {
      setError(requestError.message || "Could not load latest opponent prediction.");
    } finally {
      setIsUploadingImage(false);
    }
  }

  async function refreshLatestOpponentPrediction() {
    try {
      const prediction = await fetchLatestOpponentPrediction();
      const savedAt = Number(prediction.savedAt ?? 0);

      if (!prediction.detectedTeam?.length || savedAt <= latestOpponentPredictionSavedAtRef.current) {
        return;
      }

      const detectedMembers = prediction.detectedTeam.map(buildOpponentMember);
      latestOpponentPredictionSavedAtRef.current = savedAt;
      latestOpponentPredictionRef.current =
        prediction.filename ?? latestOpponentPredictionRef.current;
      latestOpponentUploadRef.current =
        prediction.filename ?? latestOpponentUploadRef.current;
      setOpponentTeam(detectedMembers);
      setSelectedOpponentIndex(0);
      void hydrateDetectedOpponentSprites(detectedMembers);
      setImageStatus("Phone camera opponent team loaded.");
    } catch (_error) {
      // Polling is best-effort so normal battle prep work is not interrupted.
    }
  }

  async function loadDetectedOpponentTeam(filename = null, messages = {}) {
    setError("");
    setImageStatus(messages.loadingMessage ?? "Detecting opponent team...");
    setIsUploadingImage(true);

    try {
      const detectResult = await detectOpponentTeam(filename);

      if (detectResult.detectedTeam?.length) {
        const detectedMembers =
          detectResult.detectedTeam.map(buildOpponentMember);
        latestOpponentPredictionRef.current =
          detectResult.filename ??
          filename ??
          latestOpponentPredictionRef.current;
        latestOpponentUploadRef.current =
          detectResult.filename ?? filename ?? latestOpponentUploadRef.current;
        setOpponentTeam(detectedMembers);
        setSelectedOpponentIndex(0);
        void hydrateDetectedOpponentSprites(detectedMembers);

        setImageStatus(messages.successMessage ?? "Opponent team loaded.");
      } else {
        setImageStatus(
          messages.emptyMessage ??
            "Detection finished, but no Pokemon were found.",
        );
      }
    } catch (requestError) {
      setError(requestError.message || "Opponent detection failed.");
    } finally {
      setIsUploadingImage(false);
    }
  }

  // Stores an image for preview and later upload.
  function setOpponentImage(image) {
    setError("");
    setImageStatus("");
    setSelectedImage(image);
    setImagePreview(URL.createObjectURL(image));
    setSelectedImageDetails(`${image.name} (${formatBytes(image.size)})`);
  }

  // Stores the selected opponent image for preview and upload.
  function selectOpponentImage(event) {
    const image = event.target.files?.[0];

    if (!image) {
      return;
    }

    setOpponentImage(image);
    event.target.value = "";
  }

  // Starts the guided camera preview for aligning opponent team slots.
  async function openGuidedCamera() {
    if (isUploadingImage) {
      return;
    }

    setError("");
    setCameraError("");
    releaseGuidedCameraStream();

    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraError(
        "Guided camera is not available in this browser. Use the file picker instead.",
      );
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: guidedCameraVideoConstraints,
        audio: false,
      });

      await applyBestCameraTrackSettings(stream);

      cameraStreamRef.current = stream;
      setIsCameraOpen(true);

      setTimeout(() => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      }, 0);
    } catch {
      setCameraError(
        "Camera access did not start. On phones, the guided camera usually needs HTTPS; use the file picker if the browser blocks it.",
      );
    }
  }

  function releaseGuidedCameraStream() {
    cameraStreamRef.current?.getTracks().forEach((track) => track.stop());
    cameraStreamRef.current = null;

    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.srcObject = null;
      videoRef.current.removeAttribute("src");
      videoRef.current.load();
    }
  }

  // Stops the guided camera preview.
  function closeGuidedCamera() {
    releaseGuidedCameraStream();
    setIsCameraOpen(false);
  }

  // Captures the current guided camera frame as an uploadable image.
  async function captureGuidedPhoto() {
    if (isUploadingImage) {
      return;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (!video || !canvas || !video.videoWidth || !video.videoHeight) {
      setCameraError("Camera is not ready yet.");
      return;
    }

    setIsUploadingImage(true);
    setImageStatus("Capturing best camera frame...");

    try {
      const { blob, metadata } = await captureGuidedCameraBlob(video, canvas);
      const capturedImage = new File(
        [blob],
        `opponent-team-${Date.now()}-${metadata.outputWidth}x${metadata.outputHeight}.jpg`,
        { type: "image/jpeg" },
      );

      setOpponentImage(capturedImage);
      closeGuidedCamera();

      await uploadOpponentImage(capturedImage);
    } catch (captureError) {
      setCameraError(captureError.message || "Could not capture photo.");
      setIsUploadingImage(false);
    }
  }

  async function uploadOpponentImage(imageOverride = null) {
    const imageToUpload = imageOverride ?? selectedImage;

    if (!imageToUpload) {
      setError("Take or choose a photo before uploading.");
      return;
    }

    if (!(imageToUpload instanceof File)) {
      setError(
        "Upload failed because the selected image is not a File object.",
      );
      return;
    }

    setError("");
    setImageStatus("Uploading photo...");
    setIsUploadingImage(true);

    try {
      const uploadResult = await uploadOpponentImageFile(imageToUpload);

      latestOpponentUploadRef.current =
        uploadResult.filename ?? latestOpponentUploadRef.current;
      latestOpponentPredictionRef.current =
        uploadResult.filename ?? latestOpponentPredictionRef.current;

      if (uploadResult.detectedTeam?.length) {
        const detectedMembers =
          uploadResult.detectedTeam.map(buildOpponentMember);
        setOpponentTeam(detectedMembers);
        setSelectedOpponentIndex(0);
        void hydrateDetectedOpponentSprites(detectedMembers);

        setImageStatus(
          `Saved as ${uploadResult.filename} (${formatBytes(
            uploadResult.sizeBytes,
          )}). Detection complete.`,
        );
      } else {
        setImageStatus(
          `Detection finished for ${uploadResult.filename}, but no Pokemon were found.`,
        );
      }
    } catch (requestError) {
      setError(requestError.message || "Opponent image upload failed.");
    } finally {
      setIsUploadingImage(false);
    }
  }

  // Sends a user question to the RAG explanation system.
  async function askRag(event) {
    event.preventDefault();

    if (!ragQuestion.trim()) {
      return;
    }

    setIsRagOpen(true);
    setRagError("");
    setRagAnswer("");
    setRagSource("");
    setIsAskingRag(true);

    try {
      const result = await askRagQuestion(ragQuestion);

      setRagSubmittedQuestion(result.question);
      setRagAnswer(result.answer);
      setRagSource(result.source);
    } catch (requestError) {
      setRagError(requestError.message);
    } finally {
      setIsAskingRag(false);
    }
  }

  const selectedUserMember = team[selectedUserIndex] ?? emptyMember;
  const selectedUserPointsUsed = statKeys.reduce(
    (total, statName) =>
      total + Number(selectedUserMember.statPoints?.[statName] ?? 0),
    0,
  );
  const selectedUserImage = selectedUserMember.spriteUrl || selectedUserMember.image || "";

  return (
    <main className="app-shell">
      <section className="workspace">
        <header className="topbar">
          <div className="topbar-camera">
            {onOpenPokedex && (
              <button type="button" onClick={onOpenPokedex}>
                Pokédex
              </button>
            )}

            <button
              type="button"
              onClick={detectExistingUploadedOpponentImage}
              disabled={isUploadingImage}
            >
              Retry Last Uploaded Image
            </button>
          </div>

          <form className="rag-top-form" onSubmit={askRag}>
            <input
              aria-label="Ask Questions to RAG"
              value={ragQuestion}
              placeholder="Ask Questions to RAG"
              onChange={(event) => setRagQuestion(event.target.value)}
            />
          </form>
          <div className="topbar-actions">
            {onToggleTheme && (
              <button
                className="theme-toggle"
                type="button"
                onClick={onToggleTheme}
                aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
              >
                {theme === "dark" ? (
                  <Sun size={16} aria-hidden="true" />
                ) : (
                  <Moon size={16} aria-hidden="true" />
                )}
                <span>{theme === "dark" ? "Light" : "Dark"}</span>
              </button>
            )}
            <PhoneCameraQr />
          </div>
        </header>

        <form className="analysis-layout" onSubmit={analyzeTeam}>
          <TeamForm
            onClearMember={clearUserSlot}
            onEditMember={openUserTeamDialog}
            selectedIndex={selectedUserIndex}
            team={team}
            onSelectMember={(index) => {
              setSelectedUserIndex(index);
              setSelectedMove(team[index]?.moves?.[0] ?? "");
              setSelectedMoveSource("user");
            }}
          />

          <MatchupStage
            analysis={analysis}
            isAnalyzing={isAnalyzing}
            onOpponentUpdate={updateSelectedOpponent}
            onUserFormChange={loadUserPokemonForm}
            onAnalyzeLabel={isAnalyzing ? "Analyzing" : "Analyze matchup"}
            onSelectMove={(move, source = "user") => {
              void calculateMoveDamage(move, source);
            }}
            opponentPokemon={opponentTeam[selectedOpponentIndex]}
            selectedMoveSource={selectedMoveSource}
            selectedMove={selectedMove}
            userPokemon={team[selectedUserIndex]}
          />

          <OpponentPanel
            cameraError={cameraError}
            imagePreview={imagePreview}
            imageStatus={imageStatus}
            isAnalyzing={isAnalyzing}
            isCameraOpen={isCameraOpen}
            isUploadingImage={isUploadingImage}
            isLoadingOpponentPokemon={isLoadingOpponentPokemon}
            isSearchingOpponent={isSearchingOpponent}
            onClearOpponent={clearOpponentSlot}
            onAnalyzeLabel={isAnalyzing ? "Analyzing" : "Analyze matchup"}
            onCaptureGuidedPhoto={captureGuidedPhoto}
            onCloseGuidedCamera={closeGuidedCamera}
            onOpenGuidedCamera={openGuidedCamera}
            onSelectOpponent={selectOpponentFromPanel}
            onSelectOpponentImage={selectOpponentImage}
            onSelectOpponentPokemon={setOpponentPokemonFromSearch}
            onOpponentSearchChange={setOpponentSearchQuery}
            onUploadOpponentImage={uploadOpponentImage}
            opponentTeam={opponentTeam}
            opponentSearchQuery={opponentSearchQuery}
            opponentSearchResults={opponentSearchResults}
            selectedImage={selectedImage}
            selectedImageDetails={selectedImageDetails}
            selectedOpponentIndex={selectedOpponentIndex}
            videoRef={videoRef}
            canvasRef={canvasRef}
          />
        </form>

        <ResultsGrid analysis={analysis} error={error} />
      </section>

      {isUserTeamDialogOpen && (
        <div className="team-editor-backdrop" role="presentation">
          <section
            className="team-editor-dialog"
            role="dialog"
            aria-modal="true"
            aria-label={`Edit team slot ${selectedUserIndex + 1}`}
          >
            <header className="team-editor-header">
              <div>
                <span>Team Slot {selectedUserIndex + 1}</span>
                <h2>{selectedUserMember.name || "Build Pokemon"}</h2>
              </div>

              <div className="team-editor-actions">
                <button
                  className="secondary-button"
                  type="button"
                  onClick={() => setIsUserTeamDialogOpen(false)}
                >
                  Close
                </button>
                <button
                  className="primary-button"
                  type="button"
                  disabled={isSavingUserTeam}
                  onClick={saveCurrentUserTeam}
                >
                  {isSavingUserTeam ? "Saving" : "Save Team"}
                </button>
              </div>
            </header>

            <div className="team-editor-grid">
              <aside className="team-editor-picker">
                <label className="team-editor-field">
                  <span>Pokemon</span>
                  <input
                    value={userSearchQuery}
                    placeholder={
                      selectedUserMember.name
                        ? "Search replacement"
                        : "Search Pokemon"
                    }
                    onChange={(event) => setUserSearchQuery(event.target.value)}
                  />
                </label>

                {(isSearchingUserPokemon || userSearchResults.length > 0) && (
                  <div className="team-editor-results">
                    {isSearchingUserPokemon && <span>Searching...</span>}
                    {userSearchResults.map((pokemon) => {
                      const resultImageSource = resolveApiUrl(pokemon.image);

                      return (
                        <button
                          className="team-editor-result"
                          key={pokemon.name}
                          type="button"
                          disabled={isLoadingUserPokemon}
                          onClick={() =>
                            setUserPokemonFromSearch(
                              selectedUserIndex,
                              pokemon.name,
                            )
                          }
                        >
                          <span className="team-editor-result-image">
                            {resultImageSource ? (
                              <img src={resultImageSource} alt="" />
                            ) : (
                              "IMG"
                            )}
                          </span>
                          <span>{pokemon.name}</span>
                        </button>
                      );
                    })}
                  </div>
                )}

                <div className="team-editor-preview">
                  <span className="team-editor-preview-image">
                    {selectedUserImage ? (
                      <img
                        src={resolveApiUrl(selectedUserImage)}
                        alt={`${selectedUserMember.name || "Pokemon"} sprite`}
                      />
                    ) : (
                      "+"
                    )}
                  </span>
                  <div>
                    <strong>{selectedUserMember.name || "Empty slot"}</strong>
                    <small>
                      {selectedUserMember.types?.length
                        ? selectedUserMember.types.join(" / ")
                        : "Choose a Pokemon to fill this slot"}
                    </small>
                  </div>
                </div>

                <div className="team-editor-moves">
                  {ensureFourMoves(selectedUserMember.moves).map(
                    (move, moveIndex) => (
                      <label
                        className="team-editor-field builder-move-field"
                        key={`user-move-${moveIndex}`}
                      >
                        <span>Move {moveIndex + 1}</span>
                        <input
                          value={move}
                          disabled={!selectedUserMember.name}
                          placeholder="Search moves"
                          onFocus={() => setFocusedUserMoveIndex(moveIndex)}
                          onClick={() => setFocusedUserMoveIndex(moveIndex)}
                          onBlur={() => {
                            window.setTimeout(
                              () => setFocusedUserMoveIndex(null),
                              120,
                            );
                          }}
                          onChange={(event) => {
                            setFocusedUserMoveIndex(moveIndex);
                            updateSelectedUser(
                              `moves.${moveIndex}`,
                              event.target.value,
                            );
                          }}
                        />

                        {focusedUserMoveIndex === moveIndex &&
                          selectedUserMember.name && (
                            <div className="builder-move-suggestions">
                              {filteredUserMoveOptions(move).map(
                                (moveOption) => (
                                  <button
                                    key={moveOption}
                                    type="button"
                                    onMouseDown={(event) =>
                                      event.preventDefault()
                                    }
                                    onClick={() => {
                                      updateSelectedUser(
                                        `moves.${moveIndex}`,
                                        moveOption,
                                      );
                                      setFocusedUserMoveIndex(null);
                                    }}
                                  >
                                    {moveOption}
                                  </button>
                                ),
                              )}
                            </div>
                          )}
                      </label>
                    ),
                  )}
                </div>
              </aside>

              <section className="team-editor-form">
                <div className="team-editor-row two-column">
                  <label className="team-editor-field">
                    <span>Form</span>
                    <select
                      value={
                        selectedUserMember.form ||
                        selectedUserMember.name ||
                        ""
                      }
                      disabled={
                        !selectedUserMember.name ||
                        selectedUserMember.formOptions?.length <= 1
                      }
                      onChange={(event) =>
                        updateSelectedUser("form", event.target.value)
                      }
                    >
                      {(selectedUserMember.formOptions?.length
                        ? selectedUserMember.formOptions
                        : [
                            {
                              name: selectedUserMember.name || "",
                              label: "Default",
                            },
                          ]
                      ).map((formOption) => (
                        <option key={formOption.name} value={formOption.name}>
                          {formOption.label || formOption.name}
                        </option>
                      ))}
                    </select>
                  </label>

                  <label className="team-editor-field">
                    <span>Ability</span>
                    {selectedUserMember.abilities?.length ? (
                      <select
                        value={
                          selectedUserMember.ability ||
                          selectedUserMember.abilities[0]
                        }
                        disabled={!selectedUserMember.name}
                        onChange={(event) =>
                          updateSelectedUser("ability", event.target.value)
                        }
                      >
                        {selectedUserMember.abilities.map((ability) => (
                          <option key={ability} value={ability}>
                            {ability}
                          </option>
                        ))}
                      </select>
                    ) : (
                      <input
                        value={selectedUserMember.ability ?? ""}
                        disabled={!selectedUserMember.name}
                        placeholder="Ability"
                        onChange={(event) =>
                          updateSelectedUser("ability", event.target.value)
                        }
                      />
                    )}
                  </label>
                </div>

                <div className="team-editor-row two-column">
                  <label className="team-editor-field">
                    <span>Nature</span>
                    <select
                      value={selectedUserMember.nature || "hardy"}
                      disabled={!selectedUserMember.name}
                      onChange={(event) =>
                        updateSelectedUser("nature", event.target.value)
                      }
                    >
                      {natures.map((nature) => (
                        <option key={nature} value={nature}>
                          {formatNature(nature)}
                        </option>
                      ))}
                    </select>
                  </label>

                  <label className="team-editor-field">
                    <span>Item</span>
                    <input
                      value={selectedUserMember.item ?? ""}
                      disabled={!selectedUserMember.name}
                      placeholder="Item"
                      onChange={(event) =>
                        updateSelectedUser("item", event.target.value)
                      }
                    />
                  </label>
                </div>

                <section className="builder-stat-investment">
                  <div className="builder-stat-header">
                    <span>Stat Points</span>
                    <strong>
                      {selectedUserPointsUsed} / {MAX_TOTAL_STAT_POINTS}
                    </strong>
                  </div>

                  <div className="builder-stat-grid">
                    {statKeys.map((statName) => (
                      <label
                        key={statName}
                        style={{
                          "--stat-color": statColors[statName],
                          "--stat-bar-width": calculateStatBarWidth(
                            statName,
                            selectedUserMember.stats?.[statName],
                          ),
                        }}
                      >
                        <span>{statLabels[statName]}</span>
                        <strong>
                          {Number(
                            selectedUserMember.stats?.[statName] ??
                              calculateInvestedStat(
                                selectedUserMember,
                                statName,
                              ) ??
                              0,
                          )}
                        </strong>
                        <i aria-hidden="true" />
                        <input
                          type="text"
                          inputMode="numeric"
                          pattern="[0-9]*"
                          min="0"
                          max={MAX_STAT_POINTS}
                          value={Number(
                            selectedUserMember.statPoints?.[statName] ?? 0,
                          )}
                          disabled={!selectedUserMember.name}
                          onChange={(event) =>
                            updateSelectedUser(
                              `statPoints.${statName}`,
                              event.target.value,
                            )
                          }
                        />
                      </label>
                    ))}
                  </div>
                </section>

              </section>
            </div>
          </section>
        </div>
      )}

      {isRagOpen && (
        <RagPopup
          answer={ragAnswer}
          error={ragError}
          isAsking={isAskingRag}
          onAsk={askRag}
          onClose={() => setIsRagOpen(false)}
          question={ragQuestion}
          setQuestion={setRagQuestion}
          source={ragSource}
          submittedQuestion={ragSubmittedQuestion}
        />
      )}
    </main>
  );
}

export default BattlePrepPage;
