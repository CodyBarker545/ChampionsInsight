import React, { useEffect, useMemo, useState } from "react";
import {
  fetchPokemonDetails,
  fetchPokemonLevel50Stats,
  fetchPokemonTopMoves,
  fetchUserTeam,
  resolveApiUrl,
  searchPokemon,
} from "../api/championsInsightApi.js";

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

const maxStatPoints = 32;
const maxTotalStatPoints = 66;

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
  lonely: ["Atk", "Def"],
  brave: ["Atk", "Spe"],
  adamant: ["Atk", "SpA"],
  naughty: ["Atk", "SpD"],
  bold: ["Def", "Atk"],
  relaxed: ["Def", "Spe"],
  impish: ["Def", "SpA"],
  lax: ["Def", "SpD"],
  timid: ["Spe", "Atk"],
  hasty: ["Spe", "Def"],
  jolly: ["Spe", "SpA"],
  naive: ["Spe", "SpD"],
  modest: ["SpA", "Atk"],
  mild: ["SpA", "Def"],
  quiet: ["SpA", "Spe"],
  rash: ["SpA", "SpD"],
  calm: ["SpD", "Atk"],
  gentle: ["SpD", "Def"],
  sassy: ["SpD", "Spe"],
  careful: ["SpD", "SpA"],
};

const natureStatKeys = {
  Atk: "attack",
  Def: "defense",
  SpA: "specialAttack",
  SpD: "specialDefense",
  Spe: "speed",
};

function formatNature(nature) {
  const displayName = nature.charAt(0).toUpperCase() + nature.slice(1);
  const effect = natureEffects[nature];

  if (!effect) {
    return `${displayName} (Neutral)`;
  }

  return `${displayName} (+${effect[0]} / -${effect[1]})`;
}

const emptyBuilderMember = {
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
  nature: "hardy",
  stats: {},
  baseStats: {},
  finalStats: {},
  statPoints: {},
  moves: ["", "", "", ""],
  moveOptions: [],
};

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

function topMoveNames(topMovesResult) {
  return (topMovesResult?.moves ?? [])
    .map((moveRecord) => moveRecord.move)
    .filter(Boolean)
    .slice(0, 4);
}

function ensureFourMoves(moves) {
  return [...moves, "", "", "", ""].slice(0, 4);
}

function emptyStatPoints() {
  return Object.fromEntries(statKeys.map((statName) => [statName, 0]));
}

function clampStatPointInvestment(currentPoints = {}, statName, requestedValue) {
  const requestedPoints = Math.max(0, Number(requestedValue) || 0);
  const otherPointsTotal = statKeys.reduce((total, currentStatName) => {
    if (currentStatName === statName) {
      return total;
    }

    return total + Number(currentPoints[currentStatName] ?? 0);
  }, 0);
  const remainingTotalPoints = Math.max(0, maxTotalStatPoints - otherPointsTotal);

  return Math.min(maxStatPoints, remainingTotalPoints, requestedPoints);
}

function natureMultiplier(statName, nature) {
  const effect = natureEffects[nature];

  if (!effect) {
    return 1;
  }

  if (natureStatKeys[effect[0]] === statName) {
    return 1.1;
  }

  if (natureStatKeys[effect[1]] === statName) {
    return 0.9;
  }

  return 1;
}

function calculateInvestedStat(member, statName, points = 0) {
  const baseStat = Number(member.baseStats?.[statName] ?? 0);

  if (!baseStat) {
    return Number(member.finalStats?.[statName] ?? member.stats?.[statName] ?? 0) + points;
  }

  if (statName === "hp") {
    return Math.floor(((2 * baseStat + 31) * 50) / 100) + 60 + points;
  }

  const neutralLevel50Stat = Math.floor(((2 * baseStat + 31) * 50) / 100) + 5;
  return Math.floor(
    (neutralLevel50Stat + points) * natureMultiplier(statName, member.nature || "hardy")
  );
}

function buildInvestedStats(member, statPoints = member.statPoints ?? {}) {
  return Object.fromEntries(
    statKeys.map((statName) => [
      statName,
      calculateInvestedStat(member, statName, Number(statPoints?.[statName] ?? 0)),
    ])
  );
}

function applyInvestedStats(member) {
  return {
    ...member,
    statPoints: {
      ...emptyStatPoints(),
      ...(member.statPoints ?? {}),
    },
    stats: buildInvestedStats(member),
  };
}

function buildMemberFromBackend(details, level50Stats, topMovesResult) {
  const baseStats = normalizeStats(level50Stats?.baseStats ?? details.baseStats ?? {});
  const finalStats = normalizeStats(level50Stats?.finalStats ?? {});
  const displayStats = statKeys.some((statName) => finalStats[statName] > 0)
    ? finalStats
    : baseStats;
  const commonMoves = topMoveNames(topMovesResult);

  return {
    ...emptyBuilderMember,
    id: `user-${details.name}`,
    name: level50Stats?.name ?? details.name,
    species: details.species ?? "",
    form: details.form ?? level50Stats?.form ?? "",
    formOptions: details.formOptions ?? [],
    image: details.image ?? "",
    spriteUrl: details.spriteUrl ?? "",
    types: details.types ?? [],
    ability: details.abilities?.[0] ?? "",
    abilities: details.abilities ?? [],
    nature: level50Stats?.nature ?? "hardy",
    baseStats,
    finalStats,
    statPoints: emptyStatPoints(),
    stats: displayStats,
    moves: ensureFourMoves(commonMoves.length ? commonMoves : details.moves?.slice(0, 4) ?? []),
    moveOptions: details.moves ?? [],
  };
}

function TeamBuilderPage({ initialTeam, onCancel, onSave }) {
  const [team, setTeam] = useState(() =>
    Array.from({ length: 6 }, (_, index) =>
      applyInvestedStats({
        ...emptyBuilderMember,
        ...(initialTeam?.[index] ?? {}),
        id: initialTeam?.[index]?.id ?? `builder-${index + 1}`,
        moves: ensureFourMoves(initialTeam?.[index]?.moves ?? []),
        moveOptions: initialTeam?.[index]?.moveOptions ?? initialTeam?.[index]?.moves ?? [],
      })
    )
  );
  const [selectedSlot, setSelectedSlot] = useState(0);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [isSearching, setIsSearching] = useState(false);
  const [isLoadingSavedTeam, setIsLoadingSavedTeam] = useState(false);
  const [isLoadingPokemon, setIsLoadingPokemon] = useState(false);
  const [isSavingTeam, setIsSavingTeam] = useState(false);
  const [builderError, setBuilderError] = useState("");
  const [focusedMoveIndex, setFocusedMoveIndex] = useState(null);

  const selectedMember = team[selectedSlot];
  const canSave = team.every((member) => member.name?.trim());

  const selectedMoveOptions = useMemo(
    () => Array.from(new Set(selectedMember.moveOptions ?? selectedMember.moves ?? [])).filter(Boolean),
    [selectedMember]
  );
  const selectedPointsUsed = statKeys.reduce(
    (total, statName) => total + Number(selectedMember.statPoints?.[statName] ?? 0),
    0
  );

  useEffect(() => {
    if (initialTeam?.length) {
      return undefined;
    }

    let isCurrentRequest = true;
    setIsLoadingSavedTeam(true);

    fetchUserTeam()
      .then((savedTeam) => {
        if (!isCurrentRequest) {
          return;
        }

        setTeam(
          Array.from({ length: 6 }, (_, index) =>
            applyInvestedStats({
              ...emptyBuilderMember,
              ...(savedTeam.team?.[index] ?? {}),
              id: savedTeam.team?.[index]?.id ?? `builder-${index + 1}`,
              moves: ensureFourMoves(savedTeam.team?.[index]?.moves ?? []),
              moveOptions: savedTeam.team?.[index]?.moveOptions ?? savedTeam.team?.[index]?.moves ?? [],
            })
          )
        );
      })
      .catch((error) => {
        if (isCurrentRequest) {
          setBuilderError(error.message);
        }
      })
      .finally(() => {
        if (isCurrentRequest) {
          setIsLoadingSavedTeam(false);
        }
      });

    return () => {
      isCurrentRequest = false;
    };
  }, [initialTeam]);

  useEffect(() => {
    const trimmedQuery = query.trim();
    if (trimmedQuery.length < 2) {
      setResults([]);
      return undefined;
    }

    const timeoutId = window.setTimeout(async () => {
      setIsSearching(true);
      setBuilderError("");

      try {
        const searchResult = await searchPokemon(trimmedQuery, 12);
        setResults(searchResult.results ?? []);
      } catch (error) {
        setBuilderError(error.message);
      } finally {
        setIsSearching(false);
      }
    }, 250);

    return () => window.clearTimeout(timeoutId);
  }, [query]);

  function updateSlot(index, updater) {
    setTeam((currentTeam) =>
      currentTeam.map((member, memberIndex) =>
        memberIndex === index ? updater(member) : member
      )
    );
  }

  function selectBuilderSlot(index) {
    setSelectedSlot(index);
    setQuery("");
    setResults([]);
  }

  async function selectPokemonForSlot(pokemonName) {
    setIsLoadingPokemon(true);
    setBuilderError("");

    try {
      const [details, level50Stats, topMovesResult] = await Promise.all([
        fetchPokemonDetails(pokemonName),
        fetchPokemonLevel50Stats(pokemonName, "hardy"),
        fetchPokemonTopMoves(pokemonName, 4),
      ]);
      const member = buildMemberFromBackend(details, level50Stats, topMovesResult);

      updateSlot(selectedSlot, () => ({
        ...member,
        id: `user-${selectedSlot + 1}-${member.name}`,
      }));
      setQuery("");
      setResults([]);
    } catch (error) {
      setBuilderError(error.message);
    } finally {
      setIsLoadingPokemon(false);
    }
  }

  async function updateNature(nature) {
    if (!selectedMember.name) {
      return;
    }

    updateSlot(selectedSlot, (member) => ({ ...member, nature }));

    try {
      const level50Stats = await fetchPokemonLevel50Stats(selectedMember.name, nature);
      updateSlot(selectedSlot, (member) => {
        const finalStats = normalizeStats(level50Stats.finalStats ?? {});
        const nextMember = {
          ...member,
          nature,
          baseStats: normalizeStats(level50Stats.baseStats ?? member.baseStats),
          finalStats,
        };
        return {
          ...nextMember,
          stats: buildInvestedStats(nextMember),
        };
      });
    } catch (error) {
      setBuilderError(error.message);
    }
  }

  async function updateForm(formName) {
    if (!selectedMember.name || !formName) {
      return;
    }

    updateSlot(selectedSlot, (member) => ({ ...member, form: formName }));

    try {
      const [details, level50Stats, topMovesResult] = await Promise.all([
        fetchPokemonDetails(formName),
        fetchPokemonLevel50Stats(formName, selectedMember.nature || "hardy"),
        fetchPokemonTopMoves(formName, 4),
      ]);
      const formMember = buildMemberFromBackend(details, level50Stats, topMovesResult);

      updateSlot(selectedSlot, (member) => ({
        ...member,
        ...formMember,
        id: member.id,
        item: member.item,
        moves: member.moves?.some(Boolean) ? member.moves : formMember.moves,
        moveOptions: formMember.moveOptions?.length
          ? formMember.moveOptions
          : member.moveOptions,
        statPoints: member.statPoints,
        stats: buildInvestedStats(
          { ...member, ...formMember, statPoints: member.statPoints },
          member.statPoints
        ),
      }));
    } catch (error) {
      setBuilderError(error.message);
    }
  }

  function updateMove(moveIndex, moveName) {
    updateSlot(selectedSlot, (member) => {
      const moves = ensureFourMoves(member.moves);
      moves[moveIndex] = moveName;
      const moveOptions = Array.from(new Set([...(member.moveOptions ?? []), moveName])).filter(Boolean);

      return {
        ...member,
        moves,
        moveOptions,
      };
    });
  }

  function filteredMoveOptions(moveValue) {
    const normalizedValue = moveValue.trim().toLowerCase();
    const options = selectedMoveOptions.filter((moveOption) =>
      moveOption.toLowerCase().includes(normalizedValue)
    );

    return options.slice(0, 8);
  }

  function clearSlot(index) {
    updateSlot(index, () => ({
      ...emptyBuilderMember,
      id: `builder-${index + 1}`,
      moves: ensureFourMoves([]),
      moveOptions: [],
    }));
    setSelectedSlot(index);
    setQuery("");
    setResults([]);
  }

  function updateStatPoints(statName, requestedValue) {
    if (!selectedMember.name) {
      return;
    }

    updateSlot(selectedSlot, (member) => {
      const currentPoints = {
        ...emptyStatPoints(),
        ...(member.statPoints ?? {}),
      };
      const nextPoints = {
        ...currentPoints,
        [statName]: clampStatPointInvestment(currentPoints, statName, requestedValue),
      };

      return {
        ...member,
        statPoints: nextPoints,
        stats: buildInvestedStats(member, nextPoints),
      };
    });
  }

  async function saveTeam() {
    setIsSavingTeam(true);
    setBuilderError("");

    try {
      await onSave(
        team.map((member, index) => ({
          ...member,
          id: member.id || `user-${index + 1}`,
          moves: ensureFourMoves(member.moves).filter(Boolean),
        }))
      );
    } catch (error) {
      setBuilderError(error.message);
    } finally {
      setIsSavingTeam(false);
    }
  }

  return (
    <main className="team-builder-shell">
      <section className="team-builder-panel">
        <header className="team-builder-header">
          <div>
            <span>Team Builder</span>
            <h1>Build Your Six</h1>
          </div>

          <div className="team-builder-actions">
            <button className="secondary-button" type="button" onClick={onCancel}>
              Back
            </button>
            <button
              className="primary-button"
              type="button"
              disabled={!canSave || isSavingTeam || isLoadingSavedTeam}
              onClick={saveTeam}
            >
              {isSavingTeam ? "Saving" : "Use Team"}
            </button>
          </div>
        </header>

        <div className="builder-layout">
          <section className="builder-team-panel">
            <div className="builder-slots">
              {team.map((member, index) => {
                const imageSource = resolveApiUrl(member.spriteUrl || member.image || "");
                const isSelected = selectedSlot === index;

                return (
                  <section
                    className={`builder-slot ${selectedSlot === index ? "selected" : ""}`}
                    data-slot={index + 1}
                    key={member.id || index}
                    onClick={() => selectBuilderSlot(index)}
                  >
                    {member.name && (
                      <button
                        className="builder-slot-clear"
                        type="button"
                        aria-label={`Clear ${member.name}`}
                        onClick={(event) => {
                          event.stopPropagation();
                          clearSlot(index);
                        }}
                      >
                        x
                      </button>
                    )}

                    <button
                      className="builder-slot-summary"
                      type="button"
                      onClick={() => selectBuilderSlot(index)}
                    >
                      <span className="builder-slot-image">
                        {imageSource ? <img src={imageSource} alt="" /> : "+"}
                      </span>
                      <strong>{member.name || `Slot ${index + 1}`}</strong>
                    </button>

                    {member.name && (
                      <div className="builder-slot-details">
                        <div className="builder-slot-types">
                          {(member.types ?? []).slice(0, 2).map((typeName) => (
                            <span key={typeName}>{typeName}</span>
                          ))}
                        </div>
                        <small>{member.item || "Item pending"}</small>
                        <span>{member.ability || "Ability pending"}</span>
                        <ul>
                          {ensureFourMoves(member.moves)
                            .filter(Boolean)
                            .slice(0, 4)
                            .map((move) => (
                              <li key={move}>{move}</li>
                            ))}
                        </ul>
                      </div>
                    )}

                    {isSelected && (
                      <div className="builder-slot-search">
                        <input
                          value={query}
                          placeholder={member.name ? "Search replacement" : "Search Pokemon"}
                          onClick={(event) => event.stopPropagation()}
                          onChange={(event) => setQuery(event.target.value)}
                        />

                        {(isSearching || results.length > 0) && (
                          <div className="builder-slot-results">
                            {isSearching && <span>Searching...</span>}
                            {results.map((pokemon) => {
                              const resultImageSource = resolveApiUrl(pokemon.image);

                              return (
                                <button
                                  className="builder-result"
                                  key={pokemon.name}
                                  type="button"
                                  disabled={isLoadingPokemon}
                                  onClick={(event) => {
                                    event.stopPropagation();
                                    selectPokemonForSlot(pokemon.name);
                                  }}
                                >
                                  <span className="builder-result-image">
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
                      </div>
                    )}
                  </section>
                );
              })}
            </div>

              <section className="builder-editor">
                <div className="builder-editor-title">
                  <span>Slot {selectedSlot + 1} Setup</span>
                  <strong>{selectedMember.name || `Slot ${selectedSlot + 1}`}</strong>
                </div>

                <label>
                  <span>Form</span>
                  <select
                    value={selectedMember.form || selectedMember.name || ""}
                    disabled={!selectedMember.name || selectedMember.formOptions?.length <= 1}
                    onChange={(event) => updateForm(event.target.value)}
                  >
                    {(selectedMember.formOptions?.length
                      ? selectedMember.formOptions
                      : [{ name: selectedMember.name || "", label: "Default" }]
                    ).map((formOption) => (
                      <option key={formOption.name} value={formOption.name}>
                        {formOption.label || formOption.name}
                      </option>
                    ))}
                  </select>
                </label>

                <label>
                  <span>Nature</span>
                  <select
                    value={selectedMember.nature || "hardy"}
                    disabled={!selectedMember.name}
                    onChange={(event) => updateNature(event.target.value)}
                  >
                    {natures.map((nature) => (
                      <option key={nature} value={nature}>
                        {formatNature(nature)}
                      </option>
                    ))}
                  </select>
                </label>

                <section className="builder-stat-investment">
                  <div className="builder-stat-header">
                    <span>Stat Points</span>
                    <strong>{selectedPointsUsed} / {maxTotalStatPoints}</strong>
                  </div>

                  <div className="builder-stat-grid">
                    {statKeys.map((statName) => (
                      <label key={statName}>
                        <span>{statLabels[statName]}</span>
                        <strong>
                          {Number(selectedMember.stats?.[statName] ?? calculateInvestedStat(selectedMember, statName) ?? 0)}
                        </strong>
                        <input
                          type="number"
                          min="0"
                          max={maxStatPoints}
                          value={Number(selectedMember.statPoints?.[statName] ?? 0)}
                          disabled={!selectedMember.name}
                          onChange={(event) => updateStatPoints(statName, event.target.value)}
                        />
                      </label>
                    ))}
                  </div>
                </section>

                <div className="builder-moves">
                  {ensureFourMoves(selectedMember.moves).map((move, moveIndex) => (
                    <label className="builder-move-field" key={`move-${moveIndex}`}>
                      <span>Move {moveIndex + 1}</span>
                      <input
                        value={move}
                        disabled={!selectedMember.name}
                        placeholder="Search moves"
                        onFocus={() => setFocusedMoveIndex(moveIndex)}
                        onBlur={() => {
                          window.setTimeout(() => setFocusedMoveIndex(null), 120);
                        }}
                        onChange={(event) => updateMove(moveIndex, event.target.value)}
                      />

                      {focusedMoveIndex === moveIndex && selectedMember.name && (
                        <div className="builder-move-suggestions">
                          {filteredMoveOptions(move).map((moveOption) => (
                            <button
                              key={moveOption}
                              type="button"
                              onMouseDown={(event) => event.preventDefault()}
                              onClick={() => {
                                updateMove(moveIndex, moveOption);
                                setFocusedMoveIndex(null);
                              }}
                            >
                              {moveOption}
                            </button>
                          ))}
                        </div>
                      )}
                    </label>
                  ))}
                </div>
              </section>

            {builderError && <p className="helper-text error">{builderError}</p>}
          </section>
        </div>
      </section>
    </main>
  );
}

export default TeamBuilderPage;
