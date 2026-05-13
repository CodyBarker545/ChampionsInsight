import React, { useEffect, useMemo, useState } from "react";
import {
  fetchPokedexEntries,
  fetchPokedexEntry,
  resolveApiUrl,
} from "../api/championsInsightApi.js";

const statOrder = [
  "hp",
  "attack",
  "defense",
  "specialAttack",
  "specialDefense",
  "speed",
];

const statLabels = {
  hp: "HP",
  attack: "Attack",
  defense: "Defense",
  specialAttack: "Sp. Atk",
  specialDefense: "Sp. Def",
  speed: "Speed",
};

function getSpeciesKey(pokemon) {
  return String(pokemon?.speciesName || pokemon?.name || "").toLowerCase();
}

function isBaseDisplayForm(pokemon) {
  const name = String(pokemon?.name || "").toLowerCase();
  const speciesName = String(pokemon?.speciesName || "").toLowerCase();

  return name === speciesName;
}

function getMainSpeciesEntries(entries) {
  const groupedBySpecies = new Map();

  entries.forEach((pokemon) => {
    const speciesKey = getSpeciesKey(pokemon);

    if (!speciesKey) {
      return;
    }

    const existing = groupedBySpecies.get(speciesKey);

    if (!existing) {
      groupedBySpecies.set(speciesKey, pokemon);
      return;
    }

    const currentIsBase = isBaseDisplayForm(pokemon);
    const existingIsBase = isBaseDisplayForm(existing);

    if (currentIsBase && !existingIsBase) {
      groupedBySpecies.set(speciesKey, pokemon);
      return;
    }

    if (!pokemon.isMega && existing.isMega) {
      groupedBySpecies.set(speciesKey, pokemon);
    }
  });

  return Array.from(groupedBySpecies.values());
}

function getFormLabel(pokemon) {
  const name = String(pokemon?.name || "");
  const speciesName = String(pokemon?.speciesName || "");

  if (!name) {
    return "Form";
  }

  if (!speciesName || name.toLowerCase() === speciesName.toLowerCase()) {
    return "Base";
  }

  const cleaned = name
    .replace(speciesName, "")
    .replaceAll("-", " ")
    .trim();

  if (cleaned) {
    return cleaned;
  }

  if (pokemon.isMega) {
    return "Mega";
  }

  return "Form";
}

function sameSpecies(a, b) {
  const aSpecies = String(a?.speciesName || a?.name || "").toLowerCase();
  const bSpecies = String(b?.speciesName || b?.name || "").toLowerCase();

  return aSpecies === bSpecies;
}

function normalizeType(type) {
  return String(type || "").trim().toLowerCase();
}

const typeColors = {
  bug: "#9fbe2c",
  dark: "#4b5563",
  dragon: "#7c5cff",
  electric: "#f0c419",
  fairy: "#f08bc3",
  fighting: "#d94a4a",
  fire: "#f27835",
  flying: "#91a7ff",
  ghost: "#7c62c8",
  grass: "#62b35f",
  ground: "#d0a152",
  ice: "#75d8e8",
  normal: "#a8afbd",
  poison: "#aa66d9",
  psychic: "#ec5aa6",
  rock: "#aa9251",
  steel: "#9aa8bd",
  water: "#5393e8",
};

function getTypeColor(type) {
  return typeColors[normalizeType(type)] || "#7dd3fc";
}

function formatType(type) {
  const value = String(type || "").trim();

  if (!value) {
    return "";
  }

  return value.charAt(0).toUpperCase() + value.slice(1);
}

function getStatTotal(baseStats = {}) {
  return statOrder.reduce((total, statName) => {
    return total + Number(baseStats[statName] || 0);
  }, 0);
}

function formatPercent(value) {
  const number = Number(value || 0);

  if (number <= 0) {
    return "0%";
  }

  return `${Math.round(number * 100)}%`;
}

function formatName(value) {
  return String(value || "")
    .replaceAll("-", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function normalizeLookupName(value) {
  return String(value || "").toLowerCase().replace(/[^a-z0-9]/g, "");
}

function normalizeMoveDetail(move) {
  if (typeof move === "string") {
    return {
      name: move,
      displayName: formatName(move),
      type: "",
      category: "",
      effect: "Move details unavailable.",
    };
  }

  return {
    name: move?.name || move?.displayName || "",
    displayName: move?.displayName || formatName(move?.name),
    type: move?.type || "",
    category: move?.category || "",
    effect: move?.effect || formatMoveSummary(move),
  };
}

function formatMoveSummary(move) {
  if (!move) {
    return "Move details unavailable.";
  }

  const parts = [];
  if (move.power) {
    parts.push(`${move.power} power`);
  } else if (move.category) {
    parts.push("No direct damage");
  }
  if (move.accuracy) {
    parts.push(`${move.accuracy}% accuracy`);
  }
  if (move.pp) {
    parts.push(`${move.pp} PP`);
  }
  if (move.priority) {
    parts.push(`${Number(move.priority) > 0 ? "+" : ""}${move.priority} priority`);
  }

  return parts.length ? parts.join(", ") : "Move details unavailable.";
}

function formatMoveCategory(category) {
  const normalized = String(category || "").toLowerCase();

  if (normalized === "physical") {
    return "Attack";
  }

  if (normalized === "special") {
    return "Special";
  }

  if (normalized === "status") {
    return "Status";
  }

  return "Unknown";
}

function PokedexPage({ onBack }) {
  const [entries, setEntries] = useState([]);
  const [selectedPokemon, setSelectedPokemon] = useState(null);
  const [activeTab, setActiveTab] = useState("stats");
  const [query, setQuery] = useState("");
  const [selectedType, setSelectedType] = useState("all");
  const [isLoading, setIsLoading] = useState(true);
  const [isDetailLoading, setIsDetailLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    async function loadEntries() {
      setIsLoading(true);
      setError("");

      try {
        const result = await fetchPokedexEntries();
        setEntries(result.pokemon || []);
      } catch (requestError) {
        setError(requestError.message || "Could not load Pokédex.");
      } finally {
        setIsLoading(false);
      }
    }

    void loadEntries();
  }, []);

  const mainSpeciesEntries = useMemo(() => {
    return getMainSpeciesEntries(entries);
  }, [entries]);

  const types = useMemo(() => {
    const typeSet = new Set();

    entries.forEach((pokemon) => {
      (pokemon.types || []).forEach((type) => {
        const normalized = normalizeType(type);

        if (normalized) {
          typeSet.add(normalized);
        }
      });
    });

    return ["all", ...Array.from(typeSet).sort()];
  }, [entries]);

  const filteredEntries = useMemo(() => {
    const search = query.trim().toLowerCase();

    return mainSpeciesEntries.filter((pokemon) => {
      const speciesForms = entries.filter((entry) => sameSpecies(entry, pokemon));

      const matchesSearch =
        !search ||
        speciesForms.some((form) => {
          const name = String(form.name || "").toLowerCase();
          const speciesName = String(form.speciesName || "").toLowerCase();
          const formApiName = String(form.formApiName || "").toLowerCase();

          return (
            name.includes(search) ||
            speciesName.includes(search) ||
            formApiName.includes(search)
          );
        });

      const matchesType =
        selectedType === "all" ||
        speciesForms.some((form) =>
          (form.types || []).some((type) => normalizeType(type) === selectedType)
        );

      return matchesSearch && matchesType;
    });
  }, [mainSpeciesEntries, entries, query, selectedType]);

  async function openPokemon(pokemon) {
    setIsDetailLoading(true);
    setError("");
    setActiveTab("stats");

    try {
      const detail = await fetchPokedexEntry(pokemon.name);
      setSelectedPokemon(detail);
    } catch (requestError) {
      setError(requestError.message || `Could not load ${pokemon.name}.`);
    } finally {
      setIsDetailLoading(false);
    }
  }

  function closePokemon() {
    setSelectedPokemon(null);
    setActiveTab("stats");
  }

  return (
    <main className="pokedex-shell">
      <header className="pokedex-topbar">
        <div>
          <span>Champions Insight</span>
          <h1>Pokédex</h1>
        </div>

        <div className="pokedex-topbar-actions">
          <button className="secondary-button" type="button" onClick={onBack}>
            Back to Battle
          </button>
        </div>
      </header>

      <section className="pokedex-controls">
        <label>
          <span>Search</span>
          <input
            value={query}
            placeholder="Search Pokémon"
            onChange={(event) => setQuery(event.target.value)}
          />
        </label>

        <label>
          <span>Type</span>
          <select
            value={selectedType}
            onChange={(event) => setSelectedType(event.target.value)}
          >
            {types.map((type) => (
              <option key={type} value={type}>
                {type === "all" ? "All Types" : formatType(type)}
              </option>
            ))}
          </select>
        </label>

        <div className="pokedex-count">
          <span>Showing</span>
          <strong>{filteredEntries.length}</strong>
          <small>of {mainSpeciesEntries.length}</small>
        </div>
      </section>

      {error && <p className="alert">{error}</p>}

      {isLoading ? (
        <section className="pokedex-loading">Loading Pokédex...</section>
      ) : (
        <section className="pokedex-grid">
          {filteredEntries.map((pokemon) => {
            const imageSource = resolveApiUrl(pokemon.spriteUrl || pokemon.image || "");
            const dexNumber = String(pokemon.id || "").padStart(4, "0");
            const forms = entries.filter((entry) => sameSpecies(entry, pokemon));

            return (
              <button
                className="pokedex-card"
                key={`${pokemon.id}-${pokemon.name}-${pokemon.spriteFilename || ""}`}
                style={{ "--pokemon-main-color": getTypeColor(pokemon.types?.[0]) }}
                type="button"
                onClick={() => openPokemon(pokemon)}
              >
                <div className="pokedex-card-header">
                  {forms.length > 1 ? (
                    <span className="pokedex-form-count">{forms.length} Forms</span>
                  ) : (
                    <span />
                  )}

                  <small>#{dexNumber}</small>
                </div>

                <div className="pokedex-card-image">
                  {imageSource ? (
                    <img src={imageSource} alt={`${pokemon.name} sprite`} />
                  ) : (
                    <span>IMG</span>
                  )}
                </div>

                <div className="pokedex-card-body">
                  <strong>{pokemon.speciesName || pokemon.name}</strong>

                  <div className="pokedex-type-row">
                    {(pokemon.types || []).map((type) => (
                      <span
                        className={`pokedex-type-pill type-${normalizeType(type)}`}
                        key={type}
                      >
                        {formatType(type)}
                      </span>
                    ))}
                  </div>
                </div>
              </button>
            );
          })}
        </section>
      )}

      {isDetailLoading && (
        <div className="pokedex-floating-status">Loading Pokémon details...</div>
      )}

      {selectedPokemon && (
        <PokemonDetailModal
          pokemon={selectedPokemon}
          forms={entries.filter((entry) => sameSpecies(entry, selectedPokemon))}
          entries={entries}
          activeTab={activeTab}
          onChangeTab={setActiveTab}
          onClose={closePokemon}
          onSelectForm={openPokemon}
        />
      )}
    </main>
  );
}

function PokemonDetailModal({
  pokemon,
  forms = [],
  entries = [],
  activeTab,
  onChangeTab,
  onClose,
  onSelectForm,
}) {
  const [moveSearch, setMoveSearch] = useState("");
  const imageSource = resolveApiUrl(pokemon.spriteUrl || pokemon.image || "");
  const baseStats = pokemon.baseStats || {};
  const usage = pokemon.usage || {};
  const statTotal = getStatTotal(baseStats);
  const tabs = ["stats", "moves", "abilities", "usage", "teams"];
  const activeFormIndex = Math.max(
    0,
    forms.findIndex((form) => form.name === pokemon.name),
  );
  const activeFormLabel = getFormLabel(pokemon);
  const pokemonLookup = useMemo(() => {
    const lookup = new Map();
    entries.forEach((entry) => {
      const names = [
        entry.name,
        entry.speciesName,
        entry.formApiName,
        String(entry.name || "").replaceAll(" ", "-"),
      ];
      names.forEach((name) => {
        const key = normalizeLookupName(name);
        if (key && !lookup.has(key)) {
          lookup.set(key, entry);
        }
      });
    });
    return lookup;
  }, [entries]);
  const filteredMoves = useMemo(() => {
    const search = moveSearch.trim().toLowerCase();
    const moves = (pokemon.moves || []).map(normalizeMoveDetail);

    if (!search) {
      return moves;
    }

    return moves.filter((move) => {
      const category = formatMoveCategory(move.category);
      const searchable = [
        move.displayName,
        move.name,
        move.type,
        category,
        move.effect,
      ]
        .join(" ")
        .toLowerCase();

      return searchable.includes(search);
    });
  }, [pokemon.moves, moveSearch]);

  function selectNextForm() {
    if (forms.length <= 1) {
      return;
    }

    const nextForm = forms[(activeFormIndex + 1) % forms.length];
    onSelectForm(nextForm);
  }

  return (
    <div className="pokedex-modal-backdrop" onClick={onClose}>
      <section
        className="pokedex-modal"
        role="dialog"
        aria-label={`${pokemon.name} details`}
        onClick={(event) => event.stopPropagation()}
      >
        <button
          className="pokedex-modal-close"
          type="button"
          onClick={onClose}
          aria-label="Close Pokédex details"
        >
          ×
        </button>

        <header className="pokedex-modal-header">
          <div
            className="pokedex-modal-image"
            style={{ "--pokemon-main-color": getTypeColor(pokemon.types?.[0]) }}
          >
            {imageSource ? (
              <img src={imageSource} alt={`${pokemon.name} sprite`} />
            ) : (
              <span>IMG</span>
            )}
          </div>

          <div className="pokedex-modal-title">
            <span>
              #{String(pokemon.id || "").padStart(4, "0")}
              {pokemon.speciesName ? ` · ${pokemon.speciesName}` : ""}
            </span>

            <h2>{pokemon.name}</h2>

            {forms.length > 1 && (
              <button
                type="button"
                className="pokedex-form-cycle-button"
                onClick={selectNextForm}
                aria-label={`Current form ${activeFormLabel}. Switch to next form.`}
              >
                <span>Form</span>
                <strong>{activeFormLabel}</strong>
                <small>
                  {activeFormIndex + 1}/{forms.length}
                </small>
              </button>
            )}

            <div className="pokedex-type-row">
              {(pokemon.types || []).map((type) => (
                <span
                  className={`pokedex-type-pill type-${normalizeType(type)}`}
                  key={type}
                >
                  {formatType(type)}
                </span>
              ))}
            </div>
          </div>

          <div className="pokedex-usage-summary">
            <span>Win Rate</span>
            <strong>{formatPercent(usage.winRate)}</strong>
            <small>{usage.appearances || 0} appearances</small>
          </div>
        </header>

        <nav className="pokedex-tabs">
          {tabs.map((tab) => (
            <button
              className={activeTab === tab ? "active" : ""}
              key={tab}
              type="button"
              onClick={() => onChangeTab(tab)}
            >
              {tab}
            </button>
          ))}
        </nav>

        <section className="pokedex-modal-content">
          {activeTab === "stats" && (
            <div className="pokedex-stats-panel">
              <div className="pokedex-bst-row">
                <span>Base Stat Total</span>
                <strong>{statTotal}</strong>
              </div>

              {statOrder.map((statName) => {
                const value = Number(baseStats[statName] || 0);
                const width = `${Math.min((value / 180) * 100, 100)}%`;

                return (
                  <div className="pokedex-stat-row" key={statName}>
                    <span>{statLabels[statName]}</span>
                    <strong>{value}</strong>
                    <div className="pokedex-stat-bar">
                      <i style={{ width }} />
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          {activeTab === "moves" && (
            <div className="pokedex-moves-panel">
              <label className="pokedex-move-search">
                <span>Search Moves</span>
                <input
                  value={moveSearch}
                  placeholder="Search by move, type, attack kind, or effect"
                  onChange={(event) => setMoveSearch(event.target.value)}
                />
              </label>

              {filteredMoves.length ? (
                <div className="pokedex-move-list">
                  {filteredMoves.map((move) => (
                    <article key={move.name || move.displayName}>
                      <strong>{move.displayName}</strong>
                      <span
                        className={`pokedex-type-pill type-${normalizeType(move.type)}`}
                      >
                        {move.type ? formatType(move.type) : "Unknown"}
                      </span>
                      <span className={`pokedex-move-category category-${normalizeType(move.category)}`}>
                        {formatMoveCategory(move.category)}
                      </span>
                      <p>{move.effect}</p>
                    </article>
                  ))}
                </div>
              ) : (
                <p>No moves match that search.</p>
              )}
            </div>
          )}

          {activeTab === "abilities" && (
            <div className="pokedex-list-panel">
              {(pokemon.abilities || []).length ? (
                pokemon.abilities.map((ability) => {
                  const name =
                    typeof ability === "string"
                      ? ability
                      : ability.displayName || ability.name;

                  const isHidden = typeof ability === "object" && ability.isHidden;

                  return (
                    <article key={name}>
                      <strong>{formatName(name)}</strong>
                      <span>{isHidden ? "Hidden Ability" : "Standard Ability"}</span>
                    </article>
                  );
                })
              ) : (
                <p>No ability data available.</p>
              )}
            </div>
          )}

          {activeTab === "usage" && (
            <div className="pokedex-usage-panel">
              <article>
                <span>Appearances</span>
                <strong>{usage.appearances || 0}</strong>
              </article>
              <article>
                <span>Wins</span>
                <strong>{usage.wins || 0}</strong>
              </article>
              <article>
                <span>Losses</span>
                <strong>{usage.losses || 0}</strong>
              </article>
              <article>
                <span>Win Rate</span>
                <strong>{formatPercent(usage.winRate)}</strong>
              </article>

              <section>
                <h3>Top Moves</h3>
                {(usage.topMoves || []).length ? (
                  usage.topMoves.map((move) => (
                    <div key={move.move}>
                      <span>{formatName(move.move)}</span>
                      <strong>{move.count}</strong>
                    </div>
                  ))
                ) : (
                  <p>No tournament move data available.</p>
                )}
              </section>

              <section>
                <h3>Top Items</h3>
                {(usage.topItems || []).length ? (
                  usage.topItems.map((item) => (
                    <div key={item.item}>
                      <span>{formatName(item.item)}</span>
                      <strong>{item.count}</strong>
                    </div>
                  ))
                ) : (
                  <p>No tournament item data available.</p>
                )}
              </section>
            </div>
          )}

          {activeTab === "teams" && (
            <div className="pokedex-team-panel">
              {(usage.topTeams || []).length ? (
                usage.topTeams.map((team, index) => (
                  <article key={`${team.player}-${team.tournament}-${index}`}>
                    <header>
                      <strong>{team.player || "Unknown Player"}</strong>
                      <span>{team.tournament || "Unknown Tournament"}</span>
                    </header>

                    <div className="pokedex-team-sprites">
                      {(team.pokemon || []).map((name) => {
                        const teamPokemon = pokemonLookup.get(normalizeLookupName(name));
                        const teamImage = resolveApiUrl(teamPokemon?.spriteUrl || teamPokemon?.image || "");

                        return (
                          <span className="pokedex-team-sprite" key={name} title={formatName(name)}>
                            {teamImage ? (
                              <img src={teamImage} alt={formatName(name)} />
                            ) : (
                              formatName(name).slice(0, 2)
                            )}
                          </span>
                        );
                      })}
                    </div>
                  </article>
                ))
              ) : (
                <p>No team data available.</p>
              )}
            </div>
          )}
        </section>
      </section>
    </div>
  );
}

export default PokedexPage;
