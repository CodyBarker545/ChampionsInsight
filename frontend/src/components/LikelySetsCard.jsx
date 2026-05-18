import React, { useEffect, useMemo, useState } from "react";
import { createPortal } from "react-dom";
import { fetchTeamSetProbabilities } from "../api/championsInsightApi.js";

function normalizeTeamNames(opponentTeam = []) {
  return opponentTeam
    .map((member) => String(member?.name || member?.pokemonName || "").trim())
    .filter((name) => {
      if (!name) {
        return false;
      }

      const normalized = name.toLowerCase();
      return !normalized.startsWith("opponent") && !normalized.startsWith("pokemon ");
    });
}

function formatPercent(value) {
  const number = Number(value || 0);
  if (!Number.isFinite(number)) {
    return "0%";
  }
  return `${number.toFixed(number >= 10 ? 0 : 1)}%`;
}

function formatConfidence(value, label = "") {
  const number = Number(value || 0);
  const percentage = `${Math.round(number * 100)}%`;
  return label ? `${label} ${percentage}` : percentage;
}

function highlightChange(change) {
  const number = Number(change || 0);
  if (number <= 0) {
    return "";
  }
  return `+${number.toFixed(1)}%`;
}

function renderProbabilityList(entries = [], limit = 4, emptyLabel = "No data yet.") {
  const visibleEntries = entries.slice(0, limit);
  if (!visibleEntries.length) {
    return <p className="likely-sets-empty">{emptyLabel}</p>;
  }

  return (
    <div className="likely-sets-probability-list">
      {visibleEntries.map((entry) => {
        const change = highlightChange(entry.change);
        return (
          <article className="likely-sets-probability-row" key={entry.name}>
            <div className="likely-sets-probability-copy">
              <strong>{entry.name}</strong>
              <span>{formatPercent(entry.conditionalPercentage ?? entry.percentage)}</span>
            </div>
            {change ? (
              <span className="likely-sets-shift-pill">{change}</span>
            ) : null}
          </article>
        );
      })}
    </div>
  );
}

function LikelySetsCard({
  opponentTeam,
  focusPokemonName = "",
  triggerLabel = "Likely Sets",
  triggerClassName = "",
}) {
  const teamNames = useMemo(() => normalizeTeamNames(opponentTeam), [opponentTeam]);
  const teamKey = JSON.stringify(teamNames).toLowerCase();
  const focusName = String(focusPokemonName || "").trim().toLowerCase();

  const [isOpen, setIsOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [openPokemon, setOpenPokemon] = useState({});
  const [portalRoot, setPortalRoot] = useState(null);

  useEffect(() => {
    if (typeof document === "undefined") {
      return undefined;
    }

    setPortalRoot(document.body);
    return undefined;
  }, []);

  useEffect(() => {
    setIsOpen(false);
    setResult(null);
    setError("");
    setIsLoading(false);
  }, [teamKey, focusName]);

  useEffect(() => {
    if (!isOpen || !teamNames.length) {
      return undefined;
    }

    const controller = new AbortController();
    let isActive = true;

    async function loadLikelySets() {
      setIsLoading(true);
      setError("");

      try {
        const nextResult = await fetchTeamSetProbabilities(teamNames, controller.signal);
        if (!isActive) {
          return;
        }

        setResult(nextResult);
        const predictionNames = Object.keys(nextResult?.predictions || {});
        setOpenPokemon(
          Object.fromEntries(
            predictionNames.map((pokemonName, index) => [
              pokemonName,
              focusName ? pokemonName.toLowerCase() === focusName : index === 0,
            ]),
          ),
        );
      } catch (requestError) {
        if (!isActive || requestError?.name === "AbortError") {
          return;
        }
        setError(
          requestError.message ||
            "Could not load likely tournament-backed sets right now.",
        );
      } finally {
        if (isActive) {
          setIsLoading(false);
        }
      }
    }

    void loadLikelySets();

    return () => {
      isActive = false;
      controller.abort();
    };
  }, [focusName, isOpen, teamKey, teamNames]);

  useEffect(() => {
    if (!isOpen || typeof document === "undefined") {
      return undefined;
    }

    const previousOverflow = document.body.style.overflow;
    const handleKeyDown = (event) => {
      if (event.key === "Escape") {
        setIsOpen(false);
      }
    };

    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", handleKeyDown);

    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [isOpen]);

  const predictions = result?.predictions || {};
  const predictionEntries = Object.entries(predictions);

  const modalContent =
    isOpen && portalRoot
      ? createPortal(
          <div
            className="archetype-insight-backdrop"
            onClick={() => setIsOpen(false)}
            role="presentation"
          >
            <section
              className="archetype-insight-modal likely-sets-modal"
              role="dialog"
              aria-modal="true"
              aria-label="Likely opponent sets"
              onClick={(event) => event.stopPropagation()}
            >
              <header className="archetype-insight-header">
                <div>
                  <span className="archetype-insight-eyebrow">Tournament Insight</span>
                  <h3>Likely Opponent Sets</h3>
                </div>
                <div className="archetype-insight-modal-actions">
                  <button
                    type="button"
                    className="archetype-insight-close"
                    onClick={() => setIsOpen(false)}
                    aria-label="Close likely sets"
                  >
                    X
                  </button>
                </div>
              </header>

              <div className="archetype-insight-body likely-sets-body">
                {isLoading ? (
                  <p className="archetype-insight-inline-copy">
                    Reading tournament-backed likely sets...
                  </p>
                ) : null}

                {error ? (
                  <p className="helper-text warning likely-sets-warning">
                    {error}
                  </p>
                ) : null}

                {!isLoading && !error && predictionEntries.length === 0 ? (
                  <p className="archetype-insight-inline-copy">
                    Detect an opponent team to load likely sets.
                  </p>
                ) : null}

                {!isLoading && predictionEntries.length > 0 ? (
                  <>
                    {(result?.teamExplanations || []).length > 0 && (
                      <section className="archetype-score-block">
                        <h4>Team Read</h4>
                        <ul className="archetype-reason-list">
                          {result.teamExplanations.map((entry) => (
                            <li key={entry}>{entry}</li>
                          ))}
                        </ul>
                      </section>
                    )}

                    <div className="likely-sets-card-list">
                      {predictionEntries.map(([pokemonName, prediction]) => {
                        const isExpanded = Boolean(openPokemon[pokemonName]);
                        const predictedSet = prediction.predictedSet || {};

                        return (
                          <article className="likely-sets-card" key={pokemonName}>
                            <button
                              type="button"
                              className="likely-sets-card-header"
                              onClick={() =>
                                setOpenPokemon((current) => ({
                                  ...current,
                                  [pokemonName]: !current[pokemonName],
                                }))
                              }
                            >
                              <div className="likely-sets-card-copy">
                                <strong>{pokemonName}</strong>
                                <span>
                                  {formatConfidence(
                                    prediction.confidence,
                                    prediction.confidenceLabel,
                                  )}
                                </span>
                              </div>
                              <span className="likely-sets-open-label">
                                {isExpanded ? "Hide" : "View"}
                              </span>
                            </button>

                            {isExpanded ? (
                              <div className="likely-sets-card-body">
                                <section className="likely-sets-predicted-set">
                                  <h4>Predicted Set</h4>
                                  <div className="likely-sets-chip-list">
                                    {(predictedSet.moves || []).map((move) => (
                                      <span className="archetype-chip" key={`${pokemonName}-${move}`}>
                                        {move}
                                      </span>
                                    ))}
                                  </div>
                                  <p className="likely-sets-meta-copy">
                                    Item: <strong>{predictedSet.item || "Unknown"}</strong>
                                    {" · "}
                                    Ability: <strong>{predictedSet.ability || "Unknown"}</strong>
                                    {" · "}
                                    Nature: <strong>{predictedSet.nature || "Unknown"}</strong>
                                  </p>
                                </section>

                                <div className="likely-sets-grid">
                                  <section className="archetype-score-block">
                                    <h4>Likely Moves</h4>
                                    {renderProbabilityList(prediction.likelyMoves, 6)}
                                  </section>

                                  <section className="archetype-score-block">
                                    <h4>Likely Items</h4>
                                    {renderProbabilityList(prediction.likelyItems, 3)}
                                  </section>

                                  <section className="archetype-score-block">
                                    <h4>Likely Abilities</h4>
                                    {renderProbabilityList(
                                      prediction.likelyAbilities,
                                      3,
                                      "No ability probability data.",
                                    )}
                                  </section>

                                  <section className="archetype-score-block">
                                    <h4>Likely Nature</h4>
                                    {renderProbabilityList(
                                      prediction.likelyNature,
                                      3,
                                      "No nature probability data.",
                                    )}
                                  </section>
                                </div>

                                {(prediction.explanations || []).length > 0 && (
                                  <section className="archetype-score-block">
                                    <h4>Why It Changed</h4>
                                    <ul className="archetype-reason-list">
                                      {prediction.explanations.map((entry) => (
                                        <li key={`${pokemonName}-${entry}`}>{entry}</li>
                                      ))}
                                    </ul>
                                  </section>
                                )}

                                {(prediction.dataQualityNotes || []).length > 0 && (
                                  <section className="archetype-score-block">
                                    <h4>Notes</h4>
                                    <ul className="archetype-note-list">
                                      {prediction.dataQualityNotes.map((entry) => (
                                        <li key={`${pokemonName}-${entry}`}>{entry}</li>
                                      ))}
                                    </ul>
                                  </section>
                                )}
                              </div>
                            ) : null}
                          </article>
                        );
                      })}
                    </div>

                    {(result?.dataQualityNotes || []).length > 0 && (
                      <p className="helper-text warning likely-sets-warning">
                        {result.dataQualityNotes.join(" ")}
                      </p>
                    )}
                  </>
                ) : null}
              </div>
            </section>
          </div>,
          portalRoot,
        )
      : null;

  return (
    <>
      <button
        type="button"
        className={triggerClassName || "likely-sets-trigger-button"}
        disabled={!teamNames.length}
        onClick={(event) => {
          event.stopPropagation();
          setIsOpen(true);
        }}
      >
        {triggerLabel}
      </button>

      {modalContent}
    </>
  );
}

export default LikelySetsCard;
