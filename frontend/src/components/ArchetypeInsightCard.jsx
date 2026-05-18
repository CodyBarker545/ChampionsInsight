import React, { useEffect, useMemo, useState } from "react";
import { classifyTournamentArchetype } from "../api/championsInsightApi.js";

function normalizeTeamNames(teamMembers = []) {
  return teamMembers
    .map((member) => String(member?.name || member?.pokemonName || "").trim())
    .filter((name) => {
      if (!name) {
        return false;
      }

      const normalized = name.toLowerCase();
      return (
        !normalized.startsWith("opponent") &&
        !normalized.startsWith("pokemon ")
      );
    });
}

function buildClassificationPayload(teamMembers = []) {
  const pokemon = [];
  const moves = [];
  const items = [];
  const abilities = [];

  teamMembers.forEach((member) => {
    const name = String(member?.name || member?.pokemonName || "").trim();
    if (!name) {
      return;
    }

    const normalized = name.toLowerCase();
    if (normalized.startsWith("opponent") || normalized.startsWith("pokemon ")) {
      return;
    }

    pokemon.push(name);
    moves.push((member?.moves || []).filter(Boolean));
    items.push(String(member?.item || "").trim());
    abilities.push(String(member?.ability || "").trim());
  });

  return { pokemon, moves, items, abilities };
}

function formatPercent(value) {
  const number = Number(value || 0);
  if (!Number.isFinite(number) || number <= 0) {
    return "0%";
  }
  return `${Math.round(number * 100)}%`;
}

function formatLabel(value) {
  return String(value || "")
    .replaceAll("-", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function formatNote(value) {
  const raw = String(value || "").trim().toLowerCase();

  const noteMap = {
    "missing abilities": "Missing Abilities",
    "missing date": "Missing Tournament Date",
    "missing items": "Missing Items",
    "missing moves": "Missing Moves",
    "missing pokemon stats": "Missing Pokemon Stats",
    "missing win/loss data": "Missing Win/Loss Data",
  };

  return noteMap[raw] || formatLabel(raw);
}

function topScores(scores = {}, limit = 3) {
  return Object.entries(scores)
    .sort((a, b) => (b[1]?.score ?? 0) - (a[1]?.score ?? 0))
    .slice(0, limit);
}

function ArchetypeInsightCard({
  team,
  title = "Likely Archetype",
  eyebrow = "Tournament Insight",
  emptyMessage = "Detect or enter a team to see archetype context.",
}) {
  const classificationPayload = useMemo(() => buildClassificationPayload(team), [team]);
  const teamNames = classificationPayload.pokemon;
  const teamKey = JSON.stringify(classificationPayload).toLowerCase();

  const [insight, setInsight] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [isOpen, setIsOpen] = useState(false);

  useEffect(() => {
    setIsOpen(false);

    if (!teamNames.length) {
      setInsight(null);
      setError("");
      setIsLoading(false);
      return undefined;
    }

    const controller = new AbortController();
    let isActive = true;

    async function loadInsight() {
      setIsLoading(true);
      setError("");
      setInsight(null);

      try {
        const result = await classifyTournamentArchetype(
          classificationPayload,
          controller.signal,
        );
        if (!isActive) {
          return;
        }
        setInsight(result);
      } catch (requestError) {
        if (
          !isActive ||
          requestError?.name === "AbortError" ||
          requestError?.name === "TimeoutError"
        ) {
          return;
        }
        setInsight(null);
        setError(
          requestError.message ||
            "Archetype insights are temporarily unavailable.",
        );
      } finally {
        if (isActive) {
          setIsLoading(false);
        }
      }
    }

    void loadInsight();

    return () => {
      isActive = false;
      controller.abort();
    };
  }, [classificationPayload, teamKey, teamNames]);

  const rankedScores = useMemo(() => topScores(insight?.scores), [insight]);
  const matchingClusterLabel =
    insight?.matchingCluster?.suggestedLabel ||
    insight?.matchingCluster?.cluster?.suggestedLabel ||
    "";
  const clusterPokemon = insight?.matchingCluster?.cluster?.commonPokemon ?? [];
  const clusterMoves = insight?.matchingCluster?.cluster?.commonMoves ?? [];
  const clusterItems = insight?.matchingCluster?.cluster?.commonItems ?? [];
  const headline = insight?.displayLabel || insight?.primaryArchetype || "Unknown";
  const subHeadline =
    matchingClusterLabel ||
    insight?.shortExplanation ||
    insight?.secondaryArchetype ||
    "Tap to view tournament-backed details.";

  return (
    <>
      <section className="archetype-insight-dock" aria-live="polite">
        {!teamNames.length ? (
          <p className="archetype-insight-inline-copy">{emptyMessage}</p>
        ) : isLoading ? (
          <p className="archetype-insight-inline-copy">
            Reading tournament archetype...
          </p>
        ) : error ? (
          <p className="helper-text warning archetype-inline-warning">
            {error}
          </p>
        ) : insight ? (
          <button
            type="button"
            className="archetype-insight-trigger"
            onClick={() => setIsOpen(true)}
            aria-haspopup="dialog"
            aria-expanded={isOpen}
          >
            <div className="archetype-insight-trigger-copy">
              <span className="archetype-insight-eyebrow">{eyebrow}</span>
              <strong>
                {title}: {headline}
              </strong>
              <span>{subHeadline}</span>
            </div>
            <div className="archetype-insight-trigger-meta">
              <span className="archetype-confidence-pill">
                {formatPercent(insight.confidence)}
              </span>
              <span className="archetype-insight-open-label">View</span>
            </div>
          </button>
        ) : null}
      </section>

      {isOpen && insight && (
        <div
          className="archetype-insight-backdrop"
          onClick={() => setIsOpen(false)}
          role="presentation"
        >
          <section
            className="archetype-insight-modal"
            role="dialog"
            aria-modal="true"
            aria-label={title}
            onClick={(event) => event.stopPropagation()}
          >
            <header className="archetype-insight-header">
              <div>
                <span className="archetype-insight-eyebrow">{eyebrow}</span>
                <h3>{title}</h3>
              </div>
              <div className="archetype-insight-modal-actions">
                {insight?.confidence != null && (
                  <div className="archetype-confidence-pill">
                    {formatPercent(insight.confidence)}
                  </div>
                )}
                <button
                  type="button"
                  className="archetype-insight-close"
                  onClick={() => setIsOpen(false)}
                  aria-label="Close archetype insight"
                >
                  X
                </button>
              </div>
            </header>

            <div className="archetype-insight-body">
              <div className="archetype-headline">
                <strong>{headline}</strong>
                {insight.secondaryArchetype && (
                  <span>Secondary: {insight.secondaryArchetype}</span>
                )}
              </div>

              {insight?.shortExplanation && (
                <p className="archetype-cluster-copy">
                  {insight.shortExplanation}
                </p>
              )}

              {matchingClusterLabel && (
                <p className="archetype-cluster-copy">
                  Cluster match: <strong>{matchingClusterLabel}</strong>
                </p>
              )}

              {insight?.confidenceLabel && (
                <p className="archetype-cluster-copy">
                  Confidence: <strong>{insight.confidenceLabel}</strong>
                </p>
              )}

              {rankedScores.length > 0 && (
                <section className="archetype-score-block">
                  <h4>Top Scores</h4>
                  <div className="archetype-score-list">
                    {rankedScores.map(([name, value]) => (
                      <article className="archetype-score-row" key={name}>
                        <span>{name}</span>
                        <strong>{formatPercent(value?.score)}</strong>
                      </article>
                    ))}
                  </div>
                </section>
              )}

              {(insight.reasons || []).length > 0 && (
                <section className="archetype-score-block">
                  <h4>Why</h4>
                  <ul className="archetype-reason-list">
                    {insight.reasons.map((reason) => (
                      <li key={reason}>{reason}</li>
                    ))}
                  </ul>
                </section>
              )}

              {insight?.detailedReasons?.secondary?.summary && (
                <section className="archetype-score-block">
                  <h4>Secondary Read</h4>
                  <p className="archetype-insight-inline-copy">
                    {insight.detailedReasons.secondary.summary}
                  </p>
                </section>
              )}

              <div className="archetype-context-grid">
                {clusterPokemon.length > 0 && (
                  <section className="archetype-score-block">
                    <h4>Common Pokemon</h4>
                    <div className="archetype-chip-list">
                      {clusterPokemon.slice(0, 6).map((entry) => (
                        <span className="archetype-chip" key={entry.name}>
                          {formatLabel(entry.name)}
                        </span>
                      ))}
                    </div>
                  </section>
                )}

                {clusterMoves.length > 0 && (
                  <section className="archetype-score-block">
                    <h4>Common Moves</h4>
                    <div className="archetype-chip-list">
                      {clusterMoves.slice(0, 6).map((entry) => (
                        <span className="archetype-chip" key={entry.name}>
                          {formatLabel(entry.name)}
                        </span>
                      ))}
                    </div>
                  </section>
                )}

                {clusterItems.length > 0 && (
                  <section className="archetype-score-block">
                    <h4>Common Items</h4>
                    <div className="archetype-chip-list">
                      {clusterItems.slice(0, 6).map((entry) => (
                        <span className="archetype-chip" key={entry.name}>
                          {formatLabel(entry.name)}
                        </span>
                      ))}
                    </div>
                  </section>
                )}
              </div>

              {(insight.dataQualityNotes || []).length > 0 && (
                <section className="archetype-score-block">
                  <h4>Notes</h4>
                  <ul className="archetype-note-list">
                    {insight.dataQualityNotes.map((note) => (
                      <li key={note}>{formatNote(note)}</li>
                    ))}
                  </ul>
                </section>
              )}
            </div>
          </section>
        </div>
      )}
    </>
  );
}

export default ArchetypeInsightCard;
