import React, { useEffect, useMemo, useState } from "react";
import { fetchPokemonMetaIntelligence } from "../api/championsInsightApi.js";

function formatPercent(value, digits = 1) {
  const numericValue = Number(value ?? 0);
  if (!Number.isFinite(numericValue)) {
    return "N/A";
  }
  return `${(numericValue * 100).toFixed(digits)}%`;
}

function formatChangePercent(value, digits = 1) {
  const numericValue = Number(value ?? 0);
  if (!Number.isFinite(numericValue)) {
    return "N/A";
  }
  return `${numericValue > 0 ? "+" : ""}${numericValue.toFixed(digits)}%`;
}

function formatName(value = "") {
  return String(value || "")
    .replaceAll("-", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function directionLabel(direction = "stable") {
  const normalized = String(direction || "stable").toLowerCase();
  if (normalized === "rising") {
    return "Rising";
  }
  if (normalized === "falling") {
    return "Falling";
  }
  return "Stable";
}

function TrendList({ entries = [], kind = "generic", emptyLabel }) {
  if (!entries.length) {
    return <p className="pokemon-meta-empty">{emptyLabel}</p>;
  }

  return (
    <div className="pokemon-meta-list">
      {entries.map((entry) => (
        <article
          className="pokemon-meta-list-item"
          key={`${kind}-${entry.name || entry.corePokemon?.join("-")}`}
        >
          <div className="pokemon-meta-list-copy">
            <strong>
              {entry.name
                ? formatName(entry.name)
                : (entry.corePokemon || []).map(formatName).join(" + ")}
            </strong>
            {entry.explanation ? <p>{entry.explanation}</p> : null}
          </div>

          <div className="pokemon-meta-list-metrics">
            {typeof entry.currentPercentage === "number" ? (
              <span>{entry.currentPercentage.toFixed(1)}% now</span>
            ) : null}
            {typeof entry.previousPercentage === "number" ? (
              <span>{entry.previousPercentage.toFixed(1)}% prev</span>
            ) : null}
            {typeof entry.absoluteChange === "number" ? (
              <span
                className={`pokemon-meta-change is-${String(entry.trendDirection || "stable").toLowerCase()}`}
              >
                {entry.absoluteChange > 0 ? "+" : ""}
                {entry.absoluteChange.toFixed(1)} pts
              </span>
            ) : null}
            {typeof entry.count === "number" ? <span>{entry.count} teams</span> : null}
          </div>
        </article>
      ))}
    </div>
  );
}

function PokemonMetaIntelligencePanel({ pokemonName }) {
  const [meta, setMeta] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!pokemonName) {
      setMeta(null);
      return;
    }

    let isActive = true;
    setIsLoading(true);
    setError("");

    fetchPokemonMetaIntelligence(pokemonName)
      .then((result) => {
        if (!isActive) {
          return;
        }
        setMeta(result);
      })
      .catch((requestError) => {
        if (!isActive) {
          return;
        }
        const rawMessage = String(
          requestError?.message || "Could not load Pokemon meta intelligence.",
        );
        setError(
          rawMessage.includes("backend route may be unavailable")
            ? "Meta intelligence is not available from the backend yet. Restart the backend so the new Pokedex meta route is loaded."
            : rawMessage,
        );
      })
      .finally(() => {
        if (isActive) {
          setIsLoading(false);
        }
      });

    return () => {
      isActive = false;
    };
  }, [pokemonName]);

  const usageTrend = meta?.usageTrend ?? {};
  const panelTitle = useMemo(
    () => formatName(meta?.pokemon || pokemonName || "Pokemon"),
    [meta?.pokemon, pokemonName],
  );

  return (
    <section className="pokemon-meta-panel" aria-live="polite">
      <div className="pokemon-meta-header">
        <div>
          <span className="pokemon-meta-kicker">Meta Intelligence</span>
          <h3>{panelTitle}</h3>
        </div>
        {usageTrend?.trendDirection ? (
          <span
            className={`pokemon-meta-direction is-${String(usageTrend.trendDirection).toLowerCase()}`}
          >
            {directionLabel(usageTrend.trendDirection)}
          </span>
        ) : null}
      </div>

      {isLoading ? <p className="pokemon-meta-status">Loading tournament-backed meta trends...</p> : null}
      {error ? <p className="pokemon-meta-status pokemon-meta-status-error">{error}</p> : null}

      {!isLoading && !error && meta ? (
        <>
          <div className="pokemon-meta-usage-card">
            <div>
              <span>Current Usage</span>
              <strong>{formatPercent(usageTrend.currentUsageRate)}</strong>
            </div>
            <div>
              <span>Previous Usage</span>
              <strong>{formatPercent(usageTrend.previousUsageRate)}</strong>
            </div>
            <div>
              <span>Usage Change</span>
              <strong>{formatChangePercent(usageTrend.percentageChange)}</strong>
            </div>
            <div>
              <span>Confidence</span>
              <strong>{usageTrend.confidenceLabel || "Low"}</strong>
            </div>
          </div>

          {usageTrend.explanation ? (
            <p className="pokemon-meta-summary">{usageTrend.explanation}</p>
          ) : null}

          <div className="pokemon-meta-sections">
            <details className="pokemon-meta-section" open>
              <summary>Move Trends</summary>
              <TrendList
                entries={meta.moveTrends || []}
                kind="move"
                emptyLabel="No move trend signal available for the current comparison window."
              />
            </details>

            <details className="pokemon-meta-section">
              <summary>Item Trends</summary>
              <TrendList
                entries={meta.itemTrends || []}
                kind="item"
                emptyLabel="No item trend signal available for the current comparison window."
              />
            </details>

            <details className="pokemon-meta-section">
              <summary>Archetype Trends</summary>
              <TrendList
                entries={meta.archetypeTrends || []}
                kind="archetype"
                emptyLabel="No archetype trend signal available for the current comparison window."
              />
            </details>

            <details className="pokemon-meta-section">
              <summary>Win-Rate Trend</summary>
              {meta.winRateTrend && Object.keys(meta.winRateTrend).length > 0 ? (
                <div className="pokemon-meta-winrate-card">
                  <div>
                    <span>Current</span>
                    <strong>{formatPercent(meta.winRateTrend.currentWinRate)}</strong>
                  </div>
                  <div>
                    <span>Previous</span>
                    <strong>{formatPercent(meta.winRateTrend.previousWinRate)}</strong>
                  </div>
                  <div>
                    <span>Direction</span>
                    <strong>{directionLabel(meta.winRateTrend.trendDirection)}</strong>
                  </div>
                  <p>{meta.winRateTrend.explanation}</p>
                </div>
              ) : (
                <p className="pokemon-meta-empty">Win-rate trend was not available for this Pokemon.</p>
              )}
            </details>

            <details className="pokemon-meta-section">
              <summary>Common Cores</summary>
              <TrendList
                entries={meta.commonCores || []}
                kind="common-core"
                emptyLabel="No core data available for this Pokemon right now."
              />
            </details>

            <details className="pokemon-meta-section">
              <summary>New or Rising Cores</summary>
              <TrendList
                entries={meta.newCores || []}
                kind="new-core"
                emptyLabel="No new or rising core signal was detected for this Pokemon."
              />
            </details>

            <details className="pokemon-meta-section">
              <summary>Data Quality Notes</summary>
              {(meta.dataQualityNotes || []).length ? (
                <ul className="pokemon-meta-notes">
                  {meta.dataQualityNotes.map((note) => (
                    <li key={note}>{note}</li>
                  ))}
                </ul>
              ) : (
                <p className="pokemon-meta-empty">No data quality warnings for this Pokemon.</p>
              )}
            </details>
          </div>
        </>
      ) : null}
    </section>
  );
}

export default PokemonMetaIntelligencePanel;
