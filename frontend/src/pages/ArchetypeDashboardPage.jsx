import React, { useEffect, useState } from "react";
import {
  fetchTournamentArchetypeClusters,
  fetchTournamentArchetypeRules,
} from "../api/championsInsightApi.js";
import archetypeNotesData from "../../../backend/data/championsinsight_archetype_notes.json";

function normalizeKey(value = "") {
  return String(value)
    .toLowerCase()
    .replace(/pokÃ©mon/gi, "pokemon")
    .replace(/pokemon/gi, "pokemon")
    .replace(/[^a-z0-9]+/g, "");
}

function formatName(value = "") {
  return String(value)
    .replace(/pokÃ©mon/gi, "Pokémon")
    .replaceAll("-", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function formatPercent(value, digits = 1) {
  const numericValue = Number(value ?? 0);
  if (!Number.isFinite(numericValue)) {
    return "N/A";
  }

  return `${(numericValue * 100).toFixed(digits)}%`;
}

function formatCompactNumber(value) {
  const numericValue = Number(value ?? 0);
  if (!Number.isFinite(numericValue)) {
    return "0";
  }

  return new Intl.NumberFormat("en-US", {
    notation: "compact",
    maximumFractionDigits: 1,
  }).format(numericValue);
}

function formatWholePercent(value) {
  const numericValue = Number(value ?? 0);
  if (!Number.isFinite(numericValue)) {
    return "0%";
  }

  return `${Math.round(numericValue)}%`;
}

function formatStat(value) {
  const numericValue = Number(value ?? 0);
  if (!Number.isFinite(numericValue)) {
    return "N/A";
  }

  return numericValue.toFixed(1);
}

function formatDataNote(note) {
  if (typeof note === "string") {
    return formatName(note);
  }

  if (Array.isArray(note)) {
    return note.map((entry) => formatDataNote(entry)).filter(Boolean).join(" • ");
  }

  if (note && typeof note === "object") {
    const preferredKeys = [
      "message",
      "note",
      "warning",
      "summary",
      "label",
      "reason",
      "details",
      "detail",
      "missingField",
      "field",
      "source",
    ];
    const collected = preferredKeys
      .map((key) => note[key])
      .filter(Boolean)
      .map((value) => formatName(String(value)));

    if (collected.length > 0) {
      return collected.join(" • ");
    }

    const fallback = Object.entries(note)
      .filter(([, value]) => value != null && value !== "")
      .map(([key, value]) => `${formatName(key)}: ${formatName(String(value))}`);

    return fallback.join(" • ");
  }

  return "";
}

function buildDashboardNotes(ruleNotes = [], clusterNotes = []) {
  return [...new Set(
    [...ruleNotes, ...clusterNotes]
      .map((note) => formatDataNote(note))
      .filter(Boolean),
  )];
}

function createNotesLookup() {
  const notes = archetypeNotesData?.archetypes ?? {};
  const lookup = new Map();

  Object.values(notes).forEach((entry) => {
    const candidates = [
      entry?.id,
      entry?.displayName,
      entry?.shortDescription,
    ];

    candidates.forEach((candidate) => {
      const key = normalizeKey(candidate);
      if (key) {
        lookup.set(key, entry);
      }
    });

    if (entry?.displayName === "Stall / Control") {
      lookup.set(normalizeKey("Stall or Control"), entry);
    }

    if (entry?.displayName === "Hyper Offense") {
      lookup.set(normalizeKey("Hyper-Offense"), entry);
    }
  });

  return lookup;
}

function buildArchetypeRows(ruleResponse, clusterResponse) {
  const clusterGroups = clusterResponse?.clustersByParentArchetype ?? {};
  const notesLookup = createNotesLookup();
  const archetypeSummaries = Array.isArray(ruleResponse?.archetypeSummaries)
    ? ruleResponse.archetypeSummaries
    : [];

  return archetypeSummaries
    .map((summary) => {
      const archetypeName = summary?.archetype || "Unknown";
      return {
        ...summary,
        notes:
          notesLookup.get(normalizeKey(archetypeName)) ??
          notesLookup.get(normalizeKey(summary?.displayName)) ??
          null,
        clusters: clusterGroups[archetypeName] ?? [],
      };
    })
    .sort((left, right) => (right?.teamCount ?? 0) - (left?.teamCount ?? 0));
}

function InfoList({ title, entries = [], emptyLabel = "No data available." }) {
  return (
    <section className="archetype-dashboard-subcard">
      <h4>{title}</h4>
      {entries.length > 0 ? (
        <ul className="archetype-dashboard-list">
          {entries.map((entry) => (
            <li key={`${title}-${entry.name}`}>
              <span>{formatName(entry.name)}</span>
              <strong>{formatCompactNumber(entry.count)}</strong>
            </li>
          ))}
        </ul>
      ) : (
        <p className="archetype-dashboard-empty">{emptyLabel}</p>
      )}
    </section>
  );
}

function ClusterCard({ cluster }) {
  const clusterStats = cluster?.averageTeamStats ?? {};

  return (
    <article className="archetype-dashboard-cluster-card">
      <div className="archetype-dashboard-cluster-header">
        <div>
          <strong>{cluster?.suggestedLabel || cluster?.clusterId || "Cluster"}</strong>
          <span>{cluster?.teamCount ?? 0} teams</span>
        </div>
        <div className="archetype-dashboard-cluster-metrics">
          {cluster?.averageWinRate != null && (
            <span>Win Rate {formatPercent(cluster.averageWinRate)}</span>
          )}
        </div>
      </div>

      <div className="archetype-dashboard-cluster-grid">
        <InfoList title="Common Pokémon" entries={cluster?.commonPokemon?.slice(0, 6) ?? []} />
        <InfoList title="Common Moves" entries={cluster?.commonMoves?.slice(0, 6) ?? []} />
        <InfoList title="Common Items" entries={cluster?.commonItems?.slice(0, 6) ?? []} />
      </div>

      <div className="archetype-dashboard-stat-row">
        <span>Avg Speed {formatStat(clusterStats.averageSpeed)}</span>
        <span>Avg Attack {formatStat(clusterStats.averageAttack)}</span>
        <span>Avg Sp. Atk {formatStat(clusterStats.averageSpecialAttack)}</span>
        <span>Avg Bulk {formatStat(clusterStats.averageBulk)}</span>
      </div>
    </article>
  );
}

function ArchetypeDetails({ archetype, totalTeams }) {
  const percentOfField =
    totalTeams > 0 ? ((archetype?.teamCount ?? 0) / totalTeams) * 100 : 0;
  const notes = archetype?.notes;

  return (
    <article className="archetype-dashboard-card">
      <header className="archetype-dashboard-card-header">
        <div>
          <p className="archetype-dashboard-eyebrow">
            {notes?.category ? formatName(notes.category) : "Tournament Archetype"}
          </p>
          <h3>{archetype?.archetype || "Unknown"}</h3>
          <p className="archetype-dashboard-summary">
            {notes?.shortDescription || "Tournament summary for this archetype."}
          </p>
        </div>
        <div className="archetype-dashboard-card-metrics">
          <strong>{formatCompactNumber(archetype?.teamCount ?? 0)}</strong>
          <span>{formatWholePercent(percentOfField)} of teams</span>
          {archetype?.averageWinRate != null && (
            <span>Avg win rate {formatPercent(archetype.averageWinRate)}</span>
          )}
        </div>
      </header>

      <div
        className="archetype-dashboard-share-bar"
        aria-label={`${archetype?.archetype || "Unknown"} share of teams`}
      >
        <span style={{ width: `${Math.max(percentOfField, 2)}%` }} />
      </div>

      {notes && (
        <section className="archetype-dashboard-notes">
          <div>
            <h4>Team Notes</h4>
            <ul>
              {(notes.userNotes ?? []).slice(0, 3).map((note) => (
                <li key={note}>{note.replace(/PokÃ©mon/g, "Pokémon")}</li>
              ))}
            </ul>
          </div>
          <div>
            <h4>Battle Advice</h4>
            <ul>
              {(notes.battleAdvice ?? []).slice(0, 3).map((note) => (
                <li key={note}>{note.replace(/PokÃ©mon/g, "Pokémon")}</li>
              ))}
            </ul>
          </div>
        </section>
      )}

      <div className="archetype-dashboard-card-grid">
        <InfoList title="Most Common Pokémon" entries={archetype?.mostCommonPokemon?.slice(0, 6) ?? []} />
        <InfoList title="Most Common Moves" entries={archetype?.mostCommonMoves?.slice(0, 6) ?? []} />
        <InfoList title="Most Common Items" entries={archetype?.mostCommonItems?.slice(0, 6) ?? []} />
      </div>

      {archetype?.clusters?.length > 0 ? (
        <section className="archetype-dashboard-clusters">
          <div className="archetype-dashboard-section-heading">
            <h4>Cluster Summaries</h4>
            <span>{archetype.clusters.length} clusters</span>
          </div>
          <div className="archetype-dashboard-cluster-list">
            {archetype.clusters.map((cluster) => (
              <ClusterCard key={cluster.clusterId} cluster={cluster} />
            ))}
          </div>
        </section>
      ) : (
        <section className="archetype-dashboard-clusters">
          <div className="archetype-dashboard-section-heading">
            <h4>Cluster Summaries</h4>
          </div>
          <p className="archetype-dashboard-empty">
            No cluster summary is available for this archetype yet.
          </p>
        </section>
      )}
    </article>
  );
}

function ArchetypeDashboardPage({ onBack }) {
  const [dashboardData, setDashboardData] = useState({
    totalTeamsAnalyzed: 0,
    totalClusters: 0,
    archetypes: [],
    ruleNotes: [],
    clusterNotes: [],
  });
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState("");
  const [selectedArchetype, setSelectedArchetype] = useState(null);
  const topArchetype = dashboardData.archetypes[0] ?? null;
  const dashboardNotes = buildDashboardNotes(
    dashboardData.ruleNotes,
    dashboardData.clusterNotes,
  );

  useEffect(() => {
    const controller = new AbortController();
    let isActive = true;

    async function loadDashboard() {
      setIsLoading(true);
      setError("");

      try {
        const [ruleResponse, clusterResponse] = await Promise.all([
          fetchTournamentArchetypeRules(controller.signal),
          fetchTournamentArchetypeClusters(controller.signal),
        ]);

        if (!isActive) {
          return;
        }

        setDashboardData({
          totalTeamsAnalyzed:
            ruleResponse?.totalTeamsAnalyzed ??
            clusterResponse?.totalTeamsAnalyzed ??
            0,
          totalClusters: clusterResponse?.totalClusters ?? 0,
          archetypes: buildArchetypeRows(ruleResponse, clusterResponse),
          ruleNotes: ruleResponse?.dataQualityNotes ?? [],
          clusterNotes: clusterResponse?.dataQualityNotes ?? [],
        });
      } catch (requestError) {
        if (
          requestError?.name === "AbortError" ||
          requestError?.name === "TimeoutError"
        ) {
          return;
        }

        setError(
          requestError?.message ||
            "Could not load tournament archetype dashboard data.",
        );
      } finally {
        if (isActive) {
          setIsLoading(false);
        }
      }
    }

    void loadDashboard();

    return () => {
      isActive = false;
      controller.abort();
    };
  }, []);

  useEffect(() => {
    if (!selectedArchetype) {
      return undefined;
    }

    function handleKeyDown(event) {
      if (event.key === "Escape") {
        setSelectedArchetype(null);
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [selectedArchetype]);

  return (
    <main className="app-shell">
      <section className="workspace">
        <header className="topbar">
          <div className="topbar-camera">
            <button type="button" onClick={onBack}>
              Back to Battle Prep
            </button>
          </div>
          <div className="archetype-dashboard-title-block">
            <p className="archetype-dashboard-kicker">Tournament Meta Scan</p>
            <strong>Archetype Dashboard</strong>
            <span>Tournament distribution, signatures, and cluster summaries.</span>
          </div>
          <div className="topbar-actions">
            <span className="status-pill">
              {dashboardData.totalTeamsAnalyzed} teams
            </span>
          </div>
        </header>

        <section className="archetype-dashboard-page">
          {isLoading ? (
            <section className="archetype-dashboard-state">
              <h2>Loading archetype trends...</h2>
              <p>Pulling tournament rule summaries and cluster breakdowns.</p>
            </section>
          ) : error ? (
            <section className="archetype-dashboard-state archetype-dashboard-state-error">
              <h2>Dashboard unavailable</h2>
              <p>{error}</p>
            </section>
          ) : (
            <>
              <section className="archetype-dashboard-hero">
                <div className="archetype-dashboard-hero-copy">
                  <p className="archetype-dashboard-kicker">Meta Read</p>
                  <h1>See what is actually shaping the field.</h1>
                  <p>
                    Compare archetype pressure, identify dominant structures, and
                    spot the clusters driving tournament trends.
                  </p>
                </div>
                <div className="archetype-dashboard-hero-highlight">
                  <span>Current front-runner</span>
                  <strong>{topArchetype?.archetype || "No archetype loaded"}</strong>
                  <p>
                    {topArchetype
                      ? `${formatWholePercent(topArchetype.percentageOfTeams)} of teams • ${formatCompactNumber(topArchetype.teamCount)} entries`
                      : "Waiting for tournament summary data."}
                  </p>
                </div>
              </section>

              <section className="archetype-dashboard-overview">
                <article className="archetype-dashboard-overview-card archetype-dashboard-overview-card-primary">
                  <span>Total Teams Analyzed</span>
                  <strong>{formatCompactNumber(dashboardData.totalTeamsAnalyzed)}</strong>
                </article>
                <article className="archetype-dashboard-overview-card">
                  <span>Archetypes</span>
                  <strong>{dashboardData.archetypes.length}</strong>
                </article>
                <article className="archetype-dashboard-overview-card">
                  <span>Total Clusters</span>
                  <strong>{formatCompactNumber(dashboardData.totalClusters)}</strong>
                </article>
              </section>

              <section className="archetype-dashboard-distribution">
                <div className="archetype-dashboard-section-heading">
                  <h2>Archetype Distribution</h2>
                  <span>
                    {topArchetype
                      ? `${topArchetype.archetype} leads the field`
                      : "Share of the tournament field"}
                  </span>
                </div>
                <p className="archetype-dashboard-section-note">
                  Click any archetype row to open the full scouting card.
                </p>

                <div className="archetype-dashboard-table">
                  <div className="archetype-dashboard-table-head">
                    <span>Archetype</span>
                    <span>Teams</span>
                    <span>Field Share</span>
                    <span>Avg Win Rate</span>
                  </div>
                  {dashboardData.archetypes.map((archetype) => (
                    <button
                      type="button"
                      className="archetype-dashboard-table-row archetype-dashboard-table-button"
                      key={archetype.archetype}
                      onClick={() => setSelectedArchetype(archetype)}
                    >
                      <div className="archetype-dashboard-table-archetype">
                        <strong>{archetype.archetype}</strong>
                        <span>
                          {archetype.notes?.category
                            ? formatName(archetype.notes.category)
                            : "Meta profile"}
                        </span>
                      </div>
                      <span>{formatCompactNumber(archetype.teamCount ?? 0)}</span>
                      <span>{formatWholePercent(archetype.percentageOfTeams)}</span>
                      <span>
                        {archetype.averageWinRate != null
                          ? formatPercent(archetype.averageWinRate)
                          : "N/A"}
                      </span>
                    </button>
                  ))}
                </div>
              </section>

              {dashboardNotes.length > 0 && (
                <section className="archetype-dashboard-data-notes">
                  <div className="archetype-dashboard-section-heading">
                    <h2>Data Notes</h2>
                    <span>{dashboardNotes.length} quality checks</span>
                  </div>
                  <ul>
                    {dashboardNotes.map((note) => (
                      <li key={note}>{note}</li>
                    ))}
                  </ul>
                </section>
              )}

            </>
          )}
        </section>
      </section>

      {selectedArchetype ? (
        <div
          className="archetype-dashboard-modal-backdrop"
          onClick={() => setSelectedArchetype(null)}
          role="presentation"
        >
          <section
            className="archetype-dashboard-modal"
            role="dialog"
            aria-modal="true"
            aria-label={selectedArchetype.archetype || "Archetype details"}
            onClick={(event) => event.stopPropagation()}
          >
            <div className="archetype-dashboard-modal-header">
              <div>
                <p className="archetype-dashboard-kicker">Archetype Scout</p>
                <h2>{selectedArchetype.archetype}</h2>
              </div>
              <button
                type="button"
                className="archetype-dashboard-modal-close"
                onClick={() => setSelectedArchetype(null)}
                aria-label="Close archetype details"
              >
                X
              </button>
            </div>

            <div className="archetype-dashboard-modal-body">
              <ArchetypeDetails
                archetype={selectedArchetype}
                totalTeams={dashboardData.totalTeamsAnalyzed}
              />
            </div>
          </section>
        </div>
      ) : null}
    </main>
  );
}

export default ArchetypeDashboardPage;
