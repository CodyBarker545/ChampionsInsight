// Defines the grid that shows analysis outputs.
import React from "react";
import { Activity, Swords, Zap } from "lucide-react";
import ResultCard from "./ResultCard.jsx";

// Displays speed, damage, and recommendation results.
function ResultsGrid({ analysis, error }) {
  const speed = analysis?.speed ?? null;
  const damage = analysis?.damage ?? null;
  const recommendations = Array.isArray(analysis?.recommendations)
    ? analysis.recommendations
    : [];

  const damageRange =
    damage?.range ??
    damage?.damageRange ??
    damage?.rolls ??
    null;

  const percentRange =
    damage?.percentRange ??
    damage?.percentDamageRange ??
    null;

  const koChance =
    damage?.koChance ??
    damage?.chanceToKo ??
    damage?.koText ??
    "";

  return (
    <section className="results-grid" aria-live="polite">
      {error && <div className="alert">{error}</div>}

      <ResultCard icon={<Zap size={20} />} title="Speed">
        {speed ? (
          <>
            <strong>{speed.result ?? "Speed calculated"}</strong>
            <p>
              {speed.yourSpeed ?? speed.user?.modifiedSpeed ?? "?"} vs{" "}
              {speed.opponentSpeed ?? speed.opponent?.modifiedSpeed ?? "?"}
            </p>
          </>
        ) : (
          <p>Run an analysis to compare speed.</p>
        )}
      </ResultCard>

      <ResultCard icon={<Swords size={20} />} title="Damage">
        {damage ? (
          <>
            <strong>
              {Array.isArray(damageRange)
                ? `${damageRange[0]} - ${damageRange[1]}`
                : damageRange ?? "Damage calculated"}
            </strong>

            {Array.isArray(percentRange) && (
              <p>
                {percentRange[0]}% - {percentRange[1]}%
              </p>
            )}

            {koChance && <p>{koChance}</p>}

            {analysis?.move && (
              <p>
                {analysis.attacker} used {analysis.move} into {analysis.defender}
              </p>
            )}
          </>
        ) : (
          <p>Damage ranges will appear here.</p>
        )}
      </ResultCard>

      <ResultCard icon={<Activity size={20} />} title="Recommendation">
        {analysis ? (
          <>
            <strong>{analysis.summary ?? "No matchup recommendation available."}</strong>

            {recommendations.length > 0 ? (
              <ul>
                {recommendations.map((recommendation) => (
                  <li key={recommendation}>{recommendation}</li>
                ))}
              </ul>
            ) : (
              <p>
                Damage calculation completed. Use the damage card for the result.
              </p>
            )}
          </>
        ) : (
          <p>Strategic guidance will appear after analysis.</p>
        )}
      </ResultCard>
    </section>
  );
}

export default ResultsGrid;