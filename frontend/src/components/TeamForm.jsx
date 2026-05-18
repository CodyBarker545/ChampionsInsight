// Defines the clickable user team roster slots.
import React from "react";
import { resolveApiUrl } from "../api/championsInsightApi.js";


// Displays editable roster slots for the user's team.
function TeamForm({
  selectedIndex,
  team,
  onClearMember,
  onEditMember,
  onSelectMember,
}) {
  return (
    <section className="roster-panel team-panel">
      <div className="roster-list">
        {team.map((member, index) => {
          const imageSource = resolveApiUrl(member.spriteUrl || member.image || "");
          const hasPokemon = Boolean(member.name?.trim());

          return (
            <section
              className={`team-slot user-team-slot ${
                selectedIndex === index ? "selected" : ""
              } ${hasPokemon ? "filled" : "empty"}`}
              key={member.id ?? index}
              onClick={() => {
                onSelectMember(index);
                if (!hasPokemon) {
                  onEditMember?.(index);
                }
              }}
            >
              {hasPokemon && (
                <button
                  className="team-slot-clear"
                  type="button"
                  aria-label={`Clear team slot ${index + 1}`}
                  onClick={(event) => {
                    event.stopPropagation();
                    onClearMember?.(index);
                  }}
                >
                  x
                </button>
              )}

              <button
                className="team-slot-summary"
                type="button"
                onClick={(event) => {
                  event.stopPropagation();
                  onSelectMember(index);
                  if (!hasPokemon) {
                    onEditMember?.(index);
                  }
                }}
              >
              <span className="slot-index" aria-hidden="true">
                {String(index + 1).padStart(2, "0")}
              </span>
              <span className="slot-image" aria-hidden="true">
                {imageSource ? <img src={imageSource} alt="" /> : "+"}
              </span>
              <span className="slot-copy">
                <span className="slot-name">{member.name || `Pokemon ${index + 1}`}</span>
                <span className="slot-subtitle">
                  {hasPokemon
                    ? selectedIndex === index
                      ? "Active build"
                      : "Ready slot"
                    : "Click to build"}
                </span>
              </span>
              </button>
            </section>
          );
        })}
      </div>

      <div className="roster-label">My Team</div>
    </section>
  );
}


export default TeamForm;
