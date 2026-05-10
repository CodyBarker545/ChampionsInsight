// Defines the clickable user team roster slots.
import React from "react";
import { resolveApiUrl } from "../api/championsInsightApi.js";


// Displays editable roster slots for the user's team.
function TeamForm({ selectedIndex, team, onSelectMember }) {
  return (
    <section className="roster-panel team-panel">
      <div className="roster-list">
        {team.map((member, index) => {
          const imageSource = resolveApiUrl(member.spriteUrl || member.image || "");

          return (
            <button
              className={`team-slot ${selectedIndex === index ? "selected" : ""}`}
              key={member.id ?? index}
              type="button"
              onClick={() => onSelectMember(index)}
            >
              <span className="slot-index" aria-hidden="true">
                {String(index + 1).padStart(2, "0")}
              </span>
              <span className="slot-image" aria-hidden="true">
                {imageSource ? <img src={imageSource} alt="" /> : "IMG"}
              </span>
              <span className="slot-copy">
                <span className="slot-name">{member.name || `Pokemon ${index + 1}`}</span>
                <span className="slot-subtitle">
                  {selectedIndex === index ? "Active build" : "Ready slot"}
                </span>
              </span>
            </button>
          );
        })}
      </div>

      <div className="roster-label">My Team</div>
    </section>
  );
}


export default TeamForm;
