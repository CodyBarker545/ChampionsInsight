// Defines opponent roster controls.
import React from "react";
import { resolveApiUrl } from "../api/championsInsightApi.js";

// Displays opponent input controls.
function OpponentPanel({
  imageStatus,
  isUploadingImage,
  isLoadingOpponentPokemon = false,
  isSearchingOpponent = false,
  onClearOpponent,
  onOpponentSearchChange,
  onSelectOpponent,
  onSelectOpponentPokemon,
  opponentTeam,
  opponentSearchQuery = "",
  opponentSearchResults = [],
  selectedOpponentIndex,
}) {
  return (
    <section className="roster-column opponent-column">
      <div className="roster-panel opponent-panel">
        <div className="roster-list">
          {opponentTeam.map((opponent, index) => {
            const isPlaceholderName = opponent.name?.toLowerCase().startsWith("opponent");
            const hasPokemon = Boolean(opponent.name?.trim()) && !isPlaceholderName;
            const pokemonName = hasPokemon ? opponent.name : `Slot ${index + 1}`;
            const spriteSource = resolveApiUrl(opponent.spriteUrl || opponent.image || "");
            const isSelected = selectedOpponentIndex === index;

            return (
              <section
                className={`team-slot opponent-team-slot ${
                  isSelected ? "selected" : ""
                } ${hasPokemon ? "filled" : "empty"}`}
                key={opponent.id ?? pokemonName ?? index}
                onClick={() => onSelectOpponent(index)}
              >
                {hasPokemon && (
                  <button
                    className="opponent-slot-clear"
                    type="button"
                    aria-label={`Clear opponent slot ${index + 1}`}
                    onClick={(event) => {
                      event.stopPropagation();
                      onClearOpponent?.(index);
                    }}
                  >
                    x
                  </button>
                )}

                <button
                  className="opponent-slot-summary"
                  type="button"
                  onClick={(event) => {
                    event.stopPropagation();
                    onSelectOpponent(index);
                  }}
                >
                  <span className="slot-image">
                    {spriteSource ? (
                      <img
                        className="opponent-slot-sprite"
                        src={spriteSource}
                        alt={`${pokemonName} sprite`}
                      />
                    ) : (
                      "IMG"
                    )}
                  </span>

                  <span className="slot-copy">
                    <span className="slot-name">{hasPokemon ? pokemonName : `Slot ${index + 1}`}</span>
                    <span className="slot-subtitle">
                      {hasPokemon ? "Scouted target" : "Awaiting read"}
                    </span>
                  </span>
                </button>

                {isSelected && (
                  <div className="opponent-slot-search">
                    <input
                      value={opponentSearchQuery}
                      placeholder={hasPokemon ? "Search replacement" : "Search Pokemon"}
                      onClick={(event) => event.stopPropagation()}
                      onChange={(event) => onOpponentSearchChange?.(event.target.value)}
                    />

                    {(isSearchingOpponent || opponentSearchResults.length > 0) && (
                      <div className="opponent-slot-results">
                        {isSearchingOpponent && <span>Searching...</span>}
                        {opponentSearchResults.map((pokemon) => {
                          const resultImageSource = resolveApiUrl(pokemon.image);

                          return (
                            <button
                              className="opponent-result"
                              key={pokemon.name}
                              type="button"
                              disabled={isLoadingOpponentPokemon}
                              onClick={(event) => {
                                event.stopPropagation();
                                onSelectOpponentPokemon?.(index, pokemon.name);
                              }}
                            >
                              <span className="opponent-result-image">
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

        <div className="roster-label">Opponent Team</div>
      </div>

      <div className="scout-tools">
        {isUploadingImage && <p className="helper-text">Uploading image...</p>}

        {imageStatus && <p className="helper-text success">{imageStatus}</p>}
      </div>
    </section>
  );
}

export default OpponentPanel;
