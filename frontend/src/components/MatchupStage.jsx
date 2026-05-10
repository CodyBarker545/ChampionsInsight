// Defines the center battle calculator panel for selected Pokemon.
import React, { useEffect, useMemo, useState } from "react";
import { ArrowDown, ArrowRight, ArrowUp } from "lucide-react";
import { calculateDamage, resolveApiUrl } from "../api/championsInsightApi.js";

const statLabels = {
  hp: "HP",
  attack: "Atk",
  defense: "Def",
  specialAttack: "SpA",
  specialDefense: "SpD",
  speed: "Spe",
};

const statColors = {
  hp: "#ff4d5a",
  attack: "#ff7a22",
  defense: "#facc15",
  specialAttack: "#4f8fff",
  specialDefense: "#22c55e",
  speed: "#e879f9",
};

const stageLabels = {
  attack: "Atk",
  defense: "Def",
  special_attack: "SpA",
  special_defense: "SpD",
  speed: "Spe",
};

const stageValues = [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6];

const maxStatInvestment = 32;
const minimumVisibleStatBar = 10;

const statBarRanges = {
  hp: { min: 110, max: 267 },
  attack: { min: 61, max: 260 },
  defense: { min: 54, max: 310 },
  specialAttack: { min: 31, max: 249 },
  specialDefense: { min: 50, max: 226 },
  speed: { min: 36, max: 222 },
};

const defaultFieldEffects = {
  weather: "",
  terrain: "",
  trickRoom: false,
  userTailwind: false,
  opponentTailwind: false,
  userReflect: false,
  userLightScreen: false,
  userAuroraVeil: false,
  opponentReflect: false,
  opponentLightScreen: false,
  opponentAuroraVeil: false,
};

const weatherOptions = [
  ["", "Weather"],
  ["rain", "Rain"],
  ["sun", "Sun"],
  ["sand", "Sand"],
  ["snow", "Snow"],
];

const terrainOptions = [
  ["", "Terrain"],
  ["electric", "Electric"],
  ["grassy", "Grassy"],
  ["psychic", "Psychic"],
  ["misty", "Misty"],
];

const statusOptions = [
  ["burn", "Burn"],
  ["paralysis", "Para"],
];

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

function formatNature(nature) {
  const displayName = nature.charAt(0).toUpperCase() + nature.slice(1);
  const effect = natureEffects[nature];

  if (!effect) {
    return `${displayName} (Neutral)`;
  }

  return `${displayName} (+${effect[0]} / -${effect[1]})`;
}

function getSpriteSource(pokemon) {
  if (!pokemon) {
    return "";
  }

  return resolveApiUrl(pokemon.spriteUrl || pokemon.image || "");
}

function toCalculatorStats(stats = {}) {
  return {
    hp: Number(stats.hp ?? 0),
    attack: Number(stats.attack ?? 0),
    defense: Number(stats.defense ?? 0),
    special_attack: Number(stats.specialAttack ?? stats.special_attack ?? 0),
    special_defense: Number(stats.specialDefense ?? stats.special_defense ?? 0),
    speed: Number(stats.speed ?? 0),
  };
}

function hasLoadedCalculatorStats(pokemon) {
  const stats = pokemon?.stats ?? {};
  const values = Object.keys(statLabels).map((statName) => Number(stats[statName] ?? 0));

  if (values.some((value) => !value)) {
    return false;
  }

  const isPlaceholderStats = values.every((value) => value === 100);
  if (!isPlaceholderStats) {
    return true;
  }

  return Object.keys(statLabels).some(
    (statName) =>
      Number(pokemon?.baseStats?.[statName] ?? 0) > 0 ||
      Number(pokemon?.finalStats?.[statName] ?? 0) > 0
  );
}

function applyStage(stat, stage = 0) {
  const safeStat = Number(stat ?? 0);
  const safeStage = Math.max(-6, Math.min(6, Number(stage ?? 0)));

  if (!safeStat) {
    return 0;
  }

  if (safeStage >= 0) {
    return Math.max(1, Math.floor((safeStat * (2 + safeStage)) / 2));
  }

  return Math.max(1, Math.floor((safeStat * 2) / (2 - safeStage)));
}

function applyLocalStatusSpeed(speed, status = "", ability = "") {
  if (!speed) {
    return speed;
  }

  if (status === "paralysis" && String(ability).toLowerCase() !== "quick feet") {
    return Math.floor(speed * 0.5);
  }

  return speed;
}

function toCalculatorPokemon(pokemon, boosts = {}, status = "") {
  return {
    name: pokemon?.name || "Unknown Pokemon",
    level: 50,
    types: pokemon?.types ?? [],
    stats: toCalculatorStats(pokemon?.stats),
    maxHp: Number(pokemon?.stats?.hp ?? 1),
    ability: pokemon?.ability ?? "",
    item: pokemon?.item ?? "",
    status,
    nature: pokemon?.nature ?? "",
    baseStats: toCalculatorStats(pokemon?.baseStats),
    boosts,
  };
}

function formatSpeedText(userPokemon, opponentPokemon, userSpeed, opponentSpeed) {
  const userName = userPokemon?.name || "Your Pokemon";
  const opponentName = opponentPokemon?.name || "Opponent Pokemon";

  if (!userSpeed || !opponentSpeed) {
    return "Speed check pending";
  }

  if (userSpeed === opponentSpeed) {
    return `${userName} speed ties ${opponentName}`;
  }

  return userSpeed > opponentSpeed
    ? `${userName} is faster than ${opponentName}`
    : `${userName} is slower than ${opponentName}`;
}

function hasKnownDamageRange(damageResult) {
  const range = damageResult?.damage?.percentRange;
  return Boolean(range && range !== "undefined" && !range.includes("undefined"));
}

function isNoDamageImmunity(damage = {}) {
  const notes = damage?.notes ?? [];
  const typeMultiplier = Number(damage?.modifiers?.type);

  return (
    typeMultiplier === 0 ||
    notes.some((note) =>
      String(note).toLowerCase().includes("type immunity") ||
      String(note).toLowerCase().includes("no damage")
    )
  );
}

function formatEffectiveness(damage = {}) {
  if (isNoDamageImmunity(damage)) {
    return "no effect";
  }

  const multiplier = Number(damage?.modifiers?.type);

  if (!Number.isFinite(multiplier)) {
    return "";
  }

  if (multiplier === 0) {
    return "no effect";
  }

  if (multiplier >= 4) {
    return "extremely effective";
  }

  if (multiplier > 1) {
    return "super effective";
  }

  if (multiplier <= 0.25) {
    return "extremely not effective";
  }

  if (multiplier < 1) {
    return "not very effective";
  }

  return "neutral damage";
}

function getUserSpeedStatus(speedText, userPokemon, opponentPokemon) {
  const userName = userPokemon?.name || "Your Pokemon";
  const opponentName = opponentPokemon?.name || "Opponent Pokemon";

  if (speedText.includes("tie") || speedText.includes("pending")) {
    return "tie";
  }

  if (
    speedText.startsWith(`${userName} moves first`) ||
    speedText.startsWith(`${userName} is faster`)
  ) {
    return "faster";
  }

  if (
    speedText.startsWith(`${opponentName} moves first`) ||
    speedText.startsWith(`${userName} is slower`)
  ) {
    return "slower";
  }

  return "tie";
}

function buildSideEffects(fieldEffects, prefix) {
  return {
    tailwind: fieldEffects[`${prefix}Tailwind`],
    reflect: fieldEffects[`${prefix}Reflect`],
    lightScreen: fieldEffects[`${prefix}LightScreen`],
    auroraVeil: fieldEffects[`${prefix}AuroraVeil`],
  };
}

function buildCalculatorField(fieldEffects, selectedMoveSource) {
  const attackerPrefix = selectedMoveSource === "opponent" ? "opponent" : "user";
  const defenderPrefix = selectedMoveSource === "opponent" ? "user" : "opponent";

  return {
    isDoubles: true,
    weather: fieldEffects.weather,
    terrain: fieldEffects.terrain,
    trickRoom: fieldEffects.trickRoom,
    attackerSide: buildSideEffects(fieldEffects, attackerPrefix),
    defenderSide: buildSideEffects(fieldEffects, defenderPrefix),
  };
}

function ensureFourMoves(moves = []) {
  return [...moves, "", "", "", ""].slice(0, 4);
}

function emptyBoosts() {
  return Object.fromEntries(
    Object.keys(stageLabels).map((statName) => [statName, 0])
  );
}

function findKnownMove(moveOptions = [], value = "") {
  const normalizedValue = value.trim().toLowerCase();

  if (!normalizedValue) {
    return "";
  }

  return (
    moveOptions.find((moveOption) => moveOption.toLowerCase() === normalizedValue) ??
    ""
  );
}

function getFormToggleState(pokemon) {
  const options = pokemon?.formOptions ?? [];
  const defaultForm = options.find((formOption) => formOption.isDefault) ?? options[0];
  const alternateForm =
    options.find((formOption) => !formOption.isDefault) ?? options[1] ?? options[0];
  const currentFormName = pokemon?.form || pokemon?.name || "";
  const isAlternateActive = currentFormName === alternateForm?.name;

  return {
    isAlternateActive,
    label: alternateForm?.label || alternateForm?.name || "Form",
    targetForm: isAlternateActive ? defaultForm?.name : alternateForm?.name,
  };
}

function calculateStatBarWidth(statName, statValue) {
  const safeValue = Number(statValue ?? 0);
  const range = statBarRanges[statName];

  if (!safeValue || !range) {
    return "0%";
  }

  const rangeSize = Math.max(1, range.max - range.min);
  const scaledValue = (safeValue - range.min) / rangeSize;
  const width =
    minimumVisibleStatBar +
    Math.max(0, Math.min(1, scaledValue)) * (100 - minimumVisibleStatBar);

  return `${width}%`;
}

function FormToggle({ label, onToggle, pokemon }) {
  const toggleState = getFormToggleState(pokemon);

  if (!pokemon?.formOptions?.length || pokemon.formOptions.length <= 1) {
    return null;
  }

  return (
    <button
      className={`form-toggle ${toggleState.isAlternateActive ? "active" : ""}`}
      type="button"
      aria-label={label}
      aria-pressed={toggleState.isAlternateActive}
      onClick={() => onToggle?.(toggleState.targetForm)}
    >
      {toggleState.label.replace(`${pokemon.species || ""} `, "")}
    </button>
  );
}

// Displays the selected Pokemon details and future battle calculator.
function MatchupStage({
  analysis,
  onOpponentUpdate,
  onSelectMove,
  onUserFormChange,
  opponentPokemon,
  selectedMove,
  selectedMoveSource = "user",
  userPokemon,
}) {
  const [damageResult, setDamageResult] = useState(null);
  const [damageError, setDamageError] = useState("");
  const [isCalculatingDamage, setIsCalculatingDamage] = useState(false);
  const [fieldEffects, setFieldEffects] = useState(defaultFieldEffects);
  const [opponentMoveDrafts, setOpponentMoveDrafts] = useState({});
  const [userStatus, setUserStatus] = useState("");
  const [opponentStatus, setOpponentStatus] = useState("");
  const [userBoosts, setUserBoosts] = useState(emptyBoosts);
  const [opponentBoosts, setOpponentBoosts] = useState(emptyBoosts);

  const userSpeed = Number(userPokemon?.stats?.speed ?? 0);
  const opponentSpeed = Number(opponentPokemon?.stats?.speed ?? 0);
  const displayedUserSpeed = applyLocalStatusSpeed(
    applyStage(userSpeed, userBoosts.speed),
    userStatus,
    userPokemon?.ability
  );
  const displayedOpponentSpeed = applyLocalStatusSpeed(
    applyStage(opponentSpeed, opponentBoosts.speed),
    opponentStatus,
    opponentPokemon?.ability
  );
  const opponentAbilities = opponentPokemon?.abilities ?? [];
  const activeAttacker = selectedMoveSource === "opponent" ? opponentPokemon : userPokemon;
  const activeDefender = selectedMoveSource === "opponent" ? userPokemon : opponentPokemon;
  const activeAttackerBoosts =
    selectedMoveSource === "opponent" ? opponentBoosts : userBoosts;
  const activeDefenderBoosts =
    selectedMoveSource === "opponent" ? userBoosts : opponentBoosts;
  const activeAttackerStatus =
    selectedMoveSource === "opponent" ? opponentStatus : userStatus;
  const activeDefenderStatus =
    selectedMoveSource === "opponent" ? userStatus : opponentStatus;
  const calculatorField = useMemo(
    () => buildCalculatorField(fieldEffects, selectedMoveSource),
    [fieldEffects, selectedMoveSource]
  );
  const speedText =
    damageResult?.speed?.result ??
    formatSpeedText(userPokemon, opponentPokemon, displayedUserSpeed, displayedOpponentSpeed);
  const userSpeedStatus = getUserSpeedStatus(speedText, userPokemon, opponentPokemon);

  const userSpriteSource = getSpriteSource(userPokemon);
  const opponentSpriteSource = getSpriteSource(opponentPokemon);
  const damageSummary = useMemo(() => {
    if (damageError) {
      return damageError;
    }

    if (isCalculatingDamage) {
      return "Calculating damage...";
    }

    if (!selectedMove) {
      return "Select a move";
    }

    if (!damageResult || !hasKnownDamageRange(damageResult)) {
      return "Damage range pending";
    }

    if (isNoDamageImmunity(damageResult.damage)) {
      return `${damageResult.attacker}'s ${damageResult.damage.move} has no effect on ${damageResult.defender}`;
    }

    return `${damageResult.attacker} damage range: ${damageResult.damage.percentRange} vs ${damageResult.defender}`;
  }, [damageError, damageResult, isCalculatingDamage, selectedMove]);
  const damageDetail = useMemo(() => {
    if (!damageResult || !hasKnownDamageRange(damageResult)) {
      return "Click a user or opponent move to calculate";
    }

    if (isNoDamageImmunity(damageResult.damage)) {
      return `${damageResult.damage.category} ${damageResult.damage.moveType} move, no effect`;
    }

    return [
      `${damageResult.damage.category} ${damageResult.damage.moveType} move`,
      formatEffectiveness(damageResult.damage),
      damageResult.damage.koChance,
    ].filter(Boolean).join(", ");
  }, [damageResult]);

  const speedIcon = userSpeedStatus === "tie" ? (
    <ArrowRight size={22} aria-hidden="true" />
  ) : userSpeedStatus === "faster" ? (
    <ArrowUp size={22} aria-hidden="true" />
  ) : (
    <ArrowDown size={22} aria-hidden="true" />
  );

  useEffect(() => {
    if (
      !selectedMove ||
      !activeAttacker?.name ||
      !activeDefender?.name ||
      !hasLoadedCalculatorStats(activeAttacker) ||
      !hasLoadedCalculatorStats(activeDefender)
    ) {
      setDamageResult(null);
      setDamageError("");
      return undefined;
    }

    const controller = new AbortController();
    setIsCalculatingDamage(true);
    setDamageResult(null);
    setDamageError("");

    calculateDamage({
      attacker: toCalculatorPokemon(
        activeAttacker,
        activeAttackerBoosts,
        activeAttackerStatus
      ),
      defender: toCalculatorPokemon(
        activeDefender,
        activeDefenderBoosts,
        activeDefenderStatus
      ),
      move: { name: selectedMove },
      field: calculatorField,
    }, controller.signal)
      .then((result) => {
        if (!controller.signal.aborted) {
          setDamageResult(result);
        }
      })
      .catch((error) => {
        if (error.name === "AbortError") {
          return;
        }

        if (!controller.signal.aborted) {
          setDamageResult(null);
          setDamageError(error.message);
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          setIsCalculatingDamage(false);
        }
      });

    return () => controller.abort();
  }, [
    activeAttacker,
    activeAttackerBoosts,
    activeAttackerStatus,
    activeDefender,
    activeDefenderBoosts,
    activeDefenderStatus,
    calculatorField,
    selectedMove,
  ]);

  useEffect(() => {
    setOpponentMoveDrafts({});
  }, [opponentPokemon?.name]);

  useEffect(() => {
    setUserBoosts(emptyBoosts());
    setUserStatus("");
  }, [userPokemon?.name]);

  useEffect(() => {
    setOpponentBoosts(emptyBoosts());
    setOpponentStatus("");
  }, [opponentPokemon?.name]);

  function updateFieldEffect(key, value) {
    setFieldEffects((currentEffects) => ({
      ...currentEffects,
      [key]: value,
    }));
  }

  function toggleStatus(side, status) {
    const updateStatus = side === "user" ? setUserStatus : setOpponentStatus;
    updateStatus((currentStatus) => (currentStatus === status ? "" : status));
  }

  return (
    <section className="matchup-stage" aria-label="Battle calculator">
      <div className="combatant-card">
        <span>Your Pokemon</span>

        <div
          className={`combatant-sprite-frame ${
            userSpriteSource ? "has-sprite" : "empty-sprite"
          }`}
        >
          {userSpriteSource && (
            <img
              className="combatant-sprite"
              src={userSpriteSource}
              alt={`${userPokemon?.name || "Your Pokemon"} sprite`}
            />
          )}
          <StatusControls
            label="Your Pokemon status"
            selectedStatus={userStatus}
            onToggle={(status) => toggleStatus("user", status)}
          />
        </div>

        <div className="combatant-title-row">
          <strong>{userPokemon?.name || "Choose lead"}</strong>
          <FormToggle
            label="Toggle your Pokemon form"
            pokemon={userPokemon}
            onToggle={onUserFormChange}
          />
        </div>

        <div className="meta-row">
          <small>{userPokemon?.ability || "Ability pending"}</small>
          <small>{userPokemon?.item || "Item pending"}</small>
        </div>

        <StatGrid stats={userPokemon?.stats} />

        <div className="move-list" aria-label="User moves">
          {(userPokemon?.moves ?? []).map((move) => (
            <button
              className={
                selectedMoveSource === "user" && selectedMove === move
                  ? "move-button selected"
                  : "move-button"
              }
              key={move}
              type="button"
              onClick={() => onSelectMove(move, "user")}
            >
              {move}
            </button>
          ))}
        </div>

        <StageControls
          boosts={userBoosts}
          label="User Stages"
          onChange={(statName, value) =>
            setUserBoosts((currentBoosts) => ({
              ...currentBoosts,
              [statName]: value,
            }))
          }
        />
      </div>

      <div className="calculation-lane">
        <div className="calculator-sections">
          <section className="calculator-card damage-card" aria-label="Damage and speed">
            <div className={`speed-indicator ${userSpeedStatus}`}>
              {speedIcon}
              <span>{speedText}</span>
            </div>

            <div className="damage-output">
              <span>{selectedMove || "Select a move"}</span>
              <strong>{damageSummary}</strong>
              <small>
                {damageDetail}
              </small>
            </div>
          </section>

          <section className="calculator-card field-effects-panel" aria-label="Battle field effects">
            <div className="field-effects-global">
              <select
                aria-label="Weather"
                value={fieldEffects.weather}
                onChange={(event) => updateFieldEffect("weather", event.target.value)}
              >
                {weatherOptions.map(([value, label]) => (
                  <option key={label} value={value}>
                    {label}
                  </option>
                ))}
              </select>

              <select
                aria-label="Terrain"
                value={fieldEffects.terrain}
                onChange={(event) => updateFieldEffect("terrain", event.target.value)}
              >
                {terrainOptions.map(([value, label]) => (
                  <option key={label} value={value}>
                    {label}
                  </option>
                ))}
              </select>

              <EffectToggle label="Trick Room" checked={fieldEffects.trickRoom} onChange={(value) => updateFieldEffect("trickRoom", value)} />
            </div>

            <div className="field-side-grid">
              <div className="field-side-card">
                <span>User Effects</span>
                <EffectToggle label="Tailwind" checked={fieldEffects.userTailwind} onChange={(value) => updateFieldEffect("userTailwind", value)} />
                <EffectToggle label="Reflect" checked={fieldEffects.userReflect} onChange={(value) => updateFieldEffect("userReflect", value)} />
                <EffectToggle label="Light Screen" checked={fieldEffects.userLightScreen} onChange={(value) => updateFieldEffect("userLightScreen", value)} />
                <EffectToggle label="Aurora Veil" checked={fieldEffects.userAuroraVeil} onChange={(value) => updateFieldEffect("userAuroraVeil", value)} />
              </div>

              <div className="field-side-card">
                <span>Opponent Effects</span>
                <EffectToggle label="Tailwind" checked={fieldEffects.opponentTailwind} onChange={(value) => updateFieldEffect("opponentTailwind", value)} />
                <EffectToggle label="Reflect" checked={fieldEffects.opponentReflect} onChange={(value) => updateFieldEffect("opponentReflect", value)} />
                <EffectToggle label="Light Screen" checked={fieldEffects.opponentLightScreen} onChange={(value) => updateFieldEffect("opponentLightScreen", value)} />
                <EffectToggle label="Aurora Veil" checked={fieldEffects.opponentAuroraVeil} onChange={(value) => updateFieldEffect("opponentAuroraVeil", value)} />
              </div>
            </div>
          </section>
        </div>

      </div>

      <div className="combatant-card">
        <span>Opponent Pokemon</span>

        <div
          className={`combatant-sprite-frame ${
            opponentSpriteSource ? "has-sprite" : "empty-sprite"
          }`}
        >
          {opponentSpriteSource && (
            <img
              className="combatant-sprite"
              src={opponentSpriteSource}
              alt={`${opponentPokemon?.name || "Opponent Pokemon"} sprite`}
            />
          )}
          <StatusControls
            label="Opponent Pokemon status"
            selectedStatus={opponentStatus}
            onToggle={(status) => toggleStatus("opponent", status)}
          />
        </div>

        <div className="combatant-title-row">
          <input
            className="opponent-name-input"
            aria-label="Opponent name"
            value={opponentPokemon?.name ?? ""}
            placeholder="Opponent name"
            onChange={(event) => onOpponentUpdate("name", event.target.value)}
          />
          <FormToggle
            label="Toggle opponent form"
            pokemon={opponentPokemon}
            onToggle={(formName) => onOpponentUpdate("form", formName)}
          />
        </div>

        <div className="meta-row opponent-meta-row">
          {opponentAbilities.length ? (
            <select
              className="meta-control"
              aria-label="Opponent ability"
              value={opponentPokemon?.ability ?? opponentAbilities[0]}
              onChange={(event) => onOpponentUpdate("ability", event.target.value)}
            >
              {opponentAbilities.map((ability) => (
                <option key={ability} value={ability}>
                  {ability}
                </option>
              ))}
            </select>
          ) : (
            <input
              className="meta-control"
              aria-label="Opponent ability"
              value={opponentPokemon?.ability ?? ""}
              placeholder="Ability pending"
              readOnly
            />
          )}

          <select
            className="meta-control"
            aria-label="Opponent nature"
            value={opponentPokemon?.nature ?? "hardy"}
            onChange={(event) => onOpponentUpdate("nature", event.target.value)}
          >
            {natures.map((nature) => (
              <option key={nature} value={nature}>
                {formatNature(nature)}
              </option>
            ))}
          </select>

          <input
            className="meta-control"
            aria-label="Opponent item"
            value={opponentPokemon?.item ?? ""}
            placeholder="Item"
            onChange={(event) => onOpponentUpdate("item", event.target.value)}
          />
        </div>

        <StatGrid
          baseStats={opponentPokemon?.baseStats}
          points={opponentPokemon?.statPoints}
          stats={opponentPokemon?.stats}
          onUpdate={onOpponentUpdate}
        />

        <div className="move-list opponent-move-list" aria-label="Opponent common moves">
          {ensureFourMoves(opponentPokemon?.moves).map((move, moveIndex) => {
            const moveKey = `${move || "empty"}-${moveIndex}`;

            if (!move) {
              return (
                <label className="move-search-slot" key={moveKey}>
                  <input
                    aria-label={`Opponent move ${moveIndex + 1}`}
                    list={`opponent-move-options-${moveIndex}`}
                    value={opponentMoveDrafts[moveIndex] ?? ""}
                    placeholder="Search move"
                    onChange={(event) => {
                      const draftValue = event.target.value;
                      const knownMove = findKnownMove(
                        opponentPokemon?.moveOptions,
                        draftValue
                      );

                      setOpponentMoveDrafts((currentDrafts) => ({
                        ...currentDrafts,
                        [moveIndex]: draftValue,
                      }));

                      if (knownMove) {
                        onOpponentUpdate(`moves.${moveIndex}`, knownMove);
                        setOpponentMoveDrafts((currentDrafts) => ({
                          ...currentDrafts,
                          [moveIndex]: "",
                        }));
                      }
                    }}
                    onBlur={() => {
                      const knownMove = findKnownMove(
                        opponentPokemon?.moveOptions,
                        opponentMoveDrafts[moveIndex]
                      );

                      if (knownMove) {
                        onOpponentUpdate(`moves.${moveIndex}`, knownMove);
                      }

                      setOpponentMoveDrafts((currentDrafts) => ({
                        ...currentDrafts,
                        [moveIndex]: "",
                      }));
                    }}
                  />
                  <datalist id={`opponent-move-options-${moveIndex}`}>
                    {(opponentPokemon?.moveOptions ?? []).map((moveOption, optionIndex) => (
                      <option key={`${moveOption}-${optionIndex}`} value={moveOption} />
                    ))}
                  </datalist>
                </label>
              );
            }

            return (
              <div
                className={
                  selectedMoveSource === "opponent" && selectedMove === move
                    ? "move-edit-slot selected"
                    : "move-edit-slot"
                }
                key={moveKey}
              >
                <button
                  className="move-button"
                  type="button"
                  onClick={() => onSelectMove(move, "opponent")}
                >
                  {move}
                </button>
                <button
                  className="move-remove-button"
                  type="button"
                  aria-label={`Remove ${move}`}
                  onClick={() => onOpponentUpdate(`moves.${moveIndex}`, "")}
                >
                  x
                </button>
              </div>
            );
          })}
        </div>

        <StageControls
          boosts={opponentBoosts}
          label="Opponent Stages"
          onChange={(statName, value) =>
            setOpponentBoosts((currentBoosts) => ({
              ...currentBoosts,
              [statName]: value,
            }))
          }
        />
      </div>
    </section>
  );
}

function StatusControls({ label, onToggle, selectedStatus }) {
  return (
    <div className="status-controls" role="group" aria-label={label}>
      {statusOptions.map(([status, statusLabel]) => (
        <label
          className={
            selectedStatus === status
              ? `status-toggle active ${status}`
              : `status-toggle ${status}`
          }
          key={status}
        >
          <input
            type="checkbox"
            checked={selectedStatus === status}
            onChange={() => onToggle(status)}
          />
          <span>{statusLabel}</span>
        </label>
      ))}
    </div>
  );
}

function EffectToggle({ checked, label, onChange }) {
  return (
    <label className={checked ? "effect-toggle active" : "effect-toggle"}>
      <input
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange(event.target.checked)}
      />
      <span>{label}</span>
    </label>
  );
}

function StageControls({ boosts, label, onChange }) {
  return (
    <div className="stage-controls" aria-label={label}>
      <span>{label}</span>
      <div className="stage-control-grid">
        {Object.entries(stageLabels).map(([statName, statLabel]) => (
          <label className="stage-control" key={statName}>
            <span>{statLabel}</span>
            <select
              value={boosts[statName] ?? 0}
              onChange={(event) => onChange(statName, Number(event.target.value))}
            >
              {stageValues.map((stageValue) => (
                <option key={stageValue} value={stageValue}>
                  {stageValue > 0 ? `+${stageValue}` : stageValue}
                </option>
              ))}
            </select>
          </label>
        ))}
      </div>
    </div>
  );
}

// Displays stat values and optional manual stat inputs.
function StatGrid({ baseStats = {}, points = {}, stats = {}, onUpdate }) {
  return (
    <div className={`stat-grid ${onUpdate ? "editable" : "readonly"}`}>
      {Object.entries(statLabels).map(([key, label]) => {
        const statValue = Number(stats?.[key] ?? 0);
        const barWidth = calculateStatBarWidth(key, statValue);

        return (
          <label
            className="stat-cell"
            key={key}
            style={{
              "--stat-color": statColors[key],
              "--stat-bar-width": barWidth,
            }}
          >
            <span className="stat-name">{label}</span>
            <strong className="stat-total">{statValue || "-"}</strong>
            <span className="stat-bar" aria-hidden="true" />

            {onUpdate && (
              <input
                className="stat-ev-input"
                aria-label={`Opponent ${label} points`}
                type="number"
                min="0"
                max={maxStatInvestment}
                value={points[key] ?? 0}
                onChange={(event) =>
                  onUpdate(`statPoints.${key}`, Number(event.target.value))
                }
              />
            )}
          </label>
        );
      })}
    </div>
  );
}

export default MatchupStage;
