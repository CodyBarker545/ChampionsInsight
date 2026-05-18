const express = require("express");
const calc = require("./calc/dist/index.js");

const app = express();
const port = Number(process.env.DAMAGE_CALC_PORT || 3001);

const STAT_KEYS = {
  hp: "hp",
  attack: "atk",
  defense: "def",
  special_attack: "spa",
  specialAttack: "spa",
  special_defense: "spd",
  specialDefense: "spd",
  speed: "spe",
};

const WEATHER = {
  rain: "Rain",
  sun: "Sun",
  sand: "Sand",
  snow: "Snow",
  "harsh-sun": "Harsh Sunshine",
  "heavy-rain": "Heavy Rain",
};

const TERRAIN = {
  electric: "Electric",
  grassy: "Grassy",
  psychic: "Psychic",
  misty: "Misty",
};

const STATUS = {
  burn: "brn",
  poison: "psn",
  "bad-poison": "tox",
  paralysis: "par",
  sleep: "slp",
  freeze: "frz",
};

function compact(value) {
  return Object.fromEntries(Object.entries(value).filter(([, item]) => item !== undefined));
}

function toStats(values = {}) {
  const stats = {};
  for (const [source, target] of Object.entries(STAT_KEYS)) {
    if (values[source] !== undefined) stats[target] = Number(values[source]);
  }
  return stats;
}

function pokemonOptions(data = {}) {
  return compact({
    level: data.level ?? 50,
    ability: data.ability || undefined,
    item: data.item || undefined,
    nature: data.nature || "Serious",
    evs: toStats(data.evs),
    ivs: toStats(data.ivs),
    boosts: toStats(data.boosts),
    curHP: data.currentHp ?? data.current_hp,
    status: STATUS[data.status] || data.status || undefined,
    gender: data.gender || undefined,
    teraType: data.teraType || data.tera_type || undefined,
    abilityOn: data.abilityOn || data.ability_on || data.flashFireActive || data.flash_fire_active || undefined,
    alliesFainted: data.faintedAllies ?? data.fainted_allies,
  });
}

function moveOptions(data = {}, attacker = {}) {
  return compact({
    isCrit: data.isCrit ?? data.critical,
    hits: data.hits,
    timesUsedWithMetronome: data.metronomeTurns ?? data.metronome_turns,
    ability: attacker.ability || undefined,
    item: attacker.item || undefined,
    species: attacker.name || undefined,
  });
}

function fieldOptions(data = {}) {
  const attackerSide = data.attackerSide || data.attacker_side || {};
  const defenderSide = data.defenderSide || data.defender_side || {};
  return {
    weather: WEATHER[data.weather] || data.weather || undefined,
    terrain: TERRAIN[data.terrain] || data.terrain || undefined,
    gameType: data.isDoubles || data.is_doubles ? "Doubles" : "Singles",
    isGravity: data.gravity || undefined,
    isMagicRoom: data.magicRoom || data.magic_room || undefined,
    isWonderRoom: data.wonderRoom || data.wonder_room || undefined,
    attackerSide: {
      isHelpingHand: attackerSide.helpingHand || attackerSide.helping_hand || undefined,
      isBattery: attackerSide.battery || attackerSide.isBattery || attackerSide.is_battery || undefined,
      isPowerSpot: attackerSide.powerSpot || attackerSide.power_spot || attackerSide.isPowerSpot || attackerSide.is_power_spot || undefined,
      isFlowerGift: attackerSide.flowerGift || attackerSide.flower_gift || attackerSide.isFlowerGift || attackerSide.is_flower_gift || undefined,
      isSteelySpirit: attackerSide.steelySpirit || attackerSide.steely_spirit || attackerSide.isSteelySpirit || attackerSide.is_steely_spirit || undefined,
    },
    defenderSide: {
      isReflect: defenderSide.reflect || undefined,
      isLightScreen: defenderSide.lightScreen || defenderSide.light_screen || undefined,
      isAuroraVeil: defenderSide.auroraVeil || defenderSide.aurora_veil || undefined,
      isFriendGuard: defenderSide.friendGuard || defenderSide.friend_guard || undefined,
      isFlowerGift: defenderSide.flowerGift || defenderSide.flower_gift || defenderSide.isFlowerGift || defenderSide.is_flower_gift || undefined,
    },
  };
}

function flattenDamage(damage) {
  if (typeof damage === "number") return [damage];
  return damage.flat(Infinity).filter((value) => typeof value === "number").sort((a, b) => a - b);
}

function applyExplicitStats(pokemon, data = {}) {
  const explicitStats = toStats(data.stats || {});
  if (!Object.keys(explicitStats).length) return;

  pokemon.rawStats = {...pokemon.rawStats, ...explicitStats};
  pokemon.stats = {...pokemon.stats, ...explicitStats};

  const hp = explicitStats.hp;
  if (hp !== undefined) {
    const currentHp = data.currentHp ?? data.current_hp ?? hp;
    pokemon.originalCurHP = Math.min(Number(currentHp), hp);
  }
}

function calculateDamage(payload = {}) {
  if (!payload.attacker?.name) throw new Error("attacker.name is required");
  if (!payload.defender?.name) throw new Error("defender.name is required");
  if (!payload.move?.name) throw new Error("move.name is required");

  const gen = calc.Generations.get(payload.gen || 9);
  const attacker = new calc.Pokemon(gen, payload.attacker.name, pokemonOptions(payload.attacker));
  const defender = new calc.Pokemon(gen, payload.defender.name, pokemonOptions(payload.defender));
  const move = new calc.Move(gen, payload.move.name, moveOptions(payload.move, payload.attacker));
  const field = new calc.Field(fieldOptions(payload.field || {}));

  applyExplicitStats(attacker, payload.attacker);
  applyExplicitStats(defender, payload.defender);

  const result = calc.calculate(gen, attacker, defender, move, field);
  const damageValues = flattenDamage(result.damage);
  const [minDamage, maxDamage] = result.range();
  const defenderHp = defender.maxHP();

  return {
    damageValues,
    minDamage,
    maxDamage,
    category: String(move.category || "").toLowerCase(),
    attackStatUsed: move.category === "Special" ? attacker.stats.spa : attacker.stats.atk,
    defenseStatUsed: move.category === "Special" ? defender.stats.spd : defender.stats.def,
    percentRange: defenderHp ? `${((minDamage / defenderHp) * 100).toFixed(1)}% - ${((maxDamage / defenderHp) * 100).toFixed(1)}%` : "0.0% - 0.0%",
    description: result.fullDesc(),
  };
}

app.use(express.json());

app.get("/health", (_req, res) => {
  res.json({status: "ok", engine: "@smogon/calc"});
});

app.post("/calculate", (req, res) => {
  try {
    res.json(calculateDamage(req.body));
  } catch (error) {
    res.status(400).json({error: error.message});
  }
});

app.listen(port, () => {
  console.log(`Smogon damage calculator backend running on port ${port}`);
});
