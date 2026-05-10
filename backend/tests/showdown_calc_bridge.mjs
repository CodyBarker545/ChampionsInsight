import {calculate, Field, Move, Pokemon} from '../../frontend/node_modules/@smogon/calc/dist/index.js';

const STAT_KEYS = {
  hp: 'hp',
  attack: 'atk',
  defense: 'def',
  special_attack: 'spa',
  special_defense: 'spd',
  speed: 'spe',
};

const WEATHER = {
  rain: 'Rain',
  sun: 'Sun',
  sand: 'Sand',
  snow: 'Snow',
  'harsh-sun': 'Harsh Sunshine',
  'heavy-rain': 'Heavy Rain',
};

const TERRAIN = {
  electric: 'Electric',
  grassy: 'Grassy',
  psychic: 'Psychic',
  misty: 'Misty',
};

const STATUS = {
  burn: 'brn',
  poison: 'psn',
  'bad-poison': 'tox',
  paralysis: 'par',
  sleep: 'slp',
  freeze: 'frz',
};

function toStats(values = {}) {
  const stats = {};
  for (const [source, target] of Object.entries(STAT_KEYS)) {
    if (values[source] !== undefined) stats[target] = values[source];
  }
  return stats;
}

function pokemonOptions(data = {}) {
  const options = {
    level: data.level ?? 50,
    ability: data.ability || undefined,
    item: data.item || undefined,
    nature: data.nature || 'Serious',
    evs: toStats(data.evs),
    ivs: toStats(data.ivs),
    boosts: toStats(data.boosts),
    curHP: data.currentHp ?? data.current_hp,
    status: STATUS[data.status] || data.status || undefined,
    gender: data.gender || undefined,
    teraType: data.teraType || data.tera_type || undefined,
    abilityOn: data.abilityOn || data.ability_on || data.flashFireActive || data.flash_fire_active || undefined,
    alliesFainted: data.faintedAllies ?? data.fainted_allies,
  };

  return Object.fromEntries(Object.entries(options).filter(([, value]) => value !== undefined));
}

function moveOptions(data = {}) {
  const options = {
    isCrit: data.isCrit ?? data.critical,
    hits: data.hits,
    timesUsedWithMetronome: data.metronomeTurns ?? data.metronome_turns,
  };

  return Object.fromEntries(Object.entries(options).filter(([, value]) => value !== undefined));
}

function fieldOptions(data = {}) {
  const attackerSide = data.attackerSide || data.attacker_side || {};
  const defenderSide = data.defenderSide || data.defender_side || {};
  return {
    weather: WEATHER[data.weather] || data.weather || undefined,
    terrain: TERRAIN[data.terrain] || data.terrain || undefined,
    gameType: data.isDoubles || data.is_doubles ? 'Doubles' : 'Singles',
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
  if (typeof damage === 'number') return [damage];
  return damage.flat(Infinity).filter((value) => typeof value === 'number').sort((a, b) => a - b);
}

const chunks = [];
for await (const chunk of process.stdin) chunks.push(chunk);
const payload = JSON.parse(chunks.join(''));
const gen = payload.gen || 9;

const attacker = new Pokemon(gen, payload.attacker.name, pokemonOptions(payload.attacker));
const defender = new Pokemon(gen, payload.defender.name, pokemonOptions(payload.defender));
const move = new Move(gen, payload.move.name, moveOptions(payload.move));
const field = new Field(fieldOptions(payload.field || {}));
const result = calculate(gen, attacker, defender, move, field);

process.stdout.write(JSON.stringify({
  damageValues: flattenDamage(result.damage),
  minDamage: result.range()[0],
  maxDamage: result.range()[1],
  description: result.fullDesc(),
}));
