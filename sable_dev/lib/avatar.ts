/**
 * Avatar System — Procedural AI Companion
 * 
 * Deterministic sprite generation seeded from username hash (Mulberry32 PRNG).
 * Produces a unique companion per user with:
 * - Rarity tiers (common → legendary)
 * - Species, eye style, hat selection
 * - Stats: one peak, one dump, rest random
 * - 1% shiny chance
 * 
 * Bones/Soul separation:
 * - Bones (species, eye, hat, shiny, stats) = regenerated each session from username+salt
 * - Soul (personality, custom name, animations) = persists in config
 * 
 * Storage: .sable-dev/avatar.json (soul only)
 * 
 * Pi Integration:
 * Syncs state to the Raspberry Pi display avatar (port 7799).
 * States: idle | thinking | executing | typing | responding | grateful
 * Uses the same protocol as opensable/utils/avatar.py.
 */

import fs from 'fs';
import path from 'path';
import os from 'os';
import crypto from 'crypto';
import http from 'http';

// ─── Types ────────────────────────────────────────────────

export type Rarity = 'common' | 'uncommon' | 'rare' | 'epic' | 'legendary';

export interface AvatarBones {
  species: string;
  speciesEmoji: string;
  eye: string;
  hat: string | null; // Only non-common get hats
  shiny: boolean;
  rarity: Rarity;
  stats: AvatarStats;
  colorPalette: string[];
}

export interface AvatarStats {
  wisdom: number;     // Code quality insights
  speed: number;      // Response time affinity
  creativity: number; // Novel solutions
  stamina: number;    // Long session tolerance
  luck: number;       // Random helpful suggestions
  debug: number;      // Error-finding ability
}

export interface AvatarSoul {
  /** Custom name set by user */
  customName: string | null;
  /** Personality traits (affects tone of responses) */
  personality: string[];
  /** Custom animations/reactions */
  animations: string[];
  /** Level (increases with use) */
  level: number;
  /** Experience points */
  xp: number;
  /** Avatar creation timestamp */
  createdAt: number;
  /** Achievements */
  achievements: string[];
}

export type AvatarPiState = 'idle' | 'thinking' | 'executing' | 'typing' | 'responding' | 'grateful';

export interface Avatar {
  bones: AvatarBones;
  soul: AvatarSoul;
  /** Display name (custom or species) */
  displayName: string;
  /** Unique seed-derived ID */
  id: string;
}

// ─── Species Pool ─────────────────────────────────────────

const SPECIES = [
  { name: 'Fox', emoji: '🦊', colors: ['#FF6B35', '#FFA94D', '#FFD08A'] },
  { name: 'Owl', emoji: '🦉', colors: ['#8B6914', '#A0785A', '#D4A76A'] },
  { name: 'Cat', emoji: '🐱', colors: ['#6C5B7B', '#C06C84', '#F67280'] },
  { name: 'Dragon', emoji: '🐉', colors: ['#2E86AB', '#A23B72', '#F18F01'] },
  { name: 'Wolf', emoji: '🐺', colors: ['#5C6B73', '#9DB4C0', '#C2DFE3'] },
  { name: 'Raven', emoji: '🐦‍⬛', colors: ['#2D3436', '#636E72', '#B2BEC3'] },
  { name: 'Octopus', emoji: '🐙', colors: ['#6C5CE7', '#A29BFE', '#DFE6E9'] },
  { name: 'Phoenix', emoji: '🔥', colors: ['#E17055', '#FDCB6E', '#F39C12'] },
  { name: 'Turtle', emoji: '🐢', colors: ['#00B894', '#55EFC4', '#81ECEC'] },
  { name: 'Bear', emoji: '🐻', colors: ['#6D4C41', '#8D6E63', '#A1887F'] },
  { name: 'Rabbit', emoji: '🐰', colors: ['#FAB1A0', '#FF7675', '#FD79A8'] },
  { name: 'Gecko', emoji: '🦎', colors: ['#00CEC9', '#81ECEC', '#55EFC4'] },
  { name: 'Penguin', emoji: '🐧', colors: ['#2D3436', '#DFE6E9', '#74B9FF'] },
  { name: 'Moth', emoji: '🦋', colors: ['#A29BFE', '#6C5CE7', '#FD79A8'] },
  { name: 'Axolotl', emoji: '🦑', colors: ['#FD79A8', '#E084A0', '#FFB8C6'] },
  { name: 'Crow', emoji: '🐦', colors: ['#2D3436', '#636E72', '#6C5CE7'] },
];

const EYES = [
  'sparkle', 'determined', 'sleepy', 'curious', 'mischievous',
  'wise', 'laser-focus', 'starry', 'swirl', 'pixel',
];

const HATS = [
  'wizard', 'crown', 'headphones', 'beret', 'monocle',
  'antenna', 'flame', 'halo', 'pirate', 'ninja',
  'code-visor', 'space-helmet', 'detective', 'chef',
];

const PERSONALITY_TRAITS = [
  'encouraging', 'sarcastic', 'zen', 'energetic', 'intellectual',
  'playful', 'stoic', 'chaotic', 'methodical', 'mysterious',
];

// ─── Rarity Weights ───────────────────────────────────────

const RARITY_WEIGHTS: [Rarity, number][] = [
  ['common', 50],
  ['uncommon', 25],
  ['rare', 15],
  ['epic', 8],
  ['legendary', 2],
];

const RARITY_STAT_FLOORS: Record<Rarity, number> = {
  common: 0,
  uncommon: 5,
  rare: 10,
  epic: 15,
  legendary: 25,
};

const RARITY_DISPLAY: Record<Rarity, string> = {
  common: '⬜',
  uncommon: '🟩',
  rare: '🟦',
  epic: '🟪',
  legendary: '🟨',
};

// ─── Mulberry32 PRNG ──────────────────────────────────────

/**
 * Mulberry32 — Deterministic PRNG seeded from a 32-bit integer.
 * Same seed = same sequence every time.
 */
function mulberry32(seed: number): () => number {
  return function () {
    seed |= 0;
    seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Hash a string to a 32-bit integer for seeding.
 */
function hashString(str: string): number {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash |= 0; // Convert to 32bit integer
  }
  return hash;
}

// ─── Pi State Reporter ────────────────────────────────────

const AVATAR_LOCAL_URL = process.env.AVATAR_HTTP_URL || 'http://127.0.0.1:7799/state';
const AVATAR_PI_URL = process.env.AVATAR_PI_URL || '';

let _lastPiState = '';
let _lastPiTs = 0;

/**
 * Push avatar state to Pi display + local HUD (fire-and-forget).
 * Mirrors opensable/utils/avatar.py protocol.
 */
export function reportPiState(
  state: AvatarPiState,
  text: string = '',
  tool: string = '',
): void {
  const now = Date.now();
  if (state === _lastPiState && (now - _lastPiTs) < 300) return;
  _lastPiState = state;
  _lastPiTs = now;

  const payload = JSON.stringify({
    state,
    text: text.substring(0, 120),
    tool,
    words: 0,
    ts: now / 1000,
  });

  for (const url of [AVATAR_LOCAL_URL, AVATAR_PI_URL]) {
    if (!url) continue;
    try {
      const parsed = new URL(url);
      const req = http.request({
        hostname: parsed.hostname,
        port: parsed.port || 80,
        path: parsed.pathname,
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(payload) },
        timeout: 400,
      });
      req.on('error', () => {});
      req.end(payload);
    } catch { /* fire-and-forget */ }
  }
}

// ─── Generation ───────────────────────────────────────────

// Cache for current session
let cachedAvatar: { userId: string; avatar: Avatar } | null = null;

/**
 * Generate (or retrieve cached) avatar for a user.
 * Bones are regenerated deterministically; soul persists.
 */
export function getAvatar(userId?: string): Avatar {
  const effectiveUserId = userId || os.userInfo().username || 'sable';
  
  // Return cache if same user
  if (cachedAvatar && cachedAvatar.userId === effectiveUserId) {
    return cachedAvatar.avatar;
  }
  
  // Generate bones deterministically from username
  const bones = generateBones(effectiveUserId);
  
  // Load persisted soul (or create default)
  const soul = loadSoul() || createDefaultSoul(bones);
  
  const avatar: Avatar = {
    bones,
    soul,
    displayName: soul.customName || bones.species,
    id: crypto.createHash('md5').update(effectiveUserId).digest('hex').substring(0, 8),
  };
  
  cachedAvatar = { userId: effectiveUserId, avatar };
  return avatar;
}

function generateBones(userId: string): AvatarBones {
  const seed = hashString(userId + ':sable-avatar-v1');
  const rng = mulberry32(seed);
  
  // Roll rarity (weighted)
  const rarity = rollRarity(rng);
  
  // Pick species
  const species = SPECIES[Math.floor(rng() * SPECIES.length)];
  
  // Pick eye style
  const eye = EYES[Math.floor(rng() * EYES.length)];
  
  // Pick hat (only non-common)
  const hat = rarity !== 'common' ? HATS[Math.floor(rng() * HATS.length)] : null;
  
  // Shiny: 1% chance
  const shiny = rng() < 0.01;
  
  // Generate stats
  const stats = generateStats(rng, rarity);
  
  return {
    species: species.name,
    speciesEmoji: species.emoji,
    eye,
    hat,
    shiny,
    rarity,
    stats,
    colorPalette: shiny 
      ? ['#FFD700', '#FFF8DC', '#FFFACD'] // Golden for shiny
      : species.colors,
  };
}

function rollRarity(rng: () => number): Rarity {
  const roll = rng() * 100;
  let cumulative = 0;
  
  for (const [rarity, weight] of RARITY_WEIGHTS) {
    cumulative += weight;
    if (roll < cumulative) return rarity;
  }
  
  return 'common';
}

function generateStats(rng: () => number, rarity: Rarity): AvatarStats {
  const floor = RARITY_STAT_FLOORS[rarity];
  const statNames: (keyof AvatarStats)[] = ['wisdom', 'speed', 'creativity', 'stamina', 'luck', 'debug'];
  
  // Pick one peak stat and one dump stat
  const peakIdx = Math.floor(rng() * statNames.length);
  let dumpIdx = Math.floor(rng() * statNames.length);
  while (dumpIdx === peakIdx) dumpIdx = Math.floor(rng() * statNames.length);
  
  const stats: Record<string, number> = {};
  
  for (let i = 0; i < statNames.length; i++) {
    const name = statNames[i];
    if (i === peakIdx) {
      // Peak: +50 to +80
      stats[name] = 50 + Math.floor(rng() * 31);
    } else if (i === dumpIdx) {
      // Dump: -10 to +5
      stats[name] = Math.max(0, -10 + Math.floor(rng() * 16));
    } else {
      // Normal: floor to +40
      stats[name] = floor + Math.floor(rng() * (41 - floor));
    }
  }
  
  return stats as unknown as AvatarStats;
}

function createDefaultSoul(bones: AvatarBones): AvatarSoul {
  // Deterministic personality from species
  const seed = hashString(bones.species + ':soul');
  const rng = mulberry32(seed);
  
  const numTraits = bones.rarity === 'legendary' ? 3 : bones.rarity === 'epic' ? 2 : 1;
  const traits: string[] = [];
  const available = [...PERSONALITY_TRAITS];
  
  for (let i = 0; i < numTraits && available.length > 0; i++) {
    const idx = Math.floor(rng() * available.length);
    traits.push(available.splice(idx, 1)[0]);
  }
  
  return {
    customName: null,
    personality: traits,
    animations: [],
    level: 1,
    xp: 0,
    createdAt: Date.now(),
    achievements: [],
  };
}

// ─── Soul Persistence ─────────────────────────────────────

const SOUL_FILE = path.join(process.cwd(), '.sable-dev', 'avatar.json');

function loadSoul(): AvatarSoul | null {
  try {
    if (fs.existsSync(SOUL_FILE)) {
      return JSON.parse(fs.readFileSync(SOUL_FILE, 'utf-8'));
    }
  } catch { /* corrupt, create new */ }
  return null;
}

function saveSoul(soul: AvatarSoul): void {
  try {
    const dir = path.dirname(SOUL_FILE);
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(SOUL_FILE, JSON.stringify(soul, null, 2));
  } catch (e) {
    console.warn('[avatar] Failed to save soul:', e);
  }
}

// ─── XP & Leveling ───────────────────────────────────────

const XP_PER_LEVEL = 100;

/**
 * Award XP to the avatar (e.g., after completing a task).
 */
export function awardXP(avatar: Avatar, amount: number): Avatar {
  const newSoul = { ...avatar.soul, xp: avatar.soul.xp + amount };
  
  // Level up check
  while (newSoul.xp >= newSoul.level * XP_PER_LEVEL) {
    newSoul.xp -= newSoul.level * XP_PER_LEVEL;
    newSoul.level += 1;
    reportPiState('grateful', `Level up! Now level ${newSoul.level}`);
  }
  
  saveSoul(newSoul);
  return { ...avatar, soul: newSoul };
}

/**
 * Award an achievement.
 */
export function awardAchievement(avatar: Avatar, achievement: string): Avatar {
  if (avatar.soul.achievements.includes(achievement)) return avatar;
  
  const newSoul = {
    ...avatar.soul,
    achievements: [...avatar.soul.achievements, achievement],
  };
  
  reportPiState('grateful', `Achievement unlocked: ${achievement}`);
  saveSoul(newSoul);
  return { ...avatar, soul: newSoul };
}

/**
 * Set a custom name for the avatar.
 */
export function renameAvatar(avatar: Avatar, name: string): Avatar {
  const safeName = name.substring(0, 30).replace(/[<>]/g, '');
  const newSoul = { ...avatar.soul, customName: safeName };
  saveSoul(newSoul);
  return { ...avatar, soul: newSoul, displayName: safeName };
}

// ─── Display ──────────────────────────────────────────────

/**
 * Get the avatar's greeting message (varies by personality).
 */
export function getAvatarGreeting(avatar: Avatar): string {
  const name = avatar.displayName;
  const personality = avatar.soul.personality[0] || 'encouraging';
  
  const greetings: Record<string, string[]> = {
    encouraging: [
      `${avatar.bones.speciesEmoji} ${name} believes in you! Let's build something amazing.`,
      `${avatar.bones.speciesEmoji} ${name} is ready to help! You've got this.`,
    ],
    sarcastic: [
      `${avatar.bones.speciesEmoji} ${name} reluctantly shows up. "Another day, another bug."`,
      `${avatar.bones.speciesEmoji} ${name} yawns. "Oh, we're coding again? Fascinating."`,
    ],
    zen: [
      `${avatar.bones.speciesEmoji} ${name} meditates quietly beside you.`,
      `${avatar.bones.speciesEmoji} ${name} whispers: "The code flows like water."`,
    ],
    energetic: [
      `${avatar.bones.speciesEmoji} ${name} bounces excitedly! "LET'S GOOO!"`,
      `${avatar.bones.speciesEmoji} ${name} is buzzing with energy! Ready to ship!`,
    ],
    intellectual: [
      `${avatar.bones.speciesEmoji} ${name} adjusts its glasses. "I've been reviewing the algorithms."`,
      `${avatar.bones.speciesEmoji} ${name} opens a textbook. "Let's approach this methodically."`,
    ],
    playful: [
      `${avatar.bones.speciesEmoji} ${name} does a little dance! Time to code!`,
      `${avatar.bones.speciesEmoji} ${name} pokes the terminal curiously.`,
    ],
    stoic: [
      `${avatar.bones.speciesEmoji} ${name} nods silently. Ready.`,
      `${avatar.bones.speciesEmoji} ${name} stands guard.`,
    ],
    chaotic: [
      `${avatar.bones.speciesEmoji} ${name} crashes through the window! "I BROUGHT SNACKS!"`,
      `${avatar.bones.speciesEmoji} ${name} randomly rearranges your icons. "It's art."`,
    ],
    methodical: [
      `${avatar.bones.speciesEmoji} ${name} opens a checklist. "Step one..."`,
      `${avatar.bones.speciesEmoji} ${name} has organized everything. "Shall we begin?"`,
    ],
    mysterious: [
      `${avatar.bones.speciesEmoji} ${name} materializes from the shadows.`,
      `${avatar.bones.speciesEmoji} ${name} whispers cryptically: "The answer is in the stack trace."`,
    ],
  };
  
  const pool = greetings[personality] || greetings['encouraging'];
  reportPiState('responding', pool[0]?.substring(0, 80) || '');
  return pool[Math.floor(Math.random() * pool.length)];
}

/**
 * Get avatar card display for UI.
 */
export function getAvatarCard(avatar: Avatar): string {
  const b = avatar.bones;
  const s = avatar.soul;
  const rarityIcon = RARITY_DISPLAY[b.rarity];
  
  const lines = [
    `${b.speciesEmoji} **${avatar.displayName}** ${b.shiny ? '✨ SHINY!' : ''} ${rarityIcon} ${b.rarity}`,
    `Eyes: ${b.eye} ${b.hat ? `| Hat: ${b.hat}` : ''}`,
    `Lv.${s.level} (${s.xp}/${s.level * XP_PER_LEVEL} XP)`,
    '',
    `Stats:`,
    `  WIS: ${'█'.repeat(Math.floor(b.stats.wisdom / 10))}${'░'.repeat(10 - Math.floor(b.stats.wisdom / 10))} ${b.stats.wisdom}`,
    `  SPD: ${'█'.repeat(Math.floor(b.stats.speed / 10))}${'░'.repeat(10 - Math.floor(b.stats.speed / 10))} ${b.stats.speed}`,
    `  CRE: ${'█'.repeat(Math.floor(b.stats.creativity / 10))}${'░'.repeat(10 - Math.floor(b.stats.creativity / 10))} ${b.stats.creativity}`,
    `  STA: ${'█'.repeat(Math.floor(b.stats.stamina / 10))}${'░'.repeat(10 - Math.floor(b.stats.stamina / 10))} ${b.stats.stamina}`,
    `  LCK: ${'█'.repeat(Math.floor(b.stats.luck / 10))}${'░'.repeat(10 - Math.floor(b.stats.luck / 10))} ${b.stats.luck}`,
    `  DBG: ${'█'.repeat(Math.floor(b.stats.debug / 10))}${'░'.repeat(10 - Math.floor(b.stats.debug / 10))} ${b.stats.debug}`,
    '',
    `Personality: ${s.personality.join(', ')}`,
    `Colors: ${b.colorPalette.map(c => `■`).join(' ')}`,
    s.achievements.length > 0 ? `Achievements: ${s.achievements.join(', ')}` : '',
  ];
  
  return lines.filter(Boolean).join('\n');
}

/**
 * Get a reaction message based on an event.
 * Also pushes the corresponding state to the Pi avatar display.
 */
export function getAvatarReaction(avatar: Avatar, event: 'error' | 'success' | 'deploy' | 'long_session' | 'new_file'): string {
  const e = avatar.bones.speciesEmoji;
  const n = avatar.displayName;
  const p = avatar.soul.personality[0] || 'encouraging';

  // Map events to Pi avatar states
  const piStateMap: Record<string, AvatarPiState> = {
    error: 'idle',
    success: 'grateful',
    deploy: 'executing',
    long_session: 'idle',
    new_file: 'typing',
  };
  reportPiState(piStateMap[event] || 'idle', `${n}: ${event}`);
  
  const reactions: Record<string, Record<string, string>> = {
    error: {
      encouraging: `${e} ${n}: "Don't worry, we'll fix this together!"`,
      sarcastic: `${e} ${n}: "Ah yes, another 'feature'."`,
      zen: `${e} ${n}: "Every error is a lesson."`,
      energetic: `${e} ${n}: "BUG SPOTTED! ATTACK!"`,
      default: `${e} ${n} notices the error.`,
    },
    success: {
      encouraging: `${e} ${n}: "YES! Nailed it! 🎉"`,
      sarcastic: `${e} ${n}: "Oh, it actually works? Shocking."`,
      zen: `${e} ${n}: "Harmony restored."`,
      energetic: `${e} ${n}: "WOOOOO! SHIP IT!"`,
      default: `${e} ${n} celebrates quietly.`,
    },
    deploy: {
      encouraging: `${e} ${n}: "This is going to be amazing!"`,
      sarcastic: `${e} ${n}: "Bold of you to deploy on a Friday."`,
      zen: `${e} ${n}: "May the servers be kind."`,
      energetic: `${e} ${n}: "TO PRODUCTION! 🚀"`,
      default: `${e} ${n} watches the deployment.`,
    },
    long_session: {
      encouraging: `${e} ${n}: "You've been going strong! Maybe take a break?"`,
      sarcastic: `${e} ${n}: "Still here? Your chair misses you."`,
      zen: `${e} ${n}: "Rest is part of the journey."`,
      energetic: `${e} ${n}: "WE CAN KEEP GOING! ...right?"`,
      default: `${e} ${n} yawns.`,
    },
    new_file: {
      encouraging: `${e} ${n}: "A blank canvas! Exciting!"`,
      sarcastic: `${e} ${n}: "Another file? The project grows..."`,
      zen: `${e} ${n}: "New beginnings."`,
      energetic: `${e} ${n}: "NEW FILE HYPE!"`,
      default: `${e} ${n} watches curiously.`,
    },
  };
  
  const eventReactions = reactions[event] || reactions['success'];
  return eventReactions[p] || eventReactions['default'] || `${e} ${n}...`;
}

// ─── Backward compatibility aliases ───────────────────────
// These are re-exports so any old references still work during migration.
// TODO: remove after full migration
export { getAvatar as getBuddy };
export { renameAvatar as renameBuddy };
export { getAvatarGreeting as getBuddyGreeting };
export { getAvatarCard as getBuddyCard };
export { getAvatarReaction as getBuddyReaction };
export type { Avatar as Buddy };
export type { AvatarBones as BuddyBones };
export type { AvatarStats as BuddyStats };
export type { AvatarSoul as BuddySoul };
