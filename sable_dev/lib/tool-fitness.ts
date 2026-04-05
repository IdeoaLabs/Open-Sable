/**
 * Tool Fitness Tracking
 * 
 * Tracks per-tool success/failure rates and computes fitness scores
 * using an event-sourced model. Low-fitness tools are surfaced to the
 * AI model so it can adjust its strategy.
 * 
 * Fitness formula:
 *   quality    = 1 - (failures / total) * 0.5
 *   freshness  = 0.5^(cycles_since_last_use / 50)
 *   fitness    = quality * freshness
 * 
 * Persists to: .sable-dev/tool-fitness.json
 */

import fs from 'fs';
import path from 'path';

// ─── Types ────────────────────────────────────────────────

export interface ToolFitnessEntry {
  /** Tool name (e.g., 'file_read') */
  name: string;
  /** Total invocations */
  totalUses: number;
  /** Successful invocations */
  successes: number;
  /** Failed invocations */
  failures: number;
  /** Average execution time in ms */
  avgDurationMs: number;
  /** Cycle number of last use */
  lastUsedAtCycle: number;
  /** Computed fitness 0.0–1.0 */
  fitness: number;
  /** Recent outcomes (last 20, for trend detection) */
  recentOutcomes: Array<{ success: boolean; durationMs: number; cycle: number }>;
}

export interface ToolFitnessState {
  /** Per-tool fitness entries */
  tools: Record<string, ToolFitnessEntry>;
  /** Monotonic cycle counter (incremented each request) */
  cycle: number;
  /** Timestamp of last persistence */
  lastPersistedAt: number | null;
}

// ─── Fitness Constants ────────────────────────────────────

/** Freshness half-life in request cycles */
const FRESHNESS_HALF_LIFE = 50;
/** Number of recent outcomes to keep */
const MAX_RECENT_OUTCOMES = 20;
/** Fitness below this threshold triggers a warning */
const LOW_FITNESS_THRESHOLD = 0.5;
/** Minimum uses before fitness is meaningful */
const MIN_USES_FOR_SIGNAL = 3;

// ─── Persistence ──────────────────────────────────────────

const FITNESS_DIR = path.join(process.cwd(), '.sable-dev');
const FITNESS_FILE = path.join(FITNESS_DIR, 'tool-fitness.json');

function ensureDir() {
  if (!fs.existsSync(FITNESS_DIR)) {
    fs.mkdirSync(FITNESS_DIR, { recursive: true });
  }
}

// ─── State Management ─────────────────────────────────────

export function createToolFitnessState(): ToolFitnessState {
  // Try to restore from disk
  try {
    if (fs.existsSync(FITNESS_FILE)) {
      const data = JSON.parse(fs.readFileSync(FITNESS_FILE, 'utf-8'));
      if (data.tools && typeof data.cycle === 'number') {
        return data;
      }
    }
  } catch { /* corrupt, start fresh */ }

  return {
    tools: {},
    cycle: 0,
    lastPersistedAt: null,
  };
}

/**
 * Persist fitness state to disk (non-blocking best-effort).
 */
export function persistToolFitness(state: ToolFitnessState): void {
  try {
    ensureDir();
    fs.writeFileSync(FITNESS_FILE, JSON.stringify(
      { ...state, lastPersistedAt: Date.now() },
      null,
      2,
    ));
  } catch (e) {
    console.warn('[tool-fitness] Failed to persist:', e);
  }
}

// ─── Fitness Computation ──────────────────────────────────

/**
 * Compute fitness for a single tool entry.
 */
function computeFitness(entry: ToolFitnessEntry, currentCycle: number): number {
  if (entry.totalUses === 0) return 1.0; // No data = assume fine

  // Quality: penalize failures (0.5 weight so 100% fail = 0.5 not 0.0)
  const quality = 1 - (entry.failures / entry.totalUses) * 0.5;

  // Freshness: decay tools not used recently
  const cyclesSinceUse = Math.max(0, currentCycle - entry.lastUsedAtCycle);
  const freshness = Math.pow(0.5, cyclesSinceUse / FRESHNESS_HALF_LIFE);

  return quality * freshness;
}

// ─── Event Recording ──────────────────────────────────────

/**
 * Record a tool call outcome (success or failure).
 * Updates the tool's fitness entry and recomputes fitness.
 */
export function recordToolOutcome(
  state: ToolFitnessState,
  toolName: string,
  success: boolean,
  durationMs: number,
): ToolFitnessState {
  const cycle = state.cycle;
  const existing = state.tools[toolName];

  const totalUses = (existing?.totalUses || 0) + 1;
  const successes = (existing?.successes || 0) + (success ? 1 : 0);
  const failures = (existing?.failures || 0) + (success ? 0 : 1);
  
  // Running average of duration
  const prevAvg = existing?.avgDurationMs || 0;
  const avgDurationMs = prevAvg === 0 ? durationMs : prevAvg * 0.8 + durationMs * 0.2;

  // Append to recent outcomes (keep last N)
  const recentOutcomes = [
    ...(existing?.recentOutcomes || []),
    { success, durationMs, cycle },
  ].slice(-MAX_RECENT_OUTCOMES);

  const entry: ToolFitnessEntry = {
    name: toolName,
    totalUses,
    successes,
    failures,
    avgDurationMs: Math.round(avgDurationMs),
    lastUsedAtCycle: cycle,
    fitness: 0, // will be computed below
    recentOutcomes,
  };
  entry.fitness = computeFitness(entry, cycle);

  return {
    ...state,
    tools: { ...state.tools, [toolName]: entry },
  };
}

/**
 * Increment the fitness cycle counter (call once per request).
 */
export function tickFitnessCycle(state: ToolFitnessState): ToolFitnessState {
  const cycle = state.cycle + 1;
  
  // Recompute all fitness scores with new cycle
  const tools: Record<string, ToolFitnessEntry> = {};
  for (const [name, entry] of Object.entries(state.tools)) {
    tools[name] = { ...entry, fitness: computeFitness(entry, cycle) };
  }

  return { ...state, tools, cycle };
}

// ─── Prompt Integration ───────────────────────────────────

/**
 * Get low-fitness tool warnings for injection into system prompt.
 * Only includes tools with enough data AND below the fitness threshold.
 */
export function getToolFitnessWarnings(state: ToolFitnessState): string {
  const warnings: string[] = [];

  for (const entry of Object.values(state.tools)) {
    if (entry.totalUses < MIN_USES_FOR_SIGNAL) continue;
    if (entry.fitness >= LOW_FITNESS_THRESHOLD) continue;

    const failRate = Math.round((entry.failures / entry.totalUses) * 100);
    warnings.push(
      `- ${entry.name}: fitness ${entry.fitness.toFixed(2)} (${failRate}% failure rate over ${entry.totalUses} uses, avg ${entry.avgDurationMs}ms)`
    );
  }

  if (warnings.length === 0) return '';

  return `TOOL FITNESS WARNINGS (consider alternative approaches for these tools):\n${warnings.join('\n')}`;
}

/**
 * Get a full fitness report for debugging/display.
 */
export function getToolFitnessReport(state: ToolFitnessState): string {
  const entries = Object.values(state.tools)
    .sort((a, b) => b.totalUses - a.totalUses);

  if (entries.length === 0) return 'No tool usage data yet.';

  const lines = entries.map(e => {
    const status = e.fitness >= LOW_FITNESS_THRESHOLD ? '✓' : '⚠';
    const failRate = e.totalUses > 0 ? Math.round((e.failures / e.totalUses) * 100) : 0;
    // Detect trend from recent outcomes
    const recent5 = e.recentOutcomes.slice(-5);
    const recentFails = recent5.filter(o => !o.success).length;
    const trend = recentFails >= 3 ? '↓ declining' : recentFails === 0 ? '↑ stable' : '→ mixed';
    return `${status} ${e.name}: fitness=${e.fitness.toFixed(2)} uses=${e.totalUses} fail=${failRate}% avg=${e.avgDurationMs}ms ${trend}`;
  });

  return `Tool Fitness Report (cycle ${state.cycle}):\n${lines.join('\n')}`;
}
