/**
 * Prompt Evolution,  Self-Improving Prompt System
 * 
 * Tracks user satisfaction signals (accept, redo, error) and uses them
 * to evolve prompt strategies over time. During dream consolidation,
 * reviews which strategies worked and proposes prompt mutations.
 * 
 * Signals:
 *   - accept: User kept the generated code (positive)
 *   - redo:   User re-prompted or said "try again" (negative)
 *   - error:  Generation produced a runtime error (negative)
 *   - edit:   User manually edited the result (mild negative)
 * 
 * Persistence: .sable-dev/prompt-evolution.json
 */

import fs from 'fs';
import path from 'path';

// ─── Types ────────────────────────────────────────────────

export type OutcomeSignal = 'accept' | 'redo' | 'error' | 'edit';

export interface OutcomeEvent {
  signal: OutcomeSignal;
  timestamp: number;
  /** Template used (react-spa, static-site, etc.) */
  template: string;
  /** Effort level used */
  effortLevel: string;
  /** Whether tools were used */
  usedTools: boolean;
  /** Number of tool calls */
  toolCallCount: number;
  /** Short context (first 100 chars of prompt) */
  promptPreview: string;
}

export interface PromptMutation {
  /** Unique ID */
  id: string;
  /** The prompt fragment to inject */
  fragment: string;
  /** Which template/context this applies to ('*' = all) */
  appliesTo: string;
  /** Origin: 'dream' (AI-generated) or 'signal' (rule-based) */
  origin: 'dream' | 'signal';
  /** Fitness: success rate when this mutation was active */
  fitness: number;
  /** Generation number (how many times it's been evolved) */
  generation: number;
  /** Created timestamp */
  createdAt: number;
  /** Number of sessions this mutation was active */
  activeSessions: number;
  /** Positive outcomes while active */
  positiveOutcomes: number;
  /** Negative outcomes while active */
  negativeOutcomes: number;
}

export interface PromptEvolutionState {
  /** Outcome events from recent sessions */
  outcomeHistory: OutcomeEvent[];
  /** Active prompt mutations (injected into system prompt) */
  activeMutations: PromptMutation[];
  /** Archived mutations (retired or replaced) */
  archivedMutations: PromptMutation[];
  /** Total sessions tracked */
  totalSessions: number;
  /** Last evolution cycle timestamp */
  lastEvolvedAt: number | null;
  /** Current active mutation IDs (for tracking which are being tested) */
  activeMutationIds: string[];
}

// ─── Constants ────────────────────────────────────────────

/** Maximum outcome events to retain */
const MAX_OUTCOME_HISTORY = 200;
/** Maximum active mutations */
const MAX_ACTIVE_MUTATIONS = 5;
/** Minimum sessions before evolution can run */
const MIN_SESSIONS_FOR_EVOLUTION = 5;
/** Below this fitness, mutations get retired */
const MUTATION_RETIRE_THRESHOLD = 0.3;
/** Above this fitness, mutations are considered proven */
const MUTATION_PROVEN_THRESHOLD = 0.7;
/** Signal weights for computing session quality */
const SIGNAL_WEIGHTS: Record<OutcomeSignal, number> = {
  accept: 1.0,
  edit: -0.3,
  redo: -0.8,
  error: -1.0,
};

// ─── Persistence ──────────────────────────────────────────

const EVOLUTION_DIR = path.join(process.cwd(), '.sable-dev');
const EVOLUTION_FILE = path.join(EVOLUTION_DIR, 'prompt-evolution.json');

function ensureDir() {
  if (!fs.existsSync(EVOLUTION_DIR)) {
    fs.mkdirSync(EVOLUTION_DIR, { recursive: true });
  }
}

// ─── State Management ─────────────────────────────────────

export function createPromptEvolutionState(): PromptEvolutionState {
  // Try to restore from disk
  try {
    if (fs.existsSync(EVOLUTION_FILE)) {
      const data = JSON.parse(fs.readFileSync(EVOLUTION_FILE, 'utf-8'));
      if (data.outcomeHistory && Array.isArray(data.activeMutations)) {
        return data;
      }
    }
  } catch { /* corrupt, start fresh */ }

  return {
    outcomeHistory: [],
    activeMutations: [],
    archivedMutations: [],
    totalSessions: 0,
    lastEvolvedAt: null,
    activeMutationIds: [],
  };
}

export function persistPromptEvolution(state: PromptEvolutionState): void {
  try {
    ensureDir();
    fs.writeFileSync(EVOLUTION_FILE, JSON.stringify(state, null, 2));
  } catch (e) {
    console.warn('[prompt-evolution] Failed to persist:', e);
  }
}

// ─── Signal Recording ─────────────────────────────────────

/**
 * Record a user satisfaction signal.
 */
export function recordOutcome(
  state: PromptEvolutionState,
  signal: OutcomeSignal,
  context: {
    template: string;
    effortLevel: string;
    usedTools: boolean;
    toolCallCount: number;
    promptPreview: string;
  },
): PromptEvolutionState {
  const event: OutcomeEvent = {
    signal,
    timestamp: Date.now(),
    ...context,
  };

  // Update active mutation outcomes
  const isPositive = signal === 'accept';
  const activeMutations = state.activeMutations.map(m => ({
    ...m,
    positiveOutcomes: m.positiveOutcomes + (isPositive ? 1 : 0),
    negativeOutcomes: m.negativeOutcomes + (isPositive ? 0 : 1),
    fitness: computeMutationFitness(
      m.positiveOutcomes + (isPositive ? 1 : 0),
      m.negativeOutcomes + (isPositive ? 0 : 1),
    ),
  }));

  return {
    ...state,
    outcomeHistory: [...state.outcomeHistory, event].slice(-MAX_OUTCOME_HISTORY),
    activeMutations,
  };
}

/**
 * Record a session completion (increments session counter).
 */
export function recordSessionEnd(state: PromptEvolutionState): PromptEvolutionState {
  const activeMutations = state.activeMutations.map(m => ({
    ...m,
    activeSessions: m.activeSessions + 1,
  }));

  return {
    ...state,
    totalSessions: state.totalSessions + 1,
    activeMutations,
  };
}

// ─── Fitness Computation ──────────────────────────────────

function computeMutationFitness(positives: number, negatives: number): number {
  const total = positives + negatives;
  if (total === 0) return 0.5; // Unknown = neutral
  return positives / total;
}

/**
 * Compute aggregate quality score for a template/effort combo.
 */
function computeStrategyQuality(
  events: OutcomeEvent[],
  template: string,
  effortLevel: string,
): number {
  const relevant = events.filter(
    e => e.template === template && e.effortLevel === effortLevel,
  );
  if (relevant.length === 0) return 0.5;

  const totalWeight = relevant.reduce(
    (sum, e) => sum + SIGNAL_WEIGHTS[e.signal],
    0,
  );
  // Normalize to 0-1 range
  return Math.max(0, Math.min(1, 0.5 + totalWeight / (relevant.length * 2)));
}

// ─── Rule-Based Mutation Generation ───────────────────────

/**
 * Analyze outcomes and generate rule-based prompt mutations.
 * Called during each evolution cycle.
 */
function generateRuleBasedMutations(state: PromptEvolutionState): PromptMutation[] {
  const mutations: PromptMutation[] = [];
  const events = state.outcomeHistory;
  if (events.length < MIN_SESSIONS_FOR_EVOLUTION) return mutations;

  // Analyze: which templates have high redo rates?
  const templateStats: Record<string, { accepts: number; redos: number; errors: number }> = {};
  for (const e of events) {
    if (!templateStats[e.template]) templateStats[e.template] = { accepts: 0, redos: 0, errors: 0 };
    if (e.signal === 'accept') templateStats[e.template].accepts++;
    if (e.signal === 'redo') templateStats[e.template].redos++;
    if (e.signal === 'error') templateStats[e.template].errors++;
  }

  for (const [template, stats] of Object.entries(templateStats)) {
    const total = stats.accepts + stats.redos + stats.errors;
    if (total < 3) continue;

    const redoRate = stats.redos / total;
    const errorRate = stats.errors / total;

    // High redo rate → add "be more careful about requirements" mutation
    if (redoRate > 0.4) {
      mutations.push(createMutation(
        `For ${template} projects: Read the user's request very carefully. Users frequently ask you to redo ${template} work. Before generating code, restate the key requirements to ensure you understand them correctly.`,
        template,
        'signal',
      ));
    }

    // High error rate → add "test your output" mutation
    if (errorRate > 0.3) {
      mutations.push(createMutation(
        `For ${template} projects: Your code frequently produces runtime errors. Double-check imports, variable references, and API signatures before outputting code. Prefer defensive patterns.`,
        template,
        'signal',
      ));
    }
  }

  // Analyze: tools-on vs tools-off quality
  const withTools = events.filter(e => e.usedTools);
  const withoutTools = events.filter(e => !e.usedTools);
  if (withTools.length >= 3 && withoutTools.length >= 3) {
    const toolAcceptRate = withTools.filter(e => e.signal === 'accept').length / withTools.length;
    const noToolAcceptRate = withoutTools.filter(e => e.signal === 'accept').length / withoutTools.length;

    if (toolAcceptRate > noToolAcceptRate + 0.2) {
      mutations.push(createMutation(
        'Tool-assisted generations have higher success rates. When editing code, always use file_read to inspect existing files before making changes.',
        '*',
        'signal',
      ));
    }
  }

  return mutations;
}

function createMutation(
  fragment: string,
  appliesTo: string,
  origin: 'dream' | 'signal',
): PromptMutation {
  return {
    id: `mut-${Date.now()}-${Math.random().toString(36).substring(2, 6)}`,
    fragment,
    appliesTo,
    origin,
    fitness: 0.5,
    generation: 0,
    createdAt: Date.now(),
    activeSessions: 0,
    positiveOutcomes: 0,
    negativeOutcomes: 0,
  };
}

// ─── AI-Powered Evolution (called during Dream Mode) ──────

/**
 * Generate evolved mutations using AI analysis.
 * Called from dream-mode.ts during consolidation.
 */
export async function evolveWithAI(
  state: PromptEvolutionState,
  ollamaHost: string,
  model: string,
): Promise<PromptMutation[]> {
  const events = state.outcomeHistory;
  if (events.length < MIN_SESSIONS_FOR_EVOLUTION) return [];

  // Build analysis prompt
  const recentEvents = events.slice(-50);
  const eventSummary = recentEvents.map(e =>
    `${e.signal} | template=${e.template} effort=${e.effortLevel} tools=${e.usedTools} toolCalls=${e.toolCallCount} | "${e.promptPreview}"`
  ).join('\n');

  const currentMutations = state.activeMutations.map(m =>
    `[fitness=${m.fitness.toFixed(2)}, gen=${m.generation}] ${m.fragment.substring(0, 100)}`
  ).join('\n');

  const prompt = `You are a prompt evolution engine. Analyze these generation outcomes and propose improved prompt strategies.

OUTCOME HISTORY (signal | context | prompt preview):
${eventSummary}

CURRENT ACTIVE MUTATIONS:
${currentMutations || '(none)'}

Based on patterns in successes and failures, propose 1-3 short prompt instructions that would improve generation quality. Each instruction should be:
- Specific and actionable (not vague)
- Based on actual patterns in the data
- Maximum 2 sentences

Output ONLY a JSON array of objects: [{ "fragment": "...", "appliesTo": "template-name or *" }]`;

  try {
    const response = await fetch(`${ollamaHost}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model,
        messages: [
          { role: 'system', content: 'You are a prompt optimization engine. Output ONLY valid JSON arrays.' },
          { role: 'user', content: prompt },
        ],
        stream: false,
        options: { temperature: 0.4 },
      }),
    });

    if (!response.ok) return [];

    const data = await response.json();
    const rawContent = data.message?.content || '[]';

    const jsonMatch = rawContent.match(/\[[\s\S]*\]/);
    if (!jsonMatch) return [];

    const parsed = JSON.parse(jsonMatch[0]);
    return parsed.slice(0, 3).map((entry: any) =>
      createMutation(
        String(entry.fragment || '').substring(0, 300),
        String(entry.appliesTo || '*'),
        'dream',
      )
    );
  } catch (e) {
    console.warn('[prompt-evolution] AI evolution failed:', e);
    return [];
  }
}

// ─── Evolution Cycle ──────────────────────────────────────

/**
 * Run a full evolution cycle: retire low-fitness, promote proven,
 * generate new mutations. Called from dream mode.
 */
export async function runEvolutionCycle(
  state: PromptEvolutionState,
  ollamaHost?: string,
  model?: string,
): Promise<PromptEvolutionState> {
  let newState = { ...state };

  // Step 1: Retire low-fitness mutations
  const surviving: PromptMutation[] = [];
  const retiring: PromptMutation[] = [];
  
  for (const m of newState.activeMutations) {
    if (m.activeSessions >= 3 && m.fitness < MUTATION_RETIRE_THRESHOLD) {
      retiring.push(m);
    } else {
      surviving.push(m);
    }
  }
  
  newState.activeMutations = surviving;
  newState.archivedMutations = [...newState.archivedMutations, ...retiring].slice(-50);

  // Step 2: Generate rule-based mutations
  const ruleMutations = generateRuleBasedMutations(newState);

  // Step 3: Generate AI mutations (if Ollama available)
  let aiMutations: PromptMutation[] = [];
  if (ollamaHost && model) {
    aiMutations = await evolveWithAI(newState, ollamaHost, model);
  }

  // Step 4: Merge new mutations (don't exceed max)
  const allNew = [...ruleMutations, ...aiMutations];
  const slotsAvailable = MAX_ACTIVE_MUTATIONS - newState.activeMutations.length;
  
  if (slotsAvailable > 0 && allNew.length > 0) {
    // Deduplicate: don't add mutations too similar to existing ones
    const existingFragments = newState.activeMutations.map(m => m.fragment.toLowerCase().substring(0, 50));
    const unique = allNew.filter(m => 
      !existingFragments.some(ef => ef.includes(m.fragment.toLowerCase().substring(0, 30)))
    );
    
    newState.activeMutations = [
      ...newState.activeMutations,
      ...unique.slice(0, slotsAvailable),
    ];
  }

  newState.activeMutationIds = newState.activeMutations.map(m => m.id);
  newState.lastEvolvedAt = Date.now();

  return newState;
}

// ─── Prompt Integration ───────────────────────────────────

/**
 * Get active prompt mutations for injection into system prompt.
 * Optionally filtered by template.
 */
export function getEvolvedPromptFragments(
  state: PromptEvolutionState,
  template?: string,
): string {
  if (state.activeMutations.length === 0) return '';

  const applicable = state.activeMutations.filter(m =>
    m.appliesTo === '*' || m.appliesTo === template
  );

  if (applicable.length === 0) return '';

  const fragments = applicable.map(m => `- ${m.fragment}`).join('\n');
  return `LEARNED STRATEGIES (evolved from past session outcomes):\n${fragments}`;
}

/**
 * Get evolution status for debugging/display.
 */
export function getEvolutionStatus(state: PromptEvolutionState): string {
  const lines = [
    `Prompt Evolution: ${state.totalSessions} sessions tracked`,
    `Active mutations: ${state.activeMutations.length}/${MAX_ACTIVE_MUTATIONS}`,
    `Archived: ${state.archivedMutations.length}`,
    `Outcome history: ${state.outcomeHistory.length} events`,
  ];

  if (state.lastEvolvedAt) {
    const hoursAgo = ((Date.now() - state.lastEvolvedAt) / (1000 * 60 * 60)).toFixed(1);
    lines.push(`Last evolution: ${hoursAgo}h ago`);
  }

  if (state.activeMutations.length > 0) {
    lines.push('Active mutations:');
    for (const m of state.activeMutations) {
      lines.push(`  [${m.origin}] fitness=${m.fitness.toFixed(2)} gen=${m.generation}: ${m.fragment.substring(0, 80)}...`);
    }
  }

  // Compute overall quality trend
  const recent20 = state.outcomeHistory.slice(-20);
  if (recent20.length >= 5) {
    const accepts = recent20.filter(e => e.signal === 'accept').length;
    const quality = (accepts / recent20.length * 100).toFixed(0);
    lines.push(`Recent quality: ${quality}% accept rate (last ${recent20.length} generations)`);
  }

  return lines.join('\n');
}
