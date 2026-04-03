/**
 * Token Estimation & Budget Tracking
 * 
 * Estimates token counts for messages and tracks budget usage across turns.
 * Adapted for local models (Ollama) which typically have 4K-32K context windows.
 * 
 * Uses a simple character-based estimation since local models don't report
 * token counts in the same way as cloud APIs.
 */

// ─── Constants ────────────────────────────────────────────

/** Average characters per token (conservative estimate for English code) */
const CHARS_PER_TOKEN = 3.5;

/** Stop generating when this % of budget is used */
export const COMPLETION_THRESHOLD = 0.9;

/** If delta between checks is below this, consider diminishing returns */
export const DIMINISHING_THRESHOLD = 500;

/** Default context window sizes by model family */
const MODEL_CONTEXT_SIZES: Record<string, number> = {
  'qwen3.5': 32768,
  'qwen2.5': 32768,
  'qwen2.5:72b': 65536,
  'qwen2.5:14b': 32768,
  'llama3.1': 131072,
  'llama3.2': 131072,
  'mistral': 32768,
  'gemma': 8192,
  'phi': 4096,
  'deepseek': 65536,
  // Cloud models
  'gpt-5': 128000,
  'gpt-4': 128000,
  'claude': 200000,
  'gemini': 1000000,
};

/** Tokens reserved for model output */
const OUTPUT_RESERVE = 8192;

// ─── Token Estimation ─────────────────────────────────────

/**
 * Estimate token count for a string.
 * Uses character-based estimation (conservative).
 */
export function estimateTokens(text: string): number {
  if (!text) return 0;
  return Math.ceil(text.length / CHARS_PER_TOKEN);
}

/**
 * Estimate tokens for a conversation message.
 */
export function estimateMessageTokens(message: { role: string; content: string }): number {
  // Add overhead for role/formatting (~4 tokens per message)
  return estimateTokens(message.content) + 4;
}

/**
 * Estimate total tokens for an array of messages.
 */
export function estimateTotalTokens(messages: Array<{ role: string; content: string }>): number {
  return messages.reduce((sum, msg) => sum + estimateMessageTokens(msg), 0);
}

/**
 * Get context window size for a model.
 */
export function getContextWindowSize(model: string): number {
  const lower = model.toLowerCase();
  for (const [key, size] of Object.entries(MODEL_CONTEXT_SIZES)) {
    if (lower.includes(key.toLowerCase())) {
      return size;
    }
  }
  // Default: assume 32K for unknown local models
  return 32768;
}

/**
 * Get effective context window (total minus output reserve).
 */
export function getEffectiveContextSize(model: string): number {
  return getContextWindowSize(model) - OUTPUT_RESERVE;
}

// ─── Budget Tracker ───────────────────────────────────────

export interface BudgetTracker {
  /** How many continuations have occurred */
  continuationCount: number;
  /** Tokens generated in last delta check */
  lastDeltaTokens: number;
  /** Total tokens used at last check */
  lastTotalTokens: number;
  /** Timestamp when tracking started */
  startedAt: number;
  /** Model being used */
  model: string;
  /** Context window capacity */
  contextWindow: number;
  /** Effective capacity (minus output reserve) */
  effectiveCapacity: number;
}

export interface TokenBudgetDecision {
  action: 'continue' | 'stop' | 'compact';
  reason: string;
  usage: {
    totalTokens: number;
    percentUsed: number;
    remaining: number;
  };
}

/**
 * Create a new budget tracker for a model.
 */
export function createBudgetTracker(model: string): BudgetTracker {
  const contextWindow = getContextWindowSize(model);
  return {
    continuationCount: 0,
    lastDeltaTokens: 0,
    lastTotalTokens: 0,
    startedAt: Date.now(),
    model,
    contextWindow,
    effectiveCapacity: contextWindow - OUTPUT_RESERVE,
  };
}

/**
 * Check the token budget and decide what to do.
 */
export function checkTokenBudget(
  tracker: BudgetTracker,
  currentTokens: number,
): TokenBudgetDecision {
  const percentUsed = currentTokens / tracker.effectiveCapacity;
  const remaining = tracker.effectiveCapacity - currentTokens;
  const delta = currentTokens - tracker.lastTotalTokens;

  const usage = {
    totalTokens: currentTokens,
    percentUsed: Math.round(percentUsed * 100) / 100,
    remaining,
  };

  // Update tracker
  tracker.lastDeltaTokens = delta;
  tracker.lastTotalTokens = currentTokens;

  // Over budget → must compact
  if (percentUsed >= 1.0) {
    return { action: 'compact', reason: 'Context window full — compaction required', usage };
  }

  // Near limit → trigger compaction
  if (percentUsed >= COMPLETION_THRESHOLD) {
    return { action: 'compact', reason: `${Math.round(percentUsed * 100)}% of context used — auto-compacting`, usage };
  }

  // Diminishing returns detection
  if (tracker.continuationCount > 2 && delta > 0 && delta < DIMINISHING_THRESHOLD) {
    return { action: 'stop', reason: 'Diminishing returns — output delta below threshold', usage };
  }

  tracker.continuationCount++;
  return { action: 'continue', reason: 'Budget OK', usage };
}

// ─── Token Warning States ─────────────────────────────────

export type TokenWarningLevel = 'ok' | 'warning' | 'critical' | 'overflow';

/**
 * Calculate the warning level for current token usage.
 */
export function getTokenWarningLevel(
  currentTokens: number,
  model: string
): { level: TokenWarningLevel; message: string; percentUsed: number } {
  const effective = getEffectiveContextSize(model);
  const percent = currentTokens / effective;

  if (percent >= 1.0) {
    return { level: 'overflow', message: 'Context window full — messages will be lost', percentUsed: percent };
  }
  if (percent >= 0.85) {
    return { level: 'critical', message: 'Context nearly full — consider compacting', percentUsed: percent };
  }
  if (percent >= 0.7) {
    return { level: 'warning', message: 'Context usage high', percentUsed: percent };
  }
  return { level: 'ok', message: '', percentUsed: percent };
}
