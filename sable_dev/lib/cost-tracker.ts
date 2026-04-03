/**
 * Cost & Token Tracking
 * 
 * Tracks token usage per turn, per model, and per session.
 * For local models (Ollama), cost is $0 but usage metrics are still useful.
 * For cloud providers, computes approximate cost from token counts.
 */

// ─── Types ────────────────────────────────────────────────

export interface TokenUsage {
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
}

export interface TurnUsage {
  turn: number;
  timestamp: number;
  model: string;
  usage: TokenUsage;
  toolCalls: number;
  durationMs: number;
}

export interface SessionCostState {
  turns: TurnUsage[];
  totalTokens: TokenUsage;
  totalToolCalls: number;
  totalDurationMs: number;
  sessionStartedAt: number;
}

// ─── Pricing (per 1M tokens) ─────────────────────────────

interface ModelPricing {
  promptPer1M: number;
  completionPer1M: number;
}

const PRICING: Record<string, ModelPricing> = {
  // Cloud providers (approximate)
  'gpt-4o': { promptPer1M: 2.50, completionPer1M: 10.00 },
  'gpt-4o-mini': { promptPer1M: 0.15, completionPer1M: 0.60 },
  'claude-3.5-sonnet': { promptPer1M: 3.00, completionPer1M: 15.00 },
  'claude-3-haiku': { promptPer1M: 0.25, completionPer1M: 1.25 },
  'gemini-2.0-flash': { promptPer1M: 0.10, completionPer1M: 0.40 },
  // Local models = free
  'ollama': { promptPer1M: 0, completionPer1M: 0 },
};

// ─── State Management ─────────────────────────────────────

/**
 * Create initial cost tracking state.
 */
export function createCostState(): SessionCostState {
  return {
    turns: [],
    totalTokens: { promptTokens: 0, completionTokens: 0, totalTokens: 0 },
    totalToolCalls: 0,
    totalDurationMs: 0,
    sessionStartedAt: Date.now(),
  };
}

/**
 * Record token usage for a turn.
 */
export function trackUsage(
  state: SessionCostState,
  model: string,
  usage: Partial<TokenUsage>,
  toolCalls: number = 0,
  durationMs: number = 0,
): SessionCostState {
  const turnUsage: TokenUsage = {
    promptTokens: usage.promptTokens || 0,
    completionTokens: usage.completionTokens || 0,
    totalTokens: usage.totalTokens || ((usage.promptTokens || 0) + (usage.completionTokens || 0)),
  };

  const turn: TurnUsage = {
    turn: state.turns.length + 1,
    timestamp: Date.now(),
    model,
    usage: turnUsage,
    toolCalls,
    durationMs,
  };

  return {
    ...state,
    turns: [...state.turns, turn],
    totalTokens: {
      promptTokens: state.totalTokens.promptTokens + turnUsage.promptTokens,
      completionTokens: state.totalTokens.completionTokens + turnUsage.completionTokens,
      totalTokens: state.totalTokens.totalTokens + turnUsage.totalTokens,
    },
    totalToolCalls: state.totalToolCalls + toolCalls,
    totalDurationMs: state.totalDurationMs + durationMs,
  };
}

// ─── Cost Calculation ─────────────────────────────────────

/**
 * Get pricing for a model. Returns $0 for unknown/local models.
 */
function getPricing(model: string): ModelPricing {
  // Check for exact match
  if (PRICING[model]) return PRICING[model];
  
  // Check if it's a local model (ollama prefix or no recognized provider)
  if (model.startsWith('ollama/') || model.startsWith('openwebui/')) {
    return PRICING['ollama'];
  }
  
  // Check partial matches
  for (const [key, pricing] of Object.entries(PRICING)) {
    if (model.includes(key)) return pricing;
  }
  
  // Default: free (assume local)
  return { promptPer1M: 0, completionPer1M: 0 };
}

/**
 * Calculate cost for a specific usage.
 */
export function calculateCost(model: string, usage: TokenUsage): number {
  const pricing = getPricing(model);
  return (
    (usage.promptTokens / 1_000_000) * pricing.promptPer1M +
    (usage.completionTokens / 1_000_000) * pricing.completionPer1M
  );
}

/**
 * Get total session cost.
 */
export function getSessionCost(state: SessionCostState): number {
  let total = 0;
  for (const turn of state.turns) {
    total += calculateCost(turn.model, turn.usage);
  }
  return total;
}

// ─── Reporting ────────────────────────────────────────────

/**
 * Format a cost report for display.
 */
export function formatCostReport(state: SessionCostState): string {
  const cost = getSessionCost(state);
  const elapsed = Date.now() - state.sessionStartedAt;
  const minutes = Math.floor(elapsed / 60000);
  const seconds = Math.floor((elapsed % 60000) / 1000);
  
  const lines = [
    `Session Duration: ${minutes}m ${seconds}s`,
    `Turns: ${state.turns.length}`,
    `Tool Calls: ${state.totalToolCalls}`,
    ``,
    `Token Usage:`,
    `  Prompt:     ${state.totalTokens.promptTokens.toLocaleString()}`,
    `  Completion: ${state.totalTokens.completionTokens.toLocaleString()}`,
    `  Total:      ${state.totalTokens.totalTokens.toLocaleString()}`,
  ];
  
  if (cost > 0) {
    lines.push(``, `Estimated Cost: $${cost.toFixed(4)}`);
  } else {
    lines.push(``, `Cost: $0.00 (local models)`);
  }
  
  // Per-model breakdown if multiple models used
  const modelMap = new Map<string, TokenUsage>();
  for (const turn of state.turns) {
    const existing = modelMap.get(turn.model) || { promptTokens: 0, completionTokens: 0, totalTokens: 0 };
    modelMap.set(turn.model, {
      promptTokens: existing.promptTokens + turn.usage.promptTokens,
      completionTokens: existing.completionTokens + turn.usage.completionTokens,
      totalTokens: existing.totalTokens + turn.usage.totalTokens,
    });
  }
  
  if (modelMap.size > 1) {
    lines.push(``, `By Model:`);
    for (const [model, usage] of modelMap) {
      const modelCost = calculateCost(model, usage);
      const shortName = model.split('/').pop() || model;
      lines.push(`  ${shortName}: ${usage.totalTokens.toLocaleString()} tokens${modelCost > 0 ? ` ($${modelCost.toFixed(4)})` : ''}`);
    }
  }
  
  return lines.join('\n');
}

/**
 * Get a compact one-line summary.
 */
export function getCompactCostSummary(state: SessionCostState): string {
  const cost = getSessionCost(state);
  const total = state.totalTokens.totalTokens;
  
  if (cost > 0) {
    return `${total.toLocaleString()} tokens | $${cost.toFixed(4)}`;
  }
  return `${total.toLocaleString()} tokens`;
}
