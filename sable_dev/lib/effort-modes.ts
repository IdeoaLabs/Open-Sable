/**
 * Effort Modes
 * 
 * Controls how much effort the AI puts into responses.
 * Maps to system prompt adjustments and token budget constraints.
 * 
 * Levels:
 *   low    - Quick, concise responses. Minimal exploration.
 *   medium - Balanced (default). Normal tool usage.
 *   high   - Thorough analysis. More context gathering, deeper reasoning.
 */

// ─── Types ────────────────────────────────────────────────

export type EffortLevel = 'low' | 'medium' | 'high';

export interface EffortConfig {
  level: EffortLevel;
  /** Max tokens for AI response */
  maxTokens: number;
  /** Temperature adjustment */
  temperature: number;
  /** Max tool call rounds (maxSteps) */
  maxSteps: number;
  /** System prompt modifier */
  systemPromptSuffix: string;
  /** Whether to run context pipeline */
  useContextPipeline: boolean;
  /** Whether to auto-suggest plans */
  suggestPlans: boolean;
}

// ─── Configurations ───────────────────────────────────────

const EFFORT_CONFIGS: Record<EffortLevel, EffortConfig> = {
  low: {
    level: 'low',
    maxTokens: 2048,
    temperature: 0.3,
    maxSteps: 3,
    systemPromptSuffix: 'Be concise. Give direct answers. Minimize explanation. Use fewer tool calls.',
    useContextPipeline: false,
    suggestPlans: false,
  },
  medium: {
    level: 'medium',
    maxTokens: 8000,
    temperature: 0.7,
    maxSteps: 8,
    systemPromptSuffix: '',
    useContextPipeline: true,
    suggestPlans: true,
  },
  high: {
    level: 'high',
    maxTokens: 16000,
    temperature: 0.8,
    maxSteps: 15,
    systemPromptSuffix: 'Be thorough. Gather full context before making changes. Verify your work. Consider edge cases. Read files completely before editing.',
    useContextPipeline: true,
    suggestPlans: true,
  },
};

// ─── State ────────────────────────────────────────────────

export interface EffortState {
  current: EffortLevel;
  history: Array<{ level: EffortLevel; timestamp: number }>;
}

/**
 * Create initial effort state.
 */
export function createEffortState(initial: EffortLevel = 'medium'): EffortState {
  return {
    current: initial,
    history: [{ level: initial, timestamp: Date.now() }],
  };
}

// ─── Operations ───────────────────────────────────────────

/**
 * Get the configuration for an effort level.
 */
export function getEffortConfig(level?: EffortLevel): EffortConfig {
  return EFFORT_CONFIGS[level || 'medium'];
}

/**
 * Set effort level.
 */
export function setEffortLevel(state: EffortState, level: EffortLevel): EffortState {
  if (state.current === level) return state;
  
  return {
    current: level,
    history: [...state.history, { level, timestamp: Date.now() }],
  };
}

/**
 * Get current effort level.
 */
export function getCurrentEffort(state: EffortState): EffortLevel {
  return state.current;
}

// ─── Effort Resolution ───────────────────────────────────

/**
 * Resolve effort from user prompt (auto-detect).
 * Only suggests,  caller decides whether to override.
 */
export function resolveEffortFromPrompt(prompt: string): EffortLevel | null {
  const lower = prompt.toLowerCase().trim();
  
  // Explicit effort markers
  if (/\b(quick|brief|short|fast|tl;?dr)\b/.test(lower)) return 'low';
  if (/\b(thorough|detailed|careful|in[- ]depth|comprehensive)\b/.test(lower)) return 'high';
  
  // Length-based heuristic
  if (lower.length < 30 && !lower.includes('implement') && !lower.includes('create')) {
    return 'low';
  }
  
  if (lower.length > 200 || (lower.match(/\n/g)?.length || 0) > 3) {
    return 'high';
  }
  
  return null; // No suggestion, use current setting
}

/**
 * Apply effort config to AI generation parameters.
 */
export function applyEffortToParams(
  params: {
    maxTokens?: number;
    temperature?: number;
    maxSteps?: number;
    systemPrompt?: string;
  },
  effort: EffortConfig,
): typeof params {
  return {
    ...params,
    maxTokens: effort.maxTokens,
    temperature: effort.temperature,
    maxSteps: effort.maxSteps,
    systemPrompt: effort.systemPromptSuffix 
      ? `${params.systemPrompt || ''}\n\n${effort.systemPromptSuffix}`.trim()
      : params.systemPrompt,
  };
}

/**
 * Format effort level for display.
 */
export function formatEffortDisplay(level: EffortLevel): string {
  const labels: Record<EffortLevel, string> = {
    low: 'Low (quick & concise)',
    medium: 'Medium (balanced)',
    high: 'High (thorough & detailed)',
  };
  return labels[level];
}

/**
 * Get all available effort levels with descriptions.
 */
export function getEffortLevels(): Array<{ level: EffortLevel; label: string; description: string }> {
  return [
    { level: 'low', label: 'Low', description: 'Quick responses, minimal tool usage' },
    { level: 'medium', label: 'Medium', description: 'Balanced approach (default)' },
    { level: 'high', label: 'High', description: 'Thorough analysis, more context gathering' },
  ];
}
