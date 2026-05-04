/**
 * Fast Mode
 * 
 * Toggle between a fast (small) model for quick tasks and a 
 * capable (larger) model for complex reasoning.
 * 
 * Includes cooldown tracking to prevent constant switching.
 */

// ─── Configuration ────────────────────────────────────────

export interface FastModeConfig {
  /** The smaller, faster model */
  fastModel: string;
  /** The default capable model */
  capableModel: string;
  /** Minimum ms between model switches to prevent thrashing */
  cooldownMs: number;
}

const DEFAULT_CONFIG: FastModeConfig = {
  fastModel: 'ollama/vaultbox/qwen3.5-uncensored:4b',
  capableModel: 'ollama/qwen2.5:14b',
  cooldownMs: 5000,
};

// ─── State ────────────────────────────────────────────────

export interface FastModeState {
  enabled: boolean;
  lastToggleAt: number | null;
  config: FastModeConfig;
}

/**
 * Create initial fast mode state.
 */
export function createFastModeState(config?: Partial<FastModeConfig>): FastModeState {
  return {
    enabled: false,
    lastToggleAt: null,
    config: { ...DEFAULT_CONFIG, ...config },
  };
}

// ─── Operations ───────────────────────────────────────────

/**
 * Check if fast mode is enabled.
 */
export function isFastModeEnabled(state: FastModeState): boolean {
  return state.enabled;
}

/**
 * Get the current model based on fast mode state.
 */
export function getFastModeModel(state: FastModeState): string {
  return state.enabled ? state.config.fastModel : state.config.capableModel;
}

/**
 * Toggle fast mode on/off with cooldown protection.
 */
export function toggleFastMode(
  state: FastModeState,
  enable?: boolean,
): { state: FastModeState; toggled: boolean; reason?: string } {
  const now = Date.now();
  const target = enable !== undefined ? enable : !state.enabled;
  
  // Already in desired state
  if (target === state.enabled) {
    return { 
      state, 
      toggled: false, 
      reason: `Fast mode is already ${target ? 'enabled' : 'disabled'}.`,
    };
  }
  
  // Check cooldown
  if (state.lastToggleAt !== null) {
    const elapsed = now - state.lastToggleAt;
    if (elapsed < state.config.cooldownMs) {
      const remaining = Math.ceil((state.config.cooldownMs - elapsed) / 1000);
      return {
        state,
        toggled: false,
        reason: `Cooldown active. Try again in ${remaining}s.`,
      };
    }
  }
  
  return {
    state: {
      ...state,
      enabled: target,
      lastToggleAt: now,
    },
    toggled: true,
  };
}

// ─── Auto Fast Mode (optional) ───────────────────────────

/**
 * Simple heuristic: suggest fast mode for short, simple prompts.
 * This is advisory,  the caller decides whether to actually enable it.
 */
export function shouldSuggestFastMode(prompt: string): boolean {
  const lower = prompt.toLowerCase().trim();
  
  // Short prompts are usually simple
  if (lower.length < 50) return true;
  
  // Quick question patterns
  const quickPatterns = [
    /^(what|how|where|which|when|who|why)\s+(is|are|was|were|do|does|did|can|should)\b/,
    /^(explain|describe|list|show|tell me)\b/,
    /^(yes|no|ok|sure|thanks|ty|thx)\b/,
  ];
  
  if (quickPatterns.some(p => p.test(lower))) return true;
  
  // Complex patterns suggest NOT using fast mode
  const complexPatterns = [
    /refactor/,
    /implement/,
    /create\s+a\s+(new\s+)?(?:module|system|component|service)/,
    /fix.*(bug|error|issue).*(in|across|throughout)/,
    /review\s+.*code/,
    /migration/,
    /multiple\s+files/,
  ];
  
  if (complexPatterns.some(p => p.test(lower))) return false;
  
  return false;
}

/**
 * Get model display name (short, human-friendly).
 */
export function getModelDisplayName(modelId: string): string {
  // Strip provider prefix
  const parts = modelId.split('/');
  const name = parts[parts.length - 1];
  return name;
}

/**
 * Format fast mode status for display.
 */
export function formatFastModeStatus(state: FastModeState): string {
  const model = getFastModeModel(state);
  const displayName = getModelDisplayName(model);
  return `Fast mode: ${state.enabled ? 'ON' : 'OFF'} (${displayName})`;
}
