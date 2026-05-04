/**
 * Error Recovery & Retry System
 * 
 * Handles:
 * - Retryable errors (429, 503, timeout) with exponential backoff
 * - Non-retryable errors (auth, not found),  fail immediately
 * - Conversation interruption detection and recovery
 * - Auto-fix loop: build errors → re-invoke model with error context
 */

export interface RetryConfig {
  maxRetries: number;
  baseDelay: number;       // ms
  maxDelay: number;        // ms
  backoffFactor: number;
  /** If true, keeps retrying even after hitting max, with maxDelay between */
  persistent?: boolean;
}

const DEFAULT_RETRY_CONFIG: RetryConfig = {
  maxRetries: 3,
  baseDelay: 1000,
  maxDelay: 30000,
  backoffFactor: 2,
  persistent: false,
};

export type ErrorCategory = 'retryable' | 'permanent' | 'context_overflow' | 'rate_limit';

/**
 * Classify an error to determine retry strategy.
 */
export function classifyError(error: any): ErrorCategory {
  const message = (error?.message || '').toLowerCase();
  const status = error?.status || error?.statusCode || 0;

  // Rate limits
  if (status === 429 || message.includes('rate limit') || message.includes('too many requests')) {
    return 'rate_limit';
  }

  // Context overflow,  need to reduce context, not retry blindly
  if (message.includes('context length') || message.includes('token limit') || 
      message.includes('maximum context') || message.includes('too long')) {
    return 'context_overflow';
  }

  // Retryable server errors
  if ([500, 502, 503, 529].includes(status) || 
      message.includes('service unavailable') || 
      message.includes('timeout') ||
      message.includes('econnreset') ||
      message.includes('econnrefused') ||
      message.includes('network') ||
      message.includes('temporarily')) {
    return 'retryable';
  }

  // Everything else is permanent
  return 'permanent';
}

/**
 * Execute a function with retry logic.
 */
export async function withRetry<T>(
  fn: () => Promise<T>,
  config: Partial<RetryConfig> = {},
  onRetry?: (attempt: number, error: any, category: ErrorCategory, delay: number) => void,
): Promise<T> {
  const cfg = { ...DEFAULT_RETRY_CONFIG, ...config };
  let attempt = 0;
  let consecutiveServerErrors = 0;

  while (true) {
    try {
      const result = await fn();
      consecutiveServerErrors = 0;
      return result;
    } catch (error: any) {
      attempt++;
      const category = classifyError(error);

      // Permanent errors,  don't retry
      if (category === 'permanent') {
        throw error;
      }

      // Context overflow,  don't retry (caller needs to reduce context)
      if (category === 'context_overflow') {
        throw Object.assign(error, { isContextOverflow: true });
      }

      // Track consecutive server errors
      if (category === 'retryable') {
        consecutiveServerErrors++;
      }

      // Check if we've exhausted retries
      if (attempt > cfg.maxRetries && !cfg.persistent) {
        throw error;
      }

      // Calculate delay with exponential backoff
      let delay = Math.min(
        cfg.baseDelay * Math.pow(cfg.backoffFactor, attempt - 1),
        cfg.maxDelay
      );

      // Rate limit: use retry-after header if available
      if (category === 'rate_limit' && error?.headers?.['retry-after']) {
        const retryAfter = parseInt(error.headers['retry-after'], 10);
        if (!isNaN(retryAfter)) {
          delay = retryAfter * 1000;
        }
      }

      // Add jitter (±20%)
      delay = delay * (0.8 + Math.random() * 0.4);

      onRetry?.(attempt, error, category, delay);

      await sleep(delay);
    }
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// ─── Conversation Recovery ────────────────────────────────

export interface InterruptionState {
  type: 'mid-turn' | 'mid-prompt' | 'clean';
  lastAssistantMessage?: string;
  wasGenerating: boolean;
  timestamp: number;
}

/**
 * Detect the type of conversation interruption.
 * @param messages The conversation messages so far  
 * @param wasStreaming Whether the AI was mid-stream when interrupted
 */
export function detectInterruption(
  messages: Array<{ role: string; content: string }>,
  wasStreaming: boolean,
): InterruptionState {
  if (messages.length === 0) {
    return { type: 'clean', wasGenerating: false, timestamp: Date.now() };
  }

  const lastMsg = messages[messages.length - 1];

  // If last message is from user and we were streaming → mid-turn
  if (lastMsg.role === 'user' && wasStreaming) {
    return { 
      type: 'mid-turn', 
      wasGenerating: true, 
      timestamp: Date.now() 
    };
  }

  // If last message is from assistant and looks truncated → mid-turn
  if (lastMsg.role === 'assistant') {
    const content = lastMsg.content;
    const looksIncomplete = 
      content.includes('<file path=') && !content.includes('</file>') ||
      content.endsWith(',') ||
      content.endsWith('{') ||
      content.endsWith('(');

    if (looksIncomplete) {
      return {
        type: 'mid-turn',
        lastAssistantMessage: content,
        wasGenerating: true,
        timestamp: Date.now(),
      };
    }
  }

  // If last message is from user but we weren't streaming → mid-prompt
  if (lastMsg.role === 'user' && !wasStreaming) {
    return { type: 'mid-prompt', wasGenerating: false, timestamp: Date.now() };
  }

  return { type: 'clean', wasGenerating: false, timestamp: Date.now() };
}

/**
 * Build a recovery message to inject when resuming an interrupted conversation.
 */
export function buildRecoveryMessage(state: InterruptionState): string | null {
  if (state.type === 'clean') return null;

  if (state.type === 'mid-turn') {
    if (state.lastAssistantMessage) {
      return `Continue from where you left off. Your previous response was interrupted. Here is what you had so far:\n\n${state.lastAssistantMessage.slice(-500)}`;
    }
    return 'Continue generating the code. Your previous response was interrupted before completion.';
  }

  if (state.type === 'mid-prompt') {
    return 'Continue with the user\'s last request.';
  }

  return null;
}

// ─── Auto-Fix Loop ────────────────────────────────────────

export interface AutoFixResult {
  fixed: boolean;
  attempts: number;
  lastError?: string;
}

/**
 * Detect if a tool result contains a build/runtime error that should trigger auto-fix.
 */
export function isToolError(result: { success: boolean; output?: string; error?: string }): boolean {
  if (!result.success) return true;
  
  const output = (result.output || '').toLowerCase();
  const errorPatterns = [
    'syntaxerror',
    'referenceerror', 
    'typeerror',
    'module not found',
    'cannot find module',
    'unexpected token',
    'failed to compile',
    'build failed',
    'enoent',
  ];

  return errorPatterns.some(p => output.includes(p));
}

/**
 * Format an error for injection into the model's next turn.
 * The model will see this and reformulate its approach.
 */
export function formatErrorForModel(toolName: string, error: string): string {
  return `[TOOL ERROR] The ${toolName} tool returned an error:\n\`\`\`\n${error.slice(0, 1000)}\n\`\`\`\nPlease fix the issue and try again with a corrected approach.`;
}
