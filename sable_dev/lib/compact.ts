/**
 * Context Compaction System
 * 
 * Multi-level pruning to keep conversations within token limits:
 *   1. Auto-compact  — Triggered when context reaches 90% of capacity
 *   2. Micro-compact — Clears old tool results between turns
 *   3. Manual compact — User-triggered via /compact command
 * 
 * Adapted for local models which have 4K-32K context windows.
 * Uses a summarization approach: sends old messages to a fast local model
 * to generate a compact summary, then replaces them.
 */

import {
  estimateTokens,
  estimateMessageTokens,
  estimateTotalTokens,
  getEffectiveContextSize,
  type BudgetTracker,
} from './token-budget';

// ─── Constants ────────────────────────────────────────────

/** Target budget after compaction (tokens) */
const POST_COMPACT_TOKEN_BUDGET = 12_000;

/** Max tokens per restored file after compaction */
const POST_COMPACT_MAX_TOKENS_PER_FILE = 2_000;

/** Max files to keep in context after compaction */
const POST_COMPACT_MAX_FILES = 5;

/** Max consecutive auto-compact failures before giving up */
const MAX_CONSECUTIVE_FAILURES = 3;

/** Tool results older than this many turns get micro-compacted */
const MICRO_COMPACT_TURN_AGE = 4;

/** Placeholder for cleared tool results */
const CLEARED_MESSAGE = '[Previous tool output cleared to save context]';

// ─── Types ────────────────────────────────────────────────

export interface CompactionResult {
  /** The compacted message array */
  messages: CompactMessage[];
  /** Summary of what was compacted */
  summary: string;
  /** Token count before compaction */
  preCompactTokens: number;
  /** Token count after compaction */
  postCompactTokens: number;
  /** Number of messages removed */
  messagesRemoved: number;
}

export interface CompactMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  metadata?: Record<string, unknown>;
}

export interface AutoCompactState {
  /** Whether we just auto-compacted */
  compacted: boolean;
  /** Turn counter since last compact */
  turnCounter: number;
  /** Consecutive failures */
  consecutiveFailures: number;
  /** Last compact timestamp */
  lastCompactAt?: number;
}

// ─── Micro-Compact ────────────────────────────────────────

/**
 * Micro-compact: Clear old tool results from messages to save tokens.
 * This is a lightweight operation that runs between turns.
 * 
 * Strategy: Replace tool outputs older than MICRO_COMPACT_TURN_AGE turns
 * with a short placeholder, preserving the tool call info.
 */
export function microCompact(
  messages: CompactMessage[],
  currentTurn: number,
): { messages: CompactMessage[]; tokensFreed: number } {
  let tokensFreed = 0;
  
  const result = messages.map((msg, idx) => {
    // Only compact assistant messages with tool results
    if (msg.role !== 'assistant') return msg;
    
    const turnAge = currentTurn - idx;
    if (turnAge < MICRO_COMPACT_TURN_AGE) return msg;
    
    // Check if this looks like a tool result (contains tool output patterns)
    const hasToolOutput = msg.content.includes('[TOOL ') || 
                          msg.content.includes('```\n') && msg.content.length > 2000;
    
    if (hasToolOutput && msg.content.length > 500) {
      const originalTokens = estimateTokens(msg.content);
      // Keep first 200 chars + tool name, replace the rest
      const toolNameMatch = msg.content.match(/\[TOOL (\w+)\]/);
      const toolName = toolNameMatch ? toolNameMatch[1] : 'unknown';
      const shortened = `[Tool: ${toolName}] ${CLEARED_MESSAGE}`;
      const newTokens = estimateTokens(shortened);
      tokensFreed += originalTokens - newTokens;
      
      return { ...msg, content: shortened };
    }
    
    return msg;
  });
  
  return { messages: result, tokensFreed };
}

// ─── Auto-Compact ─────────────────────────────────────────

/**
 * Check if auto-compact should trigger.
 */
export function shouldAutoCompact(
  messages: CompactMessage[],
  model: string,
  state: AutoCompactState,
): boolean {
  if (state.consecutiveFailures >= MAX_CONSECUTIVE_FAILURES) {
    return false; // Circuit breaker
  }
  
  const totalTokens = estimateTotalTokens(messages);
  const effectiveSize = getEffectiveContextSize(model);
  const threshold = effectiveSize * 0.85; // Trigger at 85%
  
  return totalTokens >= threshold;
}

/**
 * Perform auto-compaction by summarizing old messages.
 * 
 * Strategy:
 *   1. Keep the system prompt (first message)
 *   2. Keep the last N messages (recent context)
 *   3. Summarize everything in between into a compact summary
 *   4. Return: [system, summary, ...recent]
 */
export async function compactConversation(
  messages: CompactMessage[],
  model: string,
  summarizer?: (text: string) => Promise<string>,
): Promise<CompactionResult> {
  const preCompactTokens = estimateTotalTokens(messages);
  
  if (messages.length <= 4) {
    return {
      messages,
      summary: 'Too few messages to compact',
      preCompactTokens,
      postCompactTokens: preCompactTokens,
      messagesRemoved: 0,
    };
  }
  
  // Separate messages into: system, old, recent
  const systemMessages = messages.filter(m => m.role === 'system');
  const nonSystem = messages.filter(m => m.role !== 'system');
  
  // Keep last 4 messages as recent context
  const recentCount = Math.min(4, Math.floor(nonSystem.length / 2));
  const oldMessages = nonSystem.slice(0, nonSystem.length - recentCount);
  const recentMessages = nonSystem.slice(nonSystem.length - recentCount);
  
  // Build summary of old messages
  let summaryText: string;
  
  if (summarizer) {
    // Use AI summarizer (preferred — calls fast local model)
    const oldContent = oldMessages
      .map(m => `[${m.role}]: ${m.content.substring(0, 500)}`)
      .join('\n');
    
    try {
      summaryText = await summarizer(oldContent);
    } catch {
      // Fallback to extractive summary
      summaryText = buildExtractedSummary(oldMessages);
    }
  } else {
    // No summarizer — use extractive approach
    summaryText = buildExtractedSummary(oldMessages);
  }
  
  // Build compacted message array
  const compactSummary: CompactMessage = {
    role: 'system',
    content: `CONVERSATION SUMMARY (${oldMessages.length} messages compacted):\n${summaryText}`,
    metadata: { isCompactionSummary: true, compactedAt: Date.now() },
  };
  
  const compactedMessages: CompactMessage[] = [
    ...systemMessages,
    compactSummary,
    ...recentMessages,
  ];
  
  const postCompactTokens = estimateTotalTokens(compactedMessages);
  
  return {
    messages: compactedMessages,
    summary: `Compacted ${oldMessages.length} messages → ${estimateTokens(summaryText)} tokens`,
    preCompactTokens,
    postCompactTokens,
    messagesRemoved: oldMessages.length,
  };
}

// ─── Extractive Summary ───────────────────────────────────

/**
 * Build an extractive summary without using AI.
 * Extracts key facts: file edits, user requests, errors, decisions.
 */
function buildExtractedSummary(messages: CompactMessage[]): string {
  const facts: string[] = [];
  const filesEdited = new Set<string>();
  const userRequests: string[] = [];
  const errors: string[] = [];
  
  for (const msg of messages) {
    const content = msg.content;
    
    // Extract file paths
    const fileMatches = content.match(/(?:file|path|src\/|components\/)[^\s"'<>]+/gi);
    if (fileMatches) {
      fileMatches.slice(0, 5).forEach(f => filesEdited.add(f.replace(/[,"']/g, '')));
    }
    
    // Extract user requests (first line of user messages)
    if (msg.role === 'user' && content.length > 10) {
      const firstLine = content.split('\n')[0].substring(0, 150);
      userRequests.push(firstLine);
    }
    
    // Extract errors
    if (content.toLowerCase().includes('error') || content.toLowerCase().includes('failed')) {
      const errorLine = content.split('\n').find(l => 
        l.toLowerCase().includes('error') || l.toLowerCase().includes('failed')
      );
      if (errorLine) {
        errors.push(errorLine.substring(0, 150));
      }
    }
  }
  
  // Build summary
  const parts: string[] = [];
  
  if (userRequests.length > 0) {
    parts.push(`User requests:\n${userRequests.slice(-5).map(r => `- ${r}`).join('\n')}`);
  }
  
  if (filesEdited.size > 0) {
    parts.push(`Files mentioned: ${Array.from(filesEdited).slice(0, 10).join(', ')}`);
  }
  
  if (errors.length > 0) {
    parts.push(`Errors encountered:\n${errors.slice(-3).map(e => `- ${e}`).join('\n')}`);
  }
  
  return parts.join('\n\n') || 'Previous conversation context (no extractable details).';
}

// ─── Strip Utilities ──────────────────────────────────────

/**
 * Strip image/base64 content from messages to save tokens.
 */
export function stripImagesFromMessages(messages: CompactMessage[]): CompactMessage[] {
  return messages.map(msg => {
    if (msg.content.includes('data:image/') || msg.content.includes('base64,')) {
      const stripped = msg.content.replace(
        /data:image\/[^;]+;base64,[A-Za-z0-9+/=]+/g,
        '[image removed]'
      );
      return { ...msg, content: stripped };
    }
    return msg;
  });
}

/**
 * Truncate messages from the head (oldest) when prompt-too-long error occurs.
 * Keeps system messages and last N non-system messages.
 */
export function truncateHead(
  messages: CompactMessage[],
  keepLast: number = 4,
): CompactMessage[] {
  const system = messages.filter(m => m.role === 'system');
  const nonSystem = messages.filter(m => m.role !== 'system');
  
  if (nonSystem.length <= keepLast) return messages;
  
  return [...system, ...nonSystem.slice(-keepLast)];
}

// ─── State Management ─────────────────────────────────────

/**
 * Create initial auto-compact state.
 */
export function createAutoCompactState(): AutoCompactState {
  return {
    compacted: false,
    turnCounter: 0,
    consecutiveFailures: 0,
  };
}

/**
 * Record a successful compaction.
 */
export function recordCompactSuccess(state: AutoCompactState): AutoCompactState {
  return {
    ...state,
    compacted: true,
    turnCounter: 0,
    consecutiveFailures: 0,
    lastCompactAt: Date.now(),
  };
}

/**
 * Record a failed compaction.
 */
export function recordCompactFailure(state: AutoCompactState): AutoCompactState {
  return {
    ...state,
    consecutiveFailures: state.consecutiveFailures + 1,
  };
}

/**
 * Advance turn counter.
 */
export function advanceTurn(state: AutoCompactState): AutoCompactState {
  return {
    ...state,
    turnCounter: state.turnCounter + 1,
    compacted: false,
  };
}
