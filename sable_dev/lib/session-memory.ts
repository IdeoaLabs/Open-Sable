/**
 * Session Memory Extraction
 * 
 * Auto-extracts useful insights from conversations into persistent notes.
 * Runs as a background task (non-blocking) after a threshold of activity.
 * 
 * Pattern: After enough tokens are generated / tool calls made,
 * a forked request to a fast local model summarizes key learnings
 * and appends them to a session memory file.
 * 
 * Stores to: .sable-dev/session-memory.md
 */

import { estimateTotalTokens } from './token-budget';

// ─── Configuration ────────────────────────────────────────

interface SessionMemoryConfig {
  /** Minimum total context tokens before first extraction */
  minimumTokensToInit: number;
  /** Token growth since last extraction before re-extracting */
  minimumTokensBetweenUpdates: number;
  /** Tool calls between updates */
  toolCallsBetweenUpdates: number;
}

const DEFAULT_CONFIG: SessionMemoryConfig = {
  minimumTokensToInit: 5000,
  minimumTokensBetweenUpdates: 3000,
  toolCallsBetweenUpdates: 3,
};

// ─── State ────────────────────────────────────────────────

export interface SessionMemoryState {
  /** Last extraction timestamp */
  lastExtractionAt: number | null;
  /** Tokens at last extraction */
  tokensAtLastExtraction: number;
  /** Tool calls since last extraction */
  toolCallsSinceExtraction: number;
  /** Whether extraction is currently running */
  isExtracting: boolean;
  /** Number of extractions done */
  extractionCount: number;
  /** Accumulated memory entries */
  entries: MemoryEntry[];
}

export interface MemoryEntry {
  timestamp: number;
  type: 'preference' | 'pattern' | 'decision' | 'error' | 'file_context';
  content: string;
}

/**
 * Create initial session memory state.
 */
export function createSessionMemoryState(): SessionMemoryState {
  return {
    lastExtractionAt: null,
    tokensAtLastExtraction: 0,
    toolCallsSinceExtraction: 0,
    isExtracting: false,
    extractionCount: 0,
    entries: [],
  };
}

// ─── Threshold Checks ─────────────────────────────────────

/**
 * Check if session memory extraction should be triggered.
 */
export function shouldExtractMemory(
  state: SessionMemoryState,
  currentTokens: number,
  config: SessionMemoryConfig = DEFAULT_CONFIG,
): boolean {
  if (state.isExtracting) return false;
  
  // First extraction: wait for minimum tokens
  if (state.lastExtractionAt === null) {
    return currentTokens >= config.minimumTokensToInit;
  }
  
  // Subsequent: check token growth or tool call threshold
  const tokenGrowth = currentTokens - state.tokensAtLastExtraction;
  const toolCallsReady = state.toolCallsSinceExtraction >= config.toolCallsBetweenUpdates;
  
  return tokenGrowth >= config.minimumTokensBetweenUpdates || toolCallsReady;
}

/**
 * Record a tool call (advances the tool call counter).
 */
export function recordToolCall(state: SessionMemoryState): SessionMemoryState {
  return {
    ...state,
    toolCallsSinceExtraction: state.toolCallsSinceExtraction + 1,
  };
}

// ─── Extraction ───────────────────────────────────────────

/**
 * Extract session memory from conversation messages.
 * This can either use AI (via summarizer callback) or use rule-based extraction.
 * 
 * @param messages - The conversation messages
 * @param state - Current memory state
 * @param summarizer - Optional AI summarizer (calls fast local model)
 * @returns Updated state with new entries
 */
export async function extractSessionMemory(
  messages: Array<{ role: string; content: string }>,
  state: SessionMemoryState,
  summarizer?: (prompt: string) => Promise<string>,
): Promise<SessionMemoryState> {
  const newState = { ...state, isExtracting: true };
  
  try {
    let newEntries: MemoryEntry[];
    
    if (summarizer) {
      newEntries = await extractWithAI(messages, state, summarizer);
    } else {
      newEntries = extractWithRules(messages, state);
    }
    
    const currentTokens = estimateTotalTokens(messages);
    
    return {
      ...newState,
      isExtracting: false,
      lastExtractionAt: Date.now(),
      tokensAtLastExtraction: currentTokens,
      toolCallsSinceExtraction: 0,
      extractionCount: state.extractionCount + 1,
      entries: [...state.entries, ...newEntries],
    };
  } catch {
    return { ...newState, isExtracting: false };
  }
}

/**
 * Extract memory using an AI model (fast local model).
 */
async function extractWithAI(
  messages: Array<{ role: string; content: string }>,
  state: SessionMemoryState,
  summarizer: (prompt: string) => Promise<string>,
): Promise<MemoryEntry[]> {
  // Only look at messages since last extraction
  const startIdx = state.lastExtractionAt 
    ? Math.max(0, messages.length - 10) // Last 10 messages
    : 0;
  
  const recentMessages = messages.slice(startIdx)
    .map(m => `[${m.role}]: ${m.content.substring(0, 300)}`)
    .join('\n');
  
  const prompt = `Extract key facts from this conversation that would be useful to remember for future interactions. Focus on:
- User preferences (coding style, framework choices, naming conventions)
- Important decisions made
- Patterns in what the user asks for
- Files that are important to the project
- Errors that were resolved and how

Output as a bullet list. Be concise — one line per fact. Only include genuinely useful insights.

Conversation:
${recentMessages}`;

  const result = await summarizer(prompt);
  
  // Parse bullet points from response
  const entries: MemoryEntry[] = [];
  const lines = result.split('\n').filter(l => l.trim().startsWith('-') || l.trim().startsWith('*'));
  
  for (const line of lines) {
    const content = line.replace(/^[\s\-\*]+/, '').trim();
    if (!content || content.length < 10) continue;
    
    const type = classifyEntry(content);
    entries.push({ timestamp: Date.now(), type, content });
  }
  
  return entries;
}

/**
 * Extract memory using rule-based patterns (no AI needed).
 */
function extractWithRules(
  messages: Array<{ role: string; content: string }>,
  state: SessionMemoryState,
): MemoryEntry[] {
  const entries: MemoryEntry[] = [];
  const seen = new Set(state.entries.map(e => e.content));
  
  for (const msg of messages) {
    const content = msg.content;
    
    // Detect file edit patterns
    if (msg.role === 'user' && content.match(/<file path="[^"]+"/)) {
      const files = content.match(/path="([^"]+)"/g)?.map(m => m.replace(/path="|"/g, ''));
      if (files && files.length > 0) {
        const entry = `Files edited: ${files.join(', ')}`;
        if (!seen.has(entry)) {
          entries.push({ timestamp: Date.now(), type: 'file_context', content: entry });
          seen.add(entry);
        }
      }
    }
    
    // Detect preference statements
    if (msg.role === 'user') {
      const prefPatterns = [
        /(?:i prefer|always use|don't use|never use|i like|i want)\s+(.+)/i,
        /(?:use|switch to|change to)\s+(tailwind|css modules|styled-components|sass)/i,
      ];
      for (const pattern of prefPatterns) {
        const match = content.match(pattern);
        if (match) {
          const entry = `Preference: ${match[0].substring(0, 100)}`;
          if (!seen.has(entry)) {
            entries.push({ timestamp: Date.now(), type: 'preference', content: entry });
            seen.add(entry);
          }
        }
      }
    }
    
    // Detect error resolutions
    if (msg.role === 'assistant' && content.toLowerCase().includes('fixed') && content.toLowerCase().includes('error')) {
      const entry = `Error resolved: ${content.substring(0, 100)}`;
      if (!seen.has(entry)) {
        entries.push({ timestamp: Date.now(), type: 'error', content: entry });
        seen.add(entry);
      }
    }
  }
  
  return entries;
}

/**
 * Classify a memory entry by its content.
 */
function classifyEntry(content: string): MemoryEntry['type'] {
  const lower = content.toLowerCase();
  if (lower.includes('prefer') || lower.includes('style') || lower.includes('convention')) return 'preference';
  if (lower.includes('error') || lower.includes('fix') || lower.includes('bug')) return 'error';
  if (lower.includes('decided') || lower.includes('chose') || lower.includes('approach')) return 'decision';
  if (lower.includes('file') || lower.includes('component') || lower.includes('module')) return 'file_context';
  return 'pattern';
}

// ─── Serialization ────────────────────────────────────────

/**
 * Format session memory as a markdown string for persistence.
 */
export function formatMemoryAsMarkdown(state: SessionMemoryState): string {
  if (state.entries.length === 0) return '# Session Memory\n\n_No entries yet._\n';
  
  const grouped: Record<string, MemoryEntry[]> = {};
  for (const entry of state.entries) {
    if (!grouped[entry.type]) grouped[entry.type] = [];
    grouped[entry.type].push(entry);
  }
  
  let md = `# Session Memory\n\n_Updated: ${new Date().toISOString()}_\n_Entries: ${state.entries.length}_\n\n`;
  
  const typeLabels: Record<string, string> = {
    preference: 'User Preferences',
    pattern: 'Patterns',
    decision: 'Decisions',
    error: 'Error Resolutions',
    file_context: 'File Context',
  };
  
  for (const [type, entries] of Object.entries(grouped)) {
    md += `## ${typeLabels[type] || type}\n\n`;
    for (const entry of entries) {
      md += `- ${entry.content}\n`;
    }
    md += '\n';
  }
  
  return md;
}

/**
 * Format memory entries as a compact string for injection into system prompt.
 */
export function formatMemoryForPrompt(state: SessionMemoryState): string {
  if (state.entries.length === 0) return '';
  
  const recent = state.entries.slice(-15); // Last 15 entries
  const lines = recent.map(e => `- [${e.type}] ${e.content}`);
  
  return `SESSION MEMORY (remembered from this conversation):\n${lines.join('\n')}`;
}
