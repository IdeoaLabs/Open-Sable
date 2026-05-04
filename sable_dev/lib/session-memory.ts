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

// ─── Decay Constants ──────────────────────────────────────

/** STM half-life in request cycles (decays fast) */
const STM_HALF_LIFE = 10;
/** LTM half-life in request cycles (decays slowly) */
const LTM_HALF_LIFE = 100;
/** Minimum importance to promote STM → LTM */
const PROMOTION_THRESHOLD = 0.7;
/** Below this effective importance, memory is forgotten */
const FORGET_THRESHOLD = 0.1;
/** Boost given when memory is accessed */
const ACCESS_BOOST = 0.15;

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
  /** Monotonic request counter (drives decay) */
  requestCycle: number;
}

export type MemoryTier = 'stm' | 'ltm';

export interface MemoryEntry {
  timestamp: number;
  type: 'preference' | 'pattern' | 'decision' | 'error' | 'file_context';
  content: string;
  /** Importance score 0.0–1.0 (set at creation, boosted on access) */
  importance: number;
  /** Current memory tier */
  tier: MemoryTier;
  /** Number of times this memory was accessed/relevant */
  accessCount: number;
  /** Last time this memory was accessed */
  lastAccessedAt: number;
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
    requestCycle: 0,
  };
}

// ─── Decay & Promotion Engine ─────────────────────────────

/**
 * Compute effective importance with exponential decay.
 * decay_factor = 0.5^(age / half_life)
 */
function computeEffectiveImportance(entry: MemoryEntry, currentCycle: number): number {
  const halfLife = entry.tier === 'stm' ? STM_HALF_LIFE : LTM_HALF_LIFE;
  const ageCycles = Math.max(0, currentCycle - (entry.lastAccessedAt || entry.timestamp));
  const decayFactor = Math.pow(0.5, ageCycles / halfLife);
  return entry.importance * decayFactor;
}

/**
 * Run decay, promotion, and forgetting on all memories.
 * Called each request cycle.
 */
export function tickMemoryDecay(state: SessionMemoryState): SessionMemoryState {
  const cycle = state.requestCycle + 1;
  const surviving: MemoryEntry[] = [];

  for (const entry of state.entries) {
    const effective = computeEffectiveImportance(entry, cycle);

    // Forget: drop memories below threshold
    if (effective < FORGET_THRESHOLD) continue;

    // Promote: STM → LTM when importance is high enough
    if (entry.tier === 'stm' && entry.importance >= PROMOTION_THRESHOLD) {
      surviving.push({ ...entry, tier: 'ltm' });
    } else {
      surviving.push(entry);
    }
  }

  return { ...state, entries: surviving, requestCycle: cycle };
}

/**
 * Boost a memory's importance when it's relevant to the current context.
 */
export function accessMemory(state: SessionMemoryState, contentSubstring: string): SessionMemoryState {
  const lower = contentSubstring.toLowerCase();
  const updated = state.entries.map(entry => {
    if (entry.content.toLowerCase().includes(lower) || lower.includes(entry.content.toLowerCase().substring(0, 30))) {
      return {
        ...entry,
        importance: Math.min(1.0, entry.importance + ACCESS_BOOST),
        accessCount: entry.accessCount + 1,
        lastAccessedAt: state.requestCycle,
      };
    }
    return entry;
  });
  return { ...state, entries: updated };
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

Output as a bullet list. Be concise,  one line per fact. Only include genuinely useful insights.

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
    const importance = estimateImportance(content, type);
    entries.push({
      timestamp: Date.now(),
      type,
      content,
      importance,
      tier: 'stm',
      accessCount: 0,
      lastAccessedAt: 0,
    });
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
          entries.push({ timestamp: Date.now(), type: 'file_context', content: entry, importance: 0.4, tier: 'stm', accessCount: 0, lastAccessedAt: 0 });
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
            entries.push({ timestamp: Date.now(), type: 'preference', content: entry, importance: 0.8, tier: 'stm', accessCount: 0, lastAccessedAt: 0 });
            seen.add(entry);
          }
        }
      }
    }
    
    // Detect error resolutions
    if (msg.role === 'assistant' && content.toLowerCase().includes('fixed') && content.toLowerCase().includes('error')) {
      const entry = `Error resolved: ${content.substring(0, 100)}`;
      if (!seen.has(entry)) {
        entries.push({ timestamp: Date.now(), type: 'error', content: entry, importance: 0.6, tier: 'stm', accessCount: 0, lastAccessedAt: 0 });
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

/**
 * Estimate initial importance based on content and type.
 */
function estimateImportance(content: string, type: MemoryEntry['type']): number {
  const base: Record<MemoryEntry['type'], number> = {
    preference: 0.8,
    decision: 0.7,
    error: 0.6,
    pattern: 0.5,
    file_context: 0.4,
  };
  let score = base[type] || 0.5;
  // Boost for strong signal words
  const lower = content.toLowerCase();
  if (lower.includes('always') || lower.includes('never') || lower.includes('important')) score += 0.1;
  if (lower.includes('critical') || lower.includes('must')) score += 0.15;
  return Math.min(1.0, score);
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
 * Uses effective importance to select the most relevant memories.
 */
export function formatMemoryForPrompt(state: SessionMemoryState): string {
  if (state.entries.length === 0) return '';
  
  // Score and sort by effective importance
  const scored = state.entries.map(entry => ({
    entry,
    effective: computeEffectiveImportance(entry, state.requestCycle),
  }));
  
  const sorted = scored
    .filter(s => s.effective >= FORGET_THRESHOLD)
    .sort((a, b) => b.effective - a.effective)
    .slice(0, 15);
  
  if (sorted.length === 0) return '';
  
  const ltmEntries = sorted.filter(s => s.entry.tier === 'ltm');
  const stmEntries = sorted.filter(s => s.entry.tier === 'stm');
  
  let output = 'SESSION MEMORY (remembered from this conversation):\n';
  
  if (ltmEntries.length > 0) {
    output += 'Long-term (proven important):\n';
    output += ltmEntries.map(s => `- [${s.entry.type}] ${s.entry.content}`).join('\n');
    output += '\n';
  }
  
  if (stmEntries.length > 0) {
    output += 'Recent:\n';
    output += stmEntries.map(s => `- [${s.entry.type}] ${s.entry.content}`).join('\n');
  }
  
  return output;
}
