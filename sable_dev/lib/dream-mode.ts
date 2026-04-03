/**
 * Dream Mode — Autonomous Memory Consolidation
 * 
 * Runs a "dreaming" background process that reviews past conversation sessions
 * and consolidates useful insights into durable memory files.
 * 
 * Triple gate system: time gate → session gate → distributed lock
 * The dream agent gets read-only access (can only inspect, not modify).
 * Self-throttling with 10min scan intervals to prevent resource exhaustion.
 * 
 * Storage: .sable-dev/dream/
 */

import fs from 'fs';
import path from 'path';
import { estimateTotalTokens } from './token-budget';

// ─── Configuration ────────────────────────────────────────

interface DreamConfig {
  /** Hours between consolidation runs */
  minHoursBetweenRuns: number;
  /** Minimum sessions touched before triggering */
  minSessionsRequired: number;
  /** Minimum ms between gate checks (prevents spinning) */
  scanThrottleMs: number;
  /** Max turns the dream agent is allowed */
  maxDreamTurns: number;
  /** Max memory entries to produce per dream cycle */
  maxEntriesPerCycle: number;
  /** Ollama model used for dream consolidation */
  dreamModel: string;
}

const DEFAULT_CONFIG: DreamConfig = {
  minHoursBetweenRuns: 24,
  minSessionsRequired: 3,
  scanThrottleMs: 10 * 60 * 1000, // 10 minutes
  maxDreamTurns: 30,
  maxEntriesPerCycle: 10,
  dreamModel: process.env.DREAM_MODEL || 'qwen2.5:14b',
};

// ─── Types ────────────────────────────────────────────────

export type DreamPhase = 'idle' | 'checking' | 'starting' | 'reviewing' | 'updating' | 'completed' | 'failed';

export interface DreamState {
  /** Current phase */
  phase: DreamPhase;
  /** Timestamp of last completed consolidation */
  lastConsolidatedAt: number | null;
  /** Last gate-check timestamp (for scan throttle) */
  lastGateCheckAt: number | null;
  /** Sessions reviewed in current/last dream cycle */
  sessionsReviewed: number;
  /** Files touched during current dream cycle */
  filesTouched: string[];
  /** Whether a dream is currently running */
  isRunning: boolean;
  /** Abort controller for cancelling a running dream */
  abortSignal: boolean;
  /** Total dream cycles completed */
  totalCycles: number;
  /** Consolidated memory entries */
  memories: DreamMemory[];
  /** Lock acquisition timestamp */
  lockAcquiredAt: number | null;
}

export interface DreamMemory {
  id: string;
  content: string;
  source: string; // session ID
  category: 'pattern' | 'preference' | 'architecture' | 'error_lesson' | 'workflow';
  createdAt: number;
  relevanceScore: number;
}

interface SessionSummary {
  id: string;
  startedAt: number;
  lastUpdated: number;
  messageCount: number;
  preview: string;
}

// ─── Persistence Paths ────────────────────────────────────

const DREAM_DIR = path.join(process.cwd(), '.sable-dev', 'dream');
const LOCK_FILE = path.join(DREAM_DIR, '.dream.lock');
const MEMORIES_FILE = path.join(DREAM_DIR, 'consolidated-memories.json');
const STATE_FILE = path.join(DREAM_DIR, 'dream-state.json');

function ensureDreamDir() {
  if (!fs.existsSync(DREAM_DIR)) {
    fs.mkdirSync(DREAM_DIR, { recursive: true });
  }
}

// ─── State Management ─────────────────────────────────────

export function createDreamState(): DreamState {
  // Try to restore from disk
  const restored = loadDreamState();
  if (restored) return restored;
  
  return {
    phase: 'idle',
    lastConsolidatedAt: null,
    lastGateCheckAt: null,
    sessionsReviewed: 0,
    filesTouched: [],
    isRunning: false,
    abortSignal: false,
    totalCycles: 0,
    memories: [],
    lockAcquiredAt: null,
  };
}

function loadDreamState(): DreamState | null {
  try {
    if (fs.existsSync(STATE_FILE)) {
      const data = JSON.parse(fs.readFileSync(STATE_FILE, 'utf-8'));
      return {
        ...data,
        isRunning: false, // Reset running state on load
        abortSignal: false,
        lockAcquiredAt: null,
      };
    }
  } catch { /* corrupt state, start fresh */ }
  return null;
}

function saveDreamState(state: DreamState): void {
  try {
    ensureDreamDir();
    const serializable = {
      ...state,
      abortSignal: false,  // Don't persist runtime flags
    };
    fs.writeFileSync(STATE_FILE, JSON.stringify(serializable, null, 2));
  } catch (e) {
    console.warn('[dream-mode] Failed to save state:', e);
  }
}

// ─── Lock System ──────────────────────────────────────────

function tryAcquireLock(): boolean {
  try {
    ensureDreamDir();
    
    // Check if lock exists and is stale (> 1 hour = assume crashed)
    if (fs.existsSync(LOCK_FILE)) {
      const stat = fs.statSync(LOCK_FILE);
      const ageMs = Date.now() - stat.mtimeMs;
      if (ageMs < 60 * 60 * 1000) {
        return false; // Lock is still fresh, another dream is running
      }
      // Stale lock — remove it (mtime rollback for retry)
      fs.unlinkSync(LOCK_FILE);
    }
    
    // Write lock with PID for debugging
    fs.writeFileSync(LOCK_FILE, JSON.stringify({ pid: process.pid, acquiredAt: Date.now() }));
    return true;
  } catch {
    return false;
  }
}

function releaseLock(): void {
  try {
    if (fs.existsSync(LOCK_FILE)) {
      fs.unlinkSync(LOCK_FILE);
    }
  } catch { /* best effort */ }
}

// ─── Gate System (cheapest first) ─────────────────────────

/**
 * Check if dreaming should be triggered.
 * Gates are evaluated cheapest-first to abort early.
 */
export function shouldDream(
  state: DreamState,
  config: DreamConfig = DEFAULT_CONFIG,
): { shouldRun: boolean; reason?: string } {
  // Gate 0: Already running
  if (state.isRunning) {
    return { shouldRun: false, reason: 'dream already running' };
  }
  
  // Gate 1: Scan throttle — don't check gates more than every 10min
  if (state.lastGateCheckAt && (Date.now() - state.lastGateCheckAt) < config.scanThrottleMs) {
    return { shouldRun: false, reason: 'scan throttled' };
  }
  
  // Gate 2: Time gate — minimum hours since last consolidation
  if (state.lastConsolidatedAt) {
    const hoursSince = (Date.now() - state.lastConsolidatedAt) / (1000 * 60 * 60);
    if (hoursSince < config.minHoursBetweenRuns) {
      return { shouldRun: false, reason: `only ${hoursSince.toFixed(1)}h since last dream (min: ${config.minHoursBetweenRuns}h)` };
    }
  }
  
  // Gate 3: Session gate — need enough sessions to review
  const sessions = listSessionsSince(state.lastConsolidatedAt || 0);
  if (sessions.length < config.minSessionsRequired) {
    return { shouldRun: false, reason: `only ${sessions.length} sessions (min: ${config.minSessionsRequired})` };
  }
  
  // Gate 4: Distributed lock
  if (!tryAcquireLock()) {
    return { shouldRun: false, reason: 'lock held by another process' };
  }
  
  return { shouldRun: true };
}

/**
 * Update gate check timestamp (call on every check to throttle)
 */
export function markGateChecked(state: DreamState): DreamState {
  return { ...state, lastGateCheckAt: Date.now() };
}

// ─── Session Listing ──────────────────────────────────────

function listSessionsSince(since: number): SessionSummary[] {
  const sessionsDir = path.join(process.cwd(), '.sable-dev');
  const sessions: SessionSummary[] = [];
  
  try {
    if (!fs.existsSync(sessionsDir)) return sessions;
    
    // Look for session files (chat-history.json, session.json)
    const files = fs.readdirSync(sessionsDir).filter(f => 
      f.endsWith('.json') && (f.includes('session') || f.includes('chat'))
    );
    
    for (const file of files) {
      try {
        const filePath = path.join(sessionsDir, file);
        const stat = fs.statSync(filePath);
        if (stat.mtimeMs <= since) continue;
        
        const data = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
        const messages = data.messages || data.conversationState?.context?.messages || [];
        
        if (messages.length === 0) continue;
        
        sessions.push({
          id: file.replace('.json', ''),
          startedAt: data.savedAt || stat.birthtimeMs,
          lastUpdated: stat.mtimeMs,
          messageCount: messages.length,
          preview: messages.slice(0, 3).map((m: any) => 
            `${m.role || m.type}: ${(m.content || '').substring(0, 100)}`
          ).join('\n'),
        });
      } catch { /* skip corrupt files */ }
    }
  } catch { /* dir read failed */ }
  
  return sessions.sort((a, b) => b.lastUpdated - a.lastUpdated);
}

// ─── Dream Execution ──────────────────────────────────────

/**
 * Run a dream consolidation cycle.
 * This is meant to be called as a fire-and-forget background task.
 */
export async function runDream(
  state: DreamState,
  config: DreamConfig = DEFAULT_CONFIG,
): Promise<DreamState> {
  const newState: DreamState = {
    ...state,
    phase: 'starting',
    isRunning: true,
    lockAcquiredAt: Date.now(),
    sessionsReviewed: 0,
    filesTouched: [],
  };
  
  saveDreamState(newState);
  
  try {
    // Phase 1: Collect sessions to review
    newState.phase = 'reviewing';
    const sessions = listSessionsSince(state.lastConsolidatedAt || 0);
    
    if (sessions.length === 0) {
      newState.phase = 'completed';
      newState.isRunning = false;
      releaseLock();
      saveDreamState(newState);
      return newState;
    }
    
    // Build review prompt from session previews (capped at maxDreamTurns)
    const sessionSummaries = sessions
      .slice(0, config.maxDreamTurns)
      .map(s => `Session "${s.id}" (${s.messageCount} messages, last: ${new Date(s.lastUpdated).toISOString()}):\n${s.preview}`)
      .join('\n\n---\n\n');
    
    // Phase 2: Call the dream model to extract insights
    newState.phase = 'updating';
    newState.sessionsReviewed = Math.min(sessions.length, config.maxDreamTurns);
    saveDreamState(newState);
    
    const ollamaHost = process.env.OLLAMA_HOST || 'http://localhost:11434';
    
    const dreamPrompt = `You are a memory consolidation agent. Review the following conversation sessions and extract the most important learnings.

For each insight, categorize it as one of:
- pattern: A recurring code pattern or technique the user uses
- preference: A user preference about tools, styles, or approaches
- architecture: An architectural decision or project structure choice
- error_lesson: A lesson learned from an error or bug
- workflow: A workflow pattern or process the user follows

Output ONLY a JSON array of objects with { "content": "...", "category": "...", "relevanceScore": 0.0-1.0 }

Maximum ${config.maxEntriesPerCycle} entries. Focus on the most useful, non-obvious insights.

SESSIONS TO REVIEW:
${sessionSummaries}`;

    const response = await fetch(`${ollamaHost}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: config.dreamModel,
        messages: [
          { role: 'system', content: 'You extract and consolidate learnings from conversation sessions. Output ONLY valid JSON arrays.' },
          { role: 'user', content: dreamPrompt },
        ],
        stream: false,
        options: { temperature: 0.3 },
      }),
    });

    if (!response.ok) {
      throw new Error(`Dream model returned ${response.status}`);
    }

    const data = await response.json();
    const rawContent = data.message?.content || '[]';
    
    // Parse the dream agent's output
    let extractedMemories: DreamMemory[] = [];
    try {
      // Extract JSON array from response (may have surrounding text)
      const jsonMatch = rawContent.match(/\[[\s\S]*\]/);
      if (jsonMatch) {
        const parsed = JSON.parse(jsonMatch[0]);
        extractedMemories = parsed
          .slice(0, config.maxEntriesPerCycle)
          .map((entry: any, i: number) => ({
            id: `dream-${Date.now()}-${i}`,
            content: String(entry.content || '').substring(0, 500),
            source: sessions[0]?.id || 'unknown',
            category: ['pattern', 'preference', 'architecture', 'error_lesson', 'workflow'].includes(entry.category)
              ? entry.category
              : 'pattern',
            createdAt: Date.now(),
            relevanceScore: Math.max(0, Math.min(1, Number(entry.relevanceScore) || 0.5)),
          }));
      }
    } catch (parseErr) {
      console.warn('[dream-mode] Failed to parse dream output:', parseErr);
    }
    
    // Phase 3: Merge new memories with existing
    const existingMemories = loadConsolidatedMemories();
    const allMemories = deduplicateMemories([...existingMemories, ...extractedMemories]);
    
    // Cap total memories at 100
    const cappedMemories = allMemories
      .sort((a, b) => b.relevanceScore - a.relevanceScore)
      .slice(0, 100);
    
    // Save consolidated memories
    saveConsolidatedMemories(cappedMemories);
    newState.filesTouched.push(MEMORIES_FILE);
    
    // Also write a human-readable markdown summary
    writeDreamSummary(extractedMemories);
    newState.filesTouched.push(path.join(DREAM_DIR, 'dream-log.md'));
    
    // Complete
    newState.phase = 'completed';
    newState.isRunning = false;
    newState.lastConsolidatedAt = Date.now();
    newState.totalCycles += 1;
    newState.memories = cappedMemories;
    
  } catch (error) {
    console.error('[dream-mode] Dream cycle failed:', error);
    newState.phase = 'failed';
    newState.isRunning = false;
  } finally {
    releaseLock();
    saveDreamState(newState);
  }
  
  return newState;
}

/**
 * Stop a running dream (best effort).
 */
export function abortDream(state: DreamState): DreamState {
  releaseLock();
  return {
    ...state,
    phase: 'idle',
    isRunning: false,
    abortSignal: true,
    lockAcquiredAt: null,
  };
}

// ─── Memory Persistence ───────────────────────────────────

function loadConsolidatedMemories(): DreamMemory[] {
  try {
    if (fs.existsSync(MEMORIES_FILE)) {
      return JSON.parse(fs.readFileSync(MEMORIES_FILE, 'utf-8'));
    }
  } catch { /* corrupt, start fresh */ }
  return [];
}

function saveConsolidatedMemories(memories: DreamMemory[]): void {
  try {
    ensureDreamDir();
    fs.writeFileSync(MEMORIES_FILE, JSON.stringify(memories, null, 2));
  } catch (e) {
    console.warn('[dream-mode] Failed to save memories:', e);
  }
}

function deduplicateMemories(memories: DreamMemory[]): DreamMemory[] {
  const seen = new Set<string>();
  return memories.filter(m => {
    // Simple dedup by content similarity (first 100 chars lowercase)
    const key = m.content.toLowerCase().substring(0, 100);
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function writeDreamSummary(newMemories: DreamMemory[]): void {
  try {
    ensureDreamDir();
    const logPath = path.join(DREAM_DIR, 'dream-log.md');
    
    const entry = `\n## Dream Cycle — ${new Date().toISOString()}\n\n` +
      `Extracted ${newMemories.length} insights:\n\n` +
      newMemories.map(m => `- **[${m.category}]** ${m.content} (relevance: ${m.relevanceScore})`).join('\n') +
      '\n';
    
    fs.appendFileSync(logPath, entry);
  } catch { /* non-critical */ }
}

// ─── Prompt Integration ───────────────────────────────────

/**
 * Format dream memories for injection into system prompt.
 * Only includes high-relevance memories.
 */
export function formatDreamMemoriesForPrompt(state: DreamState, maxEntries: number = 8): string {
  if (state.memories.length === 0) return '';
  
  const topMemories = state.memories
    .filter(m => m.relevanceScore >= 0.5)
    .sort((a, b) => b.relevanceScore - a.relevanceScore)
    .slice(0, maxEntries);
  
  if (topMemories.length === 0) return '';
  
  return `## Consolidated Insights (from previous sessions)\n` +
    topMemories.map(m => `- [${m.category}] ${m.content}`).join('\n');
}

/**
 * Get a status summary of the dream system.
 */
export function getDreamStatus(state: DreamState): string {
  const lines: string[] = [
    `Dream Mode: ${state.phase}`,
    `Total cycles: ${state.totalCycles}`,
    `Memories stored: ${state.memories.length}`,
  ];
  
  if (state.lastConsolidatedAt) {
    const hoursAgo = ((Date.now() - state.lastConsolidatedAt) / (1000 * 60 * 60)).toFixed(1);
    lines.push(`Last dream: ${hoursAgo}h ago`);
  } else {
    lines.push('Last dream: never');
  }
  
  if (state.isRunning) {
    lines.push(`Currently reviewing ${state.sessionsReviewed} sessions`);
    lines.push(`Files touched: ${state.filesTouched.join(', ') || 'none'}`);
  }
  
  return lines.join('\n');
}
