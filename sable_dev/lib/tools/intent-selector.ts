/**
 * Intent-Based Tool Selection
 * 
 * Analyzes the user prompt to determine which tools are relevant,
 * reducing the number of tools sent to the model. Critical for
 * small local models where prompt size matters.
 */

import type { ToolDefinition } from './types';
import { ALL_TOOLS } from './registry';

export type IntentCategory = 
  | 'read'       // User wants to understand/explore code
  | 'edit'       // User wants to modify existing code
  | 'create'     // User wants to create new files
  | 'debug'      // User wants to fix errors/bugs
  | 'search'     // User wants to find something in the codebase
  | 'run'        // User wants to execute something
  | 'general';   // Unclear or multi-intent

interface IntentSignal {
  category: IntentCategory;
  keywords: string[];
  weight: number;
}

const INTENT_SIGNALS: IntentSignal[] = [
  { category: 'read', keywords: ['read', 'show', 'display', 'what is', 'what does', 'how does', 'explain', 'look at', 'check', 'view', 'contents'], weight: 1 },
  { category: 'edit', keywords: ['change', 'update', 'modify', 'edit', 'replace', 'refactor', 'rename', 'move', 'fix the', 'adjust', 'tweak'], weight: 1 },
  { category: 'create', keywords: ['create', 'add', 'new file', 'generate', 'make', 'build', 'write', 'scaffold', 'setup'], weight: 1 },
  { category: 'debug', keywords: ['error', 'bug', 'fix', 'broken', 'not working', 'crash', 'fails', 'wrong', 'issue', 'debug', 'stack trace', 'exception'], weight: 1.5 },
  { category: 'search', keywords: ['find', 'search', 'where', 'locate', 'grep', 'which file', 'look for', 'contains'], weight: 1 },
  { category: 'run', keywords: ['run', 'execute', 'test', 'build', 'compile', 'start', 'install', 'npm', 'command'], weight: 1 },
];

/** Map from intent category to relevant tool names */
const INTENT_TOOL_MAP: Record<IntentCategory, string[]> = {
  read:    ['file_read', 'list_files', 'glob'],
  edit:    ['file_read', 'file_edit', 'file_write', 'grep', 'list_files'],
  create:  ['file_write', 'list_files', 'glob', 'bash'],
  debug:   ['file_read', 'file_edit', 'grep', 'bash', 'list_files', 'agent_spawn'],
  search:  ['grep', 'glob', 'list_files', 'file_read', 'web_fetch', 'tool_search'],
  run:     ['bash', 'file_read', 'list_files'],
  general: [], // All tools
};

/**
 * Classify the user's intent from their prompt.
 */
export function classifyIntent(prompt: string): { primary: IntentCategory; scores: Record<IntentCategory, number> } {
  const lower = prompt.toLowerCase();
  const scores: Record<string, number> = {};

  for (const signal of INTENT_SIGNALS) {
    let score = 0;
    for (const kw of signal.keywords) {
      if (lower.includes(kw)) {
        score += signal.weight;
      }
    }
    scores[signal.category] = (scores[signal.category] || 0) + score;
  }

  // Find highest scoring category
  let primary: IntentCategory = 'general';
  let maxScore = 0;
  for (const [cat, score] of Object.entries(scores)) {
    if (score > maxScore) {
      maxScore = score;
      primary = cat as IntentCategory;
    }
  }

  // If no strong signal, default to general (all tools)
  if (maxScore < 1) {
    primary = 'general';
  }

  return { primary, scores: scores as Record<IntentCategory, number> };
}

/**
 * Select tools relevant to the user's intent.
 * Returns a filtered subset of ALL_TOOLS.
 */
export function selectToolsForIntent(prompt: string): ToolDefinition[] {
  const { primary } = classifyIntent(prompt);

  // 'general' means include all tools
  if (primary === 'general') {
    return ALL_TOOLS;
  }

  const relevantNames = INTENT_TOOL_MAP[primary];
  if (!relevantNames || relevantNames.length === 0) {
    return ALL_TOOLS;
  }

  return ALL_TOOLS.filter(t => relevantNames.includes(t.name));
}

/**
 * Get a compact summary of which tools were selected and why.
 */
export function getSelectionSummary(prompt: string): string {
  const { primary, scores } = classifyIntent(prompt);
  const tools = selectToolsForIntent(prompt);
  return `Intent: ${primary} | Tools: ${tools.map(t => t.name).join(', ')} | Scores: ${JSON.stringify(scores)}`;
}
