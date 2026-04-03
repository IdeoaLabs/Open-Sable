/**
 * Tool Search Tool
 * 
 * A tool that the AI model can call to discover available tools by name or keyword.
 * Uses exact match → keyword scoring → fallback to show all tools.
 * This helps models with limited context find the right tool for the job.
 */

import type { ToolDefinition, ToolInput, ToolResult, ToolContext } from './types';
import { ALL_TOOLS } from './registry';

// ─── Scoring Weights ──────────────────────────────────────

const SCORE_EXACT_NAME = 100;
const SCORE_NAME_CONTAINS = 12;
const SCORE_DESCRIPTION_CONTAINS = 10;
const SCORE_PARAM_NAME_MATCH = 4;
const SCORE_PARAM_DESC_MATCH = 3;
const SCORE_ALIAS_MATCH = 2;

// ─── Memoized Tool Descriptions ───────────────────────────

let cachedDescriptions: string | null = null;

function getAllToolDescriptions(): string {
  if (cachedDescriptions) return cachedDescriptions;
  
  cachedDescriptions = ALL_TOOLS.map(t => {
    const params = t.inputSchema.properties
      ? Object.keys(t.inputSchema.properties as Record<string, unknown>).join(', ')
      : 'none';
    return `- ${t.name}: ${t.description} (params: ${params})`;
  }).join('\n');
  
  return cachedDescriptions;
}

// ─── Search Logic ─────────────────────────────────────────

interface ScoredTool {
  tool: ToolDefinition;
  score: number;
  matchReason: string;
}

function scoreTool(tool: ToolDefinition, query: string): ScoredTool {
  const q = query.toLowerCase();
  const name = tool.name.toLowerCase();
  const desc = tool.description.toLowerCase();
  let score = 0;
  const reasons: string[] = [];

  // Exact name match
  if (name === q) {
    score += SCORE_EXACT_NAME;
    reasons.push('exact name match');
  }

  // Name contains query
  if (name.includes(q)) {
    score += SCORE_NAME_CONTAINS;
    reasons.push('name match');
  }

  // Query contains tool name
  if (q.includes(name)) {
    score += SCORE_NAME_CONTAINS;
    reasons.push('query includes name');
  }

  // Description match
  const queryWords = q.split(/\s+/).filter(w => w.length > 2);
  for (const word of queryWords) {
    if (desc.includes(word)) {
      score += SCORE_DESCRIPTION_CONTAINS;
      reasons.push(`description match: "${word}"`);
    }
  }

  // Parameter name match
  if (tool.inputSchema.properties) {
    const paramNames = Object.keys(tool.inputSchema.properties as Record<string, unknown>);
    for (const paramName of paramNames) {
      if (paramName.toLowerCase().includes(q) || q.includes(paramName.toLowerCase())) {
        score += SCORE_PARAM_NAME_MATCH;
        reasons.push(`param name: ${paramName}`);
      }
    }

    // Parameter description match
    const paramEntries = Object.entries(tool.inputSchema.properties as Record<string, { description?: string }>);
    for (const [, param] of paramEntries) {
      if (param.description) {
        for (const word of queryWords) {
          if (param.description.toLowerCase().includes(word)) {
            score += SCORE_PARAM_DESC_MATCH;
            reasons.push(`param desc match: "${word}"`);
            break; // Only count once per param
          }
        }
      }
    }
  }

  // Common alias matching
  const aliases: Record<string, string[]> = {
    file_read: ['read', 'cat', 'view', 'open', 'show'],
    file_edit: ['edit', 'modify', 'change', 'update', 'replace', 'patch'],
    file_write: ['write', 'create', 'save', 'new file'],
    bash: ['run', 'exec', 'execute', 'shell', 'command', 'terminal'],
    grep: ['search', 'find', 'regex', 'look for', 'pattern'],
    glob: ['files', 'list', 'directory', 'find files'],
    list_files: ['ls', 'dir', 'tree', 'browse'],
    web_fetch: ['fetch', 'url', 'http', 'download', 'web'],
    agent_spawn: ['agent', 'delegate', 'subtask', 'parallel'],
  };

  const toolAliases = aliases[tool.name] || [];
  for (const alias of toolAliases) {
    if (q.includes(alias) || alias.includes(q)) {
      score += SCORE_ALIAS_MATCH;
      reasons.push(`alias: ${alias}`);
    }
  }

  return {
    tool,
    score,
    matchReason: reasons.join(', ') || 'no match',
  };
}

/**
 * Search tools by query string. Returns scored results sorted by relevance.
 */
export function searchTools(query: string): ScoredTool[] {
  if (!query.trim()) {
    return ALL_TOOLS.map(t => ({ tool: t, score: 1, matchReason: 'listed all' }));
  }

  const scored = ALL_TOOLS
    .map(t => scoreTool(t, query))
    .filter(s => s.score > 0)
    .sort((a, b) => b.score - a.score);

  // If no matches, return all tools as fallback
  if (scored.length === 0) {
    return ALL_TOOLS.map(t => ({ tool: t, score: 0, matchReason: 'fallback: no match found' }));
  }

  return scored;
}

// ─── Tool Definition ──────────────────────────────────────

export const ToolSearchTool: ToolDefinition = {
  name: 'tool_search',
  description: 'Search for available tools by keyword or description. Use this when you need to find the right tool for a task. Returns matching tools with their parameters.',
  inputSchema: {
    type: 'object',
    properties: {
      query: {
        type: 'string',
        description: 'Search query — a keyword, tool name, or description of what you want to do (e.g., "read file", "run command", "search code")',
      },
    },
    required: ['query'],
  },
  isConcurrencySafe: true,
  execute: async (input: ToolInput, _context: ToolContext): Promise<ToolResult> => {
    const query = (input.query as string) || '';

    if (!query.trim()) {
      return {
        success: true,
        output: `Available tools:\n${getAllToolDescriptions()}`,
      };
    }

    const results = searchTools(query);
    const topResults = results.slice(0, 5);

    const formatted = topResults.map(r => {
      const params = r.tool.inputSchema.properties
        ? Object.entries(r.tool.inputSchema.properties as Record<string, { type: string; description?: string }>)
            .map(([name, schema]) => `    ${name} (${schema.type}): ${schema.description || ''}`)
            .join('\n')
        : '    (no parameters)';
      const required = (r.tool.inputSchema.required as string[])?.join(', ') || 'none';
      return `${r.tool.name} (score: ${r.score})\n  ${r.tool.description}\n  Parameters:\n${params}\n  Required: ${required}`;
    }).join('\n\n');

    return {
      success: true,
      output: `Found ${results.length} matching tool(s) for "${query}":\n\n${formatted}`,
    };
  },
};
