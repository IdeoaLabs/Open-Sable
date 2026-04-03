/**
 * Grep Tool — Search for text patterns in sandbox files.
 * 
 * Uses ripgrep-style search with regex support.
 * Returns matching lines with filename, line number, and context.
 */

import type { ToolDefinition, ToolInput, ToolContext, ToolResult } from './types';

export const GrepTool: ToolDefinition = {
  name: 'grep',
  description: 'Search for a text pattern across files in the sandbox. Returns matching lines with file paths and line numbers. Supports regex patterns. Use this to find where things are defined or used.',
  inputSchema: {
    type: 'object',
    properties: {
      pattern: { type: 'string', description: 'Search pattern (regex supported)' },
      path: { type: 'string', description: 'Optional: limit search to this directory or file path' },
      include: { type: 'string', description: 'Optional: glob pattern for file types (e.g., "*.tsx", "*.css")' },
    },
    required: ['pattern'],
  },
  isConcurrencySafe: true,

  async execute(input: ToolInput, context: ToolContext): Promise<ToolResult> {
    const pattern = input.pattern as string;
    const searchPath = (input.path as string) || '.';
    const include = input.include as string | undefined;

    try {
      // Build grep command, preferring ripgrep if available
      let cmd: string;
      const escapedPattern = pattern.replace(/'/g, "'\\''");
      
      if (include) {
        cmd = `grep -rn --include='${include}' '${escapedPattern}' ${searchPath} 2>/dev/null || true`;
      } else {
        cmd = `grep -rn --exclude-dir=node_modules --exclude-dir=.git --exclude-dir=dist --exclude-dir=build '${escapedPattern}' ${searchPath} 2>/dev/null || true`;
      }

      const result = await context.runCommand(cmd);
      const output = result.stdout.trim();

      if (!output) {
        return {
          success: true,
          output: `No matches found for pattern: ${pattern}`,
        };
      }

      // Limit results
      const lines = output.split('\n');
      const maxResults = 50;
      const truncated = lines.length > maxResults;
      const displayLines = lines.slice(0, maxResults);

      let formattedOutput = `Found ${lines.length} match${lines.length !== 1 ? 'es' : ''} for "${pattern}":\n\n`;
      formattedOutput += displayLines.join('\n');
      if (truncated) {
        formattedOutput += `\n\n... and ${lines.length - maxResults} more matches (showing first ${maxResults})`;
      }

      return {
        success: true,
        output: formattedOutput,
      };
    } catch (error) {
      return {
        success: false,
        output: '',
        error: `Search failed: ${(error as Error).message}`,
      };
    }
  },
};
