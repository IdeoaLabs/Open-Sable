/**
 * Glob Tool — Find files by pattern matching.
 * 
 * Uses shell glob patterns to find files in the sandbox.
 * Useful for discovering project structure and finding files by extension.
 */

import type { ToolDefinition, ToolInput, ToolContext, ToolResult } from './types';

export const GlobTool: ToolDefinition = {
  name: 'glob',
  description: 'Find files matching a glob pattern in the sandbox. Use this to discover files by extension or name pattern. Examples: "**/*.tsx", "src/**/*.css", "**/package.json".',
  inputSchema: {
    type: 'object',
    properties: {
      pattern: { type: 'string', description: 'Glob pattern to match files (e.g., "**/*.tsx")' },
      path: { type: 'string', description: 'Optional: base directory to search from' },
    },
    required: ['pattern'],
  },
  isConcurrencySafe: true,

  async execute(input: ToolInput, context: ToolContext): Promise<ToolResult> {
    const pattern = input.pattern as string;
    const basePath = (input.path as string) || '.';

    try {
      // Use find with name pattern, excluding common non-essential dirs
      const cmd = `find ${basePath} -type f -name '${pattern.replace(/\*\*\//g, '')}' ` +
        `-not -path '*/node_modules/*' -not -path '*/.git/*' ` +
        `-not -path '*/dist/*' -not -path '*/build/*' -not -path '*/.vite/*' ` +
        `2>/dev/null | sort | head -200`;

      const result = await context.runCommand(cmd);
      const files = result.stdout.trim().split('\n').filter(Boolean);

      if (files.length === 0) {
        return {
          success: true,
          output: `No files matching pattern: ${pattern}`,
        };
      }

      return {
        success: true,
        output: `Found ${files.length} file${files.length !== 1 ? 's' : ''}:\n${files.join('\n')}`,
      };
    } catch (error) {
      return {
        success: false,
        output: '',
        error: `Glob failed: ${(error as Error).message}`,
      };
    }
  },
};
