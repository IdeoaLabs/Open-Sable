/**
 * List Files Tool — List directory contents with tree structure.
 * 
 * Shows the file structure of the sandbox project,
 * useful for understanding the project layout.
 */

import type { ToolDefinition, ToolInput, ToolContext, ToolResult } from './types';

export const ListFilesTool: ToolDefinition = {
  name: 'list_files',
  description: 'List files and directories in the sandbox. Returns a tree-like structure showing the project layout. Use this to understand the project structure before making changes.',
  inputSchema: {
    type: 'object',
    properties: {
      path: { type: 'string', description: 'Optional: directory to list (default: project root)' },
      depth: { type: 'number', description: 'Optional: maximum depth to show (default: 3)' },
    },
  },
  isConcurrencySafe: true,

  async execute(input: ToolInput, context: ToolContext): Promise<ToolResult> {
    const dirPath = (input.path as string) || '.';
    const depth = (input.depth as number) || 3;

    try {
      const cmd = `find ${dirPath} -maxdepth ${depth} -not -path '*/node_modules/*' -not -path '*/.git/*' -not -path '*/dist/*' -not -path '*/.vite/*' 2>/dev/null | head -100 | sort`;
      const result = await context.runCommand(cmd);
      
      const lines = result.stdout.trim().split('\n').filter(Boolean);
      if (lines.length === 0) {
        return { success: true, output: `Directory is empty: ${dirPath}` };
      }

      // Format as tree
      const tree = lines.map(line => {
        const depth = line.split('/').length - 1;
        const name = line.split('/').pop() || line;
        const indent = '  '.repeat(Math.max(0, depth));
        const isDir = !name.includes('.');
        return `${indent}${isDir ? '📁 ' : '📄 '}${name}`;
      }).join('\n');

      return {
        success: true,
        output: `Project structure:\n${tree}`,
      };
    } catch (error) {
      return {
        success: false,
        output: '',
        error: `Failed to list files: ${(error as Error).message}`,
      };
    }
  },
};
