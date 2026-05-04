/**
 * File Read Tool,  Read file contents with line numbers.
 * 
 * Supports reading full files or specific line ranges.
 * Handles binary detection (returns error for binary files).
 * Returns content with line numbers for AI reference.
 */

import type { ToolDefinition, ToolInput, ToolContext, ToolResult } from './types';

export const FileReadTool: ToolDefinition = {
  name: 'file_read',
  description: 'Read the contents of a file in the sandbox. Returns the file content with line numbers. Use this to understand existing code before making changes.',
  inputSchema: {
    type: 'object',
    properties: {
      path: { type: 'string', description: 'Relative path to the file to read' },
      startLine: { type: 'number', description: 'Optional: start reading from this line (1-indexed)' },
      endLine: { type: 'number', description: 'Optional: stop reading at this line (1-indexed, inclusive)' },
    },
    required: ['path'],
  },
  isConcurrencySafe: true,

  async execute(input: ToolInput, context: ToolContext): Promise<ToolResult> {
    const filePath = input.path as string;
    const startLine = input.startLine as number | undefined;
    const endLine = input.endLine as number | undefined;

    try {
      const content = await context.readFile(filePath);
      
      // Check for binary content
      if (content.includes('\0')) {
        return { success: false, output: '', error: `File appears to be binary: ${filePath}` };
      }

      const lines = content.split('\n');
      const start = Math.max(1, startLine || 1);
      const end = Math.min(lines.length, endLine || lines.length);
      
      const numbered = lines
        .slice(start - 1, end)
        .map((line, i) => `${String(start + i).padStart(4)} | ${line}`)
        .join('\n');

      return {
        success: true,
        output: `File: ${filePath} (${lines.length} lines)\n${'─'.repeat(60)}\n${numbered}`,
      };
    } catch (error) {
      return {
        success: false,
        output: '',
        error: `Failed to read ${filePath}: ${(error as Error).message}`,
      };
    }
  },
};
