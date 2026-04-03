/**
 * File Write Tool — Write or create files with complete content.
 * 
 * Used for creating new files or fully overwriting existing ones.
 * For partial modifications, use FileEditTool instead.
 */

import type { ToolDefinition, ToolInput, ToolContext, ToolResult } from './types';

export const FileWriteTool: ToolDefinition = {
  name: 'file_write',
  description: 'Create a new file or overwrite an existing file with the provided content. Use this for creating new files. For partial modifications to existing files, prefer file_edit.',
  inputSchema: {
    type: 'object',
    properties: {
      path: { type: 'string', description: 'Relative path for the file to create or overwrite' },
      content: { type: 'string', description: 'Complete file content to write' },
    },
    required: ['path', 'content'],
  },
  isConcurrencySafe: false,

  async execute(input: ToolInput, context: ToolContext): Promise<ToolResult> {
    const filePath = input.path as string;
    const content = input.content as string;

    try {
      await context.writeFile(filePath, content);
      const lineCount = content.split('\n').length;
      
      return {
        success: true,
        output: `Wrote ${filePath} (${lineCount} lines, ${content.length} bytes)`,
        modifiedFiles: [filePath],
      };
    } catch (error) {
      return {
        success: false,
        output: '',
        error: `Failed to write ${filePath}: ${(error as Error).message}`,
      };
    }
  },
};
