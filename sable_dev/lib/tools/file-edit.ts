/**
 * File Edit Tool,  Edit file contents using search-and-replace.
 * 
 * Uses an exact string match approach (old_string → new_string).
 * Generates unified diffs for display and tracking.
 * Supports creating new files when old_string is empty.
 */

import type { ToolDefinition, ToolInput, ToolContext, ToolResult } from './types';

export const FileEditTool: ToolDefinition = {
  name: 'file_edit',
  description: 'Edit a file by replacing exact text. Provide the old_string to find and new_string to replace it with. If old_string is empty, the file will be created with new_string as content. For precise edits, include a few lines of context around the target.',
  inputSchema: {
    type: 'object',
    properties: {
      path: { type: 'string', description: 'Relative path to the file to edit' },
      old_string: { type: 'string', description: 'Exact text to find and replace. Empty string means create new file.' },
      new_string: { type: 'string', description: 'Replacement text' },
    },
    required: ['path', 'old_string', 'new_string'],
  },
  isConcurrencySafe: false,

  async execute(input: ToolInput, context: ToolContext): Promise<ToolResult> {
    const filePath = input.path as string;
    const oldString = input.old_string as string;
    const newString = input.new_string as string;

    try {
      // Create new file
      if (!oldString) {
        await context.writeFile(filePath, newString);
        return {
          success: true,
          output: `Created ${filePath} (${newString.split('\n').length} lines)`,
          modifiedFiles: [filePath],
        };
      }

      // Read existing file
      let content: string;
      try {
        content = await context.readFile(filePath);
      } catch {
        return { success: false, output: '', error: `File not found: ${filePath}` };
      }

      // Find the old string
      const index = content.indexOf(oldString);
      if (index === -1) {
        // Try normalized matching (handle different line endings)
        const normalizedContent = content.replace(/\r\n/g, '\n');
        const normalizedOld = oldString.replace(/\r\n/g, '\n');
        const normIndex = normalizedContent.indexOf(normalizedOld);
        
        if (normIndex === -1) {
          return {
            success: false,
            output: '',
            error: `Could not find the specified text in ${filePath}. Make sure old_string matches exactly (including whitespace and indentation).`,
          };
        }
        
        // Apply with normalized content
        const newContent = normalizedContent.slice(0, normIndex) + newString + normalizedContent.slice(normIndex + normalizedOld.length);
        await context.writeFile(filePath, newContent);
      } else {
        // Check for multiple matches
        const secondIndex = content.indexOf(oldString, index + 1);
        if (secondIndex !== -1) {
          return {
            success: false,
            output: '',
            error: `old_string matches multiple locations in ${filePath}. Include more context to make the match unique.`,
          };
        }

        const newContent = content.slice(0, index) + newString + content.slice(index + oldString.length);
        await context.writeFile(filePath, newContent);
      }

      // Generate a simple diff summary
      const oldLines = oldString.split('\n').length;
      const newLines = newString.split('\n').length;
      const diffSummary = oldLines === newLines
        ? `Modified ${oldLines} line(s) in ${filePath}`
        : `Replaced ${oldLines} line(s) with ${newLines} line(s) in ${filePath}`;

      return {
        success: true,
        output: diffSummary,
        modifiedFiles: [filePath],
      };
    } catch (error) {
      return {
        success: false,
        output: '',
        error: `Failed to edit ${filePath}: ${(error as Error).message}`,
      };
    }
  },
};
