/**
 * AI SDK Tool Bridge
 * 
 * Converts our ToolDefinition format to AI SDK v5 tool() format.
 * Enables native tool calling in streamText/generateText with maxSteps.
 * Tools execute against the active sandbox's filesystem.
 */

import { tool, jsonSchema } from 'ai';
import { ALL_TOOLS, executeTool } from './registry';
import type { ToolContext, ToolInput } from './types';

/**
 * Build AI SDK tool set from our registered tools.
 * Each tool wraps our execute function with the provided context.
 */
export function buildAISDKTools(context: ToolContext): Record<string, any> {
  const tools: Record<string, any> = {};

  for (const toolDef of ALL_TOOLS) {
    tools[toolDef.name] = tool({
      description: toolDef.description,
      parameters: jsonSchema(toolDef.inputSchema as any) as any,
      execute: async (input: any) => {
        const result = await executeTool(toolDef.name, input as ToolInput, context);
        if (!result.success) {
          return `ERROR: ${result.error || 'Unknown error'}`;
        }
        return result.output;
      },
    } as any);
  }

  return tools;
}

/**
 * Build a ToolContext from the active sandbox provider.
 * Used when constructing tools for a streaming call.
 */
export function buildToolContext(sandboxProvider: any): ToolContext | null {
  if (!sandboxProvider) return null;

  return {
    workDir: sandboxProvider._workDir || sandboxProvider.workDir || '/tmp',
    sandboxId: sandboxProvider._sandboxId || sandboxProvider.sandboxId || 'unknown',
    readFile: async (path: string) => {
      return sandboxProvider.readFile(path);
    },
    writeFile: async (path: string, content: string) => {
      return sandboxProvider.writeFile(path, content);
    },
    runCommand: async (cmd: string) => {
      return sandboxProvider.runCommand(cmd);
    },
    listFiles: async (dir?: string) => {
      return sandboxProvider.listFiles(dir);
    },
  };
}

/**
 * Get tool descriptions as a compact string for the system prompt.
 * This helps the model understand what tools are available.
 */
export function getToolSummaryForPrompt(): string {
  return `You have access to the following tools to explore and modify the project:

- **think**: Record your reasoning before taking action. Use this to plan, analyze, and decide your next step.
- **file_read**: Read file contents with line numbers. Use to inspect existing code.
- **file_edit**: Edit a file by replacing a specific string with new content. Use for targeted changes.
- **file_write**: Create or overwrite a file with complete content.
- **bash**: Run shell commands in the sandbox (npm install, build commands, etc).
- **grep**: Search for text patterns across files using regex.
- **glob**: Find files matching a glob pattern.
- **list_files**: List directory contents in a tree structure.
- **web_fetch**: Fetch content from a URL (documentation, API responses, reference pages).
- **agent_spawn**: Delegate a focused sub-task to a secondary AI agent (code review, architecture planning, error fixing, testing, etc).
- **tool_search**: Search for available tools by keyword. Use when you need to find the right tool for a task.

REASONING STRATEGY (Thought → Action → Observation):
You MUST use the think tool before making any changes. Follow this loop:
1. THINK: Call the think tool to reason about the problem,  what you know, what's missing, and your plan.
2. ACT: Execute your planned action (read files, edit code, run commands).
3. OBSERVE: Examine the result. If more work is needed, go back to step 1.

Always think before you act. Never edit files without first reading and reasoning about them.

TOOL USAGE STRATEGY:
1. When editing an existing project, FIRST use think to plan your approach
2. Then use file_read/grep/list_files to understand the codebase
3. Think again to plan your changes based on what you found
4. Use file_edit for targeted changes or file_write for new files
5. Use bash for package installation or build commands
6. Use web_fetch to look up documentation or API examples
7. Use agent_spawn for specialized tasks like code review or test generation
8. Use tool_search to discover tools when you're not sure which tool to use
9. After making changes, you can use file_read to verify your edits

You can call multiple read-only tools in parallel. File modifications should be done one at a time.
After all tool calls are complete, output any remaining <file> tags for new files or full rewrites.`;
}
