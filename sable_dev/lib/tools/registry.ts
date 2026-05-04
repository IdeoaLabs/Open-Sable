/**
 * Tool Registry,  Central registry for all sandbox tools.
 * 
 * Imports all tool definitions and provides lookup/dispatch.
 * Handles tool execution with permission checking and result tracking.
 */

import type { ToolDefinition, ToolContext, ToolInput, ToolResult } from './types';
import { checkPermission } from './types';
import { FileReadTool } from './file-read';
import { FileEditTool } from './file-edit';
import { FileWriteTool } from './file-write';
import { BashTool } from './bash';
import { GrepTool } from './grep';
import { GlobTool } from './glob';
import { ListFilesTool } from './list-files';
import { WebFetchTool } from './web-fetch';
import { WebSearchTool } from './web-search';
import { AgentSpawnTool } from './agent-spawn';
import { ToolSearchTool } from './tool-search';
import { ThinkTool } from './think';

/**
 * All registered tools
 */
export const ALL_TOOLS: ToolDefinition[] = [
  ThinkTool,
  FileReadTool,
  FileEditTool,
  FileWriteTool,
  BashTool,
  GrepTool,
  GlobTool,
  ListFilesTool,
  WebFetchTool,
  WebSearchTool,
  AgentSpawnTool,
  ToolSearchTool,
];

/**
 * Look up a tool by name
 */
export function findTool(name: string): ToolDefinition | undefined {
  return ALL_TOOLS.find(t => t.name === name);
}

/**
 * Get tool descriptions formatted for AI system prompt
 */
export function getToolDescriptionsForPrompt(): string {
  return ALL_TOOLS.map(tool => {
    const params = tool.inputSchema.properties 
      ? Object.entries(tool.inputSchema.properties as Record<string, { type: string; description?: string }>)
          .map(([name, schema]) => `  - ${name} (${schema.type}): ${schema.description || ''}`)
          .join('\n')
      : '  (no parameters)';
    const required = (tool.inputSchema.required as string[])?.join(', ') || 'none';
    return `### ${tool.name}\n${tool.description}\nParameters:\n${params}\nRequired: ${required}`;
  }).join('\n\n');
}

/**
 * Get tool schemas formatted for AI function calling
 */
export function getToolSchemasForAI(): Array<{
  type: 'function';
  function: { name: string; description: string; parameters: Record<string, unknown> };
}> {
  return ALL_TOOLS.map(tool => ({
    type: 'function' as const,
    function: {
      name: tool.name,
      description: tool.description,
      parameters: tool.inputSchema,
    },
  }));
}

/**
 * Execute a tool by name with input and context
 */
export async function executeTool(
  toolName: string,
  input: ToolInput,
  context: ToolContext
): Promise<ToolResult> {
  const tool = findTool(toolName);
  if (!tool) {
    return { success: false, output: '', error: `Unknown tool: ${toolName}` };
  }

  // Check permissions
  const permission = checkPermission(toolName, input);
  if (permission === 'deny') {
    return { success: false, output: '', error: `Permission denied for ${toolName}` };
  }

  // Execute
  try {
    const result = await tool.execute(input, context);
    return result;
  } catch (error) {
    return {
      success: false,
      output: '',
      error: `Tool ${toolName} failed: ${(error as Error).message}`,
    };
  }
}

/**
 * Execute multiple tools, respecting concurrency safety.
 * Read-only tools run in parallel, mutating tools run serially.
 */
export async function executeToolBatch(
  calls: Array<{ name: string; input: ToolInput }>,
  context: ToolContext
): Promise<Array<{ name: string; result: ToolResult }>> {
  const results: Array<{ name: string; result: ToolResult }> = [];

  // Partition into concurrent-safe and serial groups
  const concurrent: Array<{ name: string; input: ToolInput }> = [];
  const serial: Array<{ name: string; input: ToolInput }> = [];

  for (const call of calls) {
    const tool = findTool(call.name);
    if (tool?.isConcurrencySafe) {
      concurrent.push(call);
    } else {
      serial.push(call);
    }
  }

  // Execute concurrent tools in parallel
  if (concurrent.length > 0) {
    const concurrentResults = await Promise.all(
      concurrent.map(async (call) => ({
        name: call.name,
        result: await executeTool(call.name, call.input, context),
      }))
    );
    results.push(...concurrentResults);
  }

  // Execute serial tools one by one
  for (const call of serial) {
    const result = await executeTool(call.name, call.input, context);
    results.push({ name: call.name, result });
  }

  return results;
}
