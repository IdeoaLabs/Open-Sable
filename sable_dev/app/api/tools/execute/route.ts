/**
 * POST /api/tools/execute,  Execute a sandbox tool.
 * 
 * Called by the AI streaming routes when the model wants to
 * read files, edit code, run commands, or search.
 * 
 * Supports both single tool calls and batched execution.
 */

import { NextRequest, NextResponse } from 'next/server';
import { executeTool, executeToolBatch, findTool } from '@/lib/tools/registry';
import type { ToolContext, ToolInput } from '@/lib/tools/types';

declare global {
  var activeSandboxProvider: any;
  var sandboxData: { sandboxId: string; url: string } | null;
}

function getToolContext(): ToolContext | null {
  const provider = global.activeSandboxProvider;
  if (!provider) return null;

  return {
    workDir: provider.getWorkDir?.() || '',
    sandboxId: global.sandboxData?.sandboxId || '',
    readFile: (path: string) => provider.readFile(path),
    writeFile: (path: string, content: string) => provider.writeFile(path, content),
    runCommand: async (cmd: string) => {
      const result = await provider.runCommand(cmd);
      return {
        stdout: result.stdout || '',
        stderr: result.stderr || '',
        exitCode: result.exitCode ?? (result.success ? 0 : 1),
      };
    },
    listFiles: (dir?: string) => provider.listFiles(dir),
  };
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const context = getToolContext();

    if (!context) {
      return NextResponse.json({
        success: false,
        error: 'No active sandbox. Create a sandbox first.',
      }, { status: 400 });
    }

    // Batched execution
    if (body.batch && Array.isArray(body.batch)) {
      const calls = body.batch.map((call: { name: string; input: ToolInput }) => ({
        name: call.name,
        input: call.input || {},
      }));
      
      const results = await executeToolBatch(calls, context);
      return NextResponse.json({ success: true, results });
    }

    // Single tool execution
    const { tool, input } = body;
    if (!tool) {
      return NextResponse.json({ success: false, error: 'tool name is required' }, { status: 400 });
    }

    if (!findTool(tool)) {
      return NextResponse.json({ success: false, error: `Unknown tool: ${tool}` }, { status: 400 });
    }

    const result = await executeTool(tool, input || {}, context);
    return NextResponse.json(result);
  } catch (error) {
    console.error('[tools/execute] Error:', error);
    return NextResponse.json({
      success: false,
      error: (error as Error).message,
    }, { status: 500 });
  }
}
