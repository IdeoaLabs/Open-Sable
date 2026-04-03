/**
 * GET /api/tools — List all available sandbox tools.
 * 
 * Returns tool metadata (name, description, schema) for the
 * frontend UI and for inclusion in AI system prompts.
 */

import { NextResponse } from 'next/server';
import { ALL_TOOLS, getToolSchemasForAI, getToolDescriptionsForPrompt } from '@/lib/tools/registry';

export async function GET() {
  return NextResponse.json({
    success: true,
    tools: ALL_TOOLS.map(t => ({
      name: t.name,
      description: t.description,
      inputSchema: t.inputSchema,
      isConcurrencySafe: t.isConcurrencySafe,
    })),
    /** OpenAI function-calling format for AI providers */
    aiSchemas: getToolSchemasForAI(),
    /** Human-readable descriptions for system prompts */
    promptDescriptions: getToolDescriptionsForPrompt(),
  });
}
