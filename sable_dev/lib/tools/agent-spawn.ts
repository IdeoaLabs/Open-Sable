/**
 * Agent Spawn Tool,  Delegates a sub-task to a secondary AI agent.
 * 
 * Uses the existing agent-orchestrator to pick a model and role,
 * then runs a focused completion against that model.
 * The primary model can use this to delegate review, planning, or error-fixing.
 */

import type { ToolDefinition, ToolContext, ToolInput, ToolResult } from './types';
import {
  type AgentRole,
  getAgentDefinition,
  createAgent,
  getAgentSystemPrompt,
} from '@/lib/agents/agent-orchestrator';

const VALID_ROLES: AgentRole[] = [
  'code-reviewer',
  'design-advisor',
  'package-resolver',
  'error-fixer',
  'test-writer',
  'refactorer',
  'architect',
];

async function execute(input: ToolInput, context: ToolContext): Promise<ToolResult> {
  const role = input.role as string;
  const task = input.task as string;
  const codeContext = (input.context as string) || '';

  if (!role || !task) {
    return { success: false, output: '', error: 'Both "role" and "task" parameters are required.' };
  }

  if (!VALID_ROLES.includes(role as AgentRole)) {
    return {
      success: false,
      output: '',
      error: `Invalid role "${role}". Valid roles: ${VALID_ROLES.join(', ')}`,
    };
  }

  const agentRole = role as AgentRole;
  const def = getAgentDefinition(agentRole);
  if (!def) {
    return { success: false, output: '', error: `No definition found for role: ${role}` };
  }

  const agent = createAgent(agentRole, 'delegated', task);
  const systemPrompt = getAgentSystemPrompt(agentRole);

  // Build the prompt for the sub-agent
  const fullPrompt = codeContext
    ? `${task}\n\nCODE CONTEXT:\n${codeContext.substring(0, 8000)}`
    : task;

  // Execute via the sandbox's runCommand to call ollama directly
  // This avoids importing the full AI SDK in the tool layer.
  // We use a simple completion via Ollama's API.
  try {
    const ollamaHost = process.env.OLLAMA_HOST || 'http://localhost:11434';
    const model = process.env.AGENT_MODEL || 'qwen2.5:14b';

    const response = await fetch(`${ollamaHost}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model,
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: fullPrompt },
        ],
        stream: false,
        options: { temperature: 0.3 },
      }),
    });

    if (!response.ok) {
      const errText = await response.text();
      return {
        success: false,
        output: '',
        error: `Agent ${def.name} failed (HTTP ${response.status}): ${errText.substring(0, 200)}`,
      };
    }

    const data = await response.json();
    const result = data.message?.content || '';

    if (!result) {
      return { success: false, output: '', error: `Agent ${def.name} returned empty response.` };
    }

    agent.status = 'completed';
    agent.completedAt = Date.now();
    agent.result = result;

    // Truncate if very long
    const output = result.length > 5000
      ? result.substring(0, 5000) + '\n...(truncated)'
      : result;

    return {
      success: true,
      output: `[${def.name}] ${output}`,
    };
  } catch (err: any) {
    agent.status = 'error';
    agent.error = err.message;
    return {
      success: false,
      output: '',
      error: `Agent ${def.name} error: ${err.message}`,
    };
  }
}

export const AgentSpawnTool: ToolDefinition = {
  name: 'agent_spawn',
  description: `Spawn a secondary AI agent to handle a focused sub-task. Roles: code-reviewer (review for bugs), design-advisor (UI/UX feedback), package-resolver (identify npm packages), error-fixer (diagnose errors), test-writer (generate tests), refactorer (optimize code), architect (plan file structure).`,
  inputSchema: {
    type: 'object',
    properties: {
      role: {
        type: 'string',
        description: 'The agent role: code-reviewer, design-advisor, package-resolver, error-fixer, test-writer, refactorer, or architect',
      },
      task: {
        type: 'string',
        description: 'Description of the task for the agent to perform',
      },
      context: {
        type: 'string',
        description: 'Optional code or context to provide to the agent',
      },
    },
    required: ['role', 'task'],
  },
  isConcurrencySafe: false,
  execute,
};
