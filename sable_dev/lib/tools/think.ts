/**
 * Think Tool,  ReAct Executor
 * 
 * A reasoning tool that enables Thought→Action→Observation loops.
 * The model calls this tool to record its reasoning before taking action.
 * No side effects,  purely captures the model's chain of thought.
 */

import type { ToolDefinition, ToolInput, ToolResult } from './types';

export const ThinkTool: ToolDefinition = {
  name: 'think',
  description: 'Record your reasoning before taking action. Use this to plan your approach, analyze what you know, identify what you need to find out, and decide your next step. Call this BEFORE making changes to think through the problem.',
  inputSchema: {
    type: 'object',
    properties: {
      thought: {
        type: 'string',
        description: 'Your reasoning about the current situation,  what you know, what you need to figure out, and your analysis.',
      },
      next_step: {
        type: 'string',
        description: 'The specific action you plan to take next based on your reasoning.',
      },
    },
    required: ['thought'],
  },
  isConcurrencySafe: true,
  execute: async (input: ToolInput): Promise<ToolResult> => {
    const thought = input.thought as string;
    const nextStep = input.next_step as string | undefined;
    
    let output = `Thought recorded. `;
    if (nextStep) {
      output += `Proceeding with: ${nextStep}`;
    } else {
      output += `Continue with your plan.`;
    }
    
    return {
      success: true,
      output,
    };
  },
};
