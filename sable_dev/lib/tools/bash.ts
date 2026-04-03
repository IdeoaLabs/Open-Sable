/**
 * Bash Tool — Execute shell commands in the sandbox.
 * 
 * Runs commands in the sandboxed working directory.
 * Includes permission checking for dangerous patterns.
 * Supports timeout and output truncation.
 */

import type { ToolDefinition, ToolInput, ToolContext, ToolResult } from './types';

// Dangerous command patterns that should be blocked
const BLOCKED_PATTERNS = [
  /rm\s+-rf\s+\/(?!\w)/,    // rm -rf / (but allow rm -rf /some/path)
  /rm\s+-rf\s+~\s*/,        // rm -rf ~
  /mkfs\b/,                  // filesystem format
  /dd\s+if=/,                // raw disk write
  /:$$\)\{\s*:\|:&\s*\};:/, // fork bomb
  /chmod\s+-R\s+777\s+\//,  // permission escalation on root
  /wget.*\|\s*sh/,           // pipe remote script to shell
  /curl.*\|\s*sh/,           // pipe remote script to shell
];

export const BashTool: ToolDefinition = {
  name: 'bash',
  description: 'Execute a shell command in the sandbox working directory. Use this for running build commands, package installation, git operations, or any other CLI tasks. Commands run with a 2-minute timeout.',
  inputSchema: {
    type: 'object',
    properties: {
      command: { type: 'string', description: 'The shell command to execute' },
      description: { type: 'string', description: 'Brief description of what this command does (for logging)' },
    },
    required: ['command'],
  },
  isConcurrencySafe: false,

  async execute(input: ToolInput, context: ToolContext): Promise<ToolResult> {
    const command = input.command as string;
    const description = input.description as string | undefined;

    // Security: check for dangerous patterns
    for (const pattern of BLOCKED_PATTERNS) {
      if (pattern.test(command)) {
        return {
          success: false,
          output: '',
          error: `Blocked: command matches a dangerous pattern. For safety, this operation is not allowed.`,
        };
      }
    }

    try {
      if (description) {
        console.log(`[bash-tool] ${description}: ${command}`);
      }

      const result = await context.runCommand(command);
      
      // Truncate output if too long
      let output = '';
      if (result.stdout) {
        output += result.stdout.length > 10000
          ? result.stdout.slice(0, 10000) + '\n... (output truncated, showing first 10KB)'
          : result.stdout;
      }
      if (result.stderr) {
        const stderr = result.stderr.length > 5000
          ? result.stderr.slice(0, 5000) + '\n... (stderr truncated)'
          : result.stderr;
        output += (output ? '\n' : '') + `stderr: ${stderr}`;
      }

      return {
        success: result.exitCode === 0,
        output: output || `Command completed with exit code ${result.exitCode}`,
        error: result.exitCode !== 0 ? `Exit code: ${result.exitCode}` : undefined,
      };
    } catch (error) {
      return {
        success: false,
        output: '',
        error: `Command failed: ${(error as Error).message}`,
      };
    }
  },
};
