/**
 * Sable Dev Tool System
 * 
 * A modular tool architecture inspired by professional code editors.
 * Each tool has: name, description, input schema, execute function.
 * Tools are called by the AI model during code generation.
 * 
 * Tools are partitioned into:
 * - Read-only (concurrent-safe): FileRead, Grep, Glob, ListFiles
 * - Mutating (serial): FileEdit, FileWrite, Bash, InstallPackages
 */

export interface ToolInput {
  [key: string]: unknown;
}

export interface ToolResult {
  success: boolean;
  output: string;
  error?: string;
  /** Files that were modified by this tool */
  modifiedFiles?: string[];
}

export interface ToolDefinition {
  name: string;
  description: string;
  /** JSON Schema for the input parameters */
  inputSchema: Record<string, unknown>;
  /** Whether this tool is safe to run concurrently with other tools */
  isConcurrencySafe: boolean;
  /** Execute the tool with validated input */
  execute: (input: ToolInput, context: ToolContext) => Promise<ToolResult>;
}

export interface ToolContext {
  /** The sandbox working directory */
  workDir: string;
  /** The sandbox ID (for project store) */
  sandboxId: string;
  /** Read a file from the sandbox */
  readFile: (path: string) => Promise<string>;
  /** Write a file to the sandbox */
  writeFile: (path: string, content: string) => Promise<void>;
  /** Run a command in the sandbox */
  runCommand: (cmd: string) => Promise<{ stdout: string; stderr: string; exitCode: number }>;
  /** List files in the sandbox */
  listFiles: (dir?: string) => Promise<string[]>;
}

/**
 * Permission levels for tool operations
 */
export type PermissionLevel = 'allow' | 'ask' | 'deny';

export interface PermissionRule {
  tool: string;
  pattern?: string;
  level: PermissionLevel;
  reason?: string;
}

/**
 * Default permission rules — deny dangerous operations, allow safe reads
 */
export const DEFAULT_PERMISSIONS: PermissionRule[] = [
  // Always deny dangerous operations
  { tool: 'bash', pattern: 'rm -rf /', level: 'deny', reason: 'System destruction' },
  { tool: 'bash', pattern: 'rm -rf ~', level: 'deny', reason: 'Home destruction' },
  { tool: 'bash', pattern: 'mkfs', level: 'deny', reason: 'Filesystem format' },
  { tool: 'bash', pattern: 'dd if=', level: 'deny', reason: 'Raw disk write' },
  { tool: 'bash', pattern: ':(){:|:&};:', level: 'deny', reason: 'Fork bomb' },
  { tool: 'bash', pattern: 'chmod -R 777 /', level: 'deny', reason: 'Permission escalation' },
  
  // Allow all read operations
  { tool: 'file_read', level: 'allow' },
  { tool: 'grep', level: 'allow' },
  { tool: 'glob', level: 'allow' },
  { tool: 'list_files', level: 'allow' },
  
  // Allow file writes within sandbox
  { tool: 'file_write', level: 'allow' },
  { tool: 'file_edit', level: 'allow' },
  
  // Allow package installation
  { tool: 'install_packages', level: 'allow' },
  
  // Allow general bash commands
  { tool: 'bash', level: 'allow' },
];

/**
 * Check if a tool operation is permitted
 */
export function checkPermission(toolName: string, input: ToolInput, rules: PermissionRule[] = DEFAULT_PERMISSIONS): PermissionLevel {
  // Check deny rules first (most restrictive)
  for (const rule of rules) {
    if (rule.tool !== toolName) continue;
    if (rule.level === 'deny' && rule.pattern) {
      const inputStr = JSON.stringify(input);
      if (inputStr.includes(rule.pattern)) {
        return 'deny';
      }
    }
  }
  
  // Check allow rules
  for (const rule of rules) {
    if (rule.tool === toolName && rule.level === 'allow') {
      return 'allow';
    }
  }
  
  // Default: ask
  return 'ask';
}
