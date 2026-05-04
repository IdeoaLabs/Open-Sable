/**
 * Slash Command Router
 * 
 * Intercepts user input starting with `/` and dispatches to handlers.
 * Commands: /compact, /plan, /model, /status, /cost, /help, /clear, /fast, /effort
 */

// ─── Types ────────────────────────────────────────────────

export interface SlashCommand {
  name: string;
  aliases: string[];
  description: string;
  usage: string;
  handler: (args: string, ctx: CommandContext) => CommandResult | Promise<CommandResult>;
}

export interface CommandContext {
  /** Current model name */
  model: string;
  /** Session stats */
  tokenCount: number;
  turnCount: number;
  /** Callbacks for side-effects */
  setModel?: (model: string) => void;
  setEffort?: (level: string) => void;
  triggerCompact?: () => void;
  triggerClear?: () => void;
}

export interface CommandResult {
  /** Whether this consumed the input (don't send to AI) */
  handled: boolean;
  /** Response message to show the user */
  message: string;
  /** Optional data payload */
  data?: Record<string, unknown>;
}

// ─── Command Registry ─────────────────────────────────────

// Cache cleared on module reload (new commands added dynamically)
let cachedCommands: Map<string, SlashCommand> | null = null;

function getBuiltinCommands(): SlashCommand[] {
  return [
    {
      name: 'help',
      aliases: ['h', '?'],
      description: 'Show available commands',
      usage: '/help',
      handler: () => {
        const commands = getBuiltinCommands();
        const lines = commands.map(c => `  /${c.name},  ${c.description}`);
        return {
          handled: true,
          message: `Available commands:\n${lines.join('\n')}`,
        };
      },
    },
    {
      name: 'compact',
      aliases: ['c'],
      description: 'Compact the conversation to save context',
      usage: '/compact',
      handler: (_args, ctx) => {
        ctx.triggerCompact?.();
        return {
          handled: true,
          message: 'Compacting conversation...',
          data: { action: 'compact' },
        };
      },
    },
    {
      name: 'plan',
      aliases: ['p'],
      description: 'Enter plan mode for multi-step tasks',
      usage: '/plan <task description>',
      handler: (args) => {
        if (!args.trim()) {
          return {
            handled: true,
            message: 'Usage: /plan <describe your task>\nExample: /plan refactor the auth module to use JWT',
          };
        }
        return {
          handled: false,
          message: '',
          data: { action: 'plan', prompt: args.trim() },
        };
      },
    },
    {
      name: 'model',
      aliases: ['m'],
      description: 'Switch the active model',
      usage: '/model <model-name>',
      handler: (args, ctx) => {
        const modelName = args.trim();
        if (!modelName) {
          return {
            handled: true,
            message: `Current model: ${ctx.model}\nUsage: /model <name>`,
          };
        }
        ctx.setModel?.(modelName);
        return {
          handled: true,
          message: `Switched to model: ${modelName}`,
          data: { action: 'model', model: modelName },
        };
      },
    },
    {
      name: 'status',
      aliases: ['s'],
      description: 'Show session status',
      usage: '/status',
      handler: (_args, ctx) => {
        return {
          handled: true,
          message: [
            `Model: ${ctx.model}`,
            `Tokens: ~${ctx.tokenCount.toLocaleString()}`,
            `Turns: ${ctx.turnCount}`,
          ].join('\n'),
        };
      },
    },
    {
      name: 'cost',
      aliases: [],
      description: 'Show token usage and cost estimate',
      usage: '/cost',
      handler: () => {
        return {
          handled: true,
          message: '', // Cost report is injected by the route handler
          data: { action: 'cost' },
        };
      },
    },
    {
      name: 'clear',
      aliases: ['reset'],
      description: 'Clear the conversation',
      usage: '/clear',
      handler: (_args, ctx) => {
        ctx.triggerClear?.();
        return {
          handled: true,
          message: 'Conversation cleared.',
          data: { action: 'clear' },
        };
      },
    },
    {
      name: 'fast',
      aliases: ['f'],
      description: 'Toggle fast mode (smaller, faster model)',
      usage: '/fast [on|off]',
      handler: (args) => {
        const toggle = args.trim().toLowerCase();
        return {
          handled: true,
          message: toggle === 'off' ? 'Fast mode disabled.' : 'Fast mode enabled.',
          data: { action: 'fast', enabled: toggle !== 'off' },
        };
      },
    },
    {
      name: 'effort',
      aliases: ['e'],
      description: 'Set effort level (low, medium, high)',
      usage: '/effort <low|medium|high>',
      handler: (args, ctx) => {
        const level = args.trim().toLowerCase();
        const valid = ['low', 'medium', 'high'];
        if (!valid.includes(level)) {
          return {
            handled: true,
            message: `Usage: /effort <low|medium|high>\nSets how much effort the AI puts into responses.`,
          };
        }
        ctx.setEffort?.(level);
        return {
          handled: true,
          message: `Effort level set to: ${level}`,
          data: { action: 'effort', level },
        };
      },
    },
    {
      name: 'dream',
      aliases: ['d'],
      description: 'Show dream mode status or trigger consolidation',
      usage: '/dream [status|run|abort]',
      handler: (args) => {
        const sub = args.trim().toLowerCase() || 'status';
        return {
          handled: true,
          message: '', // Filled by route handler
          data: { action: 'dream', subcommand: sub },
        };
      },
    },
    {
      name: 'avatar',
      aliases: ['av'],
      description: 'Show your avatar companion or rename it',
      usage: '/avatar [rename <name>]',
      handler: (args) => {
        const trimmed = args.trim();
        if (trimmed.startsWith('rename ')) {
          const newName = trimmed.slice(7).trim();
          return {
            handled: true,
            message: '', // Filled by route handler
            data: { action: 'avatar', subcommand: 'rename', name: newName },
          };
        }
        return {
          handled: true,
          message: '', // Filled by route handler
          data: { action: 'avatar', subcommand: 'show' },
        };
      },
    },
    {
      name: 'teleport',
      aliases: ['tp'],
      description: 'Export or import session context',
      usage: '/teleport <export|import|list> [path]',
      handler: (args) => {
        const parts = args.trim().split(/\s+/);
        const sub = parts[0]?.toLowerCase() || 'list';
        const target = parts.slice(1).join(' ');
        return {
          handled: true,
          message: '', // Filled by route handler
          data: { action: 'teleport', subcommand: sub, target },
        };
      },
    },
    {
      name: 'coordinator',
      aliases: ['coord'],
      description: 'Toggle coordinator (multi-agent) mode',
      usage: '/coordinator [on|off|status]',
      handler: (args) => {
        const sub = args.trim().toLowerCase() || 'status';
        return {
          handled: true,
          message: '', // Filled by route handler
          data: { action: 'coordinator', subcommand: sub },
        };
      },
    },
    {
      name: 'screenshot',
      aliases: ['ss'],
      description: 'Take a screenshot of the desktop',
      usage: '/screenshot',
      handler: () => {
        return {
          handled: true,
          message: '', // Filled by route handler
          data: { action: 'screenshot' },
        };
      },
    },
    {
      name: 'undo',
      aliases: ['u', 'revert'],
      description: 'Undo the last AI change (restore previous file snapshot)',
      usage: '/undo',
      handler: () => {
        return {
          handled: true,
          message: 'Reverting to previous snapshot...',
          data: { action: 'undo' },
        };
      },
    },
    {
      name: 'redo',
      aliases: [],
      description: 'Redo the last undone change',
      usage: '/redo',
      handler: () => {
        return {
          handled: true,
          message: 'Reapplying next snapshot...',
          data: { action: 'redo' },
        };
      },
    },
    {
      name: 'review',
      aliases: ['rv'],
      description: 'Ask the AI to review your current code for issues',
      usage: '/review [focus area]',
      handler: (args) => {
        const focus = args.trim();
        return {
          handled: false, // Don't consume,  let it go to AI with a special prompt
          message: '',
          data: {
            action: 'review',
            prompt: focus
              ? `Please review the current project code, focusing on: ${focus}. Check for bugs, security issues, performance problems, and suggest improvements.`
              : `Please review all the current project code. Check for bugs, security issues, performance problems, missing error handling, and suggest improvements. Be specific about file names and line numbers.`,
          },
        };
      },
    },
    {
      name: 'download',
      aliases: ['export', 'zip'],
      description: 'Download the project as a zip file',
      usage: '/download',
      handler: () => {
        return {
          handled: true,
          message: 'Preparing download...',
          data: { action: 'download' },
        };
      },
    },
  ];
}

/**
 * Get the command registry (memoized).
 */
function getCommandMap(): Map<string, SlashCommand> {
  if (cachedCommands) return cachedCommands;
  
  const commands = getBuiltinCommands();
  const map = new Map<string, SlashCommand>();
  
  for (const cmd of commands) {
    map.set(cmd.name, cmd);
    for (const alias of cmd.aliases) {
      map.set(alias, cmd);
    }
  }
  
  cachedCommands = map;
  return map;
}

// ─── Parser ───────────────────────────────────────────────

/**
 * Check if input is a slash command.
 */
export function isSlashCommand(input: string): boolean {
  return input.trimStart().startsWith('/') && 
         input.trimStart().length > 1 &&
         /^\/[a-zA-Z]/.test(input.trimStart());
}

/**
 * Parse a slash command from user input.
 * Returns null if not a valid command.
 */
export function parseSlashCommand(input: string): { name: string; args: string } | null {
  if (!isSlashCommand(input)) return null;
  
  const trimmed = input.trimStart();
  const spaceIdx = trimmed.indexOf(' ');
  
  if (spaceIdx === -1) {
    return { name: trimmed.slice(1).toLowerCase(), args: '' };
  }
  
  return {
    name: trimmed.slice(1, spaceIdx).toLowerCase(),
    args: trimmed.slice(spaceIdx + 1),
  };
}

/**
 * Route a slash command to its handler.
 * Returns null if the command is not recognized.
 */
export async function routeSlashCommand(
  input: string,
  ctx: CommandContext,
): Promise<CommandResult | null> {
  const parsed = parseSlashCommand(input);
  if (!parsed) return null;
  
  const map = getCommandMap();
  const command = map.get(parsed.name);
  
  if (!command) {
    // Suggest closest match
    const allNames = Array.from(new Set(getBuiltinCommands().map(c => c.name)));
    const suggestion = allNames.find(n => n.startsWith(parsed.name));
    return {
      handled: true,
      message: suggestion 
        ? `Unknown command "/${parsed.name}". Did you mean /${suggestion}?`
        : `Unknown command "/${parsed.name}". Type /help for available commands.`,
    };
  }
  
  return command.handler(parsed.args, ctx);
}

/**
 * Get command names for autocomplete.
 */
export function getCommandNames(): string[] {
  return getBuiltinCommands().map(c => c.name);
}

/**
 * Get full command list with descriptions for display.
 */
export function getCommandList(): Array<{ name: string; description: string; usage: string }> {
  return getBuiltinCommands().map(c => ({
    name: c.name,
    description: c.description,
    usage: c.usage,
  }));
}
