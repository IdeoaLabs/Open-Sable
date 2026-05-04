/**
 * System Prompt Composition Engine
 * 
 * Builds the effective system prompt from multiple layers with priority overrides.
 * Inspired by professional code editors' prompt management systems.
 * 
 * Priority Order (highest wins):
 *   0. Override prompt → replaces everything (e.g. error-fix mode)
 *   1. Agent prompt → domain-specific agent instructions
 *   2. Custom prompt → user-provided via settings
 *   3. Default prompt → baseline code generation prompt
 *   4. Append prompt → always added at end (tool descriptions, context)
 * 
 * Context layers injected into the prompt:
 *   - Tool descriptions (filtered by intent)
 *   - File structure context
 *   - Git status (if available)
 *   - Recent edits history
 *   - Template-specific rules
 */

export interface PromptLayer {
  id: string;
  content: string;
  priority: number;
  /** If true, this layer replaces all lower-priority layers */
  isOverride?: boolean;
  /** If true, this layer is appended regardless of overrides */
  isAppend?: boolean;
}

export interface PromptContext {
  /** The active template ID (react-spa, static-site, node-api, etc) */
  templateId: string;
  /** Whether this is an edit of existing code */
  isEdit: boolean;
  /** Whether this is an error fix */
  isErrorFix: boolean;
  /** Whether tools are available */
  hasTools: boolean;
  /** The model being used */
  model: string;
  /** Recently edited file paths */
  recentEdits?: string[];
  /** Git status summary */
  gitStatus?: string;
  /** File structure string */
  fileStructure?: string;
}

/**
 * Build the effective system prompt from layers + context.
 */
export function buildSystemPrompt(
  layers: PromptLayer[],
  context: PromptContext
): string {
  // Sort by priority (highest first)
  const sorted = [...layers].sort((a, b) => b.priority - a.priority);

  // Check for override (replaces everything except appends)
  const override = sorted.find(l => l.isOverride);
  const appends = sorted.filter(l => l.isAppend);

  let parts: string[];

  if (override) {
    parts = [override.content];
  } else {
    // Use highest priority non-append, non-override layer
    const primary = sorted.find(l => !l.isOverride && !l.isAppend);
    parts = primary ? [primary.content] : [];
  }

  // Always add appends
  for (const append of appends) {
    parts.push(append.content);
  }

  // Add context injections
  const contextParts = buildContextInjections(context);
  parts.push(...contextParts);

  return parts.filter(Boolean).join('\n\n');
}

/**
 * Build context injections based on the current state.
 */
function buildContextInjections(context: PromptContext): string[] {
  const injections: string[] = [];

  // Date context
  injections.push(`Current date: ${new Date().toISOString().split('T')[0]}`);

  // Edit mode context
  if (context.isEdit) {
    injections.push(EDIT_MODE_RULES);
  }

  // Error fix mode 
  if (context.isErrorFix) {
    injections.push(ERROR_FIX_RULES);
  }

  // Template-specific rules
  const templateRules = TEMPLATE_RULES[context.templateId];
  if (templateRules) {
    injections.push(templateRules);
  }

  // Git status
  if (context.gitStatus) {
    injections.push(`GIT STATUS:\n${context.gitStatus}`);
  }

  // Recent edits
  if (context.recentEdits && context.recentEdits.length > 0) {
    injections.push(`RECENTLY EDITED FILES:\n${context.recentEdits.map(f => `- ${f}`).join('\n')}`);
  }

  return injections;
}

// ─── Prompt Constants ───────────────────────────────────────

const EDIT_MODE_RULES = `EDIT MODE ACTIVE:
- This is an incremental update to an existing application
- Output ONLY the files that need changes
- Each file must be COMPLETE,  no truncation
- Do NOT regenerate config files unless explicitly asked
- Preserve all existing functionality when editing`;

const ERROR_FIX_RULES = `ERROR FIX MODE:
1. Read the error message carefully
2. Identify the EXACT file and line with the error
3. Output ONLY the broken file with the fix
4. Do NOT output any other files
5. Preserve ALL existing functionality,  only fix the broken syntax`;

const TEMPLATE_RULES: Record<string, string> = {
  'react-spa': `TEMPLATE: React SPA (Vite + Tailwind)
- Files are .jsx NOT .tsx,  NEVER use TypeScript syntax
- Use Tailwind CSS classes for styling,  no inline styles
- Escape apostrophes in JSX text: use &apos; or {"'"}
- Standard Tailwind colors: bg-white, text-gray-900 (NOT bg-background)`,

  'static-site': `TEMPLATE: Static HTML/CSS/JS
- Files are .html, .css, .js,  NO React, NO JSX
- Use semantic HTML5 elements
- Use modern CSS (flexbox, grid, custom properties)
- Use vanilla ES6+ JavaScript`,

  'node-api': `TEMPLATE: Node.js API (Express)
- Backend-only project,  no HTML, no CSS, no React
- Files are .js,  standard Node.js/Express patterns
- Use proper error handling middleware`,

  'fullstack': `TEMPLATE: Fullstack (React + Express)
- Frontend files in src/ (React + Vite)
- Backend files in server/ (Express.js)
- Output ONLY files you need to modify`,

  'nextjs': `TEMPLATE: Next.js (App Router)
- Pages go in app/ directory
- app/layout.js is required
- Use standard Next.js patterns`,
};

/**
 * Create the default code generation prompt layer.
 */
export function createDefaultPromptLayer(): PromptLayer {
  return {
    id: 'default',
    content: `You are an expert full-stack developer. Generate production-quality code.

OUTPUT FORMAT:
- Output ONLY <file path="...">...complete code...</file> tags
- NO explanations, NO markdown, NO conversational text
- Each file must be COMPLETE,  all imports, all functions, all closing tags
- NEVER truncate code,  if running low on output space, close the current file properly

CODE QUALITY:
- Every function must have matching braces
- Every JSX component must have matching tags
- Test mentally: would this file parse without errors?
- ALWAYS use complete, working code,  no placeholders

CRITICAL,  NEVER TRUNCATE:
- Finish every <file> tag with a complete </file> closing tag
- If running low on output space, close current file, then STOP
- NEVER leave code mid-line, mid-function, or mid-component`,
    priority: 0,
  };
}

/**
 * Create an error fix override prompt layer.
 */
export function createErrorFixLayer(brokenFile?: string, brokenContent?: string): PromptLayer {
  let content = `You are a code fixer. Fix ONE broken file.

RULES:
1. Read the error message
2. Identify the EXACT syntax error
3. Output ONLY ONE <file> tag with the COMPLETE fixed file
4. Do NOT output any other files
5. Preserve ALL existing functionality`;

  if (brokenFile) {
    content += `\n\nTHE BROKEN FILE IS: ${brokenFile}`;
  }
  if (brokenContent) {
    content += `\n\nCURRENT CONTENT:\n${brokenContent}`;
  }

  return {
    id: 'error-fix',
    content,
    priority: 100, // Highest,  override
    isOverride: true,
  };
}

/**
 * Create a tool descriptions append layer.
 */
export function createToolAppendLayer(toolSummary: string): PromptLayer {
  return {
    id: 'tools',
    content: toolSummary,
    priority: 0,
    isAppend: true,
  };
}

/**
 * Create a custom agent prompt layer.
 */
export function createAgentLayer(agentInstructions: string): PromptLayer {
  return {
    id: 'agent',
    content: agentInstructions,
    priority: 50,
  };
}
