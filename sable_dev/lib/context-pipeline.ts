/**
 * Context Pipeline
 * 
 * Gathers runtime context (git status, recent edits, dependency info)
 * to inject into the system prompt for better AI awareness.
 */

export interface PipelineContext {
  gitStatus?: string;
  recentEdits?: string[];
  dependencies?: string[];
  fileStructure?: string;
}

/**
 * Gather git status from the sandbox.
 * Returns a short summary or null if unavailable.
 */
async function gatherGitStatus(
  runCommand: (cmd: string) => Promise<string>
): Promise<string | null> {
  try {
    const status = await runCommand('git status --porcelain 2>/dev/null | head -20');
    if (!status || !status.trim()) return null;
    
    const lines = status.trim().split('\n');
    const summary = lines.map(line => {
      const code = line.substring(0, 2).trim();
      const file = line.substring(3);
      const label = code === 'M' ? 'modified' 
        : code === 'A' ? 'added' 
        : code === 'D' ? 'deleted' 
        : code === '??' ? 'untracked' 
        : code;
      return `  ${label}: ${file}`;
    }).join('\n');

    return `${lines.length} changed file(s):\n${summary}`;
  } catch {
    return null;
  }
}

/**
 * Gather recently edited files from the sandbox.
 * Uses find to get files modified in the last 30 minutes.
 */
async function gatherRecentEdits(
  runCommand: (cmd: string) => Promise<string>
): Promise<string[]> {
  try {
    const result = await runCommand(
      'find . -name "*.jsx" -o -name "*.tsx" -o -name "*.js" -o -name "*.ts" -o -name "*.css" -o -name "*.html" | head -30 | xargs ls -t 2>/dev/null | head -10'
    );
    if (!result || !result.trim()) return [];
    return result.trim().split('\n').filter(Boolean).map(f => f.replace(/^\.\//, ''));
  } catch {
    return [];
  }
}

/**
 * Gather dependency list from package.json if present.
 */
async function gatherDependencies(
  readFile: (path: string) => Promise<string>
): Promise<string[]> {
  try {
    const content = await readFile('package.json');
    const pkg = JSON.parse(content);
    const deps = Object.keys(pkg.dependencies || {});
    const devDeps = Object.keys(pkg.devDependencies || {});
    return [...deps, ...devDeps];
  } catch {
    return [];
  }
}

/**
 * Gather compact file tree from the sandbox.
 */
async function gatherFileStructure(
  runCommand: (cmd: string) => Promise<string>
): Promise<string | null> {
  try {
    const result = await runCommand(
      'find . -not -path "*/node_modules/*" -not -path "*/.git/*" -not -path "*/dist/*" -type f | sort | head -50'
    );
    if (!result || !result.trim()) return null;
    return result.trim().split('\n').map(f => f.replace(/^\.\//, '')).join('\n');
  } catch {
    return null;
  }
}

/**
 * Run the full context pipeline. 
 * Gathers all available context from the sandbox in parallel.
 */
export async function runContextPipeline(sandbox: {
  readFile: (path: string) => Promise<string>;
  runCommand: (cmd: string) => Promise<string | { stdout: string; stderr: string; exitCode: number }>;
}): Promise<PipelineContext> {
  // Normalize runCommand to always return a string
  const run = async (cmd: string): Promise<string> => {
    const result = await sandbox.runCommand(cmd);
    if (typeof result === 'string') return result;
    return result.stdout || '';
  };

  const [gitStatus, recentEdits, dependencies, fileStructure] = await Promise.all([
    gatherGitStatus(run),
    gatherRecentEdits(run),
    gatherDependencies(sandbox.readFile),
    gatherFileStructure(run),
  ]);

  return {
    gitStatus: gitStatus || undefined,
    recentEdits: recentEdits.length > 0 ? recentEdits : undefined,
    dependencies: dependencies.length > 0 ? dependencies : undefined,
    fileStructure: fileStructure || undefined,
  };
}

/**
 * Format pipeline context into a string block for the system prompt.
 */
export function formatContextForPrompt(ctx: PipelineContext): string {
  const parts: string[] = [];

  if (ctx.gitStatus) {
    parts.push(`GIT STATUS:\n${ctx.gitStatus}`);
  }

  if (ctx.recentEdits && ctx.recentEdits.length > 0) {
    parts.push(`RECENTLY EDITED FILES:\n${ctx.recentEdits.map(f => `- ${f}`).join('\n')}`);
  }

  if (ctx.dependencies && ctx.dependencies.length > 0) {
    parts.push(`INSTALLED PACKAGES:\n${ctx.dependencies.join(', ')}`);
  }

  if (ctx.fileStructure) {
    parts.push(`PROJECT FILES:\n${ctx.fileStructure}`);
  }

  return parts.join('\n\n');
}
