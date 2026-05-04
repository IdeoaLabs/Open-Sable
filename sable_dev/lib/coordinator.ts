/**
 * Coordinator,  Multi-Agent Orchestration
 * 
 * Splits complex work across async workers; coordinator synthesizes results.
 * Workers are forked sub-agents that run independently and notify via
 * structured messages when complete.
 * 
 * Architecture:
 * - Coordinator = main thread, talks to user, makes high-level decisions
 * - Workers = forked agents via Ollama, work independently
 * - Handoff: workers notify via structured result messages
 * - Parallel launches: multiple workers in one coordinator turn
 * 
 * Optional: scratchpad directory for cross-worker durable state
 */

import fs from 'fs';
import path from 'path';

// ─── Types ────────────────────────────────────────────────

export type WorkerStatus = 'pending' | 'running' | 'completed' | 'failed' | 'killed';

export interface WorkerTask {
  id: string;
  role: string;
  task: string;
  context?: string;
  /** Allowed tools for this worker */
  allowedTools: string[];
}

export interface Worker {
  id: string;
  task: WorkerTask;
  status: WorkerStatus;
  startedAt: number;
  completedAt?: number;
  result?: string;
  error?: string;
  /** Model used for this worker */
  model: string;
}

export interface CoordinatorState {
  /** Whether coordinator mode is active */
  isActive: boolean;
  /** All workers (active + completed) */
  workers: Worker[];
  /** Pending tasks not yet assigned to workers */
  pendingTasks: WorkerTask[];
  /** Results from completed workers awaiting synthesis */
  pendingResults: WorkerResult[];
  /** Scratchpad directory path for cross-worker state */
  scratchpadDir: string;
  /** Total tasks completed across all coordinator sessions */
  totalTasksCompleted: number;
  /** Whether coordinator needs to synthesize before proceeding */
  needsSynthesis: boolean;
}

export interface WorkerResult {
  workerId: string;
  taskId: string;
  role: string;
  result: string;
  completedAt: number;
}

export interface CoordinatorPlan {
  tasks: WorkerTask[];
  synthesis: string; // What the coordinator will do with results
  parallel: boolean; // Whether workers can run in parallel
}

// ─── Allowed Tools per Worker Type ────────────────────────

const WORKER_ALLOWED_TOOLS: Record<string, string[]> = {
  'researcher': ['file_read', 'grep', 'glob', 'list_files', 'web_fetch'],
  'implementer': ['file_read', 'file_edit', 'file_write', 'grep', 'glob', 'list_files'],
  'reviewer': ['file_read', 'grep', 'glob', 'list_files'],
  'tester': ['file_read', 'file_write', 'bash', 'grep', 'glob'],
  'planner': ['file_read', 'grep', 'glob', 'list_files', 'web_fetch'],
  'debugger': ['file_read', 'file_edit', 'bash', 'grep', 'glob', 'list_files'],
};

// Internal tools explicitly hidden from workers
const HIDDEN_TOOLS = ['agent_spawn', 'tool_search'];

// ─── State Management ─────────────────────────────────────

const SCRATCHPAD_DIR = path.join(process.cwd(), '.sable-dev', 'scratchpad');

export function createCoordinatorState(): CoordinatorState {
  return {
    isActive: false,
    workers: [],
    pendingTasks: [],
    pendingResults: [],
    scratchpadDir: SCRATCHPAD_DIR,
    totalTasksCompleted: 0,
    needsSynthesis: false,
  };
}

// ─── Task Planning ────────────────────────────────────────

/**
 * Analyze a prompt and determine if it should use coordinator mode.
 * Returns true for complex, multi-step tasks.
 */
export function shouldUseCoordinator(prompt: string): boolean {
  const indicators = [
    /\b(and then|after that|once .* is done|first .* then)\b/i,
    /\b(multiple files|across .* files|several components)\b/i,
    /\b(refactor|restructure|migrate|reorganize)\b/i,
    /\b(review and fix|analyze and implement|research and build)\b/i,
    /\b(test.*and.*deploy|build.*and.*test)\b/i,
    /\b(full stack|end.to.end|comprehensive)\b/i,
  ];
  
  const matchCount = indicators.filter(r => r.test(prompt)).length;
  return matchCount >= 2;
}

/**
 * Parse a coordinator plan from AI output.
 * The coordinator expresses its plan as structured task descriptions.
 */
export function parseCoordinatorPlan(
  prompt: string,
  aiPlanOutput?: string,
): CoordinatorPlan {
  // If we have AI plan output, try to parse it
  if (aiPlanOutput) {
    try {
      const jsonMatch = aiPlanOutput.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        const parsed = JSON.parse(jsonMatch[0]);
        if (parsed.tasks && Array.isArray(parsed.tasks)) {
          return {
            tasks: parsed.tasks.map((t: any, i: number) => ({
              id: `task-${Date.now()}-${i}`,
              role: sanitizeRole(t.role || 'researcher'),
              task: String(t.task || ''),
              context: t.context,
              allowedTools: WORKER_ALLOWED_TOOLS[t.role] || WORKER_ALLOWED_TOOLS['researcher'],
            })),
            synthesis: parsed.synthesis || 'Combine worker results',
            parallel: parsed.parallel !== false,
          };
        }
      }
    } catch { /* fall through to heuristic */ }
  }
  
  // Heuristic decomposition
  return decomposeTask(prompt);
}

function sanitizeRole(role: string): string {
  const valid = Object.keys(WORKER_ALLOWED_TOOLS);
  return valid.includes(role) ? role : 'researcher';
}

/**
 * Heuristic task decomposition when no AI plan is available.
 */
function decomposeTask(prompt: string): CoordinatorPlan {
  const tasks: WorkerTask[] = [];
  const lower = prompt.toLowerCase();
  
  // Always start with a researcher
  tasks.push({
    id: `task-${Date.now()}-0`,
    role: 'researcher',
    task: `Analyze the codebase to understand the current state relevant to: ${prompt.substring(0, 200)}`,
    allowedTools: WORKER_ALLOWED_TOOLS['researcher'],
  });
  
  // Add implementer for build/create/add tasks
  if (/\b(build|create|add|implement|write|make)\b/i.test(lower)) {
    tasks.push({
      id: `task-${Date.now()}-1`,
      role: 'implementer',
      task: `Implement the changes described: ${prompt.substring(0, 200)}`,
      allowedTools: WORKER_ALLOWED_TOOLS['implementer'],
    });
  }
  
  // Add reviewer if refactoring or fixing
  if (/\b(refactor|fix|review|debug|optimize)\b/i.test(lower)) {
    tasks.push({
      id: `task-${Date.now()}-2`,
      role: 'reviewer',
      task: `Review the code related to: ${prompt.substring(0, 200)}`,
      allowedTools: WORKER_ALLOWED_TOOLS['reviewer'],
    });
  }
  
  // Add tester if test-related
  if (/\b(test|verify|check|validate)\b/i.test(lower)) {
    tasks.push({
      id: `task-${Date.now()}-3`,
      role: 'tester',
      task: `Write tests for: ${prompt.substring(0, 200)}`,
      allowedTools: WORKER_ALLOWED_TOOLS['tester'],
    });
  }
  
  return {
    tasks,
    synthesis: 'Combine findings and implementation into a final response',
    parallel: tasks.length <= 3, // Parallel if not too many
  };
}

// ─── Worker Execution ─────────────────────────────────────

/**
 * Spawn a worker agent to execute a task.
 * Uses Ollama API directly (same pattern as agent-spawn tool).
 */
export async function spawnWorker(
  state: CoordinatorState,
  task: WorkerTask,
  model?: string,
): Promise<{ state: CoordinatorState; worker: Worker }> {
  const workerModel = model || process.env.AGENT_MODEL || 'qwen2.5:14b';
  
  const worker: Worker = {
    id: `worker-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
    task,
    status: 'running',
    startedAt: Date.now(),
    model: workerModel,
  };
  
  const newState = {
    ...state,
    workers: [...state.workers, worker],
  };
  
  // Execute the worker (this blocks until complete)
  try {
    const ollamaHost = process.env.OLLAMA_HOST || 'http://localhost:11434';
    
    const systemPrompt = buildWorkerSystemPrompt(task);
    const userPrompt = task.context 
      ? `${task.task}\n\nCONTEXT:\n${task.context.substring(0, 8000)}`
      : task.task;
    
    const response = await fetch(`${ollamaHost}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: workerModel,
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: userPrompt },
        ],
        stream: false,
        options: { temperature: 0.3, num_predict: 4096 },
      }),
    });
    
    if (!response.ok) {
      throw new Error(`Worker failed: HTTP ${response.status}`);
    }
    
    const data = await response.json();
    const result = data.message?.content || '';
    
    worker.status = 'completed';
    worker.completedAt = Date.now();
    worker.result = result;
    
    newState.pendingResults.push({
      workerId: worker.id,
      taskId: task.id,
      role: task.role,
      result,
      completedAt: Date.now(),
    });
    newState.totalTasksCompleted += 1;
    newState.needsSynthesis = true;
    
  } catch (error: any) {
    worker.status = 'failed';
    worker.error = error.message;
    worker.completedAt = Date.now();
  }
  
  return { state: newState, worker };
}

/**
 * Spawn multiple workers in parallel.
 */
export async function spawnWorkersParallel(
  state: CoordinatorState,
  tasks: WorkerTask[],
  model?: string,
): Promise<CoordinatorState> {
  const currentState = state;
  
  // Launch all workers concurrently
  const results = await Promise.allSettled(
    tasks.map(task => spawnWorker(currentState, task, model))
  );
  
  // Merge all worker results into state
  const allWorkers: Worker[] = [...state.workers];
  const allResults: WorkerResult[] = [...state.pendingResults];
  let completed = state.totalTasksCompleted;
  
  for (const result of results) {
    if (result.status === 'fulfilled') {
      allWorkers.push(result.value.worker);
      allResults.push(...result.value.state.pendingResults.filter(
        r => !state.pendingResults.some(pr => pr.workerId === r.workerId)
      ));
      if (result.value.worker.status === 'completed') completed++;
    }
  }
  
  return {
    ...state,
    workers: allWorkers,
    pendingResults: allResults,
    totalTasksCompleted: completed,
    needsSynthesis: allResults.length > 0,
  };
}

function buildWorkerSystemPrompt(task: WorkerTask): string {
  const toolList = task.allowedTools.join(', ');
  
  return `You are a specialized ${task.role} agent working on a focused sub-task.

YOUR ROLE: ${task.role}
ALLOWED TOOLS: ${toolList}
RESTRICTIONS: You may only use the tools listed above. Do not attempt other operations.

IMPORTANT:
- Focus ONLY on your assigned task
- Be thorough but concise
- Report your findings in a structured format  
- If you encounter blockers, describe them clearly
- Do NOT make assumptions about what other agents are doing

OUTPUT FORMAT:
Provide your results in a clear, structured format:
1. FINDINGS: What you discovered
2. ACTIONS: What you did (if any modifications)
3. RECOMMENDATIONS: What the coordinator should know`;
}

// ─── Synthesis ────────────────────────────────────────────

/**
 * Format worker results for the coordinator to synthesize.
 */
export function formatWorkerResultsForSynthesis(state: CoordinatorState): string {
  if (state.pendingResults.length === 0) return '';
  
  const sections = state.pendingResults.map(r => {
    const worker = state.workers.find(w => w.id === r.workerId);
    return `<task-notification worker="${r.workerId}" role="${r.role}" task="${r.taskId}">
${r.result}
</task-notification>`;
  });
  
  return `## Worker Results (Awaiting Synthesis)

${sections.join('\n\n')}

COORDINATOR INSTRUCTIONS:
- Review ALL worker results above
- Synthesize findings into a coherent response
- If workers found conflicts, resolve them
- Present the final result to the user
- Mark synthesis complete after responding`;
}

/**
 * Clear pending results after synthesis.
 */
export function completeSynthesis(state: CoordinatorState): CoordinatorState {
  return {
    ...state,
    pendingResults: [],
    needsSynthesis: false,
  };
}

// ─── Scratchpad ───────────────────────────────────────────

/**
 * Write to the shared scratchpad (for cross-worker state).
 */
export function writeScratchpad(state: CoordinatorState, key: string, value: string): void {
  try {
    if (!fs.existsSync(state.scratchpadDir)) {
      fs.mkdirSync(state.scratchpadDir, { recursive: true });
    }
    // Sanitize key to prevent path traversal
    const safeKey = key.replace(/[^a-zA-Z0-9._-]/g, '_');
    fs.writeFileSync(path.join(state.scratchpadDir, safeKey), value);
  } catch (e) {
    console.warn('[coordinator] Failed to write scratchpad:', e);
  }
}

/**
 * Read from the shared scratchpad.
 */
export function readScratchpad(state: CoordinatorState, key: string): string | null {
  try {
    const safeKey = key.replace(/[^a-zA-Z0-9._-]/g, '_');
    const filePath = path.join(state.scratchpadDir, safeKey);
    if (fs.existsSync(filePath)) {
      return fs.readFileSync(filePath, 'utf-8');
    }
  } catch { /* not found */ }
  return null;
}

/**
 * Clear the scratchpad after a coordinator session ends.
 */
export function clearScratchpad(state: CoordinatorState): void {
  try {
    if (fs.existsSync(state.scratchpadDir)) {
      const files = fs.readdirSync(state.scratchpadDir);
      for (const file of files) {
        fs.unlinkSync(path.join(state.scratchpadDir, file));
      }
    }
  } catch { /* best effort */ }
}

// ─── Status ───────────────────────────────────────────────

/**
 * Get coordinator status summary.
 */
export function getCoordinatorStatus(state: CoordinatorState): string {
  if (!state.isActive) return 'Coordinator: inactive';
  
  const running = state.workers.filter(w => w.status === 'running').length;
  const completed = state.workers.filter(w => w.status === 'completed').length;
  const failed = state.workers.filter(w => w.status === 'failed').length;
  const pending = state.pendingTasks.length;
  
  return [
    `Coordinator: active`,
    `Workers: ${running} running, ${completed} completed, ${failed} failed`,
    `Pending tasks: ${pending}`,
    `Awaiting synthesis: ${state.needsSynthesis ? 'yes' : 'no'}`,
    `Total completed: ${state.totalTasksCompleted}`,
  ].join('\n');
}

/**
 * Get prompt addition for coordinator mode.
 */
export function getCoordinatorPrompt(state: CoordinatorState): string {
  if (!state.isActive) return '';
  
  let prompt = `\n## Coordinator Mode Active
You are operating as a COORDINATOR. You can delegate sub-tasks to worker agents.
Workers work independently and report results to you.

RULES:
1. Break complex tasks into focused sub-tasks
2. Assign each sub-task to the appropriate worker role
3. Wait for ALL workers to complete before synthesizing
4. ALWAYS synthesize results,  don't pass raw worker output to user
5. You may launch multiple workers in parallel for independent tasks

AVAILABLE WORKER ROLES:
${Object.entries(WORKER_ALLOWED_TOOLS).map(([role, tools]) => 
  `- ${role}: ${tools.join(', ')}`
).join('\n')}
`;

  // Add pending synthesis if needed
  if (state.needsSynthesis) {
    prompt += '\n' + formatWorkerResultsForSynthesis(state);
  }

  return prompt;
}
