/**
 * Plan Mode
 * 
 * Structured planning before execution. The model generates a plan,
 * user reviews/approves, then model executes step by step.
 * 
 * Critical for small local models which perform much better when they
 * plan first instead of jumping straight to code generation.
 * 
 * Flow:
 *   1. User prompt → detect if planning would help
 *   2. Enter plan mode → model generates structured plan
 *   3. User approves/modifies plan
 *   4. Model executes plan step by step
 *   5. Exit plan mode
 */

// ─── Types ────────────────────────────────────────────────

export interface PlanStep {
  id: number;
  description: string;
  type: 'create' | 'edit' | 'delete' | 'install' | 'run' | 'review';
  targetFiles?: string[];
  status: 'pending' | 'in-progress' | 'completed' | 'skipped';
  result?: string;
}

export interface Plan {
  id: string;
  title: string;
  steps: PlanStep[];
  createdAt: number;
  status: 'draft' | 'approved' | 'executing' | 'completed' | 'cancelled';
  currentStep: number;
  summary?: string;
}

export interface PlanModeState {
  enabled: boolean;
  currentPlan: Plan | null;
  history: Plan[];
}

// ─── State ────────────────────────────────────────────────

/**
 * Create initial plan mode state.
 */
export function createPlanModeState(): PlanModeState {
  return {
    enabled: false,
    currentPlan: null,
    history: [],
  };
}

// ─── Plan Detection ───────────────────────────────────────

/**
 * Detect if a user prompt would benefit from planning.
 * Complex, multi-file, or architectural tasks should use plan mode.
 */
export function shouldSuggestPlan(prompt: string): boolean {
  const lower = prompt.toLowerCase();
  
  // Multi-step keywords
  const planKeywords = [
    'build', 'create', 'redesign', 'refactor', 'restructure',
    'migrate', 'convert', 'rewrite', 'implement', 'architect',
    'setup', 'scaffold', 'full', 'complete', 'entire',
    'dashboard', 'application', 'website', 'platform',
    'add authentication', 'add routing', 'add database',
    'multi-page', 'multi-step',
  ];
  
  const matchCount = planKeywords.filter(kw => lower.includes(kw)).length;
  
  // Also check prompt length — long prompts usually need planning
  const isLong = prompt.length > 200;
  
  // Multiple file references
  const fileRefs = (prompt.match(/\.(jsx|tsx|js|ts|css|html)/g) || []).length;
  
  return matchCount >= 2 || (matchCount >= 1 && isLong) || fileRefs >= 3;
}

// ─── Plan Creation ────────────────────────────────────────

/**
 * Create a new plan from an AI-generated plan text.
 * Parses numbered steps from the model's output.
 */
export function parsePlanFromText(planText: string, title?: string): Plan {
  const steps: PlanStep[] = [];
  
  // Match numbered lines: "1. Do something" or "- Do something"
  const stepLines = planText.split('\n').filter(line => 
    /^\s*(\d+[\.\)]\s+|-\s+|\*\s+)/.test(line)
  );
  
  for (let i = 0; i < stepLines.length; i++) {
    const line = stepLines[i].replace(/^\s*(\d+[\.\)]\s+|-\s+|\*\s+)/, '').trim();
    if (!line) continue;
    
    // Detect step type from content
    const type = detectStepType(line);
    
    // Extract file references
    const files = line.match(/[\w/-]+\.(jsx|tsx|js|ts|css|html|json)/g) || undefined;
    
    steps.push({
      id: i + 1,
      description: line,
      type,
      targetFiles: files,
      status: 'pending',
    });
  }
  
  // If no numbered steps found, treat each sentence as a step
  if (steps.length === 0) {
    const sentences = planText.split(/[.!]\s+/).filter(s => s.trim().length > 10);
    for (let i = 0; i < sentences.length && i < 10; i++) {
      steps.push({
        id: i + 1,
        description: sentences[i].trim(),
        type: 'edit',
        status: 'pending',
      });
    }
  }
  
  return {
    id: `plan-${Date.now()}`,
    title: title || 'Execution Plan',
    steps,
    createdAt: Date.now(),
    status: 'draft',
    currentStep: 0,
  };
}

/**
 * Detect step type from description text.
 */
function detectStepType(text: string): PlanStep['type'] {
  const lower = text.toLowerCase();
  if (lower.includes('create') || lower.includes('new file') || lower.includes('scaffold')) return 'create';
  if (lower.includes('delete') || lower.includes('remove')) return 'delete';
  if (lower.includes('install') || lower.includes('npm') || lower.includes('package')) return 'install';
  if (lower.includes('run') || lower.includes('test') || lower.includes('build') || lower.includes('command')) return 'run';
  if (lower.includes('review') || lower.includes('check') || lower.includes('verify')) return 'review';
  return 'edit';
}

// ─── Plan Execution ───────────────────────────────────────

/**
 * Get the next pending step in a plan.
 */
export function getNextStep(plan: Plan): PlanStep | null {
  return plan.steps.find(s => s.status === 'pending') || null;
}

/**
 * Mark a step as in-progress.
 */
export function startStep(plan: Plan, stepId: number): Plan {
  return {
    ...plan,
    status: 'executing',
    currentStep: stepId,
    steps: plan.steps.map(s => 
      s.id === stepId ? { ...s, status: 'in-progress' as const } : s
    ),
  };
}

/**
 * Complete a step with a result.
 */
export function completeStep(plan: Plan, stepId: number, result?: string): Plan {
  const updated = {
    ...plan,
    steps: plan.steps.map(s => 
      s.id === stepId ? { ...s, status: 'completed' as const, result } : s
    ),
  };
  
  // Check if all steps are done
  const allDone = updated.steps.every(s => s.status === 'completed' || s.status === 'skipped');
  if (allDone) {
    updated.status = 'completed';
  }
  
  return updated;
}

/**
 * Skip a step.
 */
export function skipStep(plan: Plan, stepId: number): Plan {
  return {
    ...plan,
    steps: plan.steps.map(s => 
      s.id === stepId ? { ...s, status: 'skipped' as const } : s
    ),
  };
}

/**
 * Approve a plan for execution.
 */
export function approvePlan(plan: Plan): Plan {
  return { ...plan, status: 'approved' };
}

/**
 * Cancel a plan.
 */
export function cancelPlan(plan: Plan): Plan {
  return { ...plan, status: 'cancelled' };
}

// ─── Plan Prompt Generation ───────────────────────────────

/**
 * Generate the system prompt addition for plan mode.
 * Tells the model to generate a plan instead of jumping to code.
 */
export function getPlanModePrompt(): string {
  return `PLAN MODE ACTIVE:
You are in planning mode. Instead of generating code, create a structured plan.

OUTPUT FORMAT:
1. Start with a brief summary of the approach
2. List numbered steps, each describing ONE action
3. For each step, mention the target file(s) if applicable
4. End with a summary of expected outcome

EXAMPLE:
Summary: Create a responsive dashboard with sidebar navigation and data cards.

1. Create src/components/Sidebar.jsx - Navigation sidebar with links
2. Edit src/App.jsx - Add Sidebar import and layout grid
3. Create src/components/DashboardCard.jsx - Reusable stat card component
4. Create src/components/Dashboard.jsx - Main dashboard with grid of cards
5. Edit src/index.css - Add responsive grid styles and sidebar styling
6. Run npm install recharts - For chart components in cards

Expected outcome: Fully responsive dashboard with sidebar nav and 4 data cards.`;
}

/**
 * Generate a step execution prompt for the current step.
 */
export function getStepExecutionPrompt(plan: Plan, step: PlanStep): string {
  const completedSteps = plan.steps
    .filter(s => s.status === 'completed')
    .map(s => `  ✓ Step ${s.id}: ${s.description}${s.result ? ` (${s.result})` : ''}`)
    .join('\n');
  
  const remaining = plan.steps
    .filter(s => s.status === 'pending')
    .map(s => `  - Step ${s.id}: ${s.description}`)
    .join('\n');
  
  return `EXECUTING PLAN: "${plan.title}"

Progress:
${completedSteps || '  (none completed yet)'}

CURRENT STEP ${step.id}:
  ${step.description}
${step.targetFiles ? `  Target files: ${step.targetFiles.join(', ')}` : ''}

Remaining:
${remaining || '  (this is the last step)'}

Execute ONLY step ${step.id}. Output the required code changes.`;
}

/**
 * Format a plan for display in the UI.
 */
export function formatPlanForDisplay(plan: Plan): string {
  const statusIcon = {
    draft: '📋',
    approved: '✅',
    executing: '⚡',
    completed: '🎉',
    cancelled: '❌',
  };
  
  const stepIcon = {
    pending: '○',
    'in-progress': '●',
    completed: '✓',
    skipped: '⊘',
  };
  
  const header = `${statusIcon[plan.status]} ${plan.title} (${plan.status})`;
  const steps = plan.steps
    .map(s => `  ${stepIcon[s.status]} ${s.id}. ${s.description}`)
    .join('\n');
  
  return `${header}\n${steps}`;
}
