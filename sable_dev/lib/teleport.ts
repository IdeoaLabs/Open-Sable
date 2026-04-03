/**
 * Teleport — Cross-Environment Session Transfer
 * 
 * Move work context between environments (machines, directories, repos).
 * Uses SHA256 delta sync: only transfers keys whose hash differs.
 * 
 * Session context includes:
 * - Git state (repo URL, branch, push perms)
 * - Working directory context
 * - Model override configuration
 * - Custom system prompt
 * - Outcomes (created branches, files changed)
 * - Conversation summary (compacted)
 * 
 * Storage: .sable-dev/teleport/
 */

import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import { execSync } from 'child_process';
import os from 'os';

// ─── Types ────────────────────────────────────────────────

export interface TeleportContext {
  /** Unique teleport session ID */
  id: string;
  /** Source environment identifier */
  sourceEnv: string;
  /** Timestamp of export */
  exportedAt: number;
  /** Git context */
  git: GitContext | null;
  /** Working directory snapshot */
  workingDir: string;
  /** Model configuration override */
  modelOverride: string | null;
  /** Custom system prompt (if any) */
  customSystemPrompt: string | null;
  /** Conversation summary (compacted) */
  conversationSummary: string;
  /** Outcomes from the session */
  outcomes: SessionOutcome[];
  /** Key-value data blobs (files, config, etc.) */
  data: Record<string, string>;
  /** SHA256 hashes for delta sync */
  hashes: Record<string, string>;
  /** Schema version */
  version: number;
}

export interface GitContext {
  repoUrl: string;
  branch: string;
  revision: string;
  hasPushPerms: boolean;
  remotes: string[];
  uncommittedChanges: number;
}

export interface SessionOutcome {
  type: 'branch_created' | 'files_changed' | 'packages_installed' | 'config_modified';
  description: string;
  timestamp: number;
  metadata?: Record<string, string>;
}

export interface TeleportState {
  /** Last export timestamp */
  lastExportAt: number | null;
  /** Last import timestamp */
  lastImportAt: number | null;
  /** Per-key content hashes (for delta sync) */
  contentHashes: Record<string, string>;
  /** Number of teleports done */
  teleportCount: number;
  /** Max entries cap */
  maxEntries: number;
}

export type SyncDirection = 'export' | 'import';

// ─── Persistence ──────────────────────────────────────────

const TELEPORT_DIR = path.join(process.cwd(), '.sable-dev', 'teleport');
const STATE_FILE = path.join(TELEPORT_DIR, 'state.json');

function ensureTeleportDir(): void {
  if (!fs.existsSync(TELEPORT_DIR)) {
    fs.mkdirSync(TELEPORT_DIR, { recursive: true });
  }
}

// ─── State Management ─────────────────────────────────────

export function createTeleportState(): TeleportState {
  // Try to restore from disk
  try {
    if (fs.existsSync(STATE_FILE)) {
      return JSON.parse(fs.readFileSync(STATE_FILE, 'utf-8'));
    }
  } catch { /* start fresh */ }
  
  return {
    lastExportAt: null,
    lastImportAt: null,
    contentHashes: {},
    teleportCount: 0,
    maxEntries: 50,
  };
}

function saveTeleportState(state: TeleportState): void {
  try {
    ensureTeleportDir();
    fs.writeFileSync(STATE_FILE, JSON.stringify(state, null, 2));
  } catch (e) {
    console.warn('[teleport] Failed to save state:', e);
  }
}

// ─── SHA256 Hashing ───────────────────────────────────────

function sha256(content: string): string {
  return `sha256:${crypto.createHash('sha256').update(content).digest('hex')}`;
}

/**
 * Compute hashes for all data keys.
 */
function computeHashes(data: Record<string, string>): Record<string, string> {
  const hashes: Record<string, string> = {};
  for (const [key, value] of Object.entries(data)) {
    hashes[key] = sha256(value);
  }
  return hashes;
}

/**
 * Compute delta: keys whose hash differs from known state.
 */
function computeDelta(
  currentHashes: Record<string, string>,
  knownHashes: Record<string, string>,
): string[] {
  const changed: string[] = [];
  for (const [key, hash] of Object.entries(currentHashes)) {
    if (knownHashes[key] !== hash) {
      changed.push(key);
    }
  }
  return changed;
}

// ─── Git Context Detection ────────────────────────────────

function detectGitContext(): GitContext | null {
  try {
    const repoUrl = execSync('git remote get-url origin', { encoding: 'utf-8', stdio: ['pipe', 'pipe', 'pipe'] }).trim();
    const branch = execSync('git branch --show-current', { encoding: 'utf-8', stdio: ['pipe', 'pipe', 'pipe'] }).trim();
    const revision = execSync('git rev-parse HEAD', { encoding: 'utf-8', stdio: ['pipe', 'pipe', 'pipe'] }).trim();
    const remotesRaw = execSync('git remote', { encoding: 'utf-8', stdio: ['pipe', 'pipe', 'pipe'] }).trim();
    const remotes = remotesRaw.split('\n').filter(Boolean);
    
    // Count uncommitted changes
    let uncommittedChanges = 0;
    try {
      const status = execSync('git status --porcelain', { encoding: 'utf-8', stdio: ['pipe', 'pipe', 'pipe'] }).trim();
      uncommittedChanges = status ? status.split('\n').length : 0;
    } catch { /* not in git */ }
    
    // Check push permissions (best effort)
    let hasPushPerms = false;
    try {
      execSync('git push --dry-run 2>&1', { encoding: 'utf-8', stdio: ['pipe', 'pipe', 'pipe'] });
      hasPushPerms = true;
    } catch { /* can't push */ }
    
    return { repoUrl, branch, revision, hasPushPerms, remotes, uncommittedChanges };
  } catch {
    return null;
  }
}

// ─── Export (Create Teleport Package) ─────────────────────

/**
 * Export the current session context for teleportation.
 */
export function exportContext(
  state: TeleportState,
  options: {
    conversationSummary?: string;
    modelOverride?: string;
    customSystemPrompt?: string;
    outcomes?: SessionOutcome[];
    additionalData?: Record<string, string>;
  } = {},
): { context: TeleportContext; state: TeleportState } {
  ensureTeleportDir();
  
  // Collect data
  const data: Record<string, string> = {};
  
  // Include session memory if exists
  const memoryFile = path.join(process.cwd(), '.sable-dev', 'session-memory.md');
  if (fs.existsSync(memoryFile)) {
    data['session-memory'] = fs.readFileSync(memoryFile, 'utf-8');
  }
  
  // Include dream memories if exist
  const dreamFile = path.join(process.cwd(), '.sable-dev', 'dream', 'consolidated-memories.json');
  if (fs.existsSync(dreamFile)) {
    data['dream-memories'] = fs.readFileSync(dreamFile, 'utf-8');
  }
  
  // Include avatar soul if exists
  const avatarFile = path.join(process.cwd(), '.sable-dev', 'avatar.json');
  if (fs.existsSync(avatarFile)) {
    data['avatar-soul'] = fs.readFileSync(avatarFile, 'utf-8');
  }
  
  // Include any additional data
  if (options.additionalData) {
    for (const [key, value] of Object.entries(options.additionalData)) {
      // Validate key to prevent path traversal
      const safeKey = key.replace(/[^a-zA-Z0-9._-]/g, '_');
      data[safeKey] = value;
    }
  }
  
  // Compute hashes & detect delta
  const hashes = computeHashes(data);
  const changedKeys = computeDelta(hashes, state.contentHashes);
  
  // Only include changed data (delta sync)
  const deltaData: Record<string, string> = {};
  for (const key of changedKeys) {
    deltaData[key] = data[key];
  }
  
  const context: TeleportContext = {
    id: `teleport-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
    sourceEnv: getEnvironmentId(),
    exportedAt: Date.now(),
    git: detectGitContext(),
    workingDir: process.cwd(),
    modelOverride: options.modelOverride || null,
    customSystemPrompt: options.customSystemPrompt || null,
    conversationSummary: options.conversationSummary || '',
    outcomes: options.outcomes || [],
    data: deltaData,
    hashes,
    version: 1,
  };
  
  // Save to file
  const exportFile = path.join(TELEPORT_DIR, `${context.id}.json`);
  fs.writeFileSync(exportFile, JSON.stringify(context, null, 2));
  
  // Update state
  const newState: TeleportState = {
    ...state,
    lastExportAt: Date.now(),
    contentHashes: hashes,
    teleportCount: state.teleportCount + 1,
  };
  saveTeleportState(newState);
  
  console.log(`[teleport] Exported context: ${changedKeys.length} changed keys, ${Object.keys(data).length} total`);
  
  return { context, state: newState };
}

/**
 * Export to a portable file (for manual transfer).
 */
export function exportToFile(
  state: TeleportState,
  outputPath: string,
  options: Parameters<typeof exportContext>[1] = {},
): { filePath: string; state: TeleportState } {
  const { context, state: newState } = exportContext(state, options);
  
  const resolvedPath = path.resolve(outputPath);
  fs.writeFileSync(resolvedPath, JSON.stringify(context, null, 2));
  
  return { filePath: resolvedPath, state: newState };
}

// ─── Import (Restore Teleport Package) ───────────────────

/**
 * Import a teleport context from a file.
 */
export function importContext(
  state: TeleportState,
  source: string | TeleportContext,
): { imported: TeleportContext; state: TeleportState; applied: string[] } {
  let context: TeleportContext;
  
  if (typeof source === 'string') {
    // Load from file
    const resolvedPath = path.resolve(source);
    if (!fs.existsSync(resolvedPath)) {
      throw new Error(`Teleport file not found: ${resolvedPath}`);
    }
    context = JSON.parse(fs.readFileSync(resolvedPath, 'utf-8'));
  } else {
    context = source;
  }
  
  // Validate version
  if (context.version !== 1) {
    throw new Error(`Unsupported teleport version: ${context.version}`);
  }
  
  const applied: string[] = [];
  
  // Apply data
  for (const [key, value] of Object.entries(context.data)) {
    // Validate key
    const safeKey = key.replace(/[^a-zA-Z0-9._-]/g, '_');
    
    switch (safeKey) {
      case 'session-memory': {
        const memDir = path.join(process.cwd(), '.sable-dev');
        if (!fs.existsSync(memDir)) fs.mkdirSync(memDir, { recursive: true });
        fs.writeFileSync(path.join(memDir, 'session-memory.md'), value);
        applied.push('session-memory');
        break;
      }
      case 'dream-memories': {
        const dreamDir = path.join(process.cwd(), '.sable-dev', 'dream');
        if (!fs.existsSync(dreamDir)) fs.mkdirSync(dreamDir, { recursive: true });
        fs.writeFileSync(path.join(dreamDir, 'consolidated-memories.json'), value);
        applied.push('dream-memories');
        break;
      }
      case 'avatar-soul': {
        const sableDir = path.join(process.cwd(), '.sable-dev');
        if (!fs.existsSync(sableDir)) fs.mkdirSync(sableDir, { recursive: true });
        fs.writeFileSync(path.join(sableDir, 'avatar.json'), value);
        applied.push('avatar-soul');
        break;
      }
      default: {
        // Store in teleport dir as generic data
        ensureTeleportDir();
        fs.writeFileSync(path.join(TELEPORT_DIR, `imported-${safeKey}`), value);
        applied.push(safeKey);
      }
    }
  }
  
  // Update state 
  const newState: TeleportState = {
    ...state,
    lastImportAt: Date.now(),
    contentHashes: { ...state.contentHashes, ...context.hashes },
    teleportCount: state.teleportCount + 1,
  };
  saveTeleportState(newState);
  
  console.log(`[teleport] Imported context from ${context.sourceEnv}: ${applied.length} items applied`);
  
  return { imported: context, state: newState, applied };
}

// ─── List Available Teleports ─────────────────────────────

/**
 * List available teleport files (both local and imported).
 */
export function listTeleports(): Array<{
  id: string;
  sourceEnv: string;
  exportedAt: number;
  dataKeys: string[];
  filePath: string;
}> {
  ensureTeleportDir();
  
  try {
    return fs.readdirSync(TELEPORT_DIR)
      .filter(f => f.startsWith('teleport-') && f.endsWith('.json'))
      .map(f => {
        try {
          const filePath = path.join(TELEPORT_DIR, f);
          const data = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
          return {
            id: data.id || f.replace('.json', ''),
            sourceEnv: data.sourceEnv || 'unknown',
            exportedAt: data.exportedAt || 0,
            dataKeys: Object.keys(data.data || {}),
            filePath,
          };
        } catch { return null; }
      })
      .filter((x): x is NonNullable<typeof x> => x !== null)
      .sort((a, b) => b.exportedAt - a.exportedAt);
  } catch {
    return [];
  }
}

// ─── Environment Detection ────────────────────────────────

function getEnvironmentId(): string {
  const hostname = os.hostname();
  const cwd = process.cwd();
  const hash = crypto.createHash('md5').update(`${hostname}:${cwd}`).digest('hex').substring(0, 8);
  return `${hostname}:${hash}`;
}

// ─── Status ───────────────────────────────────────────────

/**
 * Get teleport status summary.
 */
export function getTeleportStatus(state: TeleportState): string {
  const lines: string[] = [
    `Teleport System`,
    `Total teleports: ${state.teleportCount}`,
    `Tracked keys: ${Object.keys(state.contentHashes).length}`,
  ];
  
  if (state.lastExportAt) {
    const ago = ((Date.now() - state.lastExportAt) / (1000 * 60)).toFixed(0);
    lines.push(`Last export: ${ago}min ago`);
  }
  
  if (state.lastImportAt) {
    const ago = ((Date.now() - state.lastImportAt) / (1000 * 60)).toFixed(0);
    lines.push(`Last import: ${ago}min ago`);
  }
  
  const available = listTeleports();
  if (available.length > 0) {
    lines.push(`Available snapshots: ${available.length}`);
  }
  
  return lines.join('\n');
}

/**
 * Format teleport context for system prompt injection.
 */
export function formatTeleportContextForPrompt(context: TeleportContext): string {
  const lines: string[] = [
    `## Teleported Session Context`,
    `Source: ${context.sourceEnv}`,
    `Exported: ${new Date(context.exportedAt).toISOString()}`,
  ];
  
  if (context.git) {
    lines.push(`Git: ${context.git.branch} @ ${context.git.revision.substring(0, 8)}`);
    if (context.git.uncommittedChanges > 0) {
      lines.push(`Uncommitted changes: ${context.git.uncommittedChanges}`);
    }
  }
  
  if (context.conversationSummary) {
    lines.push('', `### Previous Session Summary`, context.conversationSummary);
  }
  
  if (context.outcomes.length > 0) {
    lines.push('', `### Session Outcomes`);
    for (const outcome of context.outcomes) {
      lines.push(`- [${outcome.type}] ${outcome.description}`);
    }
  }
  
  if (context.modelOverride) {
    lines.push(`Model override: ${context.modelOverride}`);
  }
  
  return lines.join('\n');
}
