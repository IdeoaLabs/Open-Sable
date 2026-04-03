/**
 * File Change Detection
 * 
 * Tracks file modifications between AI turns.
 * Detects external edits (user editing while AI is working).
 * Uses simple content hashing to compare states.
 */

import { createHash } from 'crypto';

// ─── Types ────────────────────────────────────────────────

export interface FileSnapshot {
  path: string;
  hash: string;
  size: number;
  timestamp: number;
}

export interface FileChange {
  path: string;
  type: 'created' | 'modified' | 'deleted';
  previousHash?: string;
  currentHash?: string;
}

export interface FileTrackingState {
  /** Map of path → snapshot at last checkpoint */
  snapshots: Map<string, FileSnapshot>;
  /** History of detected changes per turn */
  changeHistory: Array<{
    turn: number;
    timestamp: number;
    changes: FileChange[];
  }>;
  /** Current turn number */
  currentTurn: number;
}

// ─── Hashing ──────────────────────────────────────────────

/**
 * Compute a fast hash of file content.
 */
function hashContent(content: string): string {
  return createHash('sha256').update(content).digest('hex').substring(0, 16);
}

// ─── State Management ─────────────────────────────────────

/**
 * Create initial file tracking state.
 */
export function createFileTrackingState(): FileTrackingState {
  return {
    snapshots: new Map(),
    changeHistory: [],
    currentTurn: 0,
  };
}

/**
 * Take a snapshot of a file's current state.
 */
export function snapshotFile(
  state: FileTrackingState,
  path: string,
  content: string,
): FileTrackingState {
  const snapshot: FileSnapshot = {
    path,
    hash: hashContent(content),
    size: content.length,
    timestamp: Date.now(),
  };

  const newSnapshots = new Map(state.snapshots);
  newSnapshots.set(path, snapshot);

  return { ...state, snapshots: newSnapshots };
}

/**
 * Take snapshots of multiple files at once.
 */
export function snapshotFiles(
  state: FileTrackingState,
  files: Array<{ path: string; content: string }>,
): FileTrackingState {
  let current = state;
  for (const file of files) {
    current = snapshotFile(current, file.path, file.content);
  }
  return current;
}

// ─── Change Detection ─────────────────────────────────────

/**
 * Detect changes for a single file against the stored snapshot.
 */
export function detectFileChange(
  state: FileTrackingState,
  path: string,
  currentContent: string | null,
): FileChange | null {
  const snapshot = state.snapshots.get(path);

  if (!snapshot && currentContent !== null) {
    return { path, type: 'created', currentHash: hashContent(currentContent) };
  }

  if (snapshot && currentContent === null) {
    return { path, type: 'deleted', previousHash: snapshot.hash };
  }

  if (snapshot && currentContent !== null) {
    const currentHash = hashContent(currentContent);
    if (currentHash !== snapshot.hash) {
      return {
        path,
        type: 'modified',
        previousHash: snapshot.hash,
        currentHash,
      };
    }
  }

  return null; // No change
}

/**
 * Detect changes across multiple files.
 * Pass the current state of files to compare against snapshots.
 */
export function detectChanges(
  state: FileTrackingState,
  currentFiles: Array<{ path: string; content: string | null }>,
): FileChange[] {
  const changes: FileChange[] = [];

  for (const file of currentFiles) {
    const change = detectFileChange(state, file.path, file.content);
    if (change) changes.push(change);
  }

  // Also check for files that were in snapshot but not in currentFiles (deleted)
  const currentPaths = new Set(currentFiles.map(f => f.path));
  for (const [path, snapshot] of state.snapshots) {
    if (!currentPaths.has(path)) {
      changes.push({ path, type: 'deleted', previousHash: snapshot.hash });
    }
  }

  return changes;
}

/**
 * Advance to next turn and record any changes detected.
 */
export function advanceTurnWithChanges(
  state: FileTrackingState,
  changes: FileChange[],
): FileTrackingState {
  const newTurn = state.currentTurn + 1;

  return {
    ...state,
    currentTurn: newTurn,
    changeHistory: [
      ...state.changeHistory,
      { turn: newTurn, timestamp: Date.now(), changes },
    ],
  };
}

// ─── Queries ──────────────────────────────────────────────

/**
 * Get all files that have been tracked.
 */
export function getTrackedFiles(state: FileTrackingState): string[] {
  return Array.from(state.snapshots.keys());
}

/**
 * Get the last N turns of changes.
 */
export function getRecentChanges(state: FileTrackingState, lastN: number = 5): FileChange[] {
  const recent = state.changeHistory.slice(-lastN);
  return recent.flatMap(h => h.changes);
}

/**
 * Get files modified in the current session.
 */
export function getModifiedFilesInSession(state: FileTrackingState): string[] {
  const files = new Set<string>();
  for (const entry of state.changeHistory) {
    for (const change of entry.changes) {
      if (change.type !== 'deleted') {
        files.add(change.path);
      }
    }
  }
  return Array.from(files);
}

// ─── Formatting ───────────────────────────────────────────

/**
 * Format changes for injection into the AI context.
 */
export function formatChangesForPrompt(changes: FileChange[]): string {
  if (changes.length === 0) return '';

  const lines = changes.map(c => {
    switch (c.type) {
      case 'created': return `+ ${c.path} (new file)`;
      case 'modified': return `~ ${c.path} (modified)`;
      case 'deleted': return `- ${c.path} (deleted)`;
    }
  });

  return `FILES CHANGED SINCE LAST TURN:\n${lines.join('\n')}`;
}

/**
 * Format a summary of all session changes.
 */
export function formatSessionChangeSummary(state: FileTrackingState): string {
  const modified = getModifiedFilesInSession(state);
  if (modified.length === 0) return 'No files changed this session.';

  return `Files modified this session (${modified.length}):\n${modified.map(f => `  ${f}`).join('\n')}`;
}
