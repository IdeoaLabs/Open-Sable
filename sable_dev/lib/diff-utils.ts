/**
 * Diff Utilities
 * 
 * Generates structured diffs between file versions.
 * Used to show what changed during code generation/editing.
 */

import { createPatch, structuredPatch } from 'diff';

export interface FileDiff {
  path: string;
  type: 'added' | 'modified' | 'deleted';
  hunks: DiffHunk[];
  additions: number;
  deletions: number;
}

export interface DiffHunk {
  oldStart: number;
  oldLines: number;
  newStart: number;
  newLines: number;
  lines: DiffLine[];
}

export interface DiffLine {
  type: 'add' | 'remove' | 'context';
  content: string;
  oldLineNumber?: number;
  newLineNumber?: number;
}

/**
 * Generate a structured diff between two versions of a file
 */
export function generateFileDiff(
  path: string,
  oldContent: string | null,
  newContent: string | null
): FileDiff {
  if (oldContent === null && newContent !== null) {
    // New file
    const lines = newContent.split('\n');
    return {
      path,
      type: 'added',
      hunks: [{
        oldStart: 0,
        oldLines: 0,
        newStart: 1,
        newLines: lines.length,
        lines: lines.map((line, i) => ({
          type: 'add' as const,
          content: line,
          newLineNumber: i + 1,
        })),
      }],
      additions: lines.length,
      deletions: 0,
    };
  }

  if (newContent === null && oldContent !== null) {
    // Deleted file
    const lines = oldContent.split('\n');
    return {
      path,
      type: 'deleted',
      hunks: [{
        oldStart: 1,
        oldLines: lines.length,
        newStart: 0,
        newLines: 0,
        lines: lines.map((line, i) => ({
          type: 'remove' as const,
          content: line,
          oldLineNumber: i + 1,
        })),
      }],
      additions: 0,
      deletions: lines.length,
    };
  }

  // Modified file
  const patch = structuredPatch(path, path, oldContent || '', newContent || '', '', '', { context: 3 });

  let additions = 0;
  let deletions = 0;

  const hunks: DiffHunk[] = patch.hunks.map(hunk => {
    const lines: DiffLine[] = [];
    let oldLine = hunk.oldStart;
    let newLine = hunk.newStart;

    for (const line of hunk.lines) {
      if (line.startsWith('+')) {
        additions++;
        lines.push({ type: 'add', content: line.substring(1), newLineNumber: newLine++ });
      } else if (line.startsWith('-')) {
        deletions++;
        lines.push({ type: 'remove', content: line.substring(1), oldLineNumber: oldLine++ });
      } else {
        lines.push({ type: 'context', content: line.substring(1), oldLineNumber: oldLine++, newLineNumber: newLine++ });
      }
    }

    return {
      oldStart: hunk.oldStart,
      oldLines: hunk.oldLines,
      newStart: hunk.newStart,
      newLines: hunk.newLines,
      lines,
    };
  });

  return {
    path,
    type: 'modified',
    hunks,
    additions,
    deletions,
  };
}

/**
 * Generate a unified diff string (for display or logging)
 */
export function generateUnifiedDiff(
  path: string,
  oldContent: string,
  newContent: string
): string {
  return createPatch(path, oldContent, newContent, '', '', { context: 3 });
}
