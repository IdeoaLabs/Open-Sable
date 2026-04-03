/**
 * Project Store — Persistent file storage for sandbox projects.
 * 
 * Saves ALL project files to .sable-dev/projects/{sandboxId}/ on disk
 * so they survive process restarts and can be fully restored.
 * 
 * Inspired by professional code editor persistence patterns:
 * - Files stored as real files on disk (not JSON blobs)
 * - Snapshot on every file write for crash recovery
 * - Full restore recreates sandbox with all files intact
 */

import fs from 'fs';
import path from 'path';

const PROJECTS_DIR = path.join(process.cwd(), '.sable-dev', 'projects');

export interface ProjectSnapshot {
  sandboxId: string;
  templateId: string;
  files: Record<string, string>;
  createdAt: number;
  updatedAt: number;
}

function ensureProjectsDir(): void {
  if (!fs.existsSync(PROJECTS_DIR)) {
    fs.mkdirSync(PROJECTS_DIR, { recursive: true });
  }
}

function getProjectDir(sandboxId: string): string {
  // Sanitize sandboxId for filesystem safety
  const safe = sandboxId.replace(/[^a-zA-Z0-9_-]/g, '_');
  return path.join(PROJECTS_DIR, safe);
}

/**
 * Save a single file for a project
 */
export function saveProjectFile(sandboxId: string, filePath: string, content: string): boolean {
  try {
    ensureProjectsDir();
    const projectDir = getProjectDir(sandboxId);
    const normalizedPath = filePath.startsWith('/') ? filePath.slice(1) : filePath;
    
    // Skip node_modules, .git, dist, build directories
    if (normalizedPath.startsWith('node_modules/') || 
        normalizedPath.startsWith('.git/') ||
        normalizedPath.startsWith('dist/') ||
        normalizedPath.startsWith('build/') ||
        normalizedPath.startsWith('.vite/')) {
      return true; // Skip silently
    }

    const fullPath = path.join(projectDir, 'files', normalizedPath);
    
    // Security: ensure we stay within project directory
    const resolved = path.resolve(fullPath);
    if (!resolved.startsWith(path.resolve(projectDir))) {
      console.error('[project-store] Path traversal blocked:', filePath);
      return false;
    }

    fs.mkdirSync(path.dirname(fullPath), { recursive: true });
    fs.writeFileSync(fullPath, content, 'utf-8');
    
    // Update metadata
    updateProjectMeta(sandboxId, { updatedAt: Date.now() });
    return true;
  } catch (error) {
    console.error('[project-store] Failed to save file:', filePath, error);
    return false;
  }
}

/**
 * Save all files for a project at once (bulk snapshot)
 */
export function saveProjectSnapshot(sandboxId: string, templateId: string, files: Record<string, string>): boolean {
  try {
    ensureProjectsDir();
    const projectDir = getProjectDir(sandboxId);
    const filesDir = path.join(projectDir, 'files');

    // Clear existing files directory (fresh snapshot)
    if (fs.existsSync(filesDir)) {
      fs.rmSync(filesDir, { recursive: true, force: true });
    }
    fs.mkdirSync(filesDir, { recursive: true });

    // Write all files
    let fileCount = 0;
    for (const [filePath, content] of Object.entries(files)) {
      const normalizedPath = filePath.startsWith('/') ? filePath.slice(1) : filePath;
      
      // Skip non-essential directories
      if (normalizedPath.startsWith('node_modules/') || 
          normalizedPath.startsWith('.git/') ||
          normalizedPath.startsWith('dist/') ||
          normalizedPath.startsWith('build/') ||
          normalizedPath.startsWith('.vite/')) {
        continue;
      }

      const fullPath = path.join(filesDir, normalizedPath);
      const resolved = path.resolve(fullPath);
      if (!resolved.startsWith(path.resolve(projectDir))) continue;

      fs.mkdirSync(path.dirname(fullPath), { recursive: true });
      fs.writeFileSync(fullPath, content, 'utf-8');
      fileCount++;
    }

    // Write metadata
    const meta = {
      sandboxId,
      templateId,
      fileCount,
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };
    fs.writeFileSync(path.join(projectDir, 'meta.json'), JSON.stringify(meta, null, 2), 'utf-8');
    
    console.log(`[project-store] Saved snapshot for ${sandboxId}: ${fileCount} files`);
    return true;
  } catch (error) {
    console.error('[project-store] Failed to save snapshot:', error);
    return false;
  }
}

/**
 * Load all files for a project from disk
 */
export function loadProjectFiles(sandboxId: string): Record<string, string> | null {
  try {
    const projectDir = getProjectDir(sandboxId);
    const filesDir = path.join(projectDir, 'files');
    
    if (!fs.existsSync(filesDir)) {
      console.log(`[project-store] No stored files for ${sandboxId}`);
      return null;
    }

    const files: Record<string, string> = {};
    
    function readDirRecursive(dir: string, prefix: string) {
      const entries = fs.readdirSync(dir, { withFileTypes: true });
      for (const entry of entries) {
        const relativePath = prefix ? `${prefix}/${entry.name}` : entry.name;
        const fullPath = path.join(dir, entry.name);
        
        if (entry.isDirectory()) {
          readDirRecursive(fullPath, relativePath);
        } else if (entry.isFile()) {
          try {
            files[relativePath] = fs.readFileSync(fullPath, 'utf-8');
          } catch {
            // Skip binary files or unreadable files
          }
        }
      }
    }
    
    readDirRecursive(filesDir, '');
    console.log(`[project-store] Loaded ${Object.keys(files).length} files for ${sandboxId}`);
    return files;
  } catch (error) {
    console.error('[project-store] Failed to load files:', error);
    return null;
  }
}

/**
 * Load project metadata
 */
export function loadProjectMeta(sandboxId: string): { sandboxId: string; templateId: string; fileCount: number; createdAt: number; updatedAt: number } | null {
  try {
    const projectDir = getProjectDir(sandboxId);
    const metaPath = path.join(projectDir, 'meta.json');
    
    if (!fs.existsSync(metaPath)) return null;
    return JSON.parse(fs.readFileSync(metaPath, 'utf-8'));
  } catch {
    return null;
  }
}

/**
 * List all stored projects
 */
export function listStoredProjects(): Array<{ sandboxId: string; templateId: string; fileCount: number; createdAt: number; updatedAt: number }> {
  try {
    ensureProjectsDir();
    const entries = fs.readdirSync(PROJECTS_DIR, { withFileTypes: true });
    const projects: Array<{ sandboxId: string; templateId: string; fileCount: number; createdAt: number; updatedAt: number }> = [];
    
    for (const entry of entries) {
      if (!entry.isDirectory()) continue;
      const metaPath = path.join(PROJECTS_DIR, entry.name, 'meta.json');
      if (!fs.existsSync(metaPath)) continue;
      try {
        const meta = JSON.parse(fs.readFileSync(metaPath, 'utf-8'));
        projects.push(meta);
      } catch { /* skip corrupt entries */ }
    }
    
    return projects.sort((a, b) => b.updatedAt - a.updatedAt);
  } catch {
    return [];
  }
}

/**
 * Delete a stored project
 */
export function deleteStoredProject(sandboxId: string): boolean {
  try {
    const projectDir = getProjectDir(sandboxId);
    if (fs.existsSync(projectDir)) {
      fs.rmSync(projectDir, { recursive: true, force: true });
      console.log(`[project-store] Deleted project ${sandboxId}`);
      return true;
    }
    return false;
  } catch (error) {
    console.error('[project-store] Failed to delete project:', error);
    return false;
  }
}

function updateProjectMeta(sandboxId: string, updates: Partial<{ updatedAt: number; fileCount: number }>): void {
  try {
    const projectDir = getProjectDir(sandboxId);
    const metaPath = path.join(projectDir, 'meta.json');
    
    if (!fs.existsSync(metaPath)) return;
    
    const meta = JSON.parse(fs.readFileSync(metaPath, 'utf-8'));
    Object.assign(meta, updates);
    fs.writeFileSync(metaPath, JSON.stringify(meta, null, 2), 'utf-8');
  } catch { /* silent */ }
}
