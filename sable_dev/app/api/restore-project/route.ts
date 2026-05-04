/**
 * POST /api/restore-project,  Restore a project from persistent storage.
 * 
 * Creates a new sandbox and writes all saved files into it,
 * then starts the dev server. Returns the new sandbox URL.
 */

import { NextRequest, NextResponse } from 'next/server';
import { loadProjectFiles, loadProjectMeta } from '@/lib/project-store';
import { SandboxFactory } from '@/lib/sandbox/factory';
import { sandboxManager } from '@/lib/sandbox/sandbox-manager';
import { saveSession } from '@/lib/persistence';

declare global {
  var activeSandboxProvider: any;
  var sandboxData: { sandboxId: string; url: string } | null;
  var existingFiles: Set<string>;
  var activeTemplateId: string | null;
}

export async function POST(request: NextRequest) {
  try {
    const { sandboxId } = await request.json();
    
    if (!sandboxId) {
      return NextResponse.json({ success: false, error: 'sandboxId is required' }, { status: 400 });
    }

    console.log(`[restore-project] Restoring project: ${sandboxId}`);

    // Load project metadata and files
    const meta = loadProjectMeta(sandboxId);
    if (!meta) {
      return NextResponse.json({ success: false, error: 'Project not found in store' }, { status: 404 });
    }

    const files = loadProjectFiles(sandboxId);
    if (!files || Object.keys(files).length === 0) {
      return NextResponse.json({ success: false, error: 'No files found for this project' }, { status: 404 });
    }

    console.log(`[restore-project] Found ${Object.keys(files).length} files, template: ${meta.templateId}`);

    // Terminate any existing sandbox
    try {
      await sandboxManager.terminateAll();
    } catch { /* ok */ }

    // Create a new sandbox with the same template
    const templateId = meta.templateId || 'react-spa';
    const provider = await SandboxFactory.create('local');
    const sandboxInfo = await provider.createSandbox();

    // Setup the template (installs deps, starts dev server)
    await provider.setupViteApp(templateId);

    // Now overwrite all template files with saved project files
    let restoredCount = 0;
    for (const [filePath, content] of Object.entries(files)) {
      try {
        await provider.writeFile(filePath, content);
        restoredCount++;
      } catch (e) {
        console.warn(`[restore-project] Failed to restore file: ${filePath}`, e);
      }
    }

    console.log(`[restore-project] Restored ${restoredCount}/${Object.keys(files).length} files`);

    // Update globals
    global.activeSandboxProvider = provider;
    global.sandboxData = {
      sandboxId: sandboxInfo.sandboxId,
      url: sandboxInfo.url,
    };
    global.existingFiles = new Set(Object.keys(files));
    global.activeTemplateId = templateId;

    // Save session
    saveSession();

    return NextResponse.json({
      success: true,
      sandboxId: sandboxInfo.sandboxId,
      url: sandboxInfo.url,
      restoredFiles: restoredCount,
      templateId,
      originalSandboxId: sandboxId,
    });
  } catch (error) {
    console.error('[restore-project] Error:', error);
    return NextResponse.json({
      success: false,
      error: (error as Error).message
    }, { status: 500 });
  }
}
