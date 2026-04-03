import { SandboxProvider, SandboxInfo, CommandResult, SandboxProviderConfig } from '../types';
import { spawn, exec, ChildProcess } from 'child_process';
import * as fs from 'fs/promises';
import * as fsSync from 'fs';
import * as path from 'path';
import * as os from 'os';
import * as net from 'net';
import { randomUUID } from 'crypto';
import { getTemplate, DEFAULT_TEMPLATE, type TemplateId, type ProjectTemplate } from '../templates';
import { saveProjectFile, saveProjectSnapshot } from '../../project-store';

// Global Vite error buffer,  accessible by API routes
declare global {
  var lastViteError: { error: string; file: string; timestamp: number } | null;
}
if (!global.lastViteError) global.lastViteError = null;

/**
 * LocalProcessProvider - Runs sandboxed projects on the local filesystem.
 * 
 * Uses Node.js built-ins (fs, child_process, os) to create isolated
 * project directories in the system temp folder, install dependencies,
 * and run a Vite dev server on a dynamic port.
 * 
 * NO Docker, No external services, No API keys needed.
 */
export class LocalProcessProvider extends SandboxProvider {
  private workDir: string = '';
  private _sandboxId: string = '';
  private devServerPort: number = 5173;
  private devServerProcess: ChildProcess | null = null;
  private childProcesses: ChildProcess[] = [];
  private _alive: boolean = false;
  private existingFiles: Set<string> = new Set();
  private activeTemplate: ProjectTemplate | null = null;

  async createSandbox(): Promise<SandboxInfo> {
    // Create temp directory
    this.workDir = await fs.mkdtemp(path.join(os.tmpdir(), 'sable-dev-sandbox-'));
    this._sandboxId = `local-${randomUUID().slice(0, 8)}`;
    this._alive = true;

    // Find available port
    this.devServerPort = await this.findAvailablePort(5173);
    const url = `http://localhost:${this.devServerPort}`;

    console.log(`[LocalProcessProvider] Created sandbox ${this._sandboxId} at ${this.workDir}`);
    console.log(`[LocalProcessProvider] Dev server will run on port ${this.devServerPort}`);

    this.sandboxInfo = {
      sandboxId: this._sandboxId,
      url,
      provider: 'local',
      createdAt: new Date()
    };

    return this.sandboxInfo;
  }

  async runCommand(command: string): Promise<CommandResult> {
    if (!this.workDir) {
      throw new Error('No active sandbox');
    }

    return new Promise((resolve) => {
      exec(command, {
        cwd: this.workDir,
        maxBuffer: 10 * 1024 * 1024, // 10MB
        timeout: 120000, // 2 minutes
        env: {
          ...process.env,
          HOME: process.env.HOME || os.homedir(),
          PATH: process.env.PATH,
          NODE_ENV: 'development',
        }
      }, (error, stdout, stderr) => {
        resolve({
          stdout: stdout || '',
          stderr: stderr || '',
          exitCode: error ? (error as any).code || 1 : 0,
          success: !error
        });
      });
    });
  }

  async writeFile(filePath: string, content: string): Promise<void> {
    if (!this.workDir) {
      throw new Error('No active sandbox');
    }

    // Normalize the path - strip leading slashes for relative paths
    const normalizedPath = filePath.startsWith('/') ? filePath.slice(1) : filePath;
    const fullPath = path.join(this.workDir, normalizedPath);

    // Security: ensure we don't write outside the sandbox
    const resolvedPath = path.resolve(fullPath);
    if (!resolvedPath.startsWith(this.workDir)) {
      throw new Error(`Path traversal detected: ${filePath}`);
    }

    // Ensure directory exists
    await fs.mkdir(path.dirname(fullPath), { recursive: true });
    await fs.writeFile(fullPath, content, 'utf-8');
    
    this.existingFiles.add(normalizedPath);

    // Persist to project store for crash recovery and restore
    saveProjectFile(this._sandboxId, normalizedPath, content);
  }

  async readFile(filePath: string): Promise<string> {
    if (!this.workDir) {
      throw new Error('No active sandbox');
    }

    const normalizedPath = filePath.startsWith('/') ? filePath.slice(1) : filePath;
    const fullPath = path.join(this.workDir, normalizedPath);

    // Security check
    const resolvedPath = path.resolve(fullPath);
    if (!resolvedPath.startsWith(this.workDir)) {
      throw new Error(`Path traversal detected: ${filePath}`);
    }

    return await fs.readFile(fullPath, 'utf-8');
  }

  async listFiles(directory?: string): Promise<string[]> {
    if (!this.workDir) {
      throw new Error('No active sandbox');
    }

    const dir = directory
      ? path.join(this.workDir, directory)
      : this.workDir;

    try {
      const result = await this.runCommand(
        `find ${dir} -type f -not -path "*/node_modules/*" -not -path "*/.git/*" -not -path "*/dist/*" -not -path "*/build/*" | sed "s|^${this.workDir}/||"`
      );
      
      if (result.success) {
        return result.stdout.split('\n').filter(line => line.trim() !== '');
      }
      return [];
    } catch {
      return [];
    }
  }

  async installPackages(packages: string[]): Promise<CommandResult> {
    if (!this.workDir) {
      throw new Error('No active sandbox');
    }

    const command = `npm install ${packages.join(' ')}`;
    console.log(`[LocalProcessProvider] Installing packages: ${packages.join(', ')}`);
    
    const result = await this.runCommand(command);
    
    if (result.success) {
      console.log('[LocalProcessProvider] ✓ Packages installed');
    } else {
      console.warn('[LocalProcessProvider] ⚠ Package install had issues:', result.stderr.slice(0, 200));
    }

    return result;
  }

  getSandboxUrl(): string | null {
    return this.sandboxInfo?.url || null;
  }

  getSandboxInfo(): SandboxInfo | null {
    return this.sandboxInfo;
  }

  async terminate(): Promise<void> {
    console.log(`[LocalProcessProvider] Terminating sandbox ${this._sandboxId}...`);
    this._alive = false;

    // Kill dev server process
    if (this.devServerProcess) {
      try {
        // Try to kill the process group (detached processes)
        if (this.devServerProcess.pid) {
          process.kill(-this.devServerProcess.pid, 'SIGTERM');
        }
      } catch {
        try {
          this.devServerProcess.kill('SIGTERM');
        } catch { /* already dead */ }
      }
      this.devServerProcess = null;
    }

    // Kill all tracked child processes
    for (const proc of this.childProcesses) {
      try {
        proc.kill('SIGTERM');
      } catch { /* already dead */ }
    }
    this.childProcesses = [];

    // Also kill any processes running in the sandbox directory
    try {
      exec(`pkill -f "${this.workDir}" || true`, { timeout: 5000 }, () => {});
    } catch { /* ok if no processes found */ }

    // Clean up temp directory
    if (this.workDir) {
      try {
        await fs.rm(this.workDir, { recursive: true, force: true });
        console.log(`[LocalProcessProvider] ✓ Cleaned up ${this.workDir}`);
      } catch (e) {
        console.error('[LocalProcessProvider] Failed to clean up:', e);
      }
    }

    this.sandboxInfo = null;
  }

  isAlive(): boolean {
    return this._alive;
  }

  async setupViteApp(templateId?: string): Promise<void> {
    if (!this.workDir) {
      throw new Error('No active sandbox - call createSandbox() first');
    }

    const template = getTemplate(templateId || DEFAULT_TEMPLATE);
    this.activeTemplate = template;
    console.log(`[LocalProcessProvider] Setting up project with template: ${template.name} (${template.id})`);

    // Get files from the template
    const files = template.getFiles(this.devServerPort);

    // Write all project files
    for (const [filePath, content] of Object.entries(files)) {
      await this.writeFile(filePath, content);
    }
    console.log('[LocalProcessProvider] ✓ Project files created');

    // Inject console capture script into index.html for DevTools console panel
    await this.injectConsoleCapture();

    // Install dependencies
    console.log('[LocalProcessProvider] Installing dependencies (this may take a moment)...');
    const installCmd = this.activeTemplate?.installCommand || 'npm install --loglevel warn';
    const installResult = await this.runCommand(installCmd);
    if (installResult.success) {
      console.log('[LocalProcessProvider] ✓ Dependencies installed');
    } else {
      console.warn('[LocalProcessProvider] ⚠ npm install had issues:', installResult.stderr.slice(0, 500));
      // Try again with legacy peer deps
      console.log('[LocalProcessProvider] Retrying with --legacy-peer-deps...');
      await this.runCommand('npm install --legacy-peer-deps');
    }

    // Start Vite dev server
    console.log('[LocalProcessProvider] Starting Vite dev server...');
    await this.startDevServer();

    // Wait for the server to be ready
    await this.waitForDevServer();
    console.log(`[LocalProcessProvider] ✓ Dev server ready at http://localhost:${this.devServerPort}`);

    // Save initial project snapshot to persistent store
    try {
      const allFiles = await this.listFiles();
      const fileContents: Record<string, string> = {};
      for (const f of allFiles) {
        try {
          fileContents[f] = await this.readFile(f);
        } catch { /* skip binary/unreadable */ }
      }
      saveProjectSnapshot(this._sandboxId, template.id, fileContents);
    } catch (e) {
      console.warn('[LocalProcessProvider] Could not save initial snapshot:', e);
    }
  }

  async restartViteServer(): Promise<void> {
    console.log('[LocalProcessProvider] Restarting Vite dev server...');

    // Kill existing dev server
    if (this.devServerProcess) {
      try {
        if (this.devServerProcess.pid) {
          process.kill(-this.devServerProcess.pid, 'SIGTERM');
        }
      } catch {
        try { this.devServerProcess.kill('SIGTERM'); } catch { /* ok */ }
      }
      this.devServerProcess = null;
    }

    // Also kill any vite processes in the sandbox dir
    try {
      await this.runCommand('pkill -f "vite" || true');
    } catch { /* ok */ }

    // Wait a moment for ports to free up
    await new Promise(r => setTimeout(r, 2000));

    // Restart
    await this.startDevServer();
    await this.waitForDevServer();
    console.log('[LocalProcessProvider] ✓ Dev server restarted');
  }

  // ==========================================
  // Private helpers
  // ==========================================

  /**
   * Inject a console capture script into the sandbox's index.html.
   * This overrides console.log/warn/error/info and sends messages
   * to the parent window via postMessage for the Console Panel.
   */
  private async injectConsoleCapture(): Promise<void> {
    try {
      const indexPath = 'index.html';
      const html = await this.readFile(indexPath);
      
      const consoleScript = `
<script>
// Sable Dev Console Capture
(function() {
  const origLog = console.log;
  const origWarn = console.warn;
  const origError = console.error;
  const origInfo = console.info;
  
  function send(method, args) {
    try {
      const serialized = Array.from(args).map(function(a) {
        if (a === null) return 'null';
        if (a === undefined) return 'undefined';
        if (typeof a === 'object') {
          try { return JSON.stringify(a); } catch(e) { return String(a); }
        }
        return String(a);
      });
      window.parent.postMessage({ type: 'sable-console', method: method, args: serialized }, '*');
    } catch(e) {}
  }
  
  console.log = function() { send('log', arguments); origLog.apply(console, arguments); };
  console.warn = function() { send('warn', arguments); origWarn.apply(console, arguments); };
  console.error = function() { send('error', arguments); origError.apply(console, arguments); };
  console.info = function() { send('info', arguments); origInfo.apply(console, arguments); };
  
  // Capture unhandled errors
  window.addEventListener('error', function(e) {
    send('error', [e.message + ' at ' + (e.filename || '') + ':' + (e.lineno || '')]);
  });
  window.addEventListener('unhandledrejection', function(e) {
    send('error', ['Unhandled Promise Rejection: ' + (e.reason?.message || e.reason || 'unknown')]);
  });
})();
</script>`;

      // Insert before </head> or before </body>
      let modified = html;
      if (html.includes('</head>')) {
        modified = html.replace('</head>', consoleScript + '\n</head>');
      } else if (html.includes('</body>')) {
        modified = html.replace('</body>', consoleScript + '\n</body>');
      } else {
        modified = consoleScript + '\n' + html;
      }

      await this.writeFile(indexPath, modified);
      console.log('[LocalProcessProvider] ✓ Console capture script injected');
    } catch (e) {
      console.warn('[LocalProcessProvider] Could not inject console capture:', e);
    }
  }

  private startDevServer(): Promise<void> {
    return new Promise((resolve) => {
      // Use the template's dev command, or fall back to vite
      const devCommand = this.activeTemplate
        ? this.activeTemplate.getDevCommand(this.devServerPort)
        : `npx vite --host --port ${this.devServerPort}`;
      
      console.log(`[LocalProcessProvider] Starting dev server: ${devCommand}`);
      
      // Split the command for spawn
      const parts = devCommand.split(' ');
      const cmd = parts[0];
      const args = parts.slice(1);

      // Add node_modules/.bin to PATH so locally installed binaries (vite, etc.) are found
      const localBinPath = path.join(this.workDir, 'node_modules', '.bin');
      const envPath = `${localBinPath}${path.delimiter}${process.env.PATH || ''}`;

      const child = spawn(cmd, args, {
        cwd: this.workDir,
        env: {
          ...process.env,
          PATH: envPath,
          HOME: process.env.HOME || os.homedir(),
          NODE_ENV: 'development',
        },
        stdio: ['pipe', 'pipe', 'pipe'],
        detached: true
      });

      this.devServerProcess = child;
      this.childProcesses.push(child);

      let resolved = false;

      const onData = (data: Buffer) => {
        const output = data.toString();
        for (const line of output.split('\n')) {
          if (line.trim()) {
            console.log('[LocalProcessProvider:dev]', line.trim());
          }
        }

        // Capture Vite compile errors for auto-fix
        if (output.includes('[plugin:vite') || output.includes('SyntaxError') || output.includes('Unexpected token') || output.includes('Missing semicolon') || output.includes('Unterminated') || output.includes('Failed to resolve import') || output.includes('parse5 error') || output.includes('Unable to parse HTML')) {
          // Extract file path from error
          const fileMatch = output.match(/\/([^\s:]+\.(jsx|tsx|js|ts|css|html))/); 
          const fileName = fileMatch ? fileMatch[1].split('/').pop() || 'unknown' : 'unknown';
          global.lastViteError = {
            error: output.slice(0, 1500),
            file: fileName,
            timestamp: Date.now()
          };
          console.log('[LocalProcessProvider] Captured Vite compile error in:', fileName);
        }

        // Clear error when Vite successfully recompiles (HMR update or page reload)
        if (global.lastViteError && (output.includes('hmr update') || output.includes('page reload') || output.includes('vite:hmr'))) {
          console.log('[LocalProcessProvider] Vite recompiled successfully,  clearing error');
          global.lastViteError = null;
        }

        // Detect server ready: Vite prints "Local:", Express prints "listening", Next.js prints "Ready"
        if (!resolved && (output.includes('Local:') || output.includes('localhost') || output.includes('listening') || output.includes('Ready'))) {
          resolved = true;
          // Clear any stale errors on successful start
          global.lastViteError = null;
          resolve();
        }
      };

      child.stdout?.on('data', onData);
      child.stderr?.on('data', onData);

      child.on('error', (err) => {
        console.error('[LocalProcessProvider] Dev server spawn error:', err);
        if (!resolved) {
          resolved = true;
          resolve(); // Resolve anyway to not hang forever
        }
      });

      child.on('exit', (code) => {
        if (code !== null && code !== 0) {
          console.error(`[LocalProcessProvider] Dev server exited with code ${code}`);
        }
        if (!resolved) {
          resolved = true;
          resolve();
        }
      });

      // Safety timeout - don't wait more than 60s
      setTimeout(() => {
        if (!resolved) {
          console.warn('[LocalProcessProvider] Dev server start timeout (60s) - continuing anyway');
          resolved = true;
          resolve();
        }
      }, 60000);
    });
  }

  private async waitForDevServer(maxRetries = 30): Promise<void> {
    for (let i = 0; i < maxRetries; i++) {
      try {
        const response = await fetch(`http://localhost:${this.devServerPort}`, {
          signal: AbortSignal.timeout(2000)
        });
        if (response.ok || response.status === 304) {
          return; // Server is ready
        }
      } catch {
        // Not ready yet, wait and retry
      }
      await new Promise(r => setTimeout(r, 1000));
    }
    console.warn('[LocalProcessProvider] Dev server health check timed out, continuing anyway');
  }

  private findAvailablePort(start: number): Promise<number> {
    return new Promise((resolve) => {
      const server = net.createServer();
      server.listen(start, () => {
        server.close(() => resolve(start));
      });
      server.on('error', () => {
        this.findAvailablePort(start + 1).then(resolve);
      });
    });
  }

  /**
   * Get the sandbox working directory path.
   * Useful for debugging and for the compatibility wrapper.
   */
  getWorkDir(): string {
    return this.workDir;
  }

  /**
   * Returns the active template ID or null if no template was set.
   */
  getActiveTemplateId(): string | null {
    return this.activeTemplate?.id || null;
  }

  /**
   * Returns the active template's system prompt addition for AI context.
   */
  getTemplateSystemPrompt(): string {
    return this.activeTemplate?.systemPromptAddition || '';
  }

  /**
   * Returns the active template's file format instructions for AI context.
   */
  getTemplateFileFormat(): string {
    return this.activeTemplate?.fileFormatInstructions || '';
  }

  /**
   * Creates a compatibility wrapper that mimics the Vercel Sandbox SDK API.
   * This allows legacy API routes that use `global.activeSandbox.runCommand({cmd, args})`
   * to work seamlessly with the local provider.
   */
  createLegacyCompatWrapper(): any {
    // eslint-disable-next-line @typescript-eslint/no-this-alias
    const provider = this;
    const workDir = this.workDir;

    return {
      sandboxId: this._sandboxId,

      /**
       * Mimics Vercel SDK's runCommand({cmd, args, cwd, env, detached})
       * Returns { exitCode, stdout: () => Promise<string>, stderr: () => Promise<string> }
       */
      runCommand: async (opts: {
        cmd: string;
        args?: string[];
        cwd?: string;
        env?: Record<string, string>;
        detached?: boolean;
      }) => {
        const { cmd, args = [], cwd, env = {}, detached = false } = opts;

        // Build the full command string
        const fullCommand = [cmd, ...args].join(' ');
        
        // Determine working directory
        // Legacy routes expect /vercel/sandbox as the default cwd
        // Map that to our local workDir
        let effectiveCwd = workDir;
        if (cwd && cwd !== '/vercel/sandbox' && cwd !== '/') {
          effectiveCwd = path.join(workDir, cwd);
        }

        if (detached) {
          // Start process in background (like vite dev server)
          // Add node_modules/.bin to PATH so locally installed binaries are found
          const spawnLocalBin = path.join(workDir, 'node_modules', '.bin');
          const spawnPath = `${spawnLocalBin}${path.delimiter}${process.env.PATH || ''}`;

          const child = spawn(cmd, args, {
            cwd: effectiveCwd,
            env: { ...process.env, PATH: spawnPath, ...env },
            detached: true,
            stdio: ['pipe', 'pipe', 'pipe']
          });

          provider.childProcesses.push(child);

          let stdoutData = '';
          let stderrData = '';
          child.stdout?.on('data', (d: Buffer) => { stdoutData += d.toString(); });
          child.stderr?.on('data', (d: Buffer) => { stderrData += d.toString(); });

          return {
            exitCode: 0,
            stdout: async () => stdoutData,
            stderr: async () => stderrData
          };
        }

        // Synchronous execution
        return new Promise((resolve) => {
          exec(fullCommand, {
            cwd: effectiveCwd,
            maxBuffer: 10 * 1024 * 1024,
            timeout: 120000,
            env: { ...process.env, ...env }
          }, (error, stdout, stderr) => {
            const stdoutStr = stdout || '';
            const stderrStr = stderr || '';
            resolve({
              exitCode: error ? (error as any).code || 1 : 0,
              stdout: async () => stdoutStr,
              stderr: async () => stderrStr
            });
          });
        });
      },

      /**
       * Mimics Vercel SDK's writeFiles([{path, content: Buffer}])
       */
      writeFiles: async (files: Array<{ path: string; content: Buffer }>) => {
        for (const file of files) {
          const normalizedPath = file.path.startsWith('/') ? file.path.slice(1) : file.path;
          const fullPath = path.join(workDir, normalizedPath);

          // Ensure directory exists
          fsSync.mkdirSync(path.dirname(fullPath), { recursive: true });
          fsSync.writeFileSync(fullPath, file.content);
        }
      },

      /**
       * Mimics Vercel SDK's domain(port) method
       */
      domain: (port: number) => {
        return `http://localhost:${provider.devServerPort}`;
      },

      /**
       * Mimics Vercel SDK's stop() method
       */
      stop: async () => {
        await provider.terminate();
      }
    };
  }
}
