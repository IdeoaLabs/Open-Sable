/**
 * Computer Use — Desktop Automation
 * 
 * Provides screenshot capture, mouse control, keyboard input, and
 * application detection for desktop automation workflows.
 * 
 * This is a Linux-first implementation using xdotool + scrot for now.
 * Designed for the SableCore workstation environment.
 * 
 * Features:
 * - Screenshot capture (full screen or region)
 * - Mouse movement with animated easing
 * - Keyboard input via xdotool
 * - Clipboard verify protocol (safe paste)
 * - Active window detection
 * - Terminal emulator awareness
 */

import { execSync, exec } from 'child_process';
import fs from 'fs';
import path from 'path';
import crypto from 'crypto';

// ─── Types ────────────────────────────────────────────────

export interface ScreenshotResult {
  /** Path to the saved screenshot */
  path: string;
  /** Base64 encoded image data */
  base64: string;
  /** Image dimensions */
  width: number;
  height: number;
  /** Timestamp */
  capturedAt: number;
}

export interface MousePosition {
  x: number;
  y: number;
}

export interface ComputerUseState {
  /** Whether computer use capabilities are available */
  available: boolean;
  /** Detected display info */
  display: { width: number; height: number } | null;
  /** Current mouse position */
  mousePos: MousePosition | null;
  /** Detected terminal emulator */
  terminalEmulator: string | null;
  /** Screenshot history (last 5 paths) */
  screenshotHistory: string[];
  /** Error log */
  lastError: string | null;
}

export interface ClickOptions {
  button?: 'left' | 'right' | 'middle';
  doubleClick?: boolean;
}

export interface TypeOptions {
  /** Use clipboard paste instead of key-by-key typing */
  useClipboard?: boolean;
  /** Delay between keystrokes in ms (key-by-key mode) */
  delay?: number;
}

// ─── Known Terminal Emulators ─────────────────────────────

const TERMINAL_EMULATORS = [
  'gnome-terminal', 'konsole', 'xfce4-terminal', 'tilix',
  'alacritty', 'kitty', 'wezterm', 'foot', 'st',
  'xterm', 'urxvt', 'terminator', 'guake',
  'code', 'vscodium', // VS Code integrated terminal
];

// ─── Screenshots Directory ────────────────────────────────

const SCREENSHOTS_DIR = path.join(process.cwd(), '.sable-dev', 'screenshots');

function ensureScreenshotsDir(): void {
  if (!fs.existsSync(SCREENSHOTS_DIR)) {
    fs.mkdirSync(SCREENSHOTS_DIR, { recursive: true });
  }
}

// ─── Availability Detection ──────────────────────────────

/**
 * Check which tools are available for computer use.
 */
export function detectCapabilities(): ComputerUseState {
  const state: ComputerUseState = {
    available: false,
    display: null,
    mousePos: null,
    terminalEmulator: null,
    screenshotHistory: [],
    lastError: null,
  };
  
  try {
    // Check if we have a display
    const display = process.env.DISPLAY || process.env.WAYLAND_DISPLAY;
    if (!display) {
      state.lastError = 'No display server detected (no DISPLAY or WAYLAND_DISPLAY)';
      return state;
    }
    
    // Check for xdotool
    try {
      execSync('which xdotool', { stdio: 'pipe' });
    } catch {
      state.lastError = 'xdotool not installed (required for mouse/keyboard control)';
      return state;
    }
    
    // Get display dimensions
    try {
      const output = execSync('xdotool getdisplaygeometry', { encoding: 'utf-8' }).trim();
      const [w, h] = output.split(' ').map(Number);
      if (w && h) {
        state.display = { width: w, height: h };
      }
    } catch { /* non-critical */ }
    
    // Get current mouse position
    try {
      const output = execSync('xdotool getmouselocation --shell', { encoding: 'utf-8' });
      const xMatch = output.match(/X=(\d+)/);
      const yMatch = output.match(/Y=(\d+)/);
      if (xMatch && yMatch) {
        state.mousePos = { x: parseInt(xMatch[1]), y: parseInt(yMatch[1]) };
      }
    } catch { /* non-critical */ }
    
    // Detect terminal emulator
    try {
      const activeWindow = execSync('xdotool getactivewindow getwindowname', { encoding: 'utf-8' }).trim().toLowerCase();
      for (const term of TERMINAL_EMULATORS) {
        if (activeWindow.includes(term)) {
          state.terminalEmulator = term;
          break;
        }
      }
    } catch { /* non-critical */ }
    
    state.available = true;
    
  } catch (err: any) {
    state.lastError = err.message;
  }
  
  return state;
}

// ─── Screenshot ───────────────────────────────────────────

/**
 * Take a screenshot of the entire screen or a region.
 * Uses scrot (or import from ImageMagick as fallback).
 */
export async function takeScreenshot(
  region?: { x: number; y: number; width: number; height: number },
): Promise<ScreenshotResult> {
  ensureScreenshotsDir();
  
  const filename = `screenshot-${Date.now()}.png`;
  const filepath = path.join(SCREENSHOTS_DIR, filename);
  
  // Try scrot first, then import (ImageMagick), then gnome-screenshot
  const tools = [
    {
      name: 'scrot',
      fullCmd: region
        ? `scrot -a ${region.x},${region.y},${region.width},${region.height} "${filepath}"`
        : `scrot "${filepath}"`,
    },
    {
      name: 'import',
      fullCmd: region
        ? `import -window root -crop ${region.width}x${region.height}+${region.x}+${region.y} "${filepath}"`
        : `import -window root "${filepath}"`,
    },
    {
      name: 'gnome-screenshot',
      fullCmd: `gnome-screenshot -f "${filepath}"`,
    },
  ];
  
  let captured = false;
  for (const tool of tools) {
    try {
      execSync(`which ${tool.name}`, { stdio: 'pipe' });
      execSync(tool.fullCmd, { stdio: 'pipe', timeout: 10000 });
      captured = true;
      break;
    } catch { continue; }
  }
  
  if (!captured) {
    throw new Error('No screenshot tool available. Install scrot: sudo apt install scrot');
  }
  
  // Read the captured image
  const imageBuffer = fs.readFileSync(filepath);
  const base64 = imageBuffer.toString('base64');
  
  // Try to get dimensions (approximate from file size if no identify tool)
  let width = 0, height = 0;
  try {
    const dims = execSync(`identify -format "%wx%h" "${filepath}"`, { encoding: 'utf-8' }).trim();
    const [w, h] = dims.split('x').map(Number);
    width = w || 0;
    height = h || 0;
  } catch {
    // Fallback: use display dimensions
    try {
      const output = execSync('xdotool getdisplaygeometry', { encoding: 'utf-8' }).trim();
      const [w, h] = output.split(' ').map(Number);
      width = w || 1920;
      height = h || 1080;
    } catch {
      width = 1920;
      height = 1080;
    }
  }
  
  return {
    path: filepath,
    base64,
    width,
    height,
    capturedAt: Date.now(),
  };
}

// ─── Mouse Control ────────────────────────────────────────

/**
 * Move the mouse to a position with optional animation.
 * Uses ease-out-cubic for smooth movement.
 */
export async function moveMouse(
  x: number,
  y: number,
  animate: boolean = true,
): Promise<void> {
  if (!animate) {
    // Instant move + 50ms settle
    execSync(`xdotool mousemove ${x} ${y}`);
    await sleep(50);
    return;
  }
  
  // Get current position
  let startX = 0, startY = 0;
  try {
    const output = execSync('xdotool getmouselocation --shell', { encoding: 'utf-8' });
    const xMatch = output.match(/X=(\d+)/);
    const yMatch = output.match(/Y=(\d+)/);
    startX = xMatch ? parseInt(xMatch[1]) : 0;
    startY = yMatch ? parseInt(yMatch[1]) : 0;
  } catch { /* start from 0,0 */ }
  
  // Calculate animation duration: proportional to distance, capped at 500ms
  const distance = Math.sqrt((x - startX) ** 2 + (y - startY) ** 2);
  const duration = Math.min(500, Math.max(100, distance * 0.5));
  const steps = Math.max(10, Math.floor(duration / 16)); // ~60fps
  const stepDelay = duration / steps;
  
  // Ease-out-cubic animation
  for (let i = 1; i <= steps; i++) {
    const t = i / steps;
    const eased = 1 - Math.pow(1 - t, 3); // ease-out-cubic
    
    const currentX = Math.round(startX + (x - startX) * eased);
    const currentY = Math.round(startY + (y - startY) * eased);
    
    execSync(`xdotool mousemove ${currentX} ${currentY}`);
    await sleep(stepDelay);
  }
  
  // Final settle
  await sleep(50);
}

/**
 * Click at the current mouse position or at specified coordinates.
 */
export async function click(
  options?: ClickOptions & { x?: number; y?: number },
): Promise<void> {
  const button = options?.button === 'right' ? 3 : options?.button === 'middle' ? 2 : 1;
  
  if (options?.x !== undefined && options?.y !== undefined) {
    await moveMouse(options.x, options.y);
  }
  
  const cmd = options?.doubleClick
    ? `xdotool click --repeat 2 --delay 50 ${button}`
    : `xdotool click ${button}`;
  
  execSync(cmd);
  await sleep(100); // Post-click settle
}

/**
 * Drag from current position to target.
 */
export async function drag(
  toX: number,
  toY: number,
  button: number = 1,
): Promise<void> {
  // Get current position for drag start
  const output = execSync('xdotool getmouselocation --shell', { encoding: 'utf-8' });
  const xMatch = output.match(/X=(\d+)/);
  const yMatch = output.match(/Y=(\d+)/);
  const startX = xMatch ? parseInt(xMatch[1]) : 0;
  const startY = yMatch ? parseInt(yMatch[1]) : 0;
  
  // Press button
  execSync(`xdotool mousedown ${button}`);
  await sleep(50);
  
  // Animated drag
  await moveMouse(toX, toY, true);
  
  // Release button
  execSync(`xdotool mouseup ${button}`);
  await sleep(100);
}

// ─── Keyboard Control ─────────────────────────────────────

/**
 * Type text using keyboard simulation.
 * 
 * Clipboard verify protocol (when useClipboard=true):
 * 1. Save user's clipboard
 * 2. Write text, verify read-back (clipboard writes can fail silently)
 * 3. Ctrl+V
 * 4. Wait 100ms (paste-effect vs restore race)
 * 5. Restore clipboard in finally (even if throw)
 */
export async function typeText(
  text: string,
  options?: TypeOptions,
): Promise<void> {
  if (options?.useClipboard) {
    await typeViaClipboard(text);
  } else {
    // Key-by-key typing via xdotool
    const delay = options?.delay || 12;
    // Use xdotool type with delay
    // Escape single quotes for shell safety
    const escaped = text.replace(/'/g, "'\\''");
    execSync(`xdotool type --delay ${delay} '${escaped}'`);
  }
}

/**
 * Clipboard verify protocol — safe paste
 */
async function typeViaClipboard(text: string): Promise<void> {
  let savedClipboard: string | null = null;
  
  try {
    // 1. Save current clipboard
    try {
      savedClipboard = execSync('xclip -selection clipboard -o', { encoding: 'utf-8', timeout: 2000 });
    } catch {
      savedClipboard = null; // Empty clipboard
    }
    
    // 2. Write text to clipboard
    execSync(`echo -n '${text.replace(/'/g, "'\\''")}' | xclip -selection clipboard`, { timeout: 2000 });
    
    // 2b. Verify read-back (crucial — clipboard writes can fail silently)
    const readBack = execSync('xclip -selection clipboard -o', { encoding: 'utf-8', timeout: 2000 }).trim();
    if (readBack !== text.trim()) {
      throw new Error('Clipboard verify failed: written text does not match read-back');
    }
    
    // 3. Ctrl+V to paste
    execSync('xdotool key ctrl+v');
    
    // 4. Wait for paste effect
    await sleep(100);
    
  } finally {
    // 5. Restore clipboard (even on error)
    try {
      if (savedClipboard !== null) {
        execSync(`echo -n '${savedClipboard.replace(/'/g, "'\\''")}' | xclip -selection clipboard`, { timeout: 2000 });
      }
    } catch { /* best-effort restore */ }
  }
}

/**
 * Press a key combination (e.g. 'ctrl+c', 'alt+tab', 'Return').
 */
export async function pressKey(keys: string): Promise<void> {
  execSync(`xdotool key ${keys}`);
  await sleep(50);
}

/**
 * Hold and release modifier keys (LIFO order for safety).
 */
export async function withModifiers(
  modifiers: string[],
  action: () => Promise<void>,
): Promise<void> {
  const pressed: string[] = [];
  
  try {
    // Press modifiers in order
    for (const mod of modifiers) {
      execSync(`xdotool keydown ${mod}`);
      pressed.push(mod);
      await sleep(20);
    }
    
    // Execute the action
    await action();
    
  } finally {
    // Release in LIFO order (reverse) — ensures no stuck keys
    for (let i = pressed.length - 1; i >= 0; i--) {
      try {
        execSync(`xdotool keyup ${pressed[i]}`);
      } catch { /* best-effort, swallow errors */ }
    }
  }
}

// ─── Window Management ────────────────────────────────────

/**
 * Get the active window information.
 */
export function getActiveWindow(): { id: string; name: string; pid: number } | null {
  try {
    const winId = execSync('xdotool getactivewindow', { encoding: 'utf-8' }).trim();
    const name = execSync(`xdotool getactivewindow getwindowname`, { encoding: 'utf-8' }).trim();
    const pid = parseInt(execSync(`xdotool getactivewindow getwindowpid`, { encoding: 'utf-8' }).trim());
    return { id: winId, name, pid };
  } catch {
    return null;
  }
}

/**
 * Focus a window by name pattern.
 */
export async function focusWindow(namePattern: string): Promise<boolean> {
  try {
    execSync(`xdotool search --name "${namePattern.replace(/"/g, '\\"')}" windowactivate`);
    await sleep(200); // Wait for window to come to front
    return true;
  } catch {
    return false;
  }
}

/**
 * Check if the active window is a terminal.
 */
export function isTerminalActive(): boolean {
  const win = getActiveWindow();
  if (!win) return false;
  
  const nameLower = win.name.toLowerCase();
  return TERMINAL_EMULATORS.some(term => nameLower.includes(term));
}

// ─── Utilities ────────────────────────────────────────────

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Get computer use status summary.
 */
export function getComputerUseStatus(): string {
  const state = detectCapabilities();
  
  if (!state.available) {
    return `Computer Use: unavailable (${state.lastError})`;
  }
  
  return [
    'Computer Use: available',
    state.display ? `Display: ${state.display.width}x${state.display.height}` : 'Display: unknown',
    state.mousePos ? `Mouse: (${state.mousePos.x}, ${state.mousePos.y})` : 'Mouse: unknown',
    state.terminalEmulator ? `Terminal: ${state.terminalEmulator}` : 'Terminal: not detected',
  ].join('\n');
}
