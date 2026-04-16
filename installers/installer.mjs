#!/usr/bin/env node
/**
 * Open-Sable Installer — Browser-based GUI (Windows + Linux + macOS)
 *
 * Usage:
 *   node installer.mjs                    # interactive (opens browser)
 *   node installer.mjs --headless         # non-interactive (terminal only)
 *   node installer.mjs --install-dir ~/opensable
 *
 * Requires: Node.js 18+, git, python3.11+
 */

import { createServer } from 'node:http'
import { execSync, spawn } from 'node:child_process'
import { existsSync, mkdirSync, writeFileSync, readFileSync, copyFileSync, chmodSync, rmSync } from 'node:fs'
import { join, dirname, resolve } from 'node:path'
import { homedir, platform, arch } from 'node:os'
import { fileURLToPath } from 'node:url'
import { createHash } from 'node:crypto'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

// ─── Constants ──────────────────────────────────────────────────────────────
const APP_NAME = 'Open-Sable'
const APP_VERSION = '1.7.0'
const REPO_URL = 'https://github.com/IdeoaLabs/Open-Sable.git'
const REPO_BRANCH = 'master'
const IS_WIN = platform() === 'win32'
const IS_MAC = platform() === 'darwin'
const DEFAULT_INSTALL_DIR = IS_WIN
  ? join(homedir(), 'opensable')
  : join(homedir(), 'opensable')
const INSTALLER_PORT = 18730
const ICON_SRC = join(__dirname, 'assets', 'icon_source.png')
const LOGO_SRC = join(__dirname, 'assets', 'logo.png')
const ICON_ICO = join(__dirname, 'assets', 'icon.ico')
const OLLAMA_WIN_URL = 'https://ollama.com/download/OllamaSetup.exe'

// ─── State ──────────────────────────────────────────────────────────────────
let clients = [] // SSE connections
let installDir = DEFAULT_INSTALL_DIR
let isHeadless = false
let installLog = []
let installStatus = 'idle' // idle | running | done | error
let installProgress = 0
let currentStep = ''
let systemInfo = null

// ─── Parse CLI flags ────────────────────────────────────────────────────────
for (let i = 2; i < process.argv.length; i++) {
  if (process.argv[i] === '--headless') isHeadless = true
  if (process.argv[i] === '--install-dir' && process.argv[i + 1]) {
    installDir = resolve(process.argv[++i])
  }
}

// ─── Logging ────────────────────────────────────────────────────────────────
function log(msg, level = 'info') {
  const entry = { time: Date.now(), msg, level }
  installLog.push(entry)
  // Send to all SSE clients
  const data = JSON.stringify(entry)
  clients = clients.filter(res => {
    try { res.write(`data: ${data}\n\n`); return true }
    catch { return false }
  })
  // Console output
  const prefix = { info: '  ', ok: '✔ ', warn: '⚠ ', error: '✘ ', step: '━━━ ', dim: '  ' }
  if (!isHeadless && level === 'dim') return
  console.log(`${prefix[level] || '  '}${msg}`)
}

function setProgress(pct, step) {
  installProgress = pct
  currentStep = step || currentStep
  const data = JSON.stringify({ progress: pct, step: currentStep, status: installStatus })
  clients = clients.filter(res => {
    try { res.write(`data: ${data}\n\n`); return true }
    catch { return false }
  })
}

// ─── System Detection ───────────────────────────────────────────────────────
function detectSystem() {
  const info = { os: platform(), arch: arch(), python: null, git: null, node: null, ollama: null, winget: null, errors: [] }

  // Python
  const pyCandidates = IS_WIN
    ? ['python', 'python3', 'py -3.13', 'py -3.12', 'py -3.11']
    : ['python3.13', 'python3.12', 'python3.11', 'python3']
  for (const cmd of pyCandidates) {
    try {
      const ver = execSync(`${cmd} --version 2>&1`, { encoding: 'utf-8', timeout: 5000 }).trim().split(' ').pop()
      const [maj, min] = ver.split('.').map(Number)
      if (maj >= 3 && min >= 11) { info.python = { cmd, ver }; break }
    } catch {}
  }
  if (!info.python) info.errors.push('Python 3.11+ required')

  // Git
  try {
    info.git = execSync('git --version 2>&1', { encoding: 'utf-8', timeout: 5000 }).trim().split(' ').pop()
  } catch {
    info.errors.push(IS_WIN ? 'Git not found — install from git-scm.com' : 'Git not found — install with: sudo apt install git')
  }

  // Node
  try {
    const nv = execSync('node --version 2>&1', { encoding: 'utf-8', timeout: 5000 }).trim().replace('v', '')
    if (parseInt(nv) >= 18) info.node = nv
  } catch {}

  // Ollama
  try {
    info.ollama = execSync('ollama --version 2>&1', { encoding: 'utf-8', timeout: 5000 }).trim().split(' ').pop()
  } catch {}

  // Winget (Windows only)
  if (IS_WIN) {
    try {
      execSync('winget --version 2>&1', { encoding: 'utf-8', timeout: 5000 })
      info.winget = true
    } catch {}
  }

  systemInfo = info
  return info
}

// ─── Shell Exec with Streaming ──────────────────────────────────────────────
function run(cmd, opts = {}) {
  return new Promise((resolve, reject) => {
    const shellCmd = IS_WIN ? 'cmd.exe' : 'bash'
    const shellArgs = IS_WIN ? ['/s', '/c', cmd] : ['-c', cmd]
    const shell = spawn(shellCmd, shellArgs, {
      cwd: opts.cwd || installDir,
      env: { ...process.env, PYTHONUNBUFFERED: '1', ...(IS_WIN ? {} : { DEBIAN_FRONTEND: 'noninteractive' }), ...(opts.env || {}) },
      stdio: ['ignore', 'pipe', 'pipe'],
      windowsHide: true,
    })

    let stdout = '', stderr = ''

    shell.stdout.on('data', d => {
      const s = d.toString()
      stdout += s
      s.split('\n').filter(Boolean).forEach(line => log(line, 'dim'))
    })
    shell.stderr.on('data', d => {
      const s = d.toString()
      stderr += s
      // Only show non-trivial stderr
      s.split('\n').filter(l => l.trim() && !l.includes('WARNING') && !l.includes('npm warn')).forEach(line => log(line, 'dim'))
    })

    shell.on('close', code => {
      if (code !== 0 && opts.check !== false) {
        reject(new Error(`Command failed (exit ${code}): ${cmd}\n${stderr.slice(-500)}`))
      } else {
        resolve({ stdout, stderr, code })
      }
    })

    shell.on('error', reject)
  })
}

// ─── Install Steps ──────────────────────────────────────────────────────────
async function doInstall(config) {
  installDir = config.installDir || installDir
  installStatus = 'running'
  installLog = []

  const steps = [
    ['Checking prerequisites', checkPrereqs],
    ['Downloading Open-Sable', cloneRepo],
    ['Creating Python environment', createVenv],
    ['Installing Python dependencies', installPythonDeps],
    ['Installing Ollama', installOllama],
    ['Building Dashboard', buildDashboard],
    ['Building Dev Studio', buildDevStudio],
    ['Configuring environment', configureEnv],
    ['Creating shortcuts & icon', createShortcuts],
    ['Verifying installation', verify],
  ]

  try {
    for (let i = 0; i < steps.length; i++) {
      const [name, fn] = steps[i]
      log(`Step ${i + 1}/${steps.length}: ${name}`, 'step')
      setProgress(Math.round((i / steps.length) * 100), name)
      await fn(config)
    }
    setProgress(100, 'Installation complete!')
    log(`${APP_NAME} installed successfully!`, 'ok')
    log(`Location: ${installDir}`, 'info')
    if (IS_WIN) {
      log(`Start with: cd "${installDir}" && venv\\Scripts\\activate && python -m opensable`, 'info')
    } else {
      log(`Start with: cd ${installDir} && ./start.sh run`, 'info')
    }
    installStatus = 'done'
  } catch (e) {
    log(`Installation failed: ${e.message}`, 'error')
    installStatus = 'error'
  }

  setProgress(installProgress, currentStep)
}

async function checkPrereqs() {
  const info = detectSystem()
  if (info.python) log(`Python ${info.python.ver} ✓`, 'ok')
  if (info.git) log(`Git ${info.git} ✓`, 'ok')
  if (info.node) log(`Node.js ${info.node} ✓`, 'ok')
  else log('Node.js not found — dashboard/desktop/dev-studio will be skipped', 'warn')
  if (info.ollama) log(`Ollama ${info.ollama} ✓`, 'ok')
  else log('Ollama not found — will install', 'info')
  if (info.errors.length) throw new Error(info.errors.join('\n'))
}

async function cloneRepo() {
  if (existsSync(join(installDir, '.git'))) {
    log('Repository exists — pulling latest...', 'info')
    await run(`git fetch origin ${REPO_BRANCH}`, { check: false })
    await run(`git reset --hard origin/${REPO_BRANCH}`, { check: false })
    log('Repository updated', 'ok')
    return
  }

  // Clean stale empty dir
  if (existsSync(installDir)) {
    try {
      const { readdirSync } = await import('node:fs')
      const entries = readdirSync(installDir)
      if (entries.length === 0) {
        rmSync(installDir, { recursive: true })
        log('Removed empty leftover directory', 'dim')
      } else {
        log(`Directory exists — will download archive and merge`, 'warn')
      }
    } catch {}
  }

  // Try git clone first
  if (systemInfo?.git && !existsSync(installDir)) {
    mkdirSync(dirname(installDir), { recursive: true })
    try {
      // Disable SSL verify as fallback for corporate/broken cert chains
      const gitCmd = `git clone --branch ${REPO_BRANCH} --depth 1 ${REPO_URL} "${installDir}"`
      try {
        await run(gitCmd, { cwd: dirname(installDir) })
      } catch (e1) {
        log('git clone failed, retrying with SSL verify disabled...', 'warn')
        await run(`git -c http.sslVerify=false clone --branch ${REPO_BRANCH} --depth 1 ${REPO_URL} "${installDir}"`, { cwd: dirname(installDir) })
      }
      log('Repository cloned', 'ok')
      return
    } catch (e) {
      log(`git clone failed: ${e.message}`, 'warn')
      log('Falling back to archive download...', 'info')
      // Clean up partial clone
      if (existsSync(installDir)) {
        try { rmSync(installDir, { recursive: true, force: true }) } catch {}
      }
    }
  }

  // Zip/tarball fallback — use zip on Windows (no tar/curl needed)
  mkdirSync(installDir, { recursive: true })
  const ts = Date.now()

  if (IS_WIN) {
    // Use PowerShell to download + extract zip (no curl/tar dependency)
    const zipUrl = `https://github.com/IdeoaLabs/Open-Sable/archive/refs/heads/${REPO_BRANCH}.zip`
    const zipFile = join(process.env.TEMP || 'C:\\Temp', `opensable-${ts}.zip`)
    const extractDir = join(process.env.TEMP || 'C:\\Temp', `opensable-extract-${ts}`)
    log('Downloading archive (zip)...', 'info')
    // PowerShell download with TLS 1.2 forced + progress disabled for speed
    await run(`powershell -NoProfile -Command "[Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; $ProgressPreference='SilentlyContinue'; Invoke-WebRequest -Uri '${zipUrl}' -OutFile '${zipFile}'"`, { cwd: dirname(installDir) })
    log('Extracting...', 'info')
    await run(`powershell -NoProfile -Command "Expand-Archive -Path '${zipFile}' -DestinationPath '${extractDir}' -Force"`, { cwd: dirname(installDir) })
    // Move contents from extracted subdir into installDir
    const innerDir = join(extractDir, `Open-Sable-${REPO_BRANCH}`)
    await run(`powershell -NoProfile -Command "Copy-Item -Path '${innerDir}\\*' -Destination '${installDir}' -Recurse -Force"`, { cwd: dirname(installDir) })
    // Cleanup
    try { rmSync(zipFile, { force: true }) } catch {}
    try { rmSync(extractDir, { recursive: true, force: true }) } catch {}
  } else {
    // Linux/macOS: use curl + tar
    const url = `https://github.com/IdeoaLabs/Open-Sable/archive/refs/heads/${REPO_BRANCH}.tar.gz`
    const tmp = `/tmp/opensable-${ts}.tar.gz`
    const extractDir = `/tmp/opensable-extract-${ts}`
    log('Downloading archive...', 'info')
    await run(`curl -fsSL -o "${tmp}" "${url}"`, { cwd: '/tmp' })
    mkdirSync(extractDir, { recursive: true })
    await run(`tar xzf "${tmp}" -C "${extractDir}"`, { cwd: '/tmp' })
    await run(`rsync -a "${extractDir}"/Open-Sable-${REPO_BRANCH}/ "${installDir}/"`, { cwd: '/tmp' })
    rmSync(extractDir, { recursive: true, force: true })
    rmSync(tmp, { force: true })
  }

  // Init git for future updates
  if (systemInfo?.git) {
    await run(`git init`, { check: false })
    await run(`git remote add origin ${REPO_URL}`, { check: false })
    await run(`git fetch --depth 1 origin ${REPO_BRANCH}`, { check: false })
    await run(`git reset --soft origin/${REPO_BRANCH}`, { check: false })
  }
  log('Source code downloaded', 'ok')
}

async function createVenv() {
  const pyCmd = systemInfo.python.cmd
  const venvDir = join(installDir, 'venv')
  const venvPy = IS_WIN ? join(venvDir, 'Scripts', 'python.exe') : join(venvDir, 'bin', 'python')
  if (existsSync(venvPy)) {
    log('Virtual environment already exists', 'ok')
    return
  }
  log(`Creating venv with ${pyCmd}...`, 'info')
  await run(`${pyCmd} -m venv "${venvDir}"`)
  log('Python environment created', 'ok')
}

async function installPythonDeps() {
  const pip = IS_WIN ? join(installDir, 'venv', 'Scripts', 'pip.exe') : join(installDir, 'venv', 'bin', 'pip')
  await run(`"${pip}" install --upgrade pip setuptools wheel -q`, { check: false })
  if (existsSync(join(installDir, 'pyproject.toml'))) {
    await run(`"${pip}" install -e ".[core]" -q`, { check: false })
  }
  if (existsSync(join(installDir, 'requirements.txt'))) {
    await run(`"${pip}" install -r requirements.txt -q`, { check: false })
  }
  log('Python dependencies installed', 'ok')
}

async function installOllama(config) {
  if (systemInfo?.ollama) {
    log('Ollama already installed', 'ok')
    return
  }
  if (config?.skipOllama) {
    log('Ollama installation skipped', 'info')
    return
  }
  log('Installing Ollama...', 'info')
  try {
    if (IS_WIN) {
      const dl = join(process.env.TEMP || 'C:\\Temp', 'OllamaSetup.exe')
      log('Downloading Ollama installer...', 'info')
      await run(`powershell -NoProfile -Command "[Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; $ProgressPreference='SilentlyContinue'; Invoke-WebRequest -Uri '${OLLAMA_WIN_URL}' -OutFile '${dl}'"`, { cwd: dirname(installDir) })
      log('Running Ollama setup...', 'info')
      await run(`"${dl}" /VERYSILENT /NORESTART`, { check: false })
      log('Ollama installed', 'ok')
    } else {
      await run('curl -fsSL https://ollama.com/install.sh | sh')
      log('Ollama installed', 'ok')
    }
  } catch {
    log('Ollama install failed — install manually from ollama.com', 'warn')
  }
}

async function buildDashboard() {
  if (!systemInfo?.node) { log('Skipped (Node.js not found)', 'warn'); return }
  const dashDir = join(installDir, 'dashboard')
  if (!existsSync(join(dashDir, 'package.json'))) { log('No dashboard found', 'dim'); return }
  if (existsSync(join(dashDir, 'dist', 'index.html'))) { log('Dashboard already built', 'ok'); return }
  log('Installing npm dependencies...', 'info')
  await run('npm install --legacy-peer-deps', { cwd: dashDir, check: false })
  log('Building dashboard...', 'info')
  await run('npm run build', { cwd: dashDir, check: false })
  if (existsSync(join(dashDir, 'dist', 'index.html'))) {
    log('Dashboard built', 'ok')
  } else {
    log('Dashboard build failed — can build later with: cd dashboard && npm run build', 'warn')
  }
}

async function buildDevStudio() {
  if (!systemInfo?.node) { log('Skipped (Node.js not found)', 'warn'); return }
  const devDir = join(installDir, 'sable_dev')
  if (!existsSync(join(devDir, 'package.json'))) { log('No Dev Studio found', 'dim'); return }
  log('Installing Dev Studio dependencies...', 'info')
  await run('npm install --legacy-peer-deps', { cwd: devDir, check: false })
  log('Building Dev Studio...', 'info')
  await run('npm run build', { cwd: devDir, check: false })
  log('Dev Studio ready', 'ok')
}

async function configureEnv(config) {
  const envFile = join(installDir, '.env')
  if (existsSync(envFile)) {
    log('.env already exists — preserving', 'ok')
    let env = readFileSync(envFile, 'utf-8')
    if (!env.includes('DEV_STUDIO_ENABLED')) {
      env += '\n\n# Dev Studio (Sable Dev AI app builder)\nDEV_STUDIO_ENABLED=true\n'
      writeFileSync(envFile, env)
      log('Added DEV_STUDIO_ENABLED=true', 'info')
    }
    return
  }
  const example = join(installDir, '.env.example')
  if (existsSync(example)) {
    let env = readFileSync(example, 'utf-8')
    if (config?.llmProvider) {
      env = env.replace(/^LLM_PROVIDER=.*/m, `LLM_PROVIDER=${config.llmProvider}`)
    }
    if (config?.apiKey) {
      const keyMap = { openai: 'OPENAI_API_KEY', anthropic: 'ANTHROPIC_API_KEY', gemini: 'GEMINI_API_KEY', groq: 'GROQ_API_KEY', deepseek: 'DEEPSEEK_API_KEY', ollama: '' }
      const envKey = keyMap[config.llmProvider] || 'OPENAI_API_KEY'
      if (envKey) env = env.replace(new RegExp(`^${envKey}=.*`, 'm'), `${envKey}=${config.apiKey}`)
    }
    if (!env.includes('DEV_STUDIO_ENABLED')) {
      env += '\n\n# Dev Studio (Sable Dev AI app builder)\nDEV_STUDIO_ENABLED=true\n'
    }
    const writeOpts = IS_WIN ? {} : { mode: 0o600 }
    writeFileSync(envFile, env, writeOpts)
    log('.env configured', 'ok')
  } else {
    const content = [
      `LLM_PROVIDER=${config?.llmProvider || 'ollama'}`,
      config?.apiKey ? `${(config.llmProvider || 'openai').toUpperCase()}_API_KEY=${config.apiKey}` : '',
      'WEBCHAT_PORT=8789',
      'DESKTOP_ENABLED=false',
      'DEV_STUDIO_ENABLED=true',
    ].filter(Boolean).join('\n') + '\n'
    const writeOpts = IS_WIN ? {} : { mode: 0o600 }
    writeFileSync(envFile, content, writeOpts)
    log('.env created', 'ok')
  }
}

async function createShortcuts() {
  if (IS_WIN) {
    await _shortcutsWindows()
  } else {
    await _shortcutsLinux()
  }
  log('Icon and shortcuts installed', 'ok')
}

async function _shortcutsWindows() {
  // ── Copy icon.ico to install dir ──
  const iconDest = join(installDir, 'opensable.ico')
  try {
    if (existsSync(ICON_ICO)) copyFileSync(ICON_ICO, iconDest)
  } catch {}

  // ── Create opensable.bat launcher ──
  const batPath = join(installDir, 'opensable.bat')
  writeFileSync(batPath, `@echo off\r\ncd /d "${installDir}"\r\ncall venv\\Scripts\\activate.bat\r\npython -m opensable %*\r\n`)
  log('opensable.bat created', 'ok')

  // ── Create update script ──
  const updateBat = join(installDir, 'opensable-update.bat')
  writeFileSync(updateBat, [
    '@echo off',
    'echo Updating Open-Sable...',
    `cd /d "${installDir}"`,
    `git fetch origin ${REPO_BRANCH}`,
    'git stash --include-untracked 2>nul',
    `git pull --rebase origin ${REPO_BRANCH}`,
    'git stash pop 2>nul',
    'call venv\\Scripts\\activate.bat',
    'pip install -e ".[core]" -q',
    'if exist requirements.txt pip install -r requirements.txt -q',
    'if exist dashboard\\package.json ( cd dashboard && npm install --legacy-peer-deps -q && npm run build && cd .. )',
    'echo Update complete!',
    'pause',
  ].join('\r\n') + '\r\n')
  log('Update script created', 'ok')

  // ── Desktop + Start Menu shortcuts via PowerShell COM ──
  const startMenu = join(process.env.APPDATA || '', 'Microsoft', 'Windows', 'Start Menu', 'Programs')
  const desktop = join(homedir(), 'Desktop')
  for (const [loc, label] of [[startMenu, 'Start Menu'], [desktop, 'Desktop']]) {
    if (!existsSync(loc)) continue
    const lnk = join(loc, 'Open-Sable.lnk')
    const ps = [
      `$ws=New-Object -COM WScript.Shell`,
      `$s=$ws.CreateShortcut('${lnk}')`,
      `$s.TargetPath='cmd.exe'`,
      `$s.Arguments='/k cd /d \\"${installDir}\\" && venv\\Scripts\\activate.bat && python -m opensable'`,
      `$s.WorkingDirectory='${installDir}'`,
      `$s.IconLocation='${iconDest},0'`,
      `$s.Description='Open-Sable - Your Autonomous AI Agent'`,
      `$s.Save()`,
    ].join(';')
    try {
      await run(`powershell -NoProfile -Command "${ps}"`, { cwd: installDir, check: false })
      log(`${label} shortcut created`, 'ok')
    } catch {}
  }
}

async function _shortcutsLinux() {
  // ── Copy icon to install dir ──
  const iconDest = join(installDir, 'opensable.png')
  try {
    if (existsSync(ICON_SRC)) {
      copyFileSync(ICON_SRC, iconDest)
    } else if (existsSync(LOGO_SRC)) {
      copyFileSync(LOGO_SRC, iconDest)
    }
  } catch {}

  // ── .desktop file ──
  const appsDir = join(homedir(), '.local', 'share', 'applications')
  mkdirSync(appsDir, { recursive: true })
  const desktopEntry = `[Desktop Entry]
Name=${APP_NAME}
Comment=Your Autonomous AI Agent - Think, Learn, Act
Exec=bash -c 'cd "${installDir}" && source venv/bin/activate && ./start.sh run'
Icon=${iconDest}
Terminal=true
Type=Application
Categories=Development;Utility;
StartupWMClass=opensable
`
  writeFileSync(join(appsDir, 'opensable.desktop'), desktopEntry)
  log('Desktop entry created', 'ok')

  // ── CLI link ──
  const localBin = join(homedir(), '.local', 'bin')
  mkdirSync(localBin, { recursive: true })
  const cliScript = `#!/bin/bash
cd "${installDir}" && source venv/bin/activate && ./start.sh run "$@"
`
  const cliPath = join(localBin, 'opensable')
  writeFileSync(cliPath, cliScript)
  chmodSync(cliPath, 0o755)
  log('CLI command "opensable" created', 'ok')

  // ── Make start.sh executable ──
  try { chmodSync(join(installDir, 'start.sh'), 0o755) } catch {}
}

async function verify() {
  let ok = 0, total = 0
  const py = IS_WIN ? join(installDir, 'venv', 'Scripts', 'python.exe') : join(installDir, 'venv', 'bin', 'python')
  if (existsSync(py)) { log('Python venv ✓', 'ok'); ok++ } else { log('Python venv missing', 'error') }
  total++
  try {
    const { stdout } = await run(`"${py}" -c "import opensable; print(opensable.__version__)"`, { check: false })
    if (stdout.trim()) { log(`opensable v${stdout.trim()} ✓`, 'ok'); ok++ } else { log('opensable import check — skipped', 'warn') }
  } catch { log('opensable not importable yet', 'warn') }
  total++
  if (existsSync(join(installDir, 'dashboard', 'dist', 'index.html'))) { log('Dashboard ✓', 'ok'); ok++ } else { log('Dashboard not built', 'warn') }
  total++
  if (existsSync(join(installDir, '.env'))) { log('.env ✓', 'ok'); ok++ } else { log('.env missing', 'warn') }
  total++
  const iconFile = IS_WIN ? 'opensable.ico' : 'opensable.png'
  if (existsSync(join(installDir, iconFile))) { log('Icon ✓', 'ok'); ok++ } else { log('Icon missing', 'warn') }
  total++
  log(`Verification: ${ok}/${total} checks passed`, ok === total ? 'ok' : 'warn')
}

// ─── HTML GUI ───────────────────────────────────────────────────────────────
function getHTML() {
  const sys = systemInfo || detectSystem()
  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>${APP_NAME} Installer</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  :root {
    --bg: #0a0a0f; --bg2: #12121e; --bg3: #1a1a2e;
    --border: #1e1e3a; --text: #f0f0f5; --text2: #a0a0b8; --text3: #6a6a80;
    --accent: #6c5ce7; --accent2: #a29bfe; --green: #00b894; --red: #e17055; --yellow: #fdcb6e;
  }
  body { font-family: 'Inter', -apple-system, sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; display: flex; align-items: center; justify-content: center; padding: 20px; }
  .installer { max-width: 700px; width: 100%; }
  .header { text-align: center; margin-bottom: 32px; }
  .header img { width: 80px; height: 80px; border-radius: 20px; margin-bottom: 16px; filter: drop-shadow(0 0 30px rgba(108,92,231,0.4)); }
  .header h1 { font-size: 28px; font-weight: 800; margin-bottom: 4px; }
  .header h1 span { background: linear-gradient(135deg, #6c5ce7, #a29bfe); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
  .header p { color: var(--text2); font-size: 14px; }
  .card { background: var(--bg2); border: 1px solid var(--border); border-radius: 16px; padding: 28px; margin-bottom: 16px; }
  .card h3 { font-size: 16px; font-weight: 700; margin-bottom: 16px; display: flex; align-items: center; gap: 8px; }
  .card h3 .n { background: var(--accent); color: white; width: 24px; height: 24px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 700; }
  .sys-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(140px, 100%), 1fr)); gap: 10px; margin-bottom: 20px; }
  .sys-item { background: var(--bg3); border-radius: 10px; padding: 12px; text-align: center; }
  .sys-item .label { font-size: 11px; color: var(--text3); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }
  .sys-item .val { font-size: 14px; font-weight: 600; }
  .sys-item .val.ok { color: var(--green); }
  .sys-item .val.warn { color: var(--yellow); }
  .sys-item .val.err { color: var(--red); }
  .field { margin-bottom: 16px; }
  .field label { display: block; font-size: 13px; font-weight: 600; margin-bottom: 6px; color: var(--text2); }
  .field input, .field select { width: 100%; padding: 10px 14px; background: var(--bg); border: 1px solid var(--border); border-radius: 10px; color: var(--text); font-size: 14px; outline: none; font-family: inherit; }
  .field input:focus, .field select:focus { border-color: var(--accent); }
  .field small { display: block; margin-top: 4px; font-size: 11px; color: var(--text3); }
  .btn { display: inline-flex; align-items: center; justify-content: center; gap: 8px; padding: 14px 32px; border-radius: 12px; font-weight: 700; font-size: 15px; cursor: pointer; border: none; transition: all 0.3s; font-family: inherit; width: 100%; }
  .btn-primary { background: linear-gradient(135deg, #6c5ce7, #a29bfe); color: white; box-shadow: 0 4px 20px rgba(108,92,231,0.3); }
  .btn-primary:hover { transform: translateY(-1px); box-shadow: 0 8px 30px rgba(108,92,231,0.5); }
  .btn-primary:disabled { opacity: 0.5; cursor: not-allowed; transform: none; box-shadow: none; }
  .btn-success { background: linear-gradient(135deg, #00b894, #55efc4); color: #0a0a0f; }
  .progress-wrap { margin-bottom: 16px; }
  .progress-bar { height: 6px; background: var(--bg); border-radius: 3px; overflow: hidden; margin-top: 8px; }
  .progress-fill { height: 100%; background: linear-gradient(90deg, #6c5ce7, #00b894); border-radius: 3px; transition: width 0.4s; width: 0%; }
  .progress-label { display: flex; justify-content: space-between; font-size: 12px; color: var(--text3); }
  .log { background: var(--bg); border: 1px solid var(--border); border-radius: 10px; padding: 14px; max-height: 300px; overflow-y: auto; font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 12px; line-height: 1.7; }
  .log .l { padding: 1px 0; }
  .log .l-ok { color: var(--green); }
  .log .l-error { color: var(--red); }
  .log .l-warn { color: var(--yellow); }
  .log .l-step { color: var(--accent2); font-weight: 700; margin-top: 8px; }
  .log .l-dim { color: var(--text3); }
  .log .l-info { color: var(--text2); }
  .chk { display: flex; align-items: center; gap: 8px; margin-bottom: 10px; font-size: 14px; cursor: pointer; }
  .chk input { accent-color: var(--accent); width: 16px; height: 16px; }
  .page { display: none; }
  .page.active { display: block; }
  .done-box { text-align: center; padding: 40px 20px; }
  .done-box .icon { font-size: 64px; margin-bottom: 16px; }
  .done-box h2 { font-size: 24px; margin-bottom: 8px; }
  .done-box p { color: var(--text2); margin-bottom: 24px; }
  .done-box code { background: var(--bg3); padding: 12px 20px; border-radius: 10px; display: block; margin: 16px auto; font-family: monospace; font-size: 14px; color: var(--accent2); max-width: 400px; }
</style>
</head>
<body>
<div class="installer">
  <div class="header">
    <div style="font-size:48px;margin-bottom:12px;">🐍</div>
    <h1><span>${APP_NAME}</span></h1>
    <p>v${APP_VERSION} — Installer for ${IS_WIN ? 'Windows' : IS_MAC ? 'macOS' : 'Linux'}</p>
  </div>

  <!-- Page 1: Config -->
  <div id="page-config" class="page active">
    <div class="card">
      <h3><span class="n">1</span> System Check</h3>
      <div class="sys-grid">
        <div class="sys-item"><div class="label">Python</div><div class="val ${sys.python ? 'ok' : 'err'}">${sys.python ? sys.python.ver : 'Not found'}</div></div>
        <div class="sys-item"><div class="label">Git</div><div class="val ${sys.git ? 'ok' : 'err'}">${sys.git || 'Not found'}</div></div>
        <div class="sys-item"><div class="label">Node.js</div><div class="val ${sys.node ? 'ok' : 'warn'}">${sys.node || 'Not found'}</div></div>
        <div class="sys-item"><div class="label">Ollama</div><div class="val ${sys.ollama ? 'ok' : 'warn'}">${sys.ollama || 'Not found'}</div></div>
      </div>
      ${sys.errors.length ? `<div style="padding:10px 14px;background:rgba(225,112,85,0.1);border:1px solid rgba(225,112,85,0.3);border-radius:8px;color:var(--red);font-size:13px;margin-bottom:12px;">${sys.errors.join('<br>')}</div>` : ''}
    </div>

    <div class="card">
      <h3><span class="n">2</span> Configuration</h3>
      <div class="field">
        <label>Install Directory</label>
        <input type="text" id="installDir" value="${installDir}">
      </div>
      <div class="field">
        <label>LLM Provider</label>
        <select id="llmProvider">
          <option value="ollama" selected>Ollama (Free, Local)</option>
          <option value="openai">OpenAI</option>
          <option value="anthropic">Anthropic</option>
          <option value="gemini">Google Gemini</option>
          <option value="groq">Groq</option>
          <option value="deepseek">DeepSeek</option>
        </select>
      </div>
      <div class="field" id="apiKeyField" style="display:none">
        <label>API Key</label>
        <input type="password" id="apiKey" placeholder="sk-...">
        <small>Stored locally in .env — never sent anywhere</small>
      </div>
      <label class="chk"><input type="checkbox" id="installOllama" ${sys.ollama ? '' : 'checked'}> Install Ollama (local AI models)</label>
      <label class="chk"><input type="checkbox" id="buildDev" checked> Build Dev Studio (AI app builder)</label>
    </div>

    <button class="btn btn-primary" onclick="startInstall()" ${sys.errors.length ? 'disabled' : ''}>
      Install ${APP_NAME}
    </button>
  </div>

  <!-- Page 2: Progress -->
  <div id="page-progress" class="page">
    <div class="card">
      <div class="progress-wrap">
        <div class="progress-label"><span id="stepLabel">Starting...</span><span id="pctLabel">0%</span></div>
        <div class="progress-bar"><div class="progress-fill" id="progressBar"></div></div>
      </div>
      <div class="log" id="logBox"></div>
    </div>
  </div>

  <!-- Page 3: Done -->
  <div id="page-done" class="page">
    <div class="card">
      <div class="done-box" id="doneContent"></div>
    </div>
  </div>
</div>

<script>
  // SSE connection for real-time logs
  const es = new EventSource('/events')
  const logBox = document.getElementById('logBox')
  const progressBar = document.getElementById('progressBar')
  const stepLabel = document.getElementById('stepLabel')
  const pctLabel = document.getElementById('pctLabel')
  const llmSelect = document.getElementById('llmProvider')
  const apiKeyField = document.getElementById('apiKeyField')

  llmSelect.addEventListener('change', () => {
    apiKeyField.style.display = llmSelect.value === 'ollama' ? 'none' : 'block'
  })

  es.onmessage = e => {
    const d = JSON.parse(e.data)
    if (d.progress !== undefined) {
      progressBar.style.width = d.progress + '%'
      pctLabel.textContent = d.progress + '%'
      if (d.step) stepLabel.textContent = d.step
      if (d.status === 'done') showDone(true)
      if (d.status === 'error') showDone(false)
      return
    }
    if (d.msg) {
      const div = document.createElement('div')
      div.className = 'l l-' + (d.level || 'info')
      div.textContent = d.msg
      logBox.appendChild(div)
      logBox.scrollTop = logBox.scrollHeight
    }
  }

  function showPage(name) {
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'))
    document.getElementById('page-' + name).classList.add('active')
  }

  function startInstall() {
    const config = {
      installDir: document.getElementById('installDir').value,
      llmProvider: llmSelect.value,
      apiKey: document.getElementById('apiKey')?.value || '',
      skipOllama: !document.getElementById('installOllama').checked,
      buildDev: document.getElementById('buildDev').checked,
    }
    showPage('progress')
    fetch('/install', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(config) })
  }

  function showDone(success) {
    showPage('done')
    const dir = document.getElementById('installDir').value
    const isWin = navigator.platform.startsWith('Win')
    const startCmd = isWin
      ? 'cd /d ' + dir.replace(/</g,'&lt;') + ' && venv\\\\Scripts\\\\activate && python -m opensable'
      : 'cd ' + dir.replace(/</g,'&lt;') + ' && ./start.sh run'
    const shortcutNote = isWin
      ? 'Desktop and Start Menu shortcuts have been created.'
      : 'A desktop shortcut and CLI command have been created.<br>You can also run <b>opensable</b> from any terminal.'
    document.getElementById('doneContent').innerHTML = success
      ? '<div class="icon">✅</div><h2>Installation Complete!</h2><p>${APP_NAME} has been installed successfully.</p><code>' + startCmd + '</code><p style="margin-top:20px;font-size:13px;color:var(--text3)">' + shortcutNote + '</p>'
      : '<div class="icon">❌</div><h2>Installation Failed</h2><p>Check the logs above for details.</p><button class="btn btn-primary" onclick="showPage(\\'progress\\')" style="max-width:300px;margin:0 auto;">View Logs</button>'
  }
</script>
</body>
</html>`
}

// ─── HTTP Server ────────────────────────────────────────────────────────────
function startServer() {
  const server = createServer((req, res) => {
    const url = new URL(req.url, `http://localhost:${INSTALLER_PORT}`)

    if (url.pathname === '/events') {
      res.writeHead(200, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Access-Control-Allow-Origin': '*',
      })
      clients.push(res)
      req.on('close', () => { clients = clients.filter(c => c !== res) })
      return
    }

    if (url.pathname === '/install' && req.method === 'POST') {
      let body = ''
      req.on('data', c => body += c)
      req.on('end', () => {
        res.writeHead(200, { 'Content-Type': 'application/json' })
        res.end('{"ok":true}')
        try {
          const config = JSON.parse(body)
          doInstall(config)
        } catch (e) {
          log(`Invalid config: ${e.message}`, 'error')
        }
      })
      return
    }

    if (url.pathname === '/status') {
      res.writeHead(200, { 'Content-Type': 'application/json' })
      res.end(JSON.stringify({ status: installStatus, progress: installProgress, step: currentStep, system: systemInfo }))
      return
    }

    // Serve HTML
    res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' })
    res.end(getHTML())
  })

  server.listen(INSTALLER_PORT, '127.0.0.1', () => {
    const url = `http://127.0.0.1:${INSTALLER_PORT}`
    console.log(`\n  🐍 ${APP_NAME} Installer v${APP_VERSION}`)
    console.log(`  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`)
    console.log(`  🌐 Open in browser: ${url}\n`)

    if (!isHeadless) {
      // Auto-open browser
      try {
        const openCmd = IS_WIN ? 'start' : IS_MAC ? 'open' : 'xdg-open'
        if (IS_WIN) {
          spawn('cmd.exe', ['/c', 'start', url], { detached: true, stdio: 'ignore', windowsHide: true }).unref()
        } else {
          spawn(openCmd, [url], { detached: true, stdio: 'ignore' }).unref()
        }
      } catch {}
    }
  })

  // Graceful shutdown
  process.on('SIGINT', () => { server.close(); process.exit(0) })
  process.on('SIGTERM', () => { server.close(); process.exit(0) })
}

// ─── Headless Mode ──────────────────────────────────────────────────────────
async function headlessInstall() {
  detectSystem()
  await doInstall({ installDir, skipOllama: false, buildDev: true })
  process.exit(installStatus === 'done' ? 0 : 1)
}

// ─── Main ───────────────────────────────────────────────────────────────────
detectSystem()
if (isHeadless) {
  headlessInstall()
} else {
  startServer()
}
