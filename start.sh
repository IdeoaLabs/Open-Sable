#!/bin/bash
# Open-Sable,  start / stop / status (all agents live in agents/)
# Usage:
#   ./start.sh                         → start default agent (sable)
#   ./start.sh stop                    → stop default agent (sable)
#   ./start.sh restart                 → restart default agent
#   ./start.sh status                  → check if running
#   ./start.sh logs                    → tail live logs
#   ./start.sh start --profile NAME    → start a named profile agent
#   ./start.sh stop --profile NAME     → stop a named profile agent
#   ./start.sh restart --profile NAME  → restart a named profile agent
#   ./start.sh status --profile NAME   → check if profile is running
#   ./start.sh logs --profile NAME     → tail profile logs
#   ./start.sh profiles               → list available profiles
#   ./start.sh restart --all           → restart ALL agent profiles
#   ./start.sh stop --all              → stop ALL agent profiles
#   ./start.sh start --all             → start ALL agent profiles

DIR="$(cd "$(dirname "$0")" && pwd)"

# Default profile,  all agents live in agents/
DEFAULT_PROFILE="sable"

# Parse --profile and --all flags from any position
PROFILE=""
ACTION=""
ALL_PROFILES=0
for arg in "$@"; do
    if [[ "$arg" == "--profile" ]]; then
        NEXT_IS_PROFILE=1
        continue
    fi
    if [[ "$NEXT_IS_PROFILE" == "1" ]]; then
        PROFILE="$arg"
        NEXT_IS_PROFILE=0
        continue
    fi
    if [[ "$arg" == "--all" ]]; then
        ALL_PROFILES=1
        continue
    fi
    if [[ -z "$ACTION" ]]; then
        ACTION="$arg"
    fi
done
ACTION="${ACTION:-start}"

# If no profile specified, use default
PROFILE="${PROFILE:-$DEFAULT_PROFILE}"

# Set file paths based on profile
PIDFILE="$DIR/.sable-${PROFILE}.pid"
LOGFILE="$DIR/logs/sable-${PROFILE}.log"
PROFILE_DIR="$DIR/agents/$PROFILE"

cd "$DIR"

# Check venv
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Run: python install.py"
    exit 1
fi
source venv/bin/activate

is_running() {
    if [ -f "$PIDFILE" ]; then
        pid=$(cat "$PIDFILE")
        if kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
        rm -f "$PIDFILE"
    fi
    return 1
}

ensure_aggr() {
    # Auto-install Aggr.trade if not built yet
    local aggrdir="$DIR/aggr"
    if [ -f "$aggrdir/dist/index.html" ]; then
        return 0
    fi
    if ! command -v node &>/dev/null || ! command -v npm &>/dev/null; then
        echo "⏭️  Aggr.trade skipped (Node.js not found)"
        return 0
    fi
    echo "📈 Installing Aggr.trade charts..."
    if [ ! -d "$aggrdir" ]; then
        git clone --depth=1 https://github.com/Tucsky/aggr.git "$aggrdir" || return 0
    fi
    if [ ! -d "$aggrdir/templates" ]; then
        git clone --depth=1 https://github.com/0xd3lbow/aggr.template.git "$aggrdir/templates" 2>/dev/null
    fi
    # Create .env.local with production CORS proxy
    if [ ! -f "$aggrdir/.env.local" ]; then
        cat > "$aggrdir/.env.local" << 'AGGRENV'
VITE_APP_PROXY_URL=https://cors.aggr.trade/
VITE_APP_API_URL=https://api.aggr.trade/
VITE_APP_LIB_URL=https://lib.aggr.trade/
VITE_APP_LIB_REPO_URL=https://github.com/Tucsky/aggr-lib
VITE_APP_BASE_PATH=/aggr/
VITE_APP_API_SUPPORTED_TIMEFRAMES=5,10,15,30,60,180,300,900,1260,1800,3600,7200,14400,21600,28800,43200,86400
AGGRENV
    fi
    (cd "$aggrdir" && npm install && npm run build) || {
        echo "⚠️  Aggr.trade build failed,  continuing without it"
        return 0
    }
    # Strip tracking (GTM, analytics, etc)
    patch_aggr_tracking "$aggrdir"
    [ -f "$aggrdir/dist/index.html" ] && echo "✅ Aggr.trade ready" || echo "⚠️  Aggr.trade dist not found"
}

patch_aggr_tracking() {
    local dir="$1"
    # Use portable sed -i (macOS needs '' arg, GNU does not)
    _sedi() { if [[ "$OSTYPE" == darwin* ]]; then sed -i '' "$@"; else sed -i "$@"; fi; }
    for f in "$dir/index.html" "$dir/dist/index.html"; do
        [ -f "$f" ] || continue
        # Remove GTM script block
        _sedi '/<!-- Google Tag Manager -->/,/<!-- End Google Tag Manager -->/d' "$f"
        # Remove GTM noscript block
        _sedi '/<!-- Google Tag Manager (noscript) -->/,/<!-- End Google Tag Manager (noscript) -->/d' "$f"
        # Remove any remaining GTM iframes/scripts
        _sedi '/googletagmanager/d' "$f"
        # Remove google-analytics
        _sedi '/google-analytics/d' "$f"
        # Remove zunvra / other third-party analytics
        _sedi '/zunvra/d' "$f"
        # Fix base href for /aggr/ subpath
        _sedi 's|<base href="/" />|<base href="/aggr/" />|g' "$f"
    done
    echo "   \U0001f6e1  Tracking stripped + base href fixed"
}

ensure_dashboard() {
    # Auto-build React dashboard if not built yet
    local dashdir="$DIR/dashboard"
    if [ -f "$dashdir/dist/index.html" ]; then
        return 0
    fi
    if [ ! -f "$dashdir/package.json" ]; then
        echo "⏭️  Dashboard skipped (folder not found)"
        return 0
    fi
    if ! command -v node &>/dev/null || ! command -v npm &>/dev/null; then
        echo "⏭️  Dashboard skipped (Node.js not found)"
        return 0
    fi
    echo "📊 Building React Dashboard..."
    (cd "$dashdir" && npm install && npm run build) || {
        echo "⚠️  Dashboard build failed,  continuing without it"
        return 0
    }
    [ -f "$dashdir/dist/index.html" ] && echo "✅ Dashboard ready" || echo "⚠️  Dashboard dist not found"
}

ensure_marketplace() {
    # Auto-install marketplace server deps if needed
    local srvdir="$DIR/marketplace/server"
    local clidir="$DIR/marketplace/client"
    if [ ! -f "$srvdir/package.json" ]; then
        return 0
    fi
    if ! command -v node &>/dev/null; then
        return 0
    fi
    # Install server deps if node_modules missing
    if [ ! -d "$srvdir/node_modules" ]; then
        echo "🏪 Installing Marketplace server..."
        (cd "$srvdir" && npm install) || echo "⚠️  Marketplace server install failed"
    fi
    # Build client if not built
    if [ -f "$clidir/package.json" ] && [ ! -f "$clidir/build/index.html" ]; then
        echo "🏪 Building Marketplace client..."
        (cd "$clidir" && npm install && npm run build) || echo "⚠️  Marketplace client build failed"
    fi
}

start_desktop() {
    # Start the Sable Desktop Electron app if DESKTOP_ENABLED=true
    local desktop_enabled
    desktop_enabled=$(grep -E '^DESKTOP_ENABLED=' "$DIR/.env" 2>/dev/null | cut -d= -f2 | tr -d '[:space:]' | tr '[:upper:]' '[:lower:]')
    if [ "$desktop_enabled" != "true" ]; then
        return 0
    fi

    local deskdir="$DIR/desktop"
    if [ ! -f "$deskdir/package.json" ]; then
        echo "⚠️  Desktop folder not found,  set DESKTOP_ENABLED=false"
        return 0
    fi

    if ! command -v npm &>/dev/null; then
        echo "⚠️  npm not found,  desktop requires Node.js"
        return 0
    fi

    if ! command -v electron &>/dev/null && [ ! -f "$deskdir/node_modules/.bin/electron" ]; then
        echo "🖥️  Installing Desktop dependencies..."
        (cd "$deskdir" && npm install --silent) || {
            echo "⚠️  Desktop npm install failed,  skipping"
            return 0
        }
    fi

    # Auto-build renderer if not built
    if [ ! -f "$deskdir/dist/index.html" ]; then
        echo "🖥️  Building Desktop..."
        (cd "$deskdir" && npm run build) || {
            echo "⚠️  Desktop build failed,  skipping"
            return 0
        }
    fi

    # Read gateway config
    local webchat_port webchat_host webchat_token
    webchat_port=$(grep -E '^WEBCHAT_PORT=' "$DIR/.env" 2>/dev/null | cut -d= -f2 | tr -d '[:space:]')
    webchat_host=$(grep -E '^WEBCHAT_HOST=' "$DIR/.env" 2>/dev/null | cut -d= -f2 | tr -d '[:space:]')
    webchat_token=$(grep -E '^WEBCHAT_TOKEN=' "$DIR/.env" 2>/dev/null | cut -d= -f2 | tr -d '[:space:]')

    # Kill ANY existing desktop instances before starting (prevents duplicates on restart)
    pkill -9 -f "electron.*$(basename "$deskdir")" 2>/dev/null
    sleep 1
    # Double-check nothing is still alive
    if pgrep -f "electron.*$(basename "$deskdir")" >/dev/null 2>&1; then
        pkill -9 -f "electron.*$(basename "$deskdir")" 2>/dev/null
        sleep 1
    fi

    echo "🖥️  Starting Desktop..."
    WEBCHAT_PORT="${webchat_port:-8789}" \
    WEBCHAT_HOST="${webchat_host:-localhost}" \
    WEBCHAT_TOKEN="${webchat_token}" \
    nohup "$deskdir/node_modules/.bin/electron" "$deskdir" >> "$DIR/logs/desktop.log" 2>&1 &
    echo $! > "$DIR/.desktop.pid"
    echo "✅ Desktop started (PID $(cat "$DIR/.desktop.pid"))"
}

stop_desktop() {
    local pidfile="$DIR/.desktop.pid"
    if [ -f "$pidfile" ]; then
        local pid
        pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            echo "🛑 Stopping Desktop Agent (PID $pid)..."
            kill "$pid" 2>/dev/null
            sleep 2
            kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null
        fi
        rm -f "$pidfile"
    fi
}

start_dev_studio() {
    # Start Sable Dev Studio (Next.js app builder) if DEV_STUDIO_ENABLED=true
    local dev_enabled
    dev_enabled=$(grep -E '^DEV_STUDIO_ENABLED=' "$DIR/.env" 2>/dev/null | cut -d= -f2 | tr -d '[:space:]' | tr '[:upper:]' '[:lower:]')
    if [ "$dev_enabled" != "true" ]; then
        return 0
    fi

    local devdir="$DIR/sable_dev"
    if [ ! -f "$devdir/package.json" ]; then
        echo "⚠️  sable_dev folder not found,  set DEV_STUDIO_ENABLED=false"
        return 0
    fi

    if ! command -v npm &>/dev/null; then
        echo "⚠️  npm not found,  Dev Studio requires Node.js"
        return 0
    fi

    # Install deps if needed
    if [ ! -f "$devdir/node_modules/.bin/next" ]; then
        echo "🛠️  Installing Dev Studio dependencies..."
        (cd "$devdir" && npm install --silent) || {
            echo "⚠️  Dev Studio npm install failed,  skipping"
            return 0
        }
    fi

    # Check if already running
    local pidfile="$DIR/.dev-studio.pid"
    if [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
        echo "ℹ️  Dev Studio already running (PID $(cat "$pidfile"))"
        return 0
    fi

    echo "🛠️  Starting Dev Studio..."
    (cd "$devdir" && nohup npx next dev --turbopack -p 5700 -H 0.0.0.0 >> "$DIR/logs/dev-studio.log" 2>&1 &
    echo $! > "$pidfile")
    sleep 1
    if [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
        echo "✅ Dev Studio started (PID $(cat "$pidfile")) → http://localhost:5700"
    else
        echo "⚠️  Dev Studio failed to start. Check: tail -20 $DIR/logs/dev-studio.log"
    fi
}

stop_dev_studio() {
    local pidfile="$DIR/.dev-studio.pid"
    if [ -f "$pidfile" ]; then
        local pid
        pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            echo "🛑 Stopping Dev Studio (PID $pid)..."
            kill "$pid" 2>/dev/null
            sleep 2
            kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null
        fi
        rm -f "$pidfile"
    fi
}

# Kill orphaned opensable/bridge processes for the current PROFILE
_kill_orphans() {
    local session_name
    # Determine session name from profile.env or default
    session_name=$(grep -m1 '^WHATSAPP_SESSION_NAME=' "$PROFILE_DIR/profile.env" 2>/dev/null | cut -d= -f2)
    session_name="${session_name:-opensable}"

    # Kill orphan opensable processes for this profile (parent + children it spawned)
    pgrep -f "opensable.*--profile $PROFILE" 2>/dev/null | while read -r opid; do
        # Don't kill ourselves
        [[ -f "$PIDFILE" ]] && [[ "$opid" == "$(cat "$PIDFILE" 2>/dev/null)" ]] && continue
        echo "   🧹 Killing orphan opensable (PID $opid)"
        kill "$opid" 2>/dev/null
        sleep 1
        kill -0 "$opid" 2>/dev/null && kill -9 "$opid" 2>/dev/null
    done

    # Kill orphan bridge.js + chromium for this profile's session
    if [[ "$PROFILE" == "$DEFAULT_PROFILE" ]]; then
        pgrep -f "bridge\.js.*--session $session_name" 2>/dev/null | while read -r bpid; do
            echo "   🧹 Killing orphan bridge.js (PID $bpid)"
            kill "$bpid" 2>/dev/null
        done
        pkill -f "puppeteer.*session-${session_name}" 2>/dev/null
    fi
}

# Kill ALL opensable processes across ALL profiles (nuclear option for clean restarts)
_kill_all_agents() {
    local pids
    pids=$(pgrep -f "python.*opensable" 2>/dev/null)
    if [[ -z "$pids" ]]; then
        return 0
    fi
    echo "   🧹 Killing all opensable processes..."
    echo "$pids" | while read -r opid; do
        kill "$opid" 2>/dev/null
    done
    sleep 2
    # Force kill any survivors
    pids=$(pgrep -f "python.*opensable" 2>/dev/null)
    if [[ -n "$pids" ]]; then
        echo "$pids" | while read -r opid; do
            kill -9 "$opid" 2>/dev/null
        done
        sleep 1
    fi
    # Clean up all PID files and stale sockets/locks
    rm -f "$DIR"/.sable-*.pid
    rm -f /tmp/sable-*.sock /tmp/sable-ollama.lock 2>/dev/null
}

do_start() {
    # ── Supply-chain security check ──────────────────────────────────
    if [ -f "$DIR/scripts/depshield.py" ] && [ -f "$DIR/.depshield.json" ]; then
        echo "🛡️  depshield: Pre-flight supply-chain scan..."
        if python "$DIR/scripts/depshield.py" --root "$DIR" scan 2>/dev/null; then
            echo "  ✅ Supply chain clean"
        else
            echo "  ⚠️  Dependency changes detected — review with: python scripts/depshield.py audit"
        fi
    fi

    # Clean stale PID file if process is dead
    if [ -f "$PIDFILE" ] && ! kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
        rm -f "$PIDFILE"
    fi

    if is_running; then
        echo "⚠️  Already running [$PROFILE] (PID $(cat "$PIDFILE"))"
        echo "   Use: ./start.sh stop --profile $PROFILE   or   ./start.sh restart --profile $PROFILE"
        return 1
    fi

    # Validate profile directory exists
    if [[ ! -d "$PROFILE_DIR" ]]; then
        echo "❌ Profile '$PROFILE' not found at $PROFILE_DIR"
        echo "   Available profiles:"
        ls -1 "$DIR/agents/" 2>/dev/null | grep -v '^_' | grep -v '^\.' | sed 's/^/     /'
        echo ""
        echo "   Create one: cp -r agents/_template agents/$PROFILE"
        return 1
    fi

    echo "👤 Profile: $PROFILE"

    # Ensure aggr/dashboard/marketplace are installed (only for primary agent)
    if [[ "$PROFILE" == "$DEFAULT_PROFILE" ]]; then
        ensure_aggr
        ensure_dashboard
        ensure_marketplace
    fi

    mkdir -p "$DIR/logs"
    echo "🚀 Starting Open-Sable [profile: $PROFILE]..."
    if command -v setsid &>/dev/null; then
        SABLE_PROFILE="$PROFILE" setsid nohup python -m opensable --profile "$PROFILE" >> "$LOGFILE" 2>&1 &
    else
        SABLE_PROFILE="$PROFILE" nohup python -m opensable --profile "$PROFILE" >> "$LOGFILE" 2>&1 &
    fi
    echo $! > "$PIDFILE"
    sleep 1

    if is_running; then
        echo "✅ Running (PID $(cat "$PIDFILE"))"
        echo "   Logs: ./start.sh logs --profile $PROFILE"
        echo "   Stop: ./start.sh stop --profile $PROFILE"

        # Start desktop agent if enabled (only for primary agent)
        if [[ "$PROFILE" == "$DEFAULT_PROFILE" ]]; then
            start_desktop
            start_dev_studio
        fi
    else
        echo "❌ Failed to start. Check: tail -50 $LOGFILE"
        rm -f "$PIDFILE"
        return 1
    fi
}

do_stop() {
    # Stop desktop agent and dev studio (only for primary agent)
    if [[ "$PROFILE" == "$DEFAULT_PROFILE" ]]; then
        stop_desktop
        stop_dev_studio
    fi

    if ! is_running; then
        # Check for orphan processes even without PID file
        _kill_orphans
        echo "ℹ️  Not running [$PROFILE]"
        return
    fi
    pid=$(cat "$PIDFILE")
    echo "🛑 Stopping [$PROFILE] (PID $pid)..."

    # Kill the entire process group (catches children: bridge.js, chromium, etc.)
    pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
    if [[ -n "$pgid" && "$pgid" != "0" ]]; then
        kill -- -"$pgid" 2>/dev/null
    else
        kill "$pid" 2>/dev/null
    fi

    # Also kill any child agents this profile may have spawned
    # (AgentManager spawns children like: python -m opensable --profile <child>)
    _kill_orphans

    # Kill child agent profiles (e.g. nano-sweaters when stopping nano)
    pgrep -f "opensable.*--profile ${PROFILE}-" 2>/dev/null | while read -r cpid; do
        echo "   🧹 Killing child agent (PID $cpid)"
        kill "$cpid" 2>/dev/null
        sleep 1
        kill -0 "$cpid" 2>/dev/null && kill -9 "$cpid" 2>/dev/null
    done
    rm -f "$DIR"/.sable-${PROFILE}-*.pid 2>/dev/null
    rm -f /tmp/sable-${PROFILE}-*.sock 2>/dev/null

    # Wait up to 5s for graceful shutdown
    for i in $(seq 1 5); do
        if ! kill -0 "$pid" 2>/dev/null; then
            break
        fi
        sleep 1
    done
    # Force kill if still alive
    if kill -0 "$pid" 2>/dev/null; then
        echo "   Force killing..."
        if [[ -n "$pgid" && "$pgid" != "0" ]]; then
            kill -9 -- -"$pgid" 2>/dev/null
        fi
        kill -9 "$pid" 2>/dev/null
    fi
    rm -f "$PIDFILE"

    # Final sweep for any remaining orphans
    _kill_orphans

    # Clean stale sockets for this profile
    rm -f /tmp/sable-${PROFILE}.sock 2>/dev/null
    echo "✅ Stopped"
}

do_status() {
    if is_running; then
        pid=$(cat "$PIDFILE")
        uptime=$(ps -p "$pid" -o etime= 2>/dev/null | xargs)
        mem=$(ps -p "$pid" -o rss= 2>/dev/null | awk '{printf "%.0f", $1/1024}')
        echo "✅ Running [$PROFILE] (PID $pid, uptime: $uptime, mem: ${mem}MB)"
    else
        echo "⏹️  Not running [$PROFILE]"
    fi
}

do_list_profiles() {
    echo "📂 Agent profiles (agents/):"
    echo ""
    if [ -d "$DIR/agents" ]; then
        for d in "$DIR/agents"/*/; do
            name=$(basename "$d")
            [[ "$name" == _* ]] && continue
            [[ "$name" == .* ]] && continue
            soul="❌"
            [[ -f "$d/soul.md" ]] && soul="✅"
            env_count=$(grep -c '^[A-Z]' "$d/profile.env" 2>/dev/null || echo "0")
            tools_mode=$(python3 -c "import json; d=json.load(open('$d/tools.json')); print(d.get('mode','all'))" 2>/dev/null || echo "all")
            # Check if running
            pid_file="$DIR/.sable-${name}.pid"
            status="⏹️"
            if [ -f "$pid_file" ] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
                status="🟢"
            fi
            default_tag=""
            [[ "$name" == "$DEFAULT_PROFILE" ]] && default_tag=" (default)"
            echo "  $status $name$default_tag ,  soul: $soul, env: ${env_count} vars, tools: $tools_mode"
        done
    else
        echo "  (none,  create with: cp -r agents/_template agents/my_agent)"
    fi
    echo ""
}

# Helper: get list of all profile names (excluding _template)
get_all_profiles() {
    local profiles_dir="$DIR/agents"
    if [ -d "$profiles_dir" ]; then
        for d in "$profiles_dir"/*/; do
            local name=$(basename "$d")
            [[ "$name" == _* ]] && continue
            echo "$name"
        done
    fi
}

# Helper: run action for a single profile
run_for_profile() {
    local prof="$1"
    PROFILE="$prof"
    PIDFILE="$DIR/.sable-${PROFILE}.pid"
    LOGFILE="$DIR/logs/sable-${PROFILE}.log"
    PROFILE_DIR="$DIR/agents/$PROFILE"
}

do_run() {
    # ── Foreground mode: everything stops when this process dies ──
    # Same setup as do_start, but the agent runs in the foreground.
    # When agent exits (Ctrl+C / terminal close / kill), cleanup trap fires.

    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
        echo "⚠️  Already running [$PROFILE] (PID $(cat "$PIDFILE"))"
        echo "   Stop it first: ./start.sh stop --profile $PROFILE"
        return 1
    fi

    if [[ ! -d "$PROFILE_DIR" ]]; then
        echo "❌ Profile '$PROFILE' not found at $PROFILE_DIR"
        return 1
    fi

    echo "👤 Profile: $PROFILE (foreground mode)"

    # Build dependencies for default profile
    if [[ "$PROFILE" == "$DEFAULT_PROFILE" ]]; then
        ensure_aggr
        ensure_dashboard
        ensure_marketplace
    fi

    mkdir -p "$DIR/logs"

    # Start background services for default profile
    if [[ "$PROFILE" == "$DEFAULT_PROFILE" ]]; then
        start_desktop
        start_dev_studio
    fi

    # Cleanup trap — kills everything when this process exits
    _foreground_cleanup() {
        echo ""
        echo "🛑 Shutting down all services..."
        if [[ "$PROFILE" == "$DEFAULT_PROFILE" ]]; then
            stop_desktop
            stop_dev_studio
        fi
        _kill_orphans
        rm -f "$PIDFILE"
        # Kill any remaining children of this shell
        jobs -p 2>/dev/null | xargs -r kill 2>/dev/null
        echo "✅ Stopped"
    }
    trap _foreground_cleanup EXIT INT TERM HUP

    echo "🚀 Starting Open-Sable [profile: $PROFILE]..."

    # Run agent in foreground — blocks until Ctrl+C / SIGTERM / terminal close
    SABLE_PROFILE="$PROFILE" python -m opensable --profile "$PROFILE" 2>&1 | tee -a "$LOGFILE"

    # When agent exits, trap fires and cleans up everything
}

case "$ACTION" in
    run)
        do_run
        ;;
    start)
        if [[ "$ALL_PROFILES" == "1" ]]; then
            echo "🚀 Starting ALL agents..."
            # Stop any existing agents first to avoid port conflicts
            for p in $(get_all_profiles); do
                run_for_profile "$p"
                if is_running; then
                    echo "── stopping stale $p ──"
                    do_stop
                fi
            done
            sleep 1
            for p in $(get_all_profiles); do
                run_for_profile "$p"
                echo "── $p ──"
                do_start
            done
        else
            do_start
        fi
        ;;
    stop)
        if [[ "$ALL_PROFILES" == "1" ]]; then
            echo "⏹️  Stopping ALL agents..."
            _kill_all_agents
            # Also stop desktop/dev-studio
            stop_desktop
            stop_dev_studio
            echo "✅ All agents stopped"
        else
            do_stop
        fi
        ;;
    restart)
        if [[ "$ALL_PROFILES" == "1" ]]; then
            echo "🔄 Restarting ALL agents..."
            # Nuclear kill: stop ALL opensable processes to prevent zombie accumulation
            _kill_all_agents
            echo "   ✅ All processes killed"
            sleep 2
            for p in $(get_all_profiles); do
                run_for_profile "$p"
                echo "── starting $p ──"
                do_start
            done
        else
            # For single-profile restart, also kill child agents (e.g. nano-sweaters for nano)
            do_stop
            # Kill anything still matching this profile
            pkill -9 -f "opensable.*--profile $PROFILE" 2>/dev/null
            # Kill child agents whose profile starts with this profile name (e.g. nano-*)
            pkill -9 -f "opensable.*--profile ${PROFILE}-" 2>/dev/null
            # Clean sockets for this profile AND its children
            rm -f /tmp/sable-${PROFILE}.sock /tmp/sable-${PROFILE}-*.sock /tmp/sable-ollama.lock 2>/dev/null
            # Clean child PID files
            rm -f "$DIR"/.sable-${PROFILE}-*.pid 2>/dev/null
            sleep 2
            do_start
        fi
        ;;
    status)
        do_status
        ;;
    profiles|list)
        do_list_profiles
        ;;
    logs)
        if [ -f "$LOGFILE" ]; then
            tail -f "$LOGFILE"
        else
            echo "No log file yet for profile $PROFILE"
        fi
        ;;
    *)
        echo "Usage: ./start.sh [run|start|stop|restart|status|logs|profiles] [--profile NAME] [--all]"
        echo ""
        echo "Commands:"
        echo "  run                Run in foreground (stops everything on exit)"
        echo "  start              Start the agent as daemon (default: $DEFAULT_PROFILE)"
        echo "  stop               Stop the agent"
        echo "  restart            Restart the agent"
        echo "  status             Check if the agent is running"
        echo "  logs               Tail live logs"
        echo "  profiles           List all agent profiles"
        echo ""
        echo "Options:"
        echo "  --profile NAME     Target a specific agent profile (from agents/)"
        echo "  --all              Apply command to ALL agent profiles"
        echo ""
        echo "Examples:"
        echo "  ./start.sh restart --all          Restart every agent"
        echo "  ./start.sh stop --all             Stop every agent"
        echo "  ./start.sh start --profile analyst  Start just the analyst"
        echo ""
        echo "All agents live in agents/<name>/ with their own soul.md, profile.env, tools.json, and data/"
        ;;
esac
