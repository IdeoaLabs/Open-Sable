"""
Non-blocking reporter that tells the desktop avatar what Sable is doing.

Usage:
    from opensable.utils.avatar import report
    report('thinking')
    report('executing', tool='browser_search')
    report('responding', text='Here is what I found...')
    report('idle')

States: idle | thinking | typing | executing | responding
"""

import json
import os
import threading
import time
import urllib.request
from pathlib import Path

_STATE_FILE  = Path(os.environ.get("AVATAR_STATE_FILE", "/tmp/sable-avatar-state.json"))
_BRAIN_FILE  = Path(os.environ.get("AVATAR_BRAIN_FILE", "/tmp/sable-brain-state.json"))

# Avatar HTTP endpoints,  override via env vars to avoid hardcoded IPs
# Set AVATAR_PI_URL="" to disable Pi posting entirely
_HTTP_URL    = os.environ.get("AVATAR_HTTP_URL",  "http://127.0.0.1:7799/state")
_PI_URL      = os.environ.get("AVATAR_PI_URL",    "")   # empty = disabled by default
_BRAIN_URL_L = os.environ.get("AVATAR_BRAIN_URL", "http://127.0.0.1:7799/brain")
_BRAIN_URL_P = os.environ.get("AVATAR_BRAIN_URL_PI", "")

# Debounce: skip if same state was sent < this many seconds ago
_DEBOUNCE_SECS = 0.3
_last_state: str = ""
_last_ts: float = 0.0
_lock = threading.Lock()

# Brain event ring-buffer (max 120 events)
_brain_lock   = threading.Lock()
_brain_events: list = []
_BRAIN_MAX    = 120


def report(state: str, text: str = "", tool: str = "", words: int = 0) -> None:
    """Report agent state to the desktop avatar (fire-and-forget, never raises)."""
    global _last_state, _last_ts

    with _lock:
        now = time.monotonic()
        if state == _last_state and (now - _last_ts) < _DEBOUNCE_SECS:
            return
        _last_state = state
        _last_ts = now

    payload = json.dumps({
        "state": state,
        "text": text[:120],
        "tool": tool,
        "words": max(0, int(words)),
        "ts": time.time(),
    })
    threading.Thread(target=_send, args=(payload,), daemon=True).start()


def _send(payload: str) -> None:
    # Write state file locally
    try:
        _STATE_FILE.write_text(payload)
    except Exception:
        pass

    # POST to local desktop avatar + Pi display
    for url in (_HTTP_URL, _PI_URL):
        try:
            req = urllib.request.Request(
                url,
                data=payload.encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=0.4)
        except Exception:
            pass


def brain_event(event_type: str, content: str) -> None:
    """Append a brain event and push updated list to both displays (fire-and-forget)."""
    entry = {"type": event_type, "content": content[:300], "ts": time.time()}
    with _brain_lock:
        _brain_events.append(entry)
        if len(_brain_events) > _BRAIN_MAX:
            del _brain_events[:-_BRAIN_MAX]
        payload = json.dumps({"events": list(_brain_events)})
    threading.Thread(target=_send_brain, args=(payload,), daemon=True).start()


def _send_brain(payload: str) -> None:
    try:
        _BRAIN_FILE.write_text(payload)
    except Exception:
        pass
    for url in (_BRAIN_URL_L, _BRAIN_URL_P):
        try:
            req = urllib.request.Request(
                url,
                data=payload.encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=0.4)
        except Exception:
            pass
