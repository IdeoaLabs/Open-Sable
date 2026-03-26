#!/usr/bin/env python3
"""
OpenSable Pi — HUD Dashboard Display
Renders a live status HUD on the 3.5" TFT (480×320 RGB565 @ /dev/fb1)

Environment:
  FB_DEV             /dev/fb1
  FB_WIDTH / FB_HEIGHT               480 / 320
  SABLE_LOG          /home/sable/sable-agent.log
  DISPLAY_INTERVAL   1.0
"""

import os, sys, time, struct, re
from pathlib import Path
from datetime import datetime, timedelta

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    sys.exit("pillow not found — pip install pillow")

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# ── Config ────────────────────────────────────────────────────────────────────
FB_DEV   = os.environ.get("FB_DEV",   "/dev/fb1")
W        = int(os.environ.get("FB_WIDTH",  "480"))
H        = int(os.environ.get("FB_HEIGHT", "320"))
LOG_FILE = os.environ.get(
    "SABLE_LOG",
    str(Path(__file__).resolve().parent.parent / "logs" / "opensable.log")
)
INTERVAL = float(os.environ.get("DISPLAY_INTERVAL", "1.0"))

# ── Color Palette ─────────────────────────────────────────────────────────────
C = {
    "bg":        (  6,  10,  22),
    "bg2":       ( 12,  18,  38),
    "bg3":       ( 18,  28,  58),
    "border":    ( 30,  55, 110),
    "accent":    (  0, 212, 255),
    "accent2":   ( 80, 140, 255),
    "header_bg": (  8,  16,  40),
    "ok":        ( 20, 230, 120),
    "err":       (255,  65,  65),
    "warn":      (255, 210,  40),
    "dim":       ( 70,  80, 110),
    "text":      (185, 210, 240),
    "text2":     (120, 145, 185),
    "bar_bg":    ( 22,  32,  60),
    "cpu_bar":   (  0, 180, 255),
    "ram_bar":   ( 80, 220, 140),
    "white":     (240, 245, 255),
}

# ── Fonts ─────────────────────────────────────────────────────────────────────
_MONO  = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
]
_BOLD  = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
]
_SANS  = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]
_SANS_BOLD = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]

def _font(paths, size):
    for p in paths:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()

F = {
    "title":    _font(_SANS_BOLD, 18),
    "mono_sm":  _font(_MONO,  11),
    "label":    _font(_SANS,  10),
    "label_b":  _font(_SANS_BOLD, 10),
    "stat":     _font(_SANS_BOLD, 14),
    "time":     _font(_MONO,  13),
    "card_lbl": _font(_SANS,   9),
    "card_val": _font(_MONO,  13),
    "card_big": _font(_SANS_BOLD, 12),
    "section":  _font(_SANS_BOLD,  9),
}

# ── Draw helpers ──────────────────────────────────────────────────────────────

def rect(draw, x0, y0, x1, y1, fill=None, outline=None, w=1):
    if fill:
        draw.rectangle([x0, y0, x1, y1], fill=fill)
    if outline:
        draw.rectangle([x0, y0, x1, y1], outline=outline, width=w)

def hline(draw, y, x0=0, x1=None, color=None):
    draw.line([(x0, y), (x1 or W - 1, y)], fill=color or C["border"], width=1)

def progress_bar(draw, x, y, bar_w, bar_h, frac, fill_color, bg_color=None):
    bg = bg_color or C["bar_bg"]
    rect(draw, x, y, x + bar_w, y + bar_h, fill=bg)
    filled = max(2, int(frac * bar_w))
    rect(draw, x, y, x + filled, y + bar_h, fill=fill_color)
    if filled < bar_w:
        tip = tuple(min(255, v + 60) for v in fill_color)
        rect(draw, x + filled - 2, y, x + filled, y + bar_h, fill=tip)
    rect(draw, x, y, x + bar_w, y + bar_h, outline=C["border"])

# ── Log parsing ───────────────────────────────────────────────────────────────
_TS_RE    = re.compile(r'^\[?(\d{2}:\d{2}(:\d{2})?)\]?\s*')
_ANSI_RE  = re.compile(r'\x1b\[[0-9;]*m')

def clean(line: str) -> str:
    return _ANSI_RE.sub('', line).strip()

def line_color(t: str):
    tl = t.lower()
    if any(x in tl for x in ('error', '❌', 'critical', 'exception', 'traceback', 'failed')):
        return C["err"]
    if any(x in tl for x in ('warn', '⚠')):
        return C["warn"]
    if any(x in tl for x in ('✅', '✓', 'started', 'running', 'connected',
                               'online', 'ready', 'success', 'polling', 'registered')):
        return C["ok"]
    if any(x in tl for x in ('💓', 'heartbeat')):
        return (100, 200, 255)
    if any(x in tl for x in ('📋', '🔧', 'tool', 'goal', '🧠', 'node')):
        return C["accent"]
    if any(x in tl for x in ('debug', 'trace')):
        return C["dim"]
    return C["text"]

def extract_ts(line: str):
    m = _TS_RE.match(line)
    return m.group(1)[:5] if m else None

# ── Framebuffer ───────────────────────────────────────────────────────────────
def to_rgb565(img: Image.Image) -> bytes:
    if HAS_NUMPY:
        arr = np.array(img, dtype=np.uint16)   # (H, W, 3)
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        rgb565 = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)
        return rgb565.astype('<u2').tobytes()
    # fallback: pure-python (slow)
    buf = bytearray(img.width * img.height * 2)
    pix = img.load()
    idx = 0
    for y in range(img.height):
        for x in range(img.width):
            r, g, b = pix[x, y]
            struct.pack_into('<H', buf, idx,
                             ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3))
            idx += 2
    return bytes(buf)

def write_fb(data: bytes):
    try:
        with open(FB_DEV, 'wb') as fb:
            fb.write(data)
        return True
    except PermissionError:
        print(f"[display] PermissionError on {FB_DEV} — add user to 'video' group")
        return False
    except FileNotFoundError:
        print(f"[display] {FB_DEV} not found — driver not loaded?")
        return False
    except Exception as e:
        print(f"[display] fb error: {e}")
        return False

# ── Layout constants ──────────────────────────────────────────────────────────
HDR_H   = 36
CARD_Y  = HDR_H + 2
CARD_H  = 62
STATS_Y = CARD_Y + CARD_H + 2
STATS_H = 28
LOGS_Y  = STATS_Y + STATS_H + 1
LOG_H   = 14
LOG_PAD = 4

# ── Frame renderer ────────────────────────────────────────────────────────────

def render(log_lines: list, start_time: float) -> Image.Image:
    img  = Image.new('RGB', (W, H), C["bg"])
    draw = ImageDraw.Draw(img)
    tick = int(time.time()) % 2 == 0

    # ── HEADER ────────────────────────────────────────────────────────────────
    rect(draw, 0, 0, W, HDR_H, fill=C["header_bg"])
    # gradient-ish: draw a thin bright top line
    draw.line([(0, 0), (W, 0)], fill=C["accent"], width=1)
    hline(draw, HDR_H - 1, color=C["accent"])

    draw.text((10, 8), "◈ SABLE", font=F["title"], fill=C["accent"])
    draw.text((91, 10), "AI", font=F["label_b"], fill=C["accent2"])

    # LIVE badge
    bx = 148
    lv = C["ok"] if tick else (0, 100, 50)
    rect(draw, bx, 9, bx + 44, 26, fill=(8, 24, 10))
    rect(draw, bx, 9, bx + 44, 26, outline=lv)
    draw.ellipse([bx + 5, 14, bx + 13, 22], fill=lv)
    draw.text((bx + 17, 10), "LIVE", font=F["label_b"], fill=lv)

    # Clock
    draw.text((W - 168, 6), datetime.now().strftime("%a %d %b"), font=F["time"], fill=C["text2"])
    draw.text((W - 80, 6),  datetime.now().strftime("%H:%M:%S"), font=F["time"], fill=C["white"])

    # ── CARDS ────────────────────────────────────────────────────────────────
    def card(x, w, title, rows):
        """rows: list of (text, color, font_key)"""
        rect(draw, x, CARD_Y, x + w, CARD_Y + CARD_H, fill=C["bg3"])
        rect(draw, x, CARD_Y, x + w, CARD_Y + CARD_H, outline=C["border"])
        draw.text((x + 6, CARD_Y + 4), title, font=F["card_lbl"], fill=C["text2"])
        hline(draw, CARD_Y + 16, x0=x + 1, x1=x + w - 1, color=C["border"])
        y = CARD_Y + 19
        for text, color, fkey in rows:
            draw.text((x + 6, y), text, font=F[fkey], fill=color)
            y += F[fkey].size + 2

    # Uptime
    elapsed = timedelta(seconds=int(time.time() - start_time))
    hs, rem = divmod(elapsed.seconds, 3600)
    ms, ss  = divmod(rem, 60)
    up_str  = f"{hs:02d}:{ms:02d}:{ss:02d}"
    dot_col = C["ok"]

    # Card 1 — Agent
    c1x, c1w = 2, 148
    card(c1x, c1w, "AGENT", [])
    draw.ellipse([c1x + 6, CARD_Y + 22, c1x + 15, CARD_Y + 31], fill=dot_col)
    draw.text((c1x + 19, CARD_Y + 20), "ONLINE", font=F["card_big"], fill=dot_col)
    draw.text((c1x + 6,  CARD_Y + 38), "UPTIME",  font=F["label"],    fill=C["text2"])
    draw.text((c1x + 6,  CARD_Y + 48), up_str,    font=F["card_val"], fill=C["accent"])

    # Card 2 — Model
    c2x, c2w = 154, 200
    card(c2x, c2w, "MODEL", [
        ("qwen2.5-coder",    C["accent"], "card_big"),
        ("18:7b",            C["white"],  "card_val"),
        ("sofia.zunvra.com", C["text2"],  "label"),
    ])

    # Card 3 — Telegram
    c3x, c3w = 358, W - 360
    card(c3x, c3w, "TELEGRAM", [])
    tg = (30, 160, 240)
    draw.text((c3x + 6, CARD_Y + 20), "✈ BOT",       font=F["card_big"], fill=tg)
    draw.text((c3x + 6, CARD_Y + 34), "@Sablethebot", font=F["label"],    fill=tg)
    bd = C["ok"] if tick else (0, 100, 50)
    draw.ellipse([c3x + 6, CARD_Y + 50, c3x + 14, CARD_Y + 58], fill=bd)
    draw.text((c3x + 18, CARD_Y + 48), "polling", font=F["label"], fill=C["ok"])

    # ── STATS BAR ─────────────────────────────────────────────────────────────
    rect(draw, 0, STATS_Y, W, STATS_Y + STATS_H, fill=C["bg2"])
    hline(draw, STATS_Y,            color=C["border"])
    hline(draw, STATS_Y + STATS_H,  color=C["border"])

    if HAS_PSUTIL:
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        ram_pct  = mem.percent
        ram_used = mem.used  // (1024 * 1024)
        ram_tot  = mem.total // (1024 * 1024)
    else:
        cpu = ram_pct = 0.0
        ram_used = ram_tot = 0

    draw.text((6, STATS_Y + 7), "CPU", font=F["label_b"], fill=C["text2"])
    progress_bar(draw, 34, STATS_Y + 8, 136, 12, cpu / 100, C["cpu_bar"])
    cpu_col = C["err"] if cpu > 80 else C["warn"] if cpu > 50 else C["cpu_bar"]
    draw.text((176, STATS_Y + 7), f"{cpu:.0f}%", font=F["label_b"], fill=cpu_col)

    draw.line([(W//2, STATS_Y+3), (W//2, STATS_Y+STATS_H-3)], fill=C["border"])

    rx = W // 2 + 4
    draw.text((rx, STATS_Y + 7), "RAM", font=F["label_b"], fill=C["text2"])
    progress_bar(draw, rx + 28, STATS_Y + 8, 126, 12, ram_pct / 100, C["ram_bar"])
    ram_col = C["err"] if ram_pct > 85 else C["text2"]
    draw.text((rx + 160, STATS_Y + 7), f"{ram_used}M/{ram_tot}M",
              font=F["label"], fill=ram_col)

    # ── LOG AREA ──────────────────────────────────────────────────────────────
    hline(draw, LOGS_Y, color=C["border"])
    rect(draw, 0, LOGS_Y, W, LOGS_Y + 13, fill=C["bg2"])
    draw.text((6, LOGS_Y + 2), "RECENT ACTIVITY", font=F["section"], fill=C["accent2"])
    hline(draw, LOGS_Y + 13, color=C["border"])

    body_y    = LOGS_Y + 14
    max_lines = (H - body_y) // LOG_H
    max_chars = (W - LOG_PAD * 2) // 7   # ~7px per char @ 11px mono

    visible = log_lines[-max_lines:] if len(log_lines) > max_lines else log_lines
    y = body_y + 1

    for raw in visible:
        line = clean(raw)[:max_chars]
        if not line:
            y += LOG_H
            continue
        col = line_color(line)
        ts  = extract_ts(line)
        if ts and f"[{ts}]" in line[:12]:
            draw.text((LOG_PAD, y), f"[{ts}]",      font=F["mono_sm"], fill=C["dim"])
            draw.text((LOG_PAD + 46, y), line[len(ts)+3:].strip(), font=F["mono_sm"], fill=col)
        elif ts and line.startswith(ts):
            draw.text((LOG_PAD, y), ts,              font=F["mono_sm"], fill=C["dim"])
            draw.text((LOG_PAD + 38, y), line[len(ts):].strip(), font=F["mono_sm"], fill=col)
        else:
            draw.text((LOG_PAD, y), line,            font=F["mono_sm"], fill=col)
        y += LOG_H

    return img


# ── Log reader ────────────────────────────────────────────────────────────────
_CANDIDATE_LOGS = [
    "/home/sable/sable-agent.log",
    LOG_FILE,
    str(Path(__file__).resolve().parent.parent / "logs" / "opensable.log"),
]

def find_log():
    for p in _CANDIDATE_LOGS:
        pp = Path(p)
        if pp.exists() and pp.stat().st_size > 0:
            return pp
    return None

def read_log(path: Path) -> list:
    try:
        raw = path.read_text(errors='replace').splitlines()
        out = [_ANSI_RE.sub('', l).strip() for l in raw if l.strip()]
        return out[-200:]
    except Exception:
        return []


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    start_time = time.time()
    log_path   = find_log()
    last_size  = -1
    lines: list = []

    print(f"[display] {W}×{H}  fb={FB_DEV}  interval={INTERVAL}s")
    print(f"[display] log: {log_path or '(searching)'}")
    print(f"[display] psutil: {'ok' if HAS_PSUTIL else 'missing — no cpu/ram stats'}")
    print(f"[display] numpy:  {'ok (fast rgb565)' if HAS_NUMPY else 'missing — slow pixel loop'}")

    if HAS_PSUTIL:
        psutil.cpu_percent(interval=None)  # warm up

    splash = [
        "✅ OpenSable Pi HUD starting...",
        f"   display: {W}x{H} on {FB_DEV}",
        "   waiting for agent log...",
    ]
    write_fb(to_rgb565(render(splash, start_time)))

    errs = 0
    while True:
        try:
            if log_path is None:
                log_path = find_log()

            if log_path and log_path.exists():
                sz = log_path.stat().st_size
                if sz != last_size:
                    last_size = sz
                    lines = read_log(log_path)

            write_fb(to_rgb565(render(lines or splash, start_time)))
            errs = 0

        except KeyboardInterrupt:
            print("\n[display] stopped.")
            break
        except Exception as e:
            errs += 1
            print(f"[display] error #{errs}: {e}")
            if errs > 10:
                time.sleep(5)

        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
