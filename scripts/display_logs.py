#!/usr/bin/env python3
"""
OpenSable Pi,  Framebuffer Log Viewer
Renders live OpenSable logs on the 3.5" XPT2046 display (480×320 RGB565)

Environment overrides:
  FB_DEV             /dev/fb1        framebuffer device
  FB_WIDTH / FB_HEIGHT               480 / 320 by default
  SABLE_LOG          ./logs/opensable.log
  DISPLAY_INTERVAL   1.0             seconds between refreshes
  DISPLAY_ROTATE     0               rotation: 0 | 90 | 180 | 270
"""
import os, sys, time, textwrap, struct
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    sys.exit("PIL not found. Run: pip install pillow")

# ── Config ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent.parent
FB_DEV     = os.environ.get("FB_DEV",   "/dev/fb1")
WIDTH      = int(os.environ.get("FB_WIDTH",  "480"))
HEIGHT     = int(os.environ.get("FB_HEIGHT", "320"))
LOG_FILE   = os.environ.get("SABLE_LOG", str(SCRIPT_DIR / "logs" / "opensable.log"))
INTERVAL   = float(os.environ.get("DISPLAY_INTERVAL", "1.0"))
ROTATE     = int(os.environ.get("DISPLAY_ROTATE", "0"))   # 0 / 90 / 180 / 270

# ── Palette (dark terminal theme) ────────────────────────────────────────────
BG        = (10,  14,  28)
HEADER_BG = (18,  36,  72)
HEADER_FG = (80,  180, 255)
FG        = (180, 220, 180)    # default text
OK_FG     = (60,  230, 120)    # green ,  success / started / ready
ERR_FG    = (255,  80,  80)    # red   ,  error / exception
WARN_FG   = (255, 200,  40)    # yellow,  warning
TOOL_FG   = (120, 180, 255)    # blue  ,  tool calls
TICK_FG   = (60,  220, 220)    # cyan  ,  cognitive tick
DIM_FG    = (70,   70,  90)    # dim   ,  debug / trace

# ── Font loader ───────────────────────────────────────────────────────────────
_MONO_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf",
]
_BOLD_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
]

def _load_font(paths: list, size: int) -> ImageFont.FreeTypeFont:
    for p in paths:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()

FONT_SM   = _load_font(_MONO_PATHS, 10)
FONT_MD   = _load_font(_MONO_PATHS, 12)
FONT_BOLD = _load_font(_BOLD_PATHS, 12)

# ── RGB → RGB565 framebuffer ──────────────────────────────────────────────────
def image_to_fb565(img: Image.Image) -> bytes:
    """Convert an RGB PIL Image to raw RGB565 little-endian bytes."""
    buf = bytearray(img.width * img.height * 2)
    pix = img.load()
    idx = 0
    for y in range(img.height):
        for x in range(img.width):
            r, g, b = pix[x, y]
            struct.pack_into(
                '<H', buf, idx,
                ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)
            )
            idx += 2
    return bytes(buf)

def write_fb(data: bytes, dev: str = FB_DEV) -> bool:
    try:
        with open(dev, 'wb') as fb:
            fb.write(data)
        return True
    except PermissionError:
        print(f"[display] Permission denied on {dev}.")
        print(f"          sudo usermod -aG video $USER && newgrp video")
        return False
    except FileNotFoundError:
        print(f"[display] {dev} not found,  display driver not loaded?")
        return False
    except Exception as e:
        print(f"[display] fb write error: {e}")
        return False

# ── Line colorizer ────────────────────────────────────────────────────────────
def line_color(text: str) -> tuple:
    t = text.lower()
    if any(x in t for x in ('error', 'err ', '❌', 'critical', 'traceback', 'exception')):
        return ERR_FG
    if any(x in t for x in ('warn', '⚠', '⚠️')):
        return WARN_FG
    if any(x in t for x in ('✓', '✔', '[ ok', 'ok ]', 'started', 'ready', 'success',
                              'listening', '✅', 'connected', 'online')):
        return OK_FG
    if any(x in t for x in ('🔧', 'tool', 'calling tool', 'execute', 'executing')):
        return TOOL_FG
    if any(x in t for x in ('cognitive', 'tick', '🧠', '⚙', '[phase', 'connectome',
                              'inner_life', 'meta_learner')):
        return TICK_FG
    if any(x in t for x in ('debug', 'trace', 'telemetry', 'heartbeat')):
        return DIM_FG
    return FG

# ── Frame renderer ────────────────────────────────────────────────────────────
HEADER_H = 24
LINE_H   = 14

def render(lines: list) -> Image.Image:
    w, h = (HEIGHT, WIDTH) if ROTATE in (90, 270) else (WIDTH, HEIGHT)
    img  = Image.new('RGB', (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(img)

    # Header bar
    draw.rectangle([0, 0, WIDTH - 1, HEADER_H - 1], fill=HEADER_BG)
    ts = time.strftime("%H:%M:%S")
    draw.text(
        (5, 5),
        f"◈ OpenSable  ▸  Sable  ▸  {ts}",
        fill=HEADER_FG, font=FONT_BOLD
    )
    draw.line([(0, HEADER_H), (WIDTH, HEADER_H)], fill=(30, 60, 120), width=1)

    # Log lines
    max_vis = (HEIGHT - HEADER_H - 2) // LINE_H
    max_ch  = WIDTH // 6 - 1    # ~6 px per mono char at size 10
    visible = lines[-max_vis:] if len(lines) > max_vis else lines

    y = HEADER_H + 3
    for line in visible:
        text  = line.rstrip()[:max_ch]
        color = line_color(text)
        draw.text((3, y), text, fill=color, font=FONT_SM)
        y += LINE_H

    if ROTATE:
        img = img.rotate(ROTATE, expand=True)

    return img

# ── Main loop ─────────────────────────────────────────────────────────────────
def main() -> None:
    log_path  = Path(LOG_FILE)
    last_size = -1
    lines: list = []

    print(f"OpenSable Display  |  {WIDTH}x{HEIGHT}  |  {FB_DEV}")
    print(f"Log:               {log_path}")
    print(f"Interval:          {INTERVAL}s  |  Rotate: {ROTATE}°")

    # Startup splash
    splash_lines = [
        "",
        "  ◈  OpenSable Pi Display",
        f"  Log: {log_path}",
        "",
        "  Waiting for agent...",
    ]
    write_fb(image_to_fb565(render(splash_lines)), FB_DEV)

    while True:
        try:
            if log_path.exists():
                size = log_path.stat().st_size
                if size != last_size:
                    last_size = size
                    raw = log_path.read_text(errors='replace').splitlines()
                    # Wrap long lines so they're readable on the small screen
                    wrapped: list = []
                    for l in raw:
                        if len(l) > 78:
                            wrapped.extend(textwrap.wrap(l, 78) or [''])
                        else:
                            wrapped.append(l)
                    lines = wrapped
                    write_fb(image_to_fb565(render(lines)), FB_DEV)
            elif last_size != 0:
                last_size = 0
                write_fb(image_to_fb565(render(
                    ["  Waiting for OpenSable log file...", f"  {log_path}"]
                )), FB_DEV)

        except KeyboardInterrupt:
            print("\n[display] Stopped.")
            break
        except Exception as e:
            print(f"[display] {e}")

        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
