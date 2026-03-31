"""
IntentClassifier,  Fast, zero-LLM intent detection for SableCore.

Exactly what GitHub Copilot does internally: before answering, classify
the message to decide WHAT to do and WHERE to look.

Intent hierarchy (checked in order):
  code_question     ,  "why does X fail", "how does Y work", "fix the bug"
  self_modify       ,  "add feature to yourself", "implement RAG in your code"
  desktop_screenshot,  "take a screenshot", "what's on screen"
  desktop_type      ,  "type hello world", "write this text"
  desktop_click     ,  "click on X", "click the submit button"
  desktop_hotkey    ,  "press Ctrl+C", "alt+tab", "keyboard shortcut"
  window_list       ,  "list open windows", "what's running"
  window_focus      ,  "focus on terminal", "switch to chrome"
  navigate_url      ,  "go to youtube.com", "open https://..."
  open_app          ,  "open chrome", "launch terminal"
  system_command    ,  "run ls -la", "execute bash command"
  file_operation    ,  "read file", "list files in /home"
  image_request     ,  "analyze screenshot", "what do you see"
  web_search        ,  "search for", "what's the weather", "price of BTC"
  trading           ,  "buy BTC", "portfolio", "price of ETH"
  social_media      ,  "tweet", "post to instagram", "like that"
  general_chat      ,  fallback

Returns IntentResult with:
  intent, confidence, entities (extracted values), needs_code_context, needs_web_search
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class IntentResult:
    intent: str
    confidence: float
    entities: Dict[str, Any] = field(default_factory=dict)
    needs_code_context: bool = False
    needs_web_search: bool = False

    def __repr__(self):
        return (
            f"Intent({self.intent!r}, conf={self.confidence:.2f}, "
            f"code={self.needs_code_context}, web={self.needs_web_search})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Compiled patterns
# ─────────────────────────────────────────────────────────────────────────────

_URL_RE = re.compile(
    r'https?://\S+|www\.\S+\.\S+|'
    r'\b(?:youtube|github|google|twitter|reddit|instagram|facebook|linkedin|tiktok)'
    r'\.com\b',
    re.IGNORECASE,
)

_DOMAIN_RE = re.compile(r'\b[\w-]+\.[a-z]{2,}(/\S*)?\b', re.IGNORECASE)

# ── Code ─────────────────────────────────────────────────────────────────────
_CODE_VERBS = re.compile(
    r'\b(fix|debug|why\s+(is|does|did)|how\s+(does|do|is|works?)|'
    r'what\s+(is|are)\s+the\s+(function|class|method|variable|file|module|component)|'
    r'broken|not\s+working|error|exception|bug|crash|fails?|broke|'
    r'implement|refactor|optimize|add\s+to\s+the\s+code|'
    r'how\s+is\s+.{1,30}(implemented|built|done|handled|stored|saved)|'
    r'where\s+(is|are)\s+.{1,30}(defined|stored|handled|called)|'
    r'explain\s+.{1,30}(code|function|class|method|file)|'
    r'show\s+me\s+the\s+(code|source|implementation))\b',
    re.IGNORECASE,
)

_CODE_NOUNS = re.compile(
    r'\b(function|class|method|module|import|def\b|async|await|'
    r'\.py|\.jsx|\.js|\.css|\.ts|component|hook|route|endpoint|'
    r'agent\.py|vision_tools|llm\.py|gateway|rag|codebase|'
    r'toggle|theme|sidebar|titlebar|open_app|browser_search)\b',
    re.IGNORECASE,
)

# ── Self-modification ────────────────────────────────────────────────────────
_SELF_MOD = re.compile(
    r'\b(implement\s+(in\s+)?your(self)?|add\s+(to\s+)?(your|yourself|the\s+agent|sable)|'
    r'modify\s+yourself|change\s+(how\s+you|your)|make\s+yourself|'
    r'quiero\s+que\s+implementes|implementa\s+(en\s+)?ti|'
    r'at\s+your\s+level|a\s+tu\s+nivel|como\s+lo\s+haces\s+t[uú])\b',
    re.IGNORECASE,
)

# ── Desktop: screenshot ───────────────────────────────────────────────────────
_SCREENSHOT = re.compile(
    r'\b(take\s+a?\s*screenshot|take\s+screen\s+cap|screen\s+capture|'
    r'captura\s+de\s+pantalla|toma\s+una\s+captura|screenshot(\s+now)?|'
    r'what\s+(is\s+)?(on|in)\s+(the\s+)?screen|'
    r'what\s+do\s+you\s+see(\s+on\s+(the\s+)?screen)?|'
    r'show\s+me\s+(the\s+)?screen|muestra\s+(la\s+)?pantalla)\b',
    re.IGNORECASE,
)

# ── Desktop: type text ────────────────────────────────────────────────────────
_DESKTOP_TYPE = re.compile(
    r'^(type|escribe|write|teclea|ingresa|input)\s+(.+)',
    re.IGNORECASE,
)

# ── Desktop: click ────────────────────────────────────────────────────────────
_DESKTOP_CLICK = re.compile(
    r'\b(click\s+(on|the|en)\s+\S|haz\s+clic\s+(en|sobre)\s+\S|'
    r'press\s+the\s+\S.{1,20}button|click\s+(submit|ok|cancel|close|accept|yes|no)\b)',
    re.IGNORECASE,
)
_DESKTOP_CLICK_TEXT = re.compile(
    r'click\s+(?:on\s+|the\s+)?["\']?(.+?)["\']?\s*(?:button|link|icon)?$',
    re.IGNORECASE,
)

# ── Desktop: hotkey ───────────────────────────────────────────────────────────
_HOTKEY = re.compile(
    r'\b(press\s+(ctrl|alt|shift|super|win|cmd)\s*[+\-]\w+|'
    r'hotkey|keyboard\s+shortcut|'
    r'alt\s*\+\s*tab|ctrl\s*\+\s*\w+|cmd\s*\+\s*\w+|'
    r'pulsa?\s+(ctrl|alt|shift)\s*\+\s*\w+)\b',
    re.IGNORECASE,
)
_HOTKEY_EXTRACT = re.compile(
    r'((?:ctrl|alt|shift|super|win|cmd)\s*[+\-]\s*\w+(?:\s*[+\-]\s*\w+)*)',
    re.IGNORECASE,
)

# ── Window list ───────────────────────────────────────────────────────────────
_WIN_LIST = re.compile(
    r'\b(list\s+(open\s+)?(windows|apps|applications|programs)|'
    r'what\s+(windows|apps|programs|applications)\s+(is|are)\s+(open|running)|'
    r'what.{0,10}(open|running|active)|'
    r'show\s+(open\s+)?(windows|apps)|'
    r'qu[eé]\s+(ventanas|apps|programas)\s+(hay|est[aá]n)\s+(abiertas?|corriendo|activas?))\b',
    re.IGNORECASE,
)

# ── Window focus ─────────────────────────────────────────────────────────────
_WIN_FOCUS = re.compile(
    r'\b(focus\s+(on|the)|switch\s+to|bring\s+up|bring\s+.{1,20}\s+to\s+front|'
    r'go\s+to\s+the\s+\w+\s+window|'
    r'cambia\s+a|enfoca\s+(en|la?)|trae\s+.{1,20}\s+al\s+frente)\b',
    re.IGNORECASE,
)
_WIN_FOCUS_TARGET = re.compile(
    r'(?:focus\s+(?:on\s+)?|switch\s+to\s+|bring\s+up\s+|go\s+to\s+the\s+)(\w[\w\s]*)',
    re.IGNORECASE,
)

# ── Open app ─────────────────────────────────────────────────────────────────
_OPEN_APP = re.compile(
    r'^(open|launch|start|abre|abrir|lanza|inicia|ejecuta)\s+\w',
    re.IGNORECASE,
)

# ── System command ────────────────────────────────────────────────────────────
_SYS_CMD = re.compile(
    r'\b(run\s+(the\s+)?(command|cmd|script|shell|bash|zsh)|'
    r'execute\s+(the\s+)?(command|cmd|script)|'
    r'terminal\s+(command|cmd)|'
    r'corr[eé]\s+(el\s+)?(comando|script)|'
    r'ejec[uú]ta\s+(el\s+)?(comando|script))\b',
    re.IGNORECASE,
)
# Extract the actual command from "run the command: ls -la"
_SYS_CMD_EXTRACT = re.compile(
    r'(?:run|execute|corr[eé]|ejec[uú]ta)\s+'
    r'(?:the\s+)?(?:command|cmd|script|bash|zsh|shell)?\s*[:\-]?\s*["`\']?'
    r'(.+?)["`\']?\s*$',
    re.IGNORECASE,
)

# ── File ops ─────────────────────────────────────────────────────────────────
_FILE_OPS = re.compile(
    r'\b(read\s+(the\s+)?file|write\s+(to\s+)?(the\s+)?file|create\s+(a\s+)?file|'
    r'delete\s+(the\s+)?file|list\s+(files|directory|dir)|'
    r'show\s+(files|contents)\s+(in|of)|move\s+file|copy\s+file|'
    r'(make|create|write|generate|build)\s+(me\s+)?(a\s+)?(document|doc|report|essay|letter|pdf|spreadsheet|presentation|word\s+doc|pptx?|xlsx?)|'
    r'(hazme|cr[eé]ame|genera|escr[ií]be)\s+(un\s+)?(documento|reporte|ensayo|carta|pdf|hoja\s+de\s+c[aá]lculo|presentaci[oó]n)|'
    r'leer\s+archivo|listar\s+archivos|mostrar\s+archivos)\b',
    re.IGNORECASE,
)
_FILE_OPS_SUBTYPE = re.compile(
    r'\b(read|write|create|delete|list|move|copy|leer|listar|crear|borrar)\b',
    re.IGNORECASE,
)
_FILE_PATH = re.compile(r'(/[\w./\-~]+|\./[\w./\-]+|~[\w./\-]+|\w+\.\w{1,5})')

# ── URL nav ───────────────────────────────────────────────────────────────────
_NAV = re.compile(
    r'\b(go\s+to|navigate\s+to|open\s+(the\s+)?url|open\s+https?|'
    r'ir\s+a|abre\s+https?|visita\s+)\b',
    re.IGNORECASE,
)

# ── Web search ────────────────────────────────────────────────────────────────
_WEB_SEARCH = re.compile(
    r'\b(search(\s+for)?|google|look\s+up|find\s+(me\s+)?info|'
    r'busca|buscar|qu[eé]\s+es|qui[eé]n\s+es|'
    r'weather|news|noticias|price\s+of|latest|current\s+events|'
    r'what\s+(is|are)\s+the\s+(latest|current|today)|'
    r'tell\s+me\s+about|what\s+happened|trending)\b',
    re.IGNORECASE,
)

# ── Trading ──────────────────────────────────────────────────────────────────
_TRADING = re.compile(
    r'\b(buy|sell|trade|portfolio|price\s+of|market|btc|eth|bitcoin|ethereum|'
    r'stock|crypto|comprar|vender|mercado)\b',
    re.IGNORECASE,
)

# ── Social media ─────────────────────────────────────────────────────────────
_SOCIAL = re.compile(
    r'\b(tweet|tweetea|post\s+to|post\s+on|instagram|twitter|facebook|linkedin|tiktok|'
    r'like|retweet|follow|dm\s+|comment\s+on)\b',
    re.IGNORECASE,
)

# ── Image analysis (not screenshot) ──────────────────────────────────────────
_IMAGE_ANALYZE = re.compile(
    r'\b('
    r'analyze\s+(this\s+)?(image|photo|picture)'
    r'|what.{0,20}image'
    r'|describe\s+(the\s+)?image'
    r'|generat[e]?\s+(a[n]?\s+)?(image|photo|picture|art|illustration)'
    r'|create\s+(a[n]?\s+)?(image|photo|picture|art|illustration)'
    r'|make\s+(a[n]?\s+|me\s+(a[n]?\s+)?)?(image|photo|picture|art|illustration)'
    r'|draw\s+(a[n]?\s+|me\s+(a[n]?\s+)?)?'
    r'|paint\s+(a[n]?\s+|me\s+(a[n]?\s+)?)?'
    r'|render\s+(a[n]?\s+)?(image|photo|picture|scene)'
    r'|image\s+of\b'
    r')\b',
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
class IntentClassifier:
    """
    Zero-latency (~0.1ms) intent classifier using compiled regex.
    No LLM, no async, no imports at call time.
    """

    def classify(self, message: str) -> IntentResult:
        msg = message.strip()
        msg_lower = msg.lower()

        # 1. Self-modification
        if _SELF_MOD.search(msg):
            return IntentResult(
                intent="self_modify", confidence=0.92,
                needs_code_context=True,
                entities={"original": msg},
            )

        # 2. Screenshot (before navigate_url to avoid "what's on screen" → URL)
        if _SCREENSHOT.search(msg):
            return IntentResult(
                intent="desktop_screenshot", confidence=0.95,
                entities={},
            )

        # 3. Type text on screen
        m = _DESKTOP_TYPE.match(msg)
        if m:
            return IntentResult(
                intent="desktop_type", confidence=0.93,
                entities={"text": m.group(2).strip(' "\'')},
            )

        # 4. Click on something
        if _DESKTOP_CLICK.search(msg):
            target = ""
            tm = _DESKTOP_CLICK_TEXT.search(msg)
            if tm:
                target = tm.group(1).strip()
            return IntentResult(
                intent="desktop_click", confidence=0.90,
                entities={"target": target},
            )

        # 5. Keyboard hotkey
        if _HOTKEY.search(msg):
            combo = ""
            hm = _HOTKEY_EXTRACT.search(msg)
            if hm:
                combo = re.sub(r'\s', '', hm.group(1)).lower()
            return IntentResult(
                intent="desktop_hotkey", confidence=0.92,
                entities={"keys": combo},
            )

        # 6. Window list
        if _WIN_LIST.search(msg):
            return IntentResult(
                intent="window_list", confidence=0.90,
                entities={},
            )

        # 7. Window focus/switch
        if _WIN_FOCUS.search(msg):
            target = ""
            fm = _WIN_FOCUS_TARGET.search(msg)
            if fm:
                target = fm.group(1).strip()
            return IntentResult(
                intent="window_focus", confidence=0.88,
                entities={"window": target},
            )

        # 8. URL navigation
        url_match = _URL_RE.search(msg)
        if url_match or _NAV.search(msg):
            url = url_match.group(0) if url_match else ""
            return IntentResult(
                intent="navigate_url", confidence=0.90,
                entities={"url": url},
            )

        # 9. Open app
        if _OPEN_APP.match(msg) and len(msg_lower) < 60:
            tokens = msg.split()
            app = tokens[1] if len(tokens) > 1 else ""
            return IntentResult(
                intent="open_app", confidence=0.90,
                entities={"app_name": app},
            )

        # 10. System command (explicit "run command: X")
        if _SYS_CMD.search(msg):
            cmd = ""
            cm = _SYS_CMD_EXTRACT.search(msg)
            if cm:
                cmd = cm.group(1).strip()
            return IntentResult(
                intent="system_command", confidence=0.88,
                entities={"command": cmd},
            )

        # 11. Code question
        code_verb = bool(_CODE_VERBS.search(msg))
        code_noun = bool(_CODE_NOUNS.search(msg))
        if code_verb and code_noun:
            return IntentResult(intent="code_question", confidence=0.92,
                                needs_code_context=True, entities={"query": msg})
        if code_verb:
            return IntentResult(intent="code_question", confidence=0.75,
                                needs_code_context=True, entities={"query": msg})
        if code_noun:
            return IntentResult(intent="code_question", confidence=0.65,
                                needs_code_context=True, entities={"query": msg})

        # 12. File operation
        if _FILE_OPS.search(msg):
            subtype = "list"
            sm = _FILE_OPS_SUBTYPE.search(msg)
            if sm:
                subtype = sm.group(1).lower()
            path = ""
            pm = _FILE_PATH.search(msg)
            if pm:
                path = pm.group(1)
            return IntentResult(
                intent="file_operation", confidence=0.85,
                entities={"subtype": subtype, "path": path},
            )

        # 13. Image analysis
        if _IMAGE_ANALYZE.search(msg):
            return IntentResult(
                intent="image_request", confidence=0.85,
                entities={"query": msg},
            )

        # 14. Web search
        if _WEB_SEARCH.search(msg):
            return IntentResult(
                intent="web_search", confidence=0.88,
                needs_web_search=True,
                entities={"query": msg},
            )

        # 15. Trading
        if _TRADING.search(msg):
            return IntentResult(
                intent="trading", confidence=0.80,
                needs_web_search=True,
                entities={"query": msg},
            )

        # 16. Social media
        if _SOCIAL.search(msg):
            return IntentResult(
                intent="social_media", confidence=0.82,
                entities={"query": msg},
            )

        # 17. Fallback
        return IntentResult(
            intent="general_chat", confidence=0.50,
            entities={"query": msg},
        )

    def explain(self, message: str) -> str:
        """Human-readable classification explanation (for debugging)."""
        r = self.classify(message)
        return (
            f"Intent     : {r.intent} (confidence={r.confidence:.0%})\n"
            f"Code ctx   : {r.needs_code_context}\n"
            f"Web search : {r.needs_web_search}\n"
            f"Entities   : {r.entities}"
        )
