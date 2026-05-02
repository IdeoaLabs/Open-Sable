"""
TLS Fingerprint Patch,  Makes twikit use curl_cffi instead of httpx.

Problem: twikit uses plain httpx, which has a Python TLS fingerprint (JA3/JA4).
X detects the real platform via TLS fingerprint, TCP stack, etc.
Even with a mobile User-Agent, X sees "Linux desktop" because httpx leaks it.

Solution: Replace httpx.AsyncClient with curl_cffi.AsyncSession that impersonates
Chrome 131 on Android,  the TLS handshake will match a real Android Chrome browser.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def _apply_user_legacy_defaults(data: dict) -> None:
    """Fill sparse X user payloads with safe defaults expected by twikit."""
    if not isinstance(data, dict):
        return

    data.setdefault("rest_id", "")
    data.setdefault("is_blue_verified", False)

    legacy = data.setdefault("legacy", {})
    if not isinstance(legacy, dict):
        return

    entities = legacy.setdefault("entities", {})
    if not isinstance(entities, dict):
        entities = {}
        legacy["entities"] = entities

    description = entities.setdefault("description", {})
    if not isinstance(description, dict):
        description = {}
        entities["description"] = description
    description.setdefault("urls", [])

    url_obj = entities.setdefault("url", {})
    if not isinstance(url_obj, dict):
        url_obj = {}
        entities["url"] = url_obj
    url_obj.setdefault("urls", [])

    legacy_defaults = {
        "created_at": "",
        "name": "",
        "screen_name": "",
        "profile_image_url_https": "",
        "profile_banner_url": None,
        "url": None,
        "location": "",
        "description": "",
        "pinned_tweet_ids_str": [],
        "verified": False,
        "possibly_sensitive": False,
        "can_dm": False,
        "can_media_tag": False,
        "want_retweets": False,
        "default_profile": False,
        "default_profile_image": False,
        "has_custom_timelines": False,
        "followers_count": 0,
        "fast_followers_count": 0,
        "normal_followers_count": 0,
        "friends_count": 0,
        "favourites_count": 0,
        "listed_count": 0,
        "media_count": 0,
        "statuses_count": 0,
        "is_translator": False,
        "translator_type": "none",
        "withheld_in_countries": [],
    }
    for key, value in legacy_defaults.items():
        legacy.setdefault(key, value)

CURL_CFFI_AVAILABLE = False
try:
    from curl_cffi.requests import AsyncSession as _CurlAsyncSession
    CURL_CFFI_AVAILABLE = True
except ImportError:
    logger.warning("curl_cffi not installed,  TLS fingerprint impersonation disabled")

# The impersonation target: Chrome 131 on Android
# This makes the TLS handshake (JA3/JA4) look like a real Android Chrome browser
IMPERSONATE_TARGET = "chrome131_android"


class TwikitCurlSession:
    """
    Drop-in wrapper around curl_cffi.AsyncSession that provides the
    httpx.AsyncClient interface twikit expects.

    Twikit accesses:
      - self.http.request(method, url, headers=..., data=..., **kwargs)
      - self.http.cookies          (.jar, .clear(), .update(), .get(), dict())
      - self.http.cookies.jar      (iterable of cookie objects)
      - self.http._mounts          (proxy getter/setter,  we stub this)
      - self.http.headers           (dict-like)
    """

    def __init__(self, proxy: Optional[str] = None, **kwargs):
        # Build curl_cffi session with Android Chrome TLS fingerprint
        session_kwargs = {
            "impersonate": IMPERSONATE_TARGET,
        }
        if proxy:
            session_kwargs["proxies"] = {"https": proxy, "http": proxy}

        self._session = _CurlAsyncSession(**session_kwargs)

        # Stub _mounts so twikit's proxy getter/setter doesn't crash
        self._mounts = {}

        # Expose headers dict for compatibility
        self.headers = self._session.headers if hasattr(self._session, 'headers') else {}

        logger.info(f"🛡️ TLS patch active: impersonating {IMPERSONATE_TARGET}")

    @property
    def cookies(self):
        """Expose curl_cffi cookies,  compatible with twikit's usage."""
        return self._session.cookies

    @cookies.setter
    def cookies(self, value):
        """Twikit does self.http.cookies = list(cookies.items())"""
        self._session.cookies.clear()
        if isinstance(value, list):
            for name, val in value:
                self._session.cookies.set(name, val)
        elif isinstance(value, dict):
            self._session.cookies.update(value)
        else:
            # Try to iterate as key-value pairs
            try:
                for name, val in value:
                    self._session.cookies.set(name, val)
            except (TypeError, ValueError):
                pass

    async def request(self, method: str, url: str, **kwargs) -> "TwikitCurlResponse":
        """
        Forward request to curl_cffi session.
        Maps httpx-style kwargs to curl_cffi equivalents.
        """
        # curl_cffi uses 'content' instead of 'data' for bytes in some cases,
        # but 'data' is also supported. Just pass through.
        response = await self._session.request(method, url, **kwargs)
        return TwikitCurlResponse(response)

    async def aclose(self):
        """Clean up session."""
        try:
            await self._session.close()
        except Exception:
            pass


class TwikitCurlResponse:
    """
    Wraps curl_cffi Response to match httpx.Response interface.
    Twikit accesses: .json(), .text, .content, .status_code, .headers
    """

    def __init__(self, response):
        self._response = response

    @property
    def status_code(self) -> int:
        return self._response.status_code

    @property
    def text(self) -> str:
        return self._response.text

    @property
    def content(self) -> bytes:
        return self._response.content

    @property
    def headers(self):
        return self._response.headers

    def json(self, **kwargs):
        return self._response.json(**kwargs)


_TWIKIT_CLASS_PATCHES_APPLIED = False


def _apply_twikit_class_patches():
    """
    Monkey-patch twikit classes at the CLASS level so ALL instances are covered,
    regardless of which skill created them (X skill, Grok skill, etc.).

    Safe to call multiple times — only applied once.
    """
    global _TWIKIT_CLASS_PATCHES_APPLIED
    if _TWIKIT_CLASS_PATCHES_APPLIED:
        return

    # ── ClientTransaction: bypass KEY_BYTE scraping ──────────────────────────
    # init() scrapes x.com JS for KEY_BYTE indices. X changed the page format
    # so this always fails. Replace init() with a no-op and
    # generate_transaction_id() to return an empty string.
    try:
        from twikit.x_client_transaction.transaction import ClientTransaction

        async def _noop_init(self, session, headers):
            """No-op: skip X.com JS scraping entirely."""
            pass

        def _safe_generate_tid(self, method: str, path: str, **kwargs) -> str:
            """Return empty string — X still accepts requests without this header."""
            return ""

        ClientTransaction.init = _noop_init
        ClientTransaction.generate_transaction_id = _safe_generate_tid
        logger.info("🔑 ClientTransaction KEY_BYTE bypass applied (class-level)")
    except Exception as _kb_err:
        logger.warning(f"ClientTransaction patch skipped: {_kb_err}")

    # ── twikit User: withheld_in_countries KeyError fix ──────────────────────
    try:
        import twikit.user as _twu
        import twikit.guest.user as _twgu

        _orig_user_init = _twu.User.__init__
        _orig_guest_user_init = _twgu.User.__init__

        def _safe_user_init(self_u, client_u, data):
            try:
                if "legacy" in data:
                    _apply_user_legacy_defaults(data)
                elif "result" in data and isinstance(data["result"], dict):
                    _apply_user_legacy_defaults(data["result"])
            except Exception:
                pass
            _orig_user_init(self_u, client_u, data)

        def _safe_guest_user_init(self_u, data):
            try:
                if "legacy" in data:
                    _apply_user_legacy_defaults(data)
                elif "result" in data and isinstance(data["result"], dict):
                    _apply_user_legacy_defaults(data["result"])
            except Exception:
                pass
            _orig_guest_user_init(self_u, data)

        _twu.User.__init__ = _safe_user_init
        _twgu.User.__init__ = _safe_guest_user_init
        logger.info("🛡️ withheld_in_countries patch applied (class-level)")
    except Exception as _wc_err:
        logger.warning(f"withheld_in_countries patch skipped: {_wc_err}")

    _TWIKIT_CLASS_PATCHES_APPLIED = True


def patch_twikit_client(client, proxy: Optional[str] = None):
    """
    Replaces a twikit Client's httpx backend with curl_cffi
    for Android Chrome TLS fingerprint impersonation.

    Call this AFTER creating the twikit Client but BEFORE making requests.

    Args:
        client: twikit.Client instance
        proxy: Optional proxy URL
    """
    if not CURL_CFFI_AVAILABLE:
        logger.warning("curl_cffi not available,  skipping TLS patch")
        return False

    try:
        old_http = client.http
        new_http = TwikitCurlSession(proxy=proxy)

        # Copy existing cookies from the old session
        try:
            old_cookies = dict(old_http.cookies)
            if old_cookies:
                new_http.cookies.update(old_cookies)
        except Exception:
            pass

        # Replace the HTTP backend
        client.http = new_http

        # Reset client_transaction so it re-inits with the new session
        if hasattr(client, 'client_transaction'):
            client.client_transaction.home_page_response = None

        # ── KEY_BYTE bypass (class-level, covers ALL twikit clients) ─────────
        # twikit.client_transaction.init() scrapes x.com JS for KEY_BYTE
        # indices. This fails with "Couldn't get KEY_BYTE indices" because X
        # changed the page format. Patch the CLASS so every instance (X skill,
        # Grok skill, any future client) is covered, not just this one instance.
        _apply_twikit_class_patches()

        logger.info("✅ TLS patch applied,  twikit now uses Chrome Android fingerprint")
        return True

    except Exception as e:
        logger.error(f"❌ TLS patch failed: {e}")
        return False
