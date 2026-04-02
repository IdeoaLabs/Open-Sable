"""
Browser automation engine — PinchTab preferred, Playwright fallback.

When PinchTab is available (local binary or external server), all browsing
goes through its token-efficient HTTP API (~800 tokens/page, stealth mode,
persistent sessions).  When PinchTab is not reachable, the engine falls back
to bundled Playwright automatically — zero config change needed.

Ref: https://github.com/pinchtab/pinchtab
"""

import logging
import asyncio
import os
import re
import subprocess
import sys
from typing import Dict, Optional, Any, List

logger = logging.getLogger(__name__)

# ── Prompt-injection / control-token sanitizer ──────────────────────────────
# Scraped web pages sometimes contain LLM control tokens or injected prompts.
# Strip them before feeding content to the model.
_CONTROL_TOKEN_RE = re.compile(
    r'<\|(?:endoftext|im_start|im_end|pad|sep|system|user|assistant|'  # ChatML / GPT
    r'begin_of_text|end_of_text|eot_id|start_header_id|end_header_id|'  # Llama 3
    r'eos|bos|end_turn|tool_call|tool_result)\|>',  # misc
    re.IGNORECASE,
)
# Also catch common injected role markers that aren't in <|...|> form
_ROLE_INJECTION_RE = re.compile(
    r'(?:^|\n)\s*(?:system|user|assistant)\s*:\s*$',
    re.MULTILINE | re.IGNORECASE,
)


def sanitize_web_content(text: str) -> str:
    """Remove LLM control tokens and injected role markers from scraped text."""
    if not text:
        return text
    text = _CONTROL_TOKEN_RE.sub('', text)
    text = _ROLE_INJECTION_RE.sub('', text)
    # Collapse runs of blank lines left over after stripping
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


class BrowserEngine:
    """Browser automation engine — PinchTab first, Playwright fallback."""

    def __init__(self):
        self.playwright = None
        self.browser = None
        self.page = None  # Persistent page for ref-based automation
        self._initialized = False
        self._setup_lock = asyncio.Lock()
        self._element_refs = {}  # Map ref IDs to elements
        self._ref_counter = 0
        # PinchTab client (lazy init)
        self._pinchtab = None
        self._pinchtab_available = None  # None = not checked yet

    async def _try_pinchtab(self) -> bool:
        """Try to connect to PinchTab. Returns True if available."""
        if self._pinchtab_available is not None:
            return self._pinchtab_available
        # Skip if explicitly disabled
        if os.environ.get("PINCHTAB_DISABLED", "").lower() in ("1", "true", "yes"):
            self._pinchtab_available = False
            return False
        try:
            from .pinchtab_client import PinchTabClient
            self._pinchtab = PinchTabClient()
            if await self._pinchtab.connect():
                self._pinchtab_available = True
                logger.info("🌐 Browser engine: PinchTab mode (token-efficient)")
                return True
        except Exception as e:
            logger.debug(f"PinchTab not available: {e}")
        self._pinchtab_available = False
        self._pinchtab = None
        return False

    async def _ensure_playwright_installed(self) -> bool:
        """Ensure Playwright is installed and browsers are available"""
        try:
            import playwright

            return True
        except ImportError:
            logger.info("Playwright not found. Installing...")
            try:
                # Install playwright package
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "playwright"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

                # Install chromium browser
                subprocess.check_call(
                    [sys.executable, "-m", "playwright", "install", "chromium"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

                logger.info("✅ Playwright installed successfully")
                return True
            except Exception as e:
                logger.error(f"Failed to install Playwright: {e}")
                return False

    async def initialize(self) -> bool:
        """Initialize browser engine — PinchTab first, Playwright fallback."""
        if self._initialized:
            return True

        async with self._setup_lock:
            if self._initialized:
                return True

            # Try PinchTab first (fast, token-efficient, stealth)
            if await self._try_pinchtab():
                self._initialized = True
                return True

            # Fall back to Playwright
            logger.debug("PinchTab not available, falling back to Playwright")

            # Ensure playwright is installed
            if not await self._ensure_playwright_installed():
                return False

            try:
                from playwright.async_api import async_playwright

                self.playwright = await async_playwright().start()
                self.browser = await self.playwright.chromium.launch(
                    headless=True,
                    args=[
                        "--disable-blink-features=AutomationControlled",
                        "--no-sandbox",
                        "--disable-dev-shm-usage",
                        "--disable-gpu",
                    ],
                )

                self._initialized = True
                logger.info("✅ Browser engine initialized")
                return True

            except Exception as e:
                logger.error(f"Failed to initialize browser: {e}")
                return False

    async def scrape_page(self, url: str, max_length: int = 3000) -> Dict[str, str]:
        """
        Scrape content from a web page.
        Uses PinchTab (token-efficient) when available, Playwright otherwise.
        """
        if not await self.initialize():
            return {"error": "Browser engine not available. Playwright installation failed."}

        # ── PinchTab path (token-efficient) ──────────────────────────
        if self._pinchtab and self._pinchtab.available:
            try:
                nav = await self._pinchtab.navigate(url)
                if not nav.get("success"):
                    raise RuntimeError(nav.get("error", "nav failed"))
                await asyncio.sleep(1)  # Let page render
                text_result = await self._pinchtab.text()
                content = text_result.get("text", "")
                content = sanitize_web_content(content)
                if len(content) > max_length:
                    content = content[:max_length] + "...\n\n(Content truncated for brevity)"
                # Try to get title from snapshot
                title = text_result.get("title", url)
                await self._pinchtab.close_tab()
                return {"title": title, "url": url, "content": content, "success": True,
                        "engine": "pinchtab"}
            except Exception as e:
                logger.debug(f"PinchTab scrape failed, trying Playwright: {e}")

        # ── Playwright path ──────────────────────────────────────────

        page = None
        try:
            page = await self.browser.new_page()

            # Set realistic user agent
            await page.set_extra_http_headers(
                {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                }
            )

            # Navigate to page
            await page.goto(url, wait_until="domcontentloaded", timeout=20000)

            # Extract content
            title = await page.title()

            # Get clean text content
            content = await page.evaluate("""() => {
                // Remove unwanted elements
                const unwanted = document.querySelectorAll('script, style, noscript, iframe, nav, footer, header, aside, .ad, .advertisement');
                unwanted.forEach(el => el.remove());
                
                // Get main content
                const main = document.querySelector('main, article, .content, #content') || document.body;
                
                // Extract text
                return main.innerText.trim();
            }""")

            # Sanitize: strip LLM control tokens / prompt injections
            content = sanitize_web_content(content)

            # Truncate if needed
            if len(content) > max_length:
                content = content[:max_length] + "...\n\n(Content truncated for brevity)"

            return {"title": title, "url": url, "content": content, "success": True}

        except Exception as e:
            logger.error(f"Scraping error for {url}: {e}")
            return {"error": f"Failed to scrape {url}: {str(e)}", "success": False}
        finally:
            if page:
                await page.close()

    async def search_web(self, query: str, num_results: int = 5) -> Dict[str, any]:
        """
        Search the web. PinchTab uses Brave with stealth; Playwright as fallback.
        """
        if not await self.initialize():
            return {"error": "Browser engine not available"}

        # ── PinchTab path (stealth, persistent sessions) ─────────────
        if self._pinchtab and self._pinchtab.available:
            try:
                search_url = f"https://search.brave.com/search?q={query.replace(' ', '+')}&source=web"
                nav = await self._pinchtab.navigate(search_url)
                if not nav.get("success"):
                    raise RuntimeError(nav.get("error", "nav failed"))
                await asyncio.sleep(2)  # Let results render
                # Get snapshot for structured refs
                snap = await self._pinchtab.snapshot()
                # Extract text for results
                text_result = await self._pinchtab.text()
                raw = text_result.get("text", "")
                raw = sanitize_web_content(raw)
                # Parse search results from text
                results = self._parse_search_text(raw, num_results)
                await self._pinchtab.close_tab()
                if results:
                    logger.info(f"✅ PinchTab Brave: {len(results)} results")
                    return {"query": query, "results": results, "count": len(results),
                            "success": True, "engine": "pinchtab"}
                logger.debug("PinchTab search got no results, trying Playwright")
            except Exception as e:
                logger.debug(f"PinchTab search failed: {e}")

        # ── Playwright path ──────────────────────────────────────────

        if not self.browser:
            # PinchTab-only mode — Playwright was never initialized
            try:
                from playwright.async_api import async_playwright
                pw = await async_playwright().start()
                self.browser = await pw.chromium.launch(headless=True)
                logger.info("🎭 Playwright launched as search fallback")
            except Exception as e:
                logger.warning(f"Playwright not available for fallback: {e}")
                return {"error": "Search failed: no results from PinchTab and Playwright unavailable",
                        "success": False}

        page = None
        try:
            page = await self.browser.new_page()

            # Brave Search (no captcha, no JS required)
            await page.set_extra_http_headers(
                {
                    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/115.0",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5",
                    "DNT": "1",
                }
            )

            search_url = f"https://search.brave.com/search?q={query.replace(' ', '+')}&source=web"
            await page.goto(search_url, wait_until="domcontentloaded", timeout=15000)
            await asyncio.sleep(3)

            # Extract with JavaScript
            results = await page.evaluate(
                """(maxResults) => {
                const items = [];
                const snippets = document.querySelectorAll('div.snippet');
                
                for (let i = 0; i < Math.min(maxResults, snippets.length); i++) {
                    const snippet = snippets[i];
                    const titleEl = snippet.querySelector('a');
                    const descEl = snippet.querySelector('.snippet-description');
                    
                    if (titleEl) {
                        items.push({
                            title: titleEl.textContent.trim(),
                            url: titleEl.href,
                            snippet: descEl ? descEl.textContent.trim() : ''
                        });
                    }
                }
                
                return items;
            }""",
                num_results,
            )

            # Sanitize snippets
            for r in results:
                if r.get('snippet'):
                    r['snippet'] = sanitize_web_content(r['snippet'])
                if r.get('title'):
                    r['title'] = sanitize_web_content(r['title'])

            if len(results) > 0:
                logger.info(f"✅ Brave: {len(results)} results")
                return {"query": query, "results": results, "count": len(results), "success": True}

            # Fallback to Bing
            logger.warning("Brave failed, trying Bing...")
            await page.goto(
                f"https://www.bing.com/search?q={query.replace(' ', '+')}", timeout=15000
            )
            await asyncio.sleep(2)

            results = await page.evaluate(
                """(maxResults) => {
                const items = [];
                const algos = document.querySelectorAll('li.b_algo');
                
                for (let i = 0; i < Math.min(maxResults, algos.length); i++) {
                    const algo = algos[i];
                    const h2 = algo.querySelector('h2 a');
                    const caption = algo.querySelector('.b_caption p');
                    
                    if (h2) {
                        items.push({
                            title: h2.textContent.trim(),
                            url: h2.href,
                            snippet: caption ? caption.textContent.trim() : ''
                        });
                    }
                }
                
                return items;
            }""",
                num_results,
            )

            # Sanitize Bing snippets
            for r in results:
                if r.get('snippet'):
                    r['snippet'] = sanitize_web_content(r['snippet'])
                if r.get('title'):
                    r['title'] = sanitize_web_content(r['title'])

            logger.info(f"✅ Bing: {len(results)} results")
            return {"query": query, "results": results, "count": len(results), "success": True}

        except Exception as e:
            logger.error(f"Search error: {e}")
            return {"error": f"Search failed: {str(e)}", "success": False}
        finally:
            if page:
                await page.close()

    async def get_page_screenshot(
        self, url: str, output_path: Optional[str] = None
    ) -> Dict[str, str]:
        """Take a screenshot of a web page.

        When *output_path* is ``None`` (default) a temporary file is used and
        the returned dict includes ``auto_cleanup: True`` so callers know the
        file should be deleted after consumption.
        """
        if not await self.initialize():
            return {"error": "Browser engine not available"}

        auto_cleanup = output_path is None  # caller didn't pick a path → temp

        # ── PinchTab path ──────────────────────────────────────────
        if self._pinchtab and self._pinchtab.available:
            try:
                nav = await self._pinchtab.navigate(url)
                if nav.get("success"):
                    await asyncio.sleep(1)
                    png_data = await self._pinchtab.screenshot()
                    await self._pinchtab.close_tab()
                    if png_data:
                        if not output_path:
                            output_path = f"/tmp/sable_ss_{hash(url)}.png"
                        with open(output_path, "wb") as f:
                            f.write(png_data)
                        return {"path": output_path, "url": url, "success": True,
                                "engine": "pinchtab", "auto_cleanup": auto_cleanup}
            except Exception as e:
                logger.debug(f"PinchTab screenshot failed: {e}")

        # ── Playwright path ──────────────────────────────────────────

        page = None
        try:
            page = await self.browser.new_page()
            await page.goto(url, wait_until="domcontentloaded", timeout=20000)

            if not output_path:
                output_path = f"/tmp/sable_ss_{hash(url)}.png"

            await page.screenshot(path=output_path, full_page=True)

            return {"path": output_path, "url": url, "success": True,
                    "auto_cleanup": auto_cleanup}

        except Exception as e:
            logger.error(f"Screenshot error for {url}: {e}")
            return {"error": f"Screenshot failed: {str(e)}", "success": False}
        finally:
            if page:
                await page.close()

    async def cleanup(self):
        """Cleanup browser resources (PinchTab + Playwright)."""
        try:
            if self._pinchtab:
                await self._pinchtab.shutdown()
                self._pinchtab = None
                self._pinchtab_available = None
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
            self._initialized = False
            logger.info("Browser engine cleaned up")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

    async def snapshot(self, url: str = None, format: str = "aria") -> Dict[str, Any]:
        """
        Take a snapshot of the page (accessibility tree or AI format).
        PinchTab returns stable refs natively; Playwright builds them manually.
        """
        if not await self.initialize():
            return {"success": False, "error": "Browser not available"}

        # ── PinchTab path (native stable refs) ──────────────────────
        if self._pinchtab and self._pinchtab.available:
            try:
                if url:
                    nav = await self._pinchtab.navigate(url)
                    if not nav.get("success"):
                        raise RuntimeError(nav.get("error", "nav failed"))
                    await asyncio.sleep(1)
                snap = await self._pinchtab.snapshot()
                if snap.get("success"):
                    snap["engine"] = "pinchtab"
                    return snap
            except Exception as e:
                logger.debug(f"PinchTab snapshot failed: {e}")

        # ── Playwright path ──────────────────────────────────────────

        try:
            # Create or reuse page
            if not self.page or self.page.is_closed():
                self.page = await self.browser.new_page()

            # Navigate if URL provided
            if url:
                await self.page.goto(url, wait_until="domcontentloaded", timeout=15000)
                await asyncio.sleep(1)

            # Get current URL
            current_url = self.page.url

            # Reset refs
            self._element_refs = {}
            self._ref_counter = 0

            # Get accessibility snapshot
            snapshot_data = await self.page.accessibility.snapshot()

            # Process snapshot and generate refs
            refs = []
            self._process_snapshot_node(snapshot_data, refs, depth=0)

            return {
                "success": True,
                "url": current_url,
                "format": format,
                "refs": refs,
                "count": len(refs),
            }

        except Exception as e:
            logger.error(f"Snapshot error: {e}")
            return {"success": False, "error": str(e)}

    def _process_snapshot_node(self, node: Dict, refs: List, depth: int, max_depth: int = 10):
        """Process accessibility tree node and generate refs"""
        if not node or depth > max_depth:
            return

        role = node.get("role", "")
        name = node.get("name", "")

        # Skip generic containers unless they have names
        skip_roles = {"generic", "none", "presentation"}
        if role in skip_roles and not name:
            # Process children directly
            for child in node.get("children", []):
                self._process_snapshot_node(child, refs, depth, max_depth)
            return

        # Generate ref for this element
        if role:
            self._ref_counter += 1
            ref_id = f"e{self._ref_counter}"

            ref_data = {"ref": ref_id, "role": role, "name": name, "depth": depth}

            # Add value for inputs
            if "value" in node:
                ref_data["value"] = node["value"]

            refs.append(ref_data)

            # Store for later use
            self._element_refs[ref_id] = {
                "role": role,
                "name": name,
                "selector": self._generate_selector(role, name),
            }

        # Process children
        for child in node.get("children", []):
            self._process_snapshot_node(child, refs, depth + 1, max_depth)

    def _generate_selector(self, role: str, name: str) -> str:
        """Generate CSS/playwright selector from role and name"""
        # Map ARIA roles to HTML elements
        role_map = {
            "button": "button",
            "link": "a",
            "textbox": "input[type='text'], input[type='search'], input:not([type])",
            "searchbox": "input[type='search']",
            "checkbox": "input[type='checkbox']",
            "radio": "input[type='radio']",
            "combobox": "select",
            "img": "img",
            "heading": "h1, h2, h3, h4, h5, h6",
        }

        element = role_map.get(role, f"[role='{role}']")

        if name:
            # Try to match by text content or aria-label
            return f"{element}:has-text('{name}'), {element}[aria-label*='{name}']"

        return element

    async def execute_action(
        self,
        url: str = None,
        action: str = "click",
        ref: str = None,
        selector: str = None,
        value: str = None,
    ) -> Dict[str, Any]:
        """
        Execute interactive web actions using refs or selectors.
        PinchTab handles refs natively; Playwright needs selector resolution.
        """
        if not await self.initialize():
            return {"success": False, "error": "Browser not available"}

        # ── PinchTab path (native ref support) ──────────────────────
        if self._pinchtab and self._pinchtab.available and ref:
            try:
                if url:
                    nav = await self._pinchtab.navigate(url)
                    if not nav.get("success"):
                        raise RuntimeError(nav.get("error", "nav failed"))
                    await asyncio.sleep(1)
                if action == "click":
                    result = await self._pinchtab.click(ref)
                    return {**result, "engine": "pinchtab"}
                elif action in ("fill", "type"):
                    result = await self._pinchtab.fill(ref, value or "")
                    return {**result, "engine": "pinchtab"}
                elif action == "press":
                    result = await self._pinchtab.press(value or "Enter", ref=ref)
                    return {**result, "engine": "pinchtab"}
            except Exception as e:
                logger.debug(f"PinchTab action failed, trying Playwright: {e}")

        # ── Playwright path ──────────────────────────────────────────

        try:
            # Create or reuse page
            if not self.page or self.page.is_closed():
                self.page = await self.browser.new_page()

            # Navigate if URL provided
            if url:
                await self.page.goto(url, wait_until="domcontentloaded", timeout=15000)
                await asyncio.sleep(1)

            # Resolve selector from ref
            if ref:
                if ref not in self._element_refs:
                    return {"success": False, "error": f"Invalid ref: {ref}. Use snapshot first."}
                selector = self._element_refs[ref]["selector"]

            if not selector and action not in ["wait", "evaluate"]:
                return {"success": False, "error": "Missing ref or selector"}

            # Execute action
            if action == "click":
                element = await self.page.locator(selector).first
                await element.click()
                await asyncio.sleep(0.5)
                return {"success": True, "action": "clicked", "ref": ref}

            elif action == "type" or action == "fill":
                if not value:
                    return {"success": False, "error": "Missing value to type"}
                element = await self.page.locator(selector).first
                await element.fill(value)
                return {"success": True, "action": "typed", "value": value, "ref": ref}

            elif action == "hover":
                element = await self.page.locator(selector).first
                await element.hover()
                return {"success": True, "action": "hovered", "ref": ref}

            elif action == "drag":
                if not value:
                    return {"success": False, "error": "Missing target ref/selector"}
                source = await self.page.locator(selector).first
                target = await self.page.locator(value).first
                await source.drag_to(target)
                return {"success": True, "action": "dragged", "from": ref, "to": value}

            elif action == "select":
                if not value:
                    return {"success": False, "error": "Missing option value"}
                element = await self.page.locator(selector).first
                await element.select_option(value)
                return {"success": True, "action": "selected", "value": value, "ref": ref}

            elif action == "press":
                if not value:
                    return {"success": False, "error": "Missing key to press"}
                await self.page.keyboard.press(value)
                return {"success": True, "action": "pressed", "key": value}

            elif action == "wait":
                wait_time = float(value) if value else 1.0
                await asyncio.sleep(wait_time)
                return {"success": True, "action": "waited", "seconds": wait_time}

            elif action == "evaluate":
                if not value:
                    return {"success": False, "error": "Missing JavaScript code"}
                result = await self.page.evaluate(value)
                return {"success": True, "action": "evaluated", "result": result}

            elif action == "submit":
                form = await self.page.locator(selector or "form").first
                await form.evaluate("form => form.submit()")
                await asyncio.sleep(2)
                return {"success": True, "action": "submitted"}

            else:
                return {"success": False, "error": f"Unknown action: {action}"}

        except Exception as e:
            logger.error(f"Action error: {e}")
            return {"success": False, "error": str(e)}

    @staticmethod
    def _parse_search_text(raw_text: str, max_results: int = 5) -> List[Dict[str, str]]:
        """Parse search results from PinchTab text extraction.

        Brave Search via PinchTab returns text as a continuous block with
        patterns like: ``domain.com › path   Title  snippet text...``
        This parser handles both continuous-text and line-separated formats.
        """
        import re as _re
        results: List[Dict[str, str]] = []

        # ── Strategy 1: Split by domain › pattern ────────────────────
        # Brave format: "{source label} domain.com › seg1 › seg2   Title Date - Snippet"
        tld_pat = r'(?:com|org|net|io|dev|edu|gov|co|info|me|us|uk|de|fr|es|it|ca|au|br|in|ru|jp|ch|nl|se|no|fi|dk|be|at|pt|pl|cz|hu|ro|bg|hr|si|sk|lt|lv|ee|ie|lu|mt|cy)(?:\.[a-z]{2})?'
        domain_re = _re.compile(
            r'([\w.-]+\.(?:' + tld_pat + r'))\s*›',
            _re.IGNORECASE,
        )

        # Find all domain positions to split text into result blocks
        matches = list(domain_re.finditer(raw_text))

        for idx, m in enumerate(matches):
            if len(results) >= max_results:
                break

            domain = m.group(1)
            block_start = m.end()  # after "domain ›"
            # block ends at next domain match or end of text
            block_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw_text)
            block = raw_text[block_start:block_end].strip()

            # The block is: "seg1 › seg2 › file.html   Title Date - Snippet ..."
            # Split path from title at the last multi-space gap AFTER all › segments
            # Find the position of the last › in the block
            last_arrow = block.rfind('›')
            if last_arrow >= 0:
                after_arrow = block[last_arrow + 1:]
                # Find the multi-space gap after the last path segment
                gap_match = _re.search(r'(\S)\s{3,}(\S)', after_arrow)
                if gap_match:
                    path_end = last_arrow + 1 + gap_match.start() + 1
                    path_part = block[:path_end].strip()
                    rest = block[path_end:].strip()
                else:
                    # No triple-space gap, try double-space
                    gap_match = _re.search(r'(\S)\s{2,}(\S)', after_arrow)
                    if gap_match:
                        path_end = last_arrow + 1 + gap_match.start() + 1
                        path_part = block[:path_end].strip()
                        rest = block[path_end:].strip()
                    else:
                        path_part = block[:last_arrow + 1].strip() if last_arrow > 0 else ""
                        rest = block[last_arrow + 1:].strip()
            else:
                # No › in block, try splitting at first triple-space
                gap_match = _re.search(r'(\S)\s{3,}(\S)', block)
                if gap_match:
                    path_part = block[:gap_match.start() + 1].strip()
                    rest = block[gap_match.end() - 1:].strip()
                else:
                    path_part = ""
                    rest = block.strip()

            # Build URL from path segments
            path_segs = [s.strip() for s in path_part.replace('›', '/').split('/') if s.strip()]
            path = '/'.join(path_segs)
            # URL-encode spaces in path segments
            from urllib.parse import quote
            path = quote(path, safe='/.:-_~!$&\'()*+,;=@')
            url = f"https://{domain}/{path}" if path else f"https://{domain}"
            url = url.rstrip(" .")

            # Extract title and snippet from rest
            # Format: "Title Date - Snippet" or "Title - Snippet"
            date_match = _re.search(
                r'\s+((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2},?\s+\d{4})\s*[-–—]?\s*',
                rest,
            )
            if date_match:
                title = rest[:date_match.start()].strip()
                snippet = rest[date_match.end():].strip()
            else:
                # Try dash separator: "Title - Snippet" (but title can contain dashes)
                # Use first sentence-ending dash: "Title – Source  Snippet"
                dash_match = _re.search(r'\s[-–—]\s', rest)
                if dash_match and dash_match.start() > 10:
                    title = rest[:dash_match.start()].strip()
                    snippet = rest[dash_match.end():].strip()
                else:
                    # Take first ~100 chars as title, rest as snippet
                    title = rest[:100].strip()
                    snippet = rest[100:].strip()

            title = _re.sub(r'\s{2,}', ' ', title).strip()
            snippet = _re.sub(r'\s{2,}', ' ', snippet).strip()

            if title and not any(r["url"] == url for r in results):
                results.append({
                    "title": title[:200],
                    "url": url,
                    "snippet": snippet[:500],
                })

        if results:
            return results

        # ── Strategy 2 (fallback): find literal https:// URLs ────────
        lines = raw_text.replace("  ", "\n").split("\n")
        i = 0
        while i < len(lines) and len(results) < max_results:
            line = lines[i].strip()
            url_match = _re.search(r'(https?://\S+)', line)
            if url_match:
                url = url_match.group(1).rstrip(".,;)")
                title = lines[i - 1].strip() if i > 0 else url
                snippet_parts = []
                for j in range(i + 1, min(i + 3, len(lines))):
                    sl = lines[j].strip()
                    if sl and not _re.match(r'https?://', sl) and len(sl) > 10:
                        snippet_parts.append(sl)
                    else:
                        break
                snippet = " ".join(snippet_parts)
                if not any(r["url"] == url for r in results):
                    results.append({
                        "title": title[:200],
                        "url": url,
                        "snippet": snippet[:500],
                    })
            i += 1
        return results

    async def smart_search(self, query: str) -> Dict[str, Any]:
        """Smart search - automatically scrapes top result for detailed content"""
        search_result = await self.search_web(query, num_results=3)

        if not search_result.get("success") or search_result.get("count", 0) == 0:
            return {"success": False, "error": "No results", "query": query}

        results = search_result.get("results", [])

        # Auto-scrape first result
        if len(results) > 0:
            first_url = results[0].get("url")
            logger.info(f"🧠 Auto-scraping: {first_url}")

            try:
                scrape_result = await self.scrape_page(first_url, max_length=2000)
                if scrape_result.get("success"):
                    results[0]["full_content"] = scrape_result.get("content", "")
            except:
                pass

        return {"success": True, "query": query, "results": results, "count": len(results)}
