"""
Browser automation skill for Open-Sable - Playwright integration
"""
import logging
from typing import Optional, Dict, Any
import asyncio

logger = logging.getLogger(__name__)


class BrowserSkill:
    """Browser automation using Playwright"""
    
    def __init__(self, config):
        self.config = config
        self.playwright = None
        self.browser = None
        self.context = None
    
    async def initialize(self):
        """Initialize Playwright"""
        try:
            from playwright.async_api import async_playwright
            
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(
                headless=True,
                args=['--no-sandbox', '--disable-setuid-sandbox']
            )
            self.context = await self.browser.new_context()
            
            logger.info("Browser skill initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize browser: {e}")
            logger.info("Browser skill will run in demo mode")
    
    async def search_web(self, query: str) -> Dict[str, Any]:
        """Search the web using DuckDuckGo (privacy-friendly)"""
        if not self.browser:
            return self._demo_search_results(query)
        
        try:
            page = await self.context.new_page()
            
            # Use DuckDuckGo for privacy
            await page.goto(f"https://duckduckgo.com/?q={query}")
            await page.wait_for_load_state('networkidle')
            
            # Extract results
            results = await page.query_selector_all('.result__a')
            
            search_results = []
            for i, result in enumerate(results[:5]):
                title = await result.inner_text()
                link = await result.get_attribute('href')
                search_results.append({
                    'title': title,
                    'url': link
                })
            
            await page.close()
            
            return {
                'query': query,
                'results': search_results
            }
            
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return self._demo_search_results(query)
    
    async def navigate_to(self, url: str) -> Dict[str, Any]:
        """Navigate to a URL and extract basic info"""
        if not self.browser:
            return {'url': url, 'title': 'Demo Page', 'content': 'Demo mode'}
        
        try:
            # Check if URL is allowed
            if not self._is_allowed_domain(url):
                return {'error': f'Domain not allowed: {url}'}
            
            page = await self.context.new_page()
            await page.goto(url, wait_until='networkidle')
            
            title = await page.title()
            content = await page.content()
            
            # Extract main text (simplified)
            text_content = await page.inner_text('body')
            
            await page.close()
            
            return {
                'url': url,
                'title': title,
                'text': text_content[:1000]  # First 1000 chars
            }
            
        except Exception as e:
            logger.error(f"Navigation failed: {e}")
            return {'error': str(e)}
    
    async def screenshot(self, url: str, path: str = './data/screenshot.png') -> bool:
        """Take a screenshot of a URL"""
        if not self.browser:
            logger.info(f"[DEMO] Would take screenshot of {url}")
            return True
        
        try:
            page = await self.context.new_page()
            await page.goto(url)
            await page.screenshot(path=path)
            await page.close()
            
            logger.info(f"Screenshot saved to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Screenshot failed: {e}")
            return False
    
    def _is_allowed_domain(self, url: str) -> bool:
        """Check if domain is in allowed list"""
        if not self.config.enable_sandbox:
            return True
        
        from urllib.parse import urlparse
        domain = urlparse(url).netloc
        
        # Simple wildcard matching
        for allowed in self.config.allowed_domains:
            if allowed.startswith('*.'):
                if domain.endswith(allowed[2:]):
                    return True
            elif domain == allowed:
                return True
        
        return False
    
    def _demo_search_results(self, query: str) -> Dict[str, Any]:
        """Return demo search results"""
        return {
            'query': query,
            'results': [
                {
                    'title': f'Result 1 for {query}',
                    'url': 'https://example.com/1'
                },
                {
                    'title': f'Result 2 for {query}',
                    'url': 'https://example.com/2'
                }
            ]
        }
    
    async def cleanup(self):
        """Cleanup browser resources"""
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
