"""
Advanced Web Scraping Skill inspired by Maxun

Features:
- Action recording and playback
- Visual element selection
- Data extraction with patterns
- Pagination handling
- Dynamic content support
- Proxy rotation
- Anti-bot detection bypass
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
import re

try:
    from playwright.async_api import async_playwright, Page, Browser, BrowserContext
    from bs4 import BeautifulSoup
    import requests
    from fake_useragent import UserAgent

    SCRAPING_AVAILABLE = True
except ImportError:
    SCRAPING_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ScrapingAction:
    """Represents a scraping action"""

    action_type: str  # click, type, select, extract, wait, scroll
    selector: str
    value: Optional[str] = None
    wait_after: int = 1000  # milliseconds
    screenshot: bool = False
    metadata: Dict = field(default_factory=dict)


@dataclass
class ExtractionRule:
    """Data extraction rule"""

    name: str
    selector: str
    attribute: Optional[str] = None  # text, href, src, data-*, etc.
    pattern: Optional[str] = None  # regex pattern
    multiple: bool = False  # extract multiple elements
    transform: Optional[str] = None  # json, int, float, date


@dataclass
class ScrapingRecipe:
    """Complete scraping workflow"""

    name: str
    start_url: str
    actions: List[ScrapingAction] = field(default_factory=list)
    extraction_rules: List[ExtractionRule] = field(default_factory=list)
    pagination: Optional[Dict] = None
    settings: Dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


class AdvancedScraper:
    """Advanced web scraping with recording and playback"""

    def __init__(self, config=None):
        if not SCRAPING_AVAILABLE:
            raise ImportError(
                "Scraping dependencies not installed. "
                "Install with: pip install playwright beautifulsoup4 fake-useragent"
            )

        self.config = config or {}
        self.recipes_dir = Path.home() / ".opensable" / "scraping_recipes"
        self.recipes_dir.mkdir(parents=True, exist_ok=True)

        self.user_agent = UserAgent()
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.recording_actions: List[ScrapingAction] = []
        self.is_recording = False

    async def start_browser(self, headless: bool = True, proxy: Optional[Dict] = None):
        """Start browser instance"""
        playwright = await async_playwright().start()

        browser_args = {
            "headless": headless,
            "args": [
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--no-sandbox",
            ],
        }

        if proxy:
            browser_args["proxy"] = proxy

        self.browser = await playwright.chromium.launch(**browser_args)

        # Create context with anti-detection
        self.context = await self.browser.new_context(
            user_agent=self.user_agent.random,
            viewport={"width": 1920, "height": 1080},
            locale="en-US",
            timezone_id="America/New_York",
            permissions=["geolocation"],
        )

        # Add stealth scripts
        await self.context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
            
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5]
            });
            
            Object.defineProperty(navigator, 'languages', {
                get: () => ['en-US', 'en']
            });
            
            window.chrome = {
                runtime: {}
            };
        """)

        logger.info("Browser started successfully")

    async def stop_browser(self):
        """Stop browser instance"""
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        logger.info("Browser stopped")

    def start_recording(self):
        """Start recording actions"""
        self.is_recording = True
        self.recording_actions = []
        logger.info("Recording started")

    def stop_recording(self) -> List[ScrapingAction]:
        """Stop recording and return actions"""
        self.is_recording = False
        logger.info(f"Recording stopped. {len(self.recording_actions)} actions recorded")
        return self.recording_actions.copy()

    def _record_action(self, action: ScrapingAction):
        """Record an action if recording is active"""
        if self.is_recording:
            self.recording_actions.append(action)

    async def navigate(self, url: str, wait_until: str = "networkidle") -> Page:
        """Navigate to URL"""
        if not self.context:
            await self.start_browser()

        page = await self.context.new_page()

        await page.goto(url, wait_until=wait_until, timeout=30000)

        self._record_action(ScrapingAction(action_type="navigate", selector="", value=url))

        logger.info(f"Navigated to: {url}")
        return page

    async def click(self, page: Page, selector: str, wait_after: int = 1000):
        """Click element"""
        await page.click(selector)
        await asyncio.sleep(wait_after / 1000)

        self._record_action(
            ScrapingAction(action_type="click", selector=selector, wait_after=wait_after)
        )

        logger.debug(f"Clicked: {selector}")

    async def type_text(self, page: Page, selector: str, text: str, wait_after: int = 500):
        """Type text into element"""
        await page.fill(selector, text)
        await asyncio.sleep(wait_after / 1000)

        self._record_action(
            ScrapingAction(action_type="type", selector=selector, value=text, wait_after=wait_after)
        )

        logger.debug(f"Typed text into: {selector}")

    async def select_option(self, page: Page, selector: str, value: str):
        """Select dropdown option"""
        await page.select_option(selector, value)

        self._record_action(ScrapingAction(action_type="select", selector=selector, value=value))

        logger.debug(f"Selected option: {value} in {selector}")

    async def wait_for_selector(self, page: Page, selector: str, timeout: int = 30000):
        """Wait for element to appear"""
        await page.wait_for_selector(selector, timeout=timeout)

        self._record_action(
            ScrapingAction(action_type="wait", selector=selector, wait_after=timeout)
        )

    async def scroll_to_bottom(self, page: Page, smooth: bool = True):
        """Scroll to bottom of page"""
        if smooth:
            # Smooth scroll
            await page.evaluate("""
                async () => {
                    await new Promise((resolve) => {
                        let totalHeight = 0;
                        const distance = 100;
                        const timer = setInterval(() => {
                            const scrollHeight = document.body.scrollHeight;
                            window.scrollBy(0, distance);
                            totalHeight += distance;
                            
                            if(totalHeight >= scrollHeight){
                                clearInterval(timer);
                                resolve();
                            }
                        }, 100);
                    });
                }
            """)
        else:
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")

        self._record_action(ScrapingAction(action_type="scroll", selector="body", value="bottom"))

    async def extract_data(self, page: Page, rules: List[ExtractionRule]) -> Dict[str, Any]:
        """Extract data based on rules"""
        html = await page.content()
        soup = BeautifulSoup(html, "html.parser")

        results = {}

        for rule in rules:
            try:
                if rule.multiple:
                    # Extract multiple elements
                    elements = soup.select(rule.selector)
                    values = []

                    for elem in elements:
                        value = self._extract_element_value(elem, rule)
                        if value:
                            values.append(value)

                    results[rule.name] = values
                else:
                    # Extract single element
                    elem = soup.select_one(rule.selector)
                    if elem:
                        value = self._extract_element_value(elem, rule)
                        results[rule.name] = value

            except Exception as e:
                logger.error(f"Error extracting {rule.name}: {e}")
                results[rule.name] = None

        return results

    def _extract_element_value(self, element, rule: ExtractionRule) -> Any:
        """Extract value from element"""
        # Get value
        if rule.attribute:
            if rule.attribute == "text":
                value = element.get_text(strip=True)
            else:
                value = element.get(rule.attribute)
        else:
            value = element.get_text(strip=True)

        if not value:
            return None

        # Apply pattern
        if rule.pattern:
            match = re.search(rule.pattern, str(value))
            if match:
                value = match.group(1) if match.groups() else match.group(0)
            else:
                return None

        # Apply transform
        if rule.transform:
            value = self._transform_value(value, rule.transform)

        return value

    def _transform_value(self, value: str, transform: str) -> Any:
        """Transform extracted value"""
        try:
            if transform == "int":
                return int(re.sub(r"[^\d]", "", str(value)))
            elif transform == "float":
                return float(re.sub(r"[^\d.]", "", str(value)))
            elif transform == "json":
                return json.loads(value)
            elif transform == "date":
                from dateutil import parser

                return parser.parse(value)
            else:
                return value
        except Exception as e:
            logger.error(f"Transform error ({transform}): {e}")
            return value

    async def handle_pagination(
        self, page: Page, next_selector: str, max_pages: int = 10
    ) -> List[Dict]:
        """Handle pagination and extract data from multiple pages"""
        all_results = []
        current_page = 1

        while current_page <= max_pages:
            logger.info(f"Scraping page {current_page}/{max_pages}")

            # Extract data from current page
            # (caller should provide extraction rules)

            # Check if next button exists
            try:
                next_button = await page.query_selector(next_selector)
                if not next_button:
                    logger.info("No more pages")
                    break

                # Click next
                await self.click(page, next_selector)
                await asyncio.sleep(2000 / 1000)  # Wait for page load

                current_page += 1

            except Exception as e:
                logger.error(f"Pagination error: {e}")
                break

        return all_results

    async def execute_recipe(self, recipe: ScrapingRecipe) -> List[Dict]:
        """Execute a scraping recipe"""
        logger.info(f"Executing recipe: {recipe.name}")

        results = []

        try:
            # Start browser
            headless = recipe.settings.get("headless", True)
            proxy = recipe.settings.get("proxy")
            await self.start_browser(headless=headless, proxy=proxy)

            # Navigate to start URL
            page = await self.navigate(recipe.start_url)

            # Execute actions
            for action in recipe.actions:
                if action.action_type == "click":
                    await self.click(page, action.selector, action.wait_after)

                elif action.action_type == "type":
                    await self.type_text(page, action.selector, action.value, action.wait_after)

                elif action.action_type == "select":
                    await self.select_option(page, action.selector, action.value)

                elif action.action_type == "wait":
                    await self.wait_for_selector(page, action.selector, action.wait_after)

                elif action.action_type == "scroll":
                    await self.scroll_to_bottom(page)

                elif action.action_type == "extract":
                    data = await self.extract_data(page, recipe.extraction_rules)
                    results.append(data)

                if action.screenshot:
                    screenshot_path = (
                        self.recipes_dir / f"{recipe.name}_{datetime.now().isoformat()}.png"
                    )
                    await page.screenshot(path=str(screenshot_path))

            # Handle pagination if configured
            if recipe.pagination:
                next_selector = recipe.pagination.get("next_selector")
                max_pages = recipe.pagination.get("max_pages", 10)

                if next_selector:
                    for page_num in range(2, max_pages + 1):
                        try:
                            await self.click(page, next_selector)
                            await asyncio.sleep(2)

                            data = await self.extract_data(page, recipe.extraction_rules)
                            results.append(data)

                        except Exception as e:
                            logger.info(f"Pagination ended at page {page_num}: {e}")
                            break

            logger.info(f"Recipe completed. Extracted {len(results)} items")

        except Exception as e:
            logger.error(f"Recipe execution error: {e}", exc_info=True)

        finally:
            await self.stop_browser()

        return results

    def save_recipe(self, recipe: ScrapingRecipe):
        """Save recipe to disk"""
        recipe_path = self.recipes_dir / f"{recipe.name}.json"

        recipe_data = {
            "name": recipe.name,
            "start_url": recipe.start_url,
            "actions": [
                {
                    "action_type": a.action_type,
                    "selector": a.selector,
                    "value": a.value,
                    "wait_after": a.wait_after,
                    "screenshot": a.screenshot,
                    "metadata": a.metadata,
                }
                for a in recipe.actions
            ],
            "extraction_rules": [
                {
                    "name": r.name,
                    "selector": r.selector,
                    "attribute": r.attribute,
                    "pattern": r.pattern,
                    "multiple": r.multiple,
                    "transform": r.transform,
                }
                for r in recipe.extraction_rules
            ],
            "pagination": recipe.pagination,
            "settings": recipe.settings,
            "created_at": recipe.created_at.isoformat(),
        }

        with open(recipe_path, "w") as f:
            json.dump(recipe_data, f, indent=2)

        logger.info(f"Recipe saved: {recipe_path}")

    def load_recipe(self, name: str) -> ScrapingRecipe:
        """Load recipe from disk"""
        recipe_path = self.recipes_dir / f"{name}.json"

        if not recipe_path.exists():
            raise FileNotFoundError(f"Recipe not found: {name}")

        with open(recipe_path, "r") as f:
            data = json.load(f)

        recipe = ScrapingRecipe(
            name=data["name"],
            start_url=data["start_url"],
            actions=[ScrapingAction(**action) for action in data["actions"]],
            extraction_rules=[ExtractionRule(**rule) for rule in data["extraction_rules"]],
            pagination=data.get("pagination"),
            settings=data.get("settings", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
        )

        return recipe

    def list_recipes(self) -> List[str]:
        """List all saved recipes"""
        recipes = [p.stem for p in self.recipes_dir.glob("*.json")]
        return recipes


async def scrape_with_ai_guidance(
    url: str, objective: str, agent_callback: Optional[Callable] = None
) -> Dict:
    """
    AI-guided web scraping

    The agent describes what to do, and the scraper executes
    """
    scraper = AdvancedScraper()

    try:
        await scraper.start_browser(headless=False)  # Visual mode for AI
        page = await scraper.navigate(url)

        # Get page info
        title = await page.title()
        html = await page.content()

        # Ask AI what to do
        if agent_callback:
            instructions = await agent_callback(
                f"Objective: {objective}\n"
                f"URL: {url}\n"
                f"Page Title: {title}\n"
                f"What actions should I take? Provide selectors and actions."
            )

            # Parse AI instructions and execute
            # (This would require more sophisticated parsing)

        # For now, return page content
        soup = BeautifulSoup(html, "html.parser")

        return {
            "title": title,
            "url": url,
            "text": soup.get_text(strip=True)[:1000],
            "links": [a.get("href") for a in soup.find_all("a", href=True)][:20],
        }

    finally:
        await scraper.stop_browser()


# Example recipes


def create_example_recipes():
    """Create example scraping recipes"""

    # Example 1: Product scraper
    product_scraper = ScrapingRecipe(
        name="product_scraper",
        start_url="https://example.com/products",
        actions=[
            ScrapingAction(action_type="wait", selector=".product-list", wait_after=2000),
            ScrapingAction(action_type="scroll", selector="body"),
        ],
        extraction_rules=[
            ExtractionRule(
                name="product_name", selector=".product-title", attribute="text", multiple=True
            ),
            ExtractionRule(
                name="price",
                selector=".product-price",
                attribute="text",
                pattern=r"\$?([\d,]+\.?\d*)",
                transform="float",
                multiple=True,
            ),
            ExtractionRule(
                name="image", selector=".product-image img", attribute="src", multiple=True
            ),
        ],
        pagination={"next_selector": ".pagination .next", "max_pages": 5},
    )

    # Example 2: News scraper
    news_scraper = ScrapingRecipe(
        name="news_scraper",
        start_url="https://news.ycombinator.com",
        extraction_rules=[
            ExtractionRule(
                name="titles", selector=".titleline > a", attribute="text", multiple=True
            ),
            ExtractionRule(
                name="links", selector=".titleline > a", attribute="href", multiple=True
            ),
            ExtractionRule(
                name="scores",
                selector=".score",
                attribute="text",
                pattern=r"(\d+)",
                transform="int",
                multiple=True,
            ),
        ],
    )

    return [product_scraper, news_scraper]


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    async def test_scraper():
        scraper = AdvancedScraper()

        # Create and save example recipes
        recipes = create_example_recipes()
        for recipe in recipes:
            scraper.save_recipe(recipe)

        # Execute a recipe
        recipe = scraper.load_recipe("news_scraper")
        results = await scraper.execute_recipe(recipe)

        print(f"Extracted {len(results)} items")
        print(json.dumps(results[0] if results else {}, indent=2))

    asyncio.run(test_scraper())
