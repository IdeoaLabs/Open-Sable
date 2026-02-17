"""
Advanced Web Scraping Examples using Open-Sable

Demonstrates various scraping techniques inspired by Maxun.
"""

import asyncio
import json
from skills.advanced_scraper import (
    AdvancedScraper,
    ScrapingRecipe,
    ScrapingAction,
    ExtractionRule
)


async def example_1_simple_scraping():
    """Example 1: Simple data extraction"""
    print("\n=== Example 1: Simple Scraping ===\n")
    
    scraper = AdvancedScraper()
    
    recipe = ScrapingRecipe(
        name="hackernews_top",
        start_url="https://news.ycombinator.com",
        extraction_rules=[
            ExtractionRule(
                name='titles',
                selector='.titleline > a',
                attribute='text',
                multiple=True
            ),
            ExtractionRule(
                name='urls',
                selector='.titleline > a',
                attribute='href',
                multiple=True
            ),
            ExtractionRule(
                name='scores',
                selector='.score',
                attribute='text',
                pattern=r'(\d+)',
                transform='int',
                multiple=True
            )
        ]
    )
    
    # Save recipe for reuse
    scraper.save_recipe(recipe)
    
    # Execute
    results = await scraper.execute_recipe(recipe)
    
    print(f"Extracted {len(results)} items")
    if results:
        print(json.dumps(results[0], indent=2))


async def example_2_action_recording():
    """Example 2: Record actions interactively"""
    print("\n=== Example 2: Action Recording ===\n")
    
    scraper = AdvancedScraper()
    
    # Start recording
    scraper.start_recording()
    
    # Start browser in non-headless mode (visual)
    await scraper.start_browser(headless=False)
    
    # Perform actions (these will be recorded)
    page = await scraper.navigate("https://github.com/search")
    
    await scraper.type_text(page, 'input[name="q"]', 'web scraping python')
    await scraper.click(page, 'button[type="submit"]')
    await scraper.wait_for_selector(page, '.repo-list')
    
    # Stop recording
    actions = scraper.stop_recording()
    
    print(f"Recorded {len(actions)} actions:")
    for i, action in enumerate(actions, 1):
        print(f"  {i}. {action.action_type}: {action.selector}")
    
    await scraper.stop_browser()


async def example_3_pagination_scraping():
    """Example 3: Scrape multiple pages with pagination"""
    print("\n=== Example 3: Pagination Scraping ===\n")
    
    scraper = AdvancedScraper()
    
    recipe = ScrapingRecipe(
        name="github_repos",
        start_url="https://github.com/trending",
        actions=[
            ScrapingAction(
                action_type='wait',
                selector='article.Box-row',
                wait_after=2000
            )
        ],
        extraction_rules=[
            ExtractionRule(
                name='repo_names',
                selector='h2.h3 a',
                attribute='text',
                multiple=True
            ),
            ExtractionRule(
                name='descriptions',
                selector='p.col-9',
                attribute='text',
                multiple=True
            ),
            ExtractionRule(
                name='stars',
                selector='span.d-inline-block.float-sm-right',
                attribute='text',
                pattern=r'([\d,]+)',
                multiple=True
            )
        ],
        settings={
            'headless': True
        }
    )
    
    results = await scraper.execute_recipe(recipe)
    
    print(f"Extracted {len(results)} trending repositories")
    if results:
        print("\nTop 5 repos:")
        for i, repo in enumerate(results[0].get('repo_names', [])[:5], 1):
            print(f"  {i}. {repo}")


async def example_4_dynamic_content():
    """Example 4: Handle dynamic JavaScript content"""
    print("\n=== Example 4: Dynamic Content ===\n")
    
    scraper = AdvancedScraper()
    
    recipe = ScrapingRecipe(
        name="spa_scraper",
        start_url="https://example.com/spa-app",
        actions=[
            # Wait for initial load
            ScrapingAction(
                action_type='wait',
                selector='#app',
                wait_after=3000
            ),
            # Scroll to load more content
            ScrapingAction(
                action_type='scroll',
                selector='body'
            ),
            # Wait for loaded content
            ScrapingAction(
                action_type='wait',
                selector='.loaded-items',
                wait_after=2000
            )
        ],
        extraction_rules=[
            ExtractionRule(
                name='items',
                selector='.item-card',
                attribute='text',
                multiple=True
            )
        ]
    )
    
    results = await scraper.execute_recipe(recipe)
    print(f"Extracted {len(results)} items from SPA")


async def example_5_form_interaction():
    """Example 5: Fill forms and scrape results"""
    print("\n=== Example 5: Form Interaction ===\n")
    
    scraper = AdvancedScraper()
    
    recipe = ScrapingRecipe(
        name="form_scraper",
        start_url="https://example.com/search",
        actions=[
            # Fill search form
            ScrapingAction(
                action_type='type',
                selector='input[name="q"]',
                value='machine learning',
                wait_after=500
            ),
            # Select category
            ScrapingAction(
                action_type='select',
                selector='select[name="category"]',
                value='technology'
            ),
            # Submit form
            ScrapingAction(
                action_type='click',
                selector='button[type="submit"]',
                wait_after=3000
            ),
            # Wait for results
            ScrapingAction(
                action_type='wait',
                selector='.search-results',
                wait_after=2000
            )
        ],
        extraction_rules=[
            ExtractionRule(
                name='result_titles',
                selector='.result-title',
                attribute='text',
                multiple=True
            ),
            ExtractionRule(
                name='result_links',
                selector='.result-link',
                attribute='href',
                multiple=True
            )
        ]
    )
    
    results = await scraper.execute_recipe(recipe)
    print(f"Form search returned {len(results)} results")


async def example_6_proxy_rotation():
    """Example 6: Use proxy for scraping"""
    print("\n=== Example 6: Proxy Scraping ===\n")
    
    scraper = AdvancedScraper()
    
    recipe = ScrapingRecipe(
        name="proxy_scraper",
        start_url="https://httpbin.org/ip",
        extraction_rules=[
            ExtractionRule(
                name='ip_data',
                selector='pre',
                attribute='text',
                transform='json'
            )
        ],
        settings={
            'headless': True,
            'proxy': {
                'server': 'http://proxy.example.com:8080',
                'username': 'user',
                'password': 'pass'
            }
        }
    )
    
    # Note: This requires a valid proxy
    # results = await scraper.execute_recipe(recipe)
    print("Proxy scraping recipe configured (requires valid proxy)")


async def example_7_screenshot_scraping():
    """Example 7: Take screenshots during scraping"""
    print("\n=== Example 7: Screenshot Scraping ===\n")
    
    scraper = AdvancedScraper()
    
    recipe = ScrapingRecipe(
        name="screenshot_scraper",
        start_url="https://example.com",
        actions=[
            ScrapingAction(
                action_type='wait',
                selector='body',
                wait_after=2000,
                screenshot=True  # Take screenshot
            )
        ],
        extraction_rules=[
            ExtractionRule(
                name='title',
                selector='h1',
                attribute='text'
            )
        ]
    )
    
    results = await scraper.execute_recipe(recipe)
    print("Screenshot saved during scraping")


async def example_8_list_and_reuse_recipes():
    """Example 8: List and reuse saved recipes"""
    print("\n=== Example 8: Recipe Management ===\n")
    
    scraper = AdvancedScraper()
    
    # List all saved recipes
    recipes = scraper.list_recipes()
    print(f"Saved recipes ({len(recipes)}):")
    for recipe_name in recipes:
        print(f"  - {recipe_name}")
    
    # Load and execute existing recipe
    if recipes:
        recipe_name = recipes[0]
        print(f"\nLoading recipe: {recipe_name}")
        
        recipe = scraper.load_recipe(recipe_name)
        print(f"  URL: {recipe.start_url}")
        print(f"  Actions: {len(recipe.actions)}")
        print(f"  Extraction rules: {len(recipe.extraction_rules)}")
        
        # Execute
        # results = await scraper.execute_recipe(recipe)


async def main():
    """Run all examples"""
    print("=" * 60)
    print("Open-Sable Advanced Web Scraping Examples")
    print("Inspired by Maxun - No-code web scraping")
    print("=" * 60)
    
    # Run examples
    examples = [
        ("Simple Scraping", example_1_simple_scraping),
        ("Action Recording", example_2_action_recording),
        ("Pagination", example_3_pagination_scraping),
        ("Recipe Management", example_8_list_and_reuse_recipes),
    ]
    
    for name, example_func in examples:
        print(f"\n{'=' * 60}")
        print(f"Running: {name}")
        print(f"{'=' * 60}")
        
        try:
            await example_func()
        except Exception as e:
            print(f"Error in {name}: {e}")
        
        print()
        await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())
