# Advanced Web Scraping with Open-Sable

## Overview

Open-Sable now includes advanced web scraping capabilities inspired by **Maxun**, featuring:

- 🎬 **Action Recording** - Record browser actions and replay them
- 🎯 **Smart Element Selection** - CSS selectors, XPath support
- 📊 **Data Extraction Patterns** - Regex, transforms, multiple items
- 📄 **Pagination Handling** - Automatic multi-page scraping
- 🔄 **Dynamic Content** - JavaScript-rendered pages
- 🕵️ **Anti-Bot Detection** - Stealth mode, proxy support
- 💾 **Recipe System** - Save and reuse scraping workflows

---

## Quick Start

### Installation

```bash
# Install scraping dependencies
pip install playwright fake-useragent python-dateutil

# Install Playwright browsers
playwright install chromium
```

### Basic Usage

```python
from skills.advanced_scraper import AdvancedScraper, ScrapingRecipe, ExtractionRule

# Create scraper
scraper = AdvancedScraper()

# Simple scraping
recipe = ScrapingRecipe(
    name="hackernews",
    start_url="https://news.ycombinator.com",
    extraction_rules=[
        ExtractionRule(
            name='titles',
            selector='.titleline > a',
            attribute='text',
            multiple=True
        )
    ]
)

# Execute
results = await scraper.execute_recipe(recipe)
print(f"Found {len(results)} items")
```

---

## Features

### 1. Action Recording

Record your browser interactions and replay them:

```python
scraper = AdvancedScraper()

# Start recording
scraper.start_recording()

# Start browser (visual mode)
await scraper.start_browser(headless=False)

# Perform actions
page = await scraper.navigate("https://example.com")
await scraper.click(page, '.button')
await scraper.type_text(page, 'input[name="search"]', 'query')

# Stop recording
actions = scraper.stop_recording()

# Actions are saved and can be replayed
```

### 2. Data Extraction Rules

Extract data with powerful rules:

```python
ExtractionRule(
    name='prices',
    selector='.product-price',
    attribute='text',
    pattern=r'\$?([\d,]+\.?\d*)',  # Extract number
    transform='float',  # Convert to float
    multiple=True  # Get all matches
)
```

**Supported transforms:**
- `int` - Convert to integer
- `float` - Convert to float
- `json` - Parse JSON
- `date` - Parse date

### 3. Pagination

Automatically scrape multiple pages:

```python
recipe = ScrapingRecipe(
    name="products",
    start_url="https://shop.example.com/products",
    extraction_rules=[...],
    pagination={
        'next_selector': '.pagination .next',
        'max_pages': 10
    }
)
```

### 4. Dynamic Content

Handle JavaScript-rendered content:

```python
recipe = ScrapingRecipe(
    name="spa_app",
    start_url="https://app.example.com",
    actions=[
        ScrapingAction(
            action_type='wait',
            selector='#content',
            wait_after=3000  # Wait 3 seconds
        ),
        ScrapingAction(
            action_type='scroll',
            selector='body'  # Scroll to load more
        )
    ],
    extraction_rules=[...]
)
```

### 5. Anti-Bot Detection

Built-in stealth features:

```python
await scraper.start_browser(
    headless=True,
    proxy={
        'server': 'http://proxy.example.com:8080',
        'username': 'user',
        'password': 'pass'
    }
)
```

**Stealth features:**
- Random user agents
- Navigator.webdriver hidden
- Realistic viewport sizes
- Chrome runtime emulation
- WebGL fingerprinting protection

### 6. Recipe Management

Save and reuse workflows:

```python
# Save recipe
scraper.save_recipe(recipe)

# List recipes
recipes = scraper.list_recipes()

# Load recipe
recipe = scraper.load_recipe("my_recipe")

# Execute
results = await scraper.execute_recipe(recipe)
```

---

## CLI Commands

### Scrape with Recording

```bash
# Interactive scraping (visual mode)
sable scrape https://example.com --no-headless --save-recipe my_recipe

# Execute saved recipe
sable scrape https://example.com --recipe my_recipe

# Headless mode
sable scrape https://example.com --headless
```

### List Recipes

```bash
sable scrape-recipes
```

---

## Examples

### Example 1: Hacker News Scraper

```python
recipe = ScrapingRecipe(
    name="hackernews",
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

results = await scraper.execute_recipe(recipe)
```

### Example 2: E-commerce Product Scraper

```python
recipe = ScrapingRecipe(
    name="products",
    start_url="https://shop.example.com/products",
    actions=[
        ScrapingAction(
            action_type='wait',
            selector='.product-list',
            wait_after=2000
        ),
        ScrapingAction(
            action_type='scroll',
            selector='body'
        )
    ],
    extraction_rules=[
        ExtractionRule(
            name='product_names',
            selector='.product-title',
            attribute='text',
            multiple=True
        ),
        ExtractionRule(
            name='prices',
            selector='.product-price',
            attribute='text',
            pattern=r'\$?([\d,]+\.?\d*)',
            transform='float',
            multiple=True
        ),
        ExtractionRule(
            name='images',
            selector='.product-image img',
            attribute='src',
            multiple=True
        ),
        ExtractionRule(
            name='ratings',
            selector='.rating',
            attribute='data-rating',
            transform='float',
            multiple=True
        )
    ],
    pagination={
        'next_selector': '.pagination .next',
        'max_pages': 5
    }
)
```

### Example 3: Form Interaction

```python
recipe = ScrapingRecipe(
    name="search_results",
    start_url="https://example.com/search",
    actions=[
        # Fill search form
        ScrapingAction(
            action_type='type',
            selector='input[name="q"]',
            value='python web scraping',
            wait_after=500
        ),
        # Select category
        ScrapingAction(
            action_type='select',
            selector='select[name="category"]',
            value='technology'
        ),
        # Submit
        ScrapingAction(
            action_type='click',
            selector='button[type="submit"]',
            wait_after=3000
        ),
        # Wait for results
        ScrapingAction(
            action_type='wait',
            selector='.search-results'
        )
    ],
    extraction_rules=[
        ExtractionRule(
            name='results',
            selector='.result-item',
            attribute='text',
            multiple=True
        )
    ]
)
```

### Example 4: Screenshots

```python
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
    ]
)

# Screenshots saved to ~/.opensable/scraping_recipes/
```

---

## Advanced Features

### AI-Guided Scraping

Let AI guide the scraping process:

```python
from skills.advanced_scraper import scrape_with_ai_guidance

results = await scrape_with_ai_guidance(
    url="https://example.com",
    objective="Find all product prices",
    agent_callback=my_agent_function
)
```

### Custom Actions

Extend with custom actions:

```python
async def custom_action(page, selector, value):
    # Your custom logic
    element = await page.query_selector(selector)
    await element.evaluate("el => el.classList.add('highlight')")

# Register and use in recipes
```

### Data Processing

Process extracted data:

```python
results = await scraper.execute_recipe(recipe)

# Process results
for item in results:
    for key, value in item.items():
        if isinstance(value, list):
            print(f"{key}: {len(value)} items")
        else:
            print(f"{key}: {value}")
```

---

## Comparison with Maxun

| Feature | Maxun | Open-Sable |
|---------|-------|-----------|
| Action Recording | ✅ | ✅ |
| Visual Builder | ✅ | ⚠️ CLI-based |
| Data Extraction | ✅ | ✅ |
| Pagination | ✅ | ✅ |
| Dynamic Content | ✅ | ✅ |
| Anti-Bot | ✅ | ✅ |
| Recipe System | ✅ | ✅ |
| Proxy Support | ✅ | ✅ |
| Screenshots | ✅ | ✅ |
| AI Integration | ❌ | ✅ |
| Multi-Agent | ❌ | ✅ |
| CLI Tool | ❌ | ✅ |
| Python API | ⚠️ Limited | ✅ Full |

---

## Best Practices

1. **Use headless mode in production**
   ```python
   await scraper.start_browser(headless=True)
   ```

2. **Add delays between actions**
   ```python
   ScrapingAction(..., wait_after=2000)  # 2 seconds
   ```

3. **Handle errors gracefully**
   ```python
   try:
       results = await scraper.execute_recipe(recipe)
   except Exception as e:
       logger.error(f"Scraping failed: {e}")
   ```

4. **Use specific selectors**
   ```python
   # Good
   selector='.product-list .item'
   
   # Better
   selector='div[data-testid="product-item"]'
   ```

5. **Respect robots.txt**
   ```python
   # Check before scraping
   import requests
   robots = requests.get('https://example.com/robots.txt')
   ```

6. **Use proxies for large-scale scraping**
   ```python
   proxy={'server': 'http://proxy:8080'}
   ```

---

## Troubleshooting

### Browser doesn't start

```bash
# Install Playwright browsers
playwright install chromium
```

### Elements not found

```python
# Increase wait time
ScrapingAction(
    action_type='wait',
    selector='#element',
    wait_after=5000  # 5 seconds
)
```

### Anti-bot detection

```python
# Use stealth features + proxy
await scraper.start_browser(
    headless=True,
    proxy={'server': 'http://proxy:8080'}
)
```

---

## Performance Tips

1. **Reuse browser instances**
2. **Use CSS selectors (faster than XPath)**
3. **Limit pagination pages**
4. **Enable headless mode**
5. **Use caching for repeated requests**

---

## Integration with Open-Sable

The advanced scraper integrates seamlessly with Open-Sable:

```python
# In your agent code
from skills.advanced_scraper import AdvancedScraper

async def handle_scrape_request(url, objective):
    scraper = AdvancedScraper()
    
    # Use AI to determine what to scrape
    recipe = await create_recipe_from_objective(url, objective)
    
    # Execute
    results = await scraper.execute_recipe(recipe)
    
    return results
```

---

## Future Enhancements

- [ ] Visual recipe builder UI
- [ ] Distributed scraping across multiple machines
- [ ] Built-in data validation
- [ ] Export to CSV/JSON/Excel
- [ ] Scheduled scraping (cron integration)
- [ ] Change detection and alerts
- [ ] CAPTCHA solving integration

---

For more examples, see `examples/scraping_examples.py`
