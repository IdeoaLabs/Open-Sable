/**
 * WebSearch Tool — Search the web for documentation, examples, solutions.
 * 
 * Uses DuckDuckGo Instant Answer API (no API key needed) with
 * fallback to scraping search result pages.
 * Gives the AI the ability to look up library docs, error solutions, etc.
 */

import type { ToolDefinition, ToolContext, ToolInput, ToolResult } from './types';

const SEARCH_TIMEOUT_MS = 10000;
const MAX_RESULTS = 5;

/**
 * Perform a web search using DuckDuckGo HTML and extract results.
 */
async function searchDuckDuckGo(query: string): Promise<Array<{ title: string; url: string; snippet: string }>> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), SEARCH_TIMEOUT_MS);

  try {
    // Use DuckDuckGo HTML search (no API key needed)
    const searchUrl = `https://html.duckduckgo.com/html/?q=${encodeURIComponent(query)}`;
    const response = await fetch(searchUrl, {
      signal: controller.signal,
      headers: {
        'User-Agent': 'Mozilla/5.0 (compatible; SableDev/1.0)',
      },
    });

    if (!response.ok) {
      throw new Error(`Search failed: ${response.status}`);
    }

    const html = await response.text();
    const results: Array<{ title: string; url: string; snippet: string }> = [];

    // Parse DuckDuckGo HTML results
    const resultRegex = /<a[^>]+class="result__a"[^>]*href="([^"]*)"[^>]*>([\s\S]*?)<\/a>[\s\S]*?<a[^>]+class="result__snippet"[^>]*>([\s\S]*?)<\/a>/gi;
    let match;
    while ((match = resultRegex.exec(html)) !== null && results.length < MAX_RESULTS) {
      const rawUrl = match[1];
      const title = match[2].replace(/<[^>]+>/g, '').trim();
      const snippet = match[3].replace(/<[^>]+>/g, '').trim();

      // DuckDuckGo wraps URLs in a redirect, extract the actual URL
      let url = rawUrl;
      const uddgMatch = rawUrl.match(/uddg=([^&]+)/);
      if (uddgMatch) {
        url = decodeURIComponent(uddgMatch[1]);
      }

      if (title && url && url.startsWith('http')) {
        results.push({ title, url, snippet });
      }
    }

    // Fallback: try simpler regex if the above didn't catch results
    if (results.length === 0) {
      const simpleRegex = /<a[^>]+class="result__url"[^>]*href="([^"]*)"[^>]*>/gi;
      const titleRegex = /<a[^>]+class="result__a"[^>]*>([\s\S]*?)<\/a>/gi;
      let urlMatch, titleMatch;
      while ((urlMatch = simpleRegex.exec(html)) !== null && 
             (titleMatch = titleRegex.exec(html)) !== null && 
             results.length < MAX_RESULTS) {
        const url = urlMatch[1];
        const title = titleMatch[1].replace(/<[^>]+>/g, '').trim();
        if (title && url.startsWith('http')) {
          results.push({ title, url, snippet: '' });
        }
      }
    }

    return results;
  } finally {
    clearTimeout(timeout);
  }
}

async function execute(input: ToolInput, _context: ToolContext): Promise<ToolResult> {
  const query = input.query as string;
  if (!query) {
    return { success: false, output: '', error: 'query parameter is required' };
  }

  try {
    const results = await searchDuckDuckGo(query);

    if (results.length === 0) {
      return {
        success: true,
        output: `No results found for: "${query}". Try rephrasing the search query.`,
      };
    }

    const formatted = results.map((r, i) => 
      `${i + 1}. **${r.title}**\n   URL: ${r.url}\n   ${r.snippet}`
    ).join('\n\n');

    return {
      success: true,
      output: `Search results for "${query}":\n\n${formatted}\n\nUse web_fetch to read the full content of any URL above.`,
    };
  } catch (error: any) {
    return {
      success: false,
      output: '',
      error: `Search failed: ${error.message}`,
    };
  }
}

export const WebSearchTool: ToolDefinition = {
  name: 'web_search',
  description: 'Search the web for documentation, code examples, error solutions, library APIs. Returns top results with titles, URLs, and snippets. Use web_fetch to read full content of a result.',
  inputSchema: {
    type: 'object',
    properties: {
      query: {
        type: 'string',
        description: 'The search query. Be specific — e.g. "react-router-dom v6 useNavigate example" or "tailwind css grid layout".',
      },
    },
    required: ['query'],
  },
  isConcurrencySafe: true,
  execute,
};
