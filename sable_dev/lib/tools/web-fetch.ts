/**
 * WebFetch Tool,  Fetch content from URLs with safety checks.
 * 
 * Supports:
 *  - HTML pages (returns text content, strips tags)
 *  - JSON APIs (returns formatted JSON)
 *  - Plain text (returns as-is)
 * 
 * Security:
 *  - Only http/https protocols
 *  - Blocks private/internal IPs (10.x, 172.16-31.x, 192.168.x, 127.x, 169.254.x)
 *  - Response size limit (512KB)
 *  - Configurable timeout
 */

import type { ToolDefinition, ToolContext, ToolInput, ToolResult } from './types';

const MAX_RESPONSE_SIZE = 512 * 1024; // 512KB
const FETCH_TIMEOUT_MS = 15000;

/**
 * Check if a hostname resolves to a private/internal IP.
 * Prevents SSRF attacks.
 */
function isBlockedHost(hostname: string): boolean {
  // Block obviously internal hostnames
  const blocked = [
    'localhost',
    '127.0.0.1',
    '0.0.0.0',
    '[::1]',
    'metadata.google.internal',
    'instance-data',
  ];
  if (blocked.includes(hostname.toLowerCase())) return true;

  // Block private IP ranges
  const ipMatch = hostname.match(/^(\d+)\.(\d+)\.(\d+)\.(\d+)$/);
  if (ipMatch) {
    const [, a, b] = ipMatch.map(Number);
    if (a === 10) return true;              // 10.0.0.0/8
    if (a === 172 && b >= 16 && b <= 31) return true; // 172.16.0.0/12
    if (a === 192 && b === 168) return true; // 192.168.0.0/16
    if (a === 127) return true;             // 127.0.0.0/8
    if (a === 169 && b === 254) return true; // link-local
    if (a === 0) return true;               // 0.0.0.0/8
  }

  return false;
}

/**
 * Strip HTML tags and return text content.
 */
function stripHtml(html: string): string {
  // Remove script/style blocks entirely
  let text = html.replace(/<script[\s\S]*?<\/script>/gi, '');
  text = text.replace(/<style[\s\S]*?<\/style>/gi, '');
  // Remove all tags
  text = text.replace(/<[^>]+>/g, ' ');
  // Decode basic entities
  text = text.replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/&nbsp;/g, ' ');
  // Collapse whitespace
  text = text.replace(/\s+/g, ' ').trim();
  // Limit length
  if (text.length > 10000) {
    text = text.substring(0, 10000) + '\n...(truncated)';
  }
  return text;
}

async function execute(input: ToolInput, _context: ToolContext): Promise<ToolResult> {
  const url = input.url as string;
  if (!url) {
    return { success: false, output: '', error: 'url parameter is required' };
  }

  // Validate URL
  let parsed: URL;
  try {
    parsed = new URL(url);
  } catch {
    return { success: false, output: '', error: `Invalid URL: ${url}` };
  }

  // Only allow http/https
  if (!['http:', 'https:'].includes(parsed.protocol)) {
    return { success: false, output: '', error: `Blocked protocol: ${parsed.protocol}. Only http and https are allowed.` };
  }

  // Block internal hosts
  if (isBlockedHost(parsed.hostname)) {
    return { success: false, output: '', error: `Blocked host: ${parsed.hostname}. Internal/private addresses are not allowed.` };
  }

  const format = (input.format as string) || 'auto';

  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);

    const response = await fetch(url, {
      signal: controller.signal,
      headers: {
        'User-Agent': 'SableDev/1.0 (Code Assistant)',
        'Accept': 'text/html,application/json,text/plain,*/*',
      },
      redirect: 'follow',
    });

    clearTimeout(timeout);

    if (!response.ok) {
      return { success: false, output: '', error: `HTTP ${response.status}: ${response.statusText}` };
    }

    // Check content length
    const contentLength = response.headers.get('content-length');
    if (contentLength && parseInt(contentLength) > MAX_RESPONSE_SIZE) {
      return { success: false, output: '', error: `Response too large: ${contentLength} bytes (max: ${MAX_RESPONSE_SIZE})` };
    }

    const contentType = response.headers.get('content-type') || '';
    const body = await response.text();

    if (body.length > MAX_RESPONSE_SIZE) {
      return { success: false, output: '', error: `Response body too large: ${body.length} bytes (max: ${MAX_RESPONSE_SIZE})` };
    }

    let output: string;

    if (format === 'json' || (format === 'auto' && contentType.includes('json'))) {
      try {
        const json = JSON.parse(body);
        output = JSON.stringify(json, null, 2);
        if (output.length > 10000) {
          output = output.substring(0, 10000) + '\n...(truncated)';
        }
      } catch {
        output = body.substring(0, 10000);
      }
    } else if (format === 'html' || (format === 'auto' && contentType.includes('html'))) {
      output = stripHtml(body);
    } else {
      output = body.substring(0, 10000);
      if (body.length > 10000) output += '\n...(truncated)';
    }

    return {
      success: true,
      output: `URL: ${url}\nStatus: ${response.status}\nContent-Type: ${contentType}\n\n${output}`,
    };
  } catch (err: any) {
    if (err.name === 'AbortError') {
      return { success: false, output: '', error: `Request timed out after ${FETCH_TIMEOUT_MS}ms` };
    }
    return { success: false, output: '', error: `Fetch failed: ${err.message}` };
  }
}

export const WebFetchTool: ToolDefinition = {
  name: 'web_fetch',
  description: 'Fetch content from a URL. Returns page text (HTML stripped), JSON, or raw text. Use to read documentation, API responses, or reference pages.',
  inputSchema: {
    type: 'object',
    properties: {
      url: {
        type: 'string',
        description: 'The URL to fetch (must be http or https)',
      },
      format: {
        type: 'string',
        description: 'Response format: "auto" (detect from content-type), "html" (strip tags), "json" (parse JSON), "text" (raw text). Default: auto',
      },
    },
    required: ['url'],
  },
  isConcurrencySafe: true,
  execute,
};
