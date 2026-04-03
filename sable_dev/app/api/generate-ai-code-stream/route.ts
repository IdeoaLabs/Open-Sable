import { NextRequest, NextResponse } from 'next/server';
import { createGroq } from '@ai-sdk/groq';
import { createAnthropic } from '@ai-sdk/anthropic';
import { createOpenAI } from '@ai-sdk/openai';
import { createGoogleGenerativeAI } from '@ai-sdk/google';
import { streamText } from 'ai';
import type { SandboxState } from '@/types/sandbox';
import { selectFilesForEdit, getFileContents, formatFilesForAI } from '@/lib/context-selector';
import { executeSearchPlan, formatSearchResultsForAI, selectTargetFile } from '@/lib/file-search-executor';
import { FileManifest } from '@/types/file-manifest';
import type { ConversationState, ConversationMessage, ConversationEdit } from '@/types/conversation';
import { appConfig } from '@/config/app.config';
import { getTemplate, DEFAULT_TEMPLATE } from '@/lib/sandbox/templates';
import { saveSession } from '@/lib/persistence';
import { buildAISDKTools, buildToolContext, getToolSummaryForPrompt } from '@/lib/tools/ai-sdk-tools';
import { selectToolsForIntent } from '@/lib/tools/intent-selector';
import { buildSystemPrompt, createDefaultPromptLayer, createErrorFixLayer, createToolAppendLayer, type PromptContext, type PromptLayer } from '@/lib/prompt-composer';
import { classifyError, withRetry, detectInterruption, buildRecoveryMessage, isToolError, formatErrorForModel } from '@/lib/error-recovery';
import { runContextPipeline, formatContextForPrompt } from '@/lib/context-pipeline';
import { isSlashCommand, routeSlashCommand, type CommandContext } from '@/lib/slash-commands';
import { createBudgetTracker, checkTokenBudget, estimateTotalTokens } from '@/lib/token-budget';
import { shouldAutoCompact, compactConversation, microCompact, createAutoCompactState, advanceTurn, type AutoCompactState } from '@/lib/compact';
import { shouldSuggestPlan, getPlanModePrompt, type Plan } from '@/lib/plan-mode';
import { createCostState, trackUsage, formatCostReport, getCompactCostSummary, type SessionCostState } from '@/lib/cost-tracker';
import { getEffortConfig, resolveEffortFromPrompt, type EffortLevel } from '@/lib/effort-modes';
import { createSessionMemoryState, shouldExtractMemory, recordToolCall as recordMemoryToolCall, extractSessionMemory, formatMemoryForPrompt, type SessionMemoryState } from '@/lib/session-memory';
import { isFastModeEnabled, getFastModeModel, toggleFastMode, createFastModeState, type FastModeState } from '@/lib/fast-mode';
import { createFileTrackingState, snapshotFile, detectFileChange, formatChangesForPrompt, advanceTurnWithChanges, type FileTrackingState } from '@/lib/file-change-detection';

// Force dynamic route to enable streaming
export const dynamic = 'force-dynamic';

// Check if we're using Vercel AI Gateway
const isUsingAIGateway = !!process.env.AI_GATEWAY_API_KEY;
const aiGatewayBaseURL = 'https://ai-gateway.vercel.sh/v1';

console.log('[generate-ai-code-stream] AI Gateway config:', {
  isUsingAIGateway,
  hasGroqKey: !!process.env.GROQ_API_KEY,
  hasAIGatewayKey: !!process.env.AI_GATEWAY_API_KEY
});

const groq = createGroq({
  apiKey: process.env.AI_GATEWAY_API_KEY ?? process.env.GROQ_API_KEY,
  baseURL: isUsingAIGateway ? aiGatewayBaseURL : undefined,
});

const anthropic = createAnthropic({
  apiKey: process.env.AI_GATEWAY_API_KEY ?? process.env.ANTHROPIC_API_KEY,
  baseURL: isUsingAIGateway ? aiGatewayBaseURL : (process.env.ANTHROPIC_BASE_URL || 'https://api.anthropic.com/v1'),
});

const googleGenerativeAI = createGoogleGenerativeAI({
  apiKey: process.env.AI_GATEWAY_API_KEY ?? process.env.GEMINI_API_KEY,
  baseURL: isUsingAIGateway ? aiGatewayBaseURL : undefined,
});

const openai = createOpenAI({
  apiKey: process.env.AI_GATEWAY_API_KEY ?? process.env.OPENAI_API_KEY,
  baseURL: isUsingAIGateway ? aiGatewayBaseURL : process.env.OPENAI_BASE_URL,
});

// Ollama provider (OpenAI-compatible at /v1)
const ollamaProvider = createOpenAI({
  apiKey: 'ollama',
  baseURL: `${appConfig.ai.ollama.baseUrl}/v1`,
});

// OpenWebUI provider (OpenAI-compatible)
const openWebUIProvider = createOpenAI({
  apiKey: appConfig.ai.openWebUI.apiKey || 'dummy',
  baseURL: `${appConfig.ai.openWebUI.baseUrl}/api`,
});

// Helper function to analyze user preferences from conversation history
function analyzeUserPreferences(messages: ConversationMessage[]): {
  commonPatterns: string[];
  preferredEditStyle: 'targeted' | 'comprehensive';
} {
  const userMessages = messages.filter(m => m.role === 'user');
  const patterns: string[] = [];
  
  // Count edit-related keywords
  let targetedEditCount = 0;
  let comprehensiveEditCount = 0;
  
  userMessages.forEach(msg => {
    const content = msg.content.toLowerCase();
    
    // Check for targeted edit patterns
    if (content.match(/\b(update|change|fix|modify|edit|remove|delete)\s+(\w+\s+)?(\w+)\b/)) {
      targetedEditCount++;
    }
    
    // Check for comprehensive edit patterns
    if (content.match(/\b(rebuild|recreate|redesign|overhaul|refactor)\b/)) {
      comprehensiveEditCount++;
    }
    
    // Extract common request patterns
    if (content.includes('hero')) patterns.push('hero section edits');
    if (content.includes('header')) patterns.push('header modifications');
    if (content.includes('color') || content.includes('style')) patterns.push('styling changes');
    if (content.includes('button')) patterns.push('button updates');
    if (content.includes('animation')) patterns.push('animation requests');
  });
  
  return {
    commonPatterns: [...new Set(patterns)].slice(0, 3), // Top 3 unique patterns
    preferredEditStyle: targetedEditCount > comprehensiveEditCount ? 'targeted' : 'comprehensive'
  };
}

declare global {
  var sandboxState: SandboxState;
  var conversationState: ConversationState | null;
  var activeTemplateId: string | null;
  var sableCostState: SessionCostState | null;
  var sableAutoCompactState: AutoCompactState | null;
  var sableSessionMemoryState: SessionMemoryState | null;
  var sableFastModeState: FastModeState | null;
  var sableFileTrackingState: FileTrackingState | null;
  var sableEffortLevel: EffortLevel | null;
  var sableActivePlan: Plan | null;
}

export async function POST(request: NextRequest) {
  try {
    const { prompt, model = 'openai/gpt-oss-20b', context, isEdit = false, isErrorFix = false, template } = await request.json();
    
    // Store the active template from the frontend
    if (template) {
      global.activeTemplateId = template;
    }
    
    console.log('[generate-ai-code-stream] Received request:');
    console.log('[generate-ai-code-stream] - prompt:', prompt);
    console.log('[generate-ai-code-stream] - template:', template || 'default');
    console.log('[generate-ai-code-stream] - isEdit:', isEdit);
    console.log('[generate-ai-code-stream] - isErrorFix:', isErrorFix);
    console.log('[generate-ai-code-stream] - context.sandboxId:', context?.sandboxId);
    console.log('[generate-ai-code-stream] - context.currentFiles:', context?.currentFiles ? Object.keys(context.currentFiles) : 'none');
    console.log('[generate-ai-code-stream] - currentFiles count:', context?.currentFiles ? Object.keys(context.currentFiles).length : 0);
    
    // Initialize conversation state if not exists
    if (!global.conversationState) {
      global.conversationState = {
        conversationId: `conv-${Date.now()}`,
        startedAt: Date.now(),
        lastUpdated: Date.now(),
        context: {
          messages: [],
          edits: [],
          projectEvolution: { majorChanges: [] },
          userPreferences: {}
        }
      };
    }
    
    // Initialize new subsystem states
    if (!global.sableCostState) global.sableCostState = createCostState();
    if (!global.sableAutoCompactState) global.sableAutoCompactState = createAutoCompactState();
    if (!global.sableSessionMemoryState) global.sableSessionMemoryState = createSessionMemoryState();
    if (!global.sableFastModeState) global.sableFastModeState = createFastModeState();
    if (!global.sableFileTrackingState) global.sableFileTrackingState = createFileTrackingState();
    if (!global.sableEffortLevel) global.sableEffortLevel = 'medium';
    
    // ─── Slash Command Handling ───────────────────────────
    if (isSlashCommand(prompt)) {
      const cmdCtx: CommandContext = {
        model,
        tokenCount: estimateTotalTokens(
          (global.conversationState?.context?.messages || []).map((m: any) => ({ role: m.role, content: m.content }))
        ),
        turnCount: global.conversationState?.context?.messages?.length || 0,
        setModel: (m: string) => { /* Model switching handled by frontend */ },
        setEffort: (level: string) => { global.sableEffortLevel = level as EffortLevel; },
        triggerCompact: () => { /* Handled via data.action below */ },
        triggerClear: () => { 
          if (global.conversationState) {
            global.conversationState.context.messages = [];
            global.conversationState.context.edits = [];
          }
        },
      };
      
      const cmdResult = await routeSlashCommand(prompt, cmdCtx);
      if (cmdResult?.handled) {
        // Inject cost report for /cost command
        let displayMessage = cmdResult.message;
        if (cmdResult.data?.action === 'cost' && global.sableCostState) {
          displayMessage = formatCostReport(global.sableCostState);
        }
        // Inject fast mode toggle
        if (cmdResult.data?.action === 'fast' && global.sableFastModeState) {
          const enabled = cmdResult.data.enabled as boolean;
          const result = toggleFastMode(global.sableFastModeState, enabled);
          global.sableFastModeState = result.state;
          displayMessage = result.toggled
            ? (enabled ? `Fast mode enabled. Using ${getFastModeModel(result.state)}.` : 'Fast mode disabled. Using default model.')
            : (result.reason || displayMessage);
        }
        
        // Return command result as a quick SSE response
        const encoder = new TextEncoder();
        const stream = new TransformStream();
        const writer = stream.writable.getWriter();
        await writer.write(encoder.encode(`data: ${JSON.stringify({ type: 'conversation', text: displayMessage })}\n\n`));
        if (cmdResult.data) {
          await writer.write(encoder.encode(`data: ${JSON.stringify({ type: 'command', ...cmdResult.data })}\n\n`));
        }
        await writer.write(encoder.encode(`data: ${JSON.stringify({ type: 'done' })}\n\n`));
        await writer.close();
        return new Response(stream.readable, {
          headers: { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache' },
        });
      }
    }
    
    // Add user message to conversation history
    const userMessage: ConversationMessage = {
      id: `msg-${Date.now()}`,
      role: 'user',
      content: prompt,
      timestamp: Date.now(),
      metadata: {
        sandboxId: context?.sandboxId
      }
    };
    global.conversationState.context.messages.push(userMessage);
    
    // Clean up old messages to prevent unbounded growth
    if (global.conversationState.context.messages.length > 20) {
      // Keep only the last 15 messages
      global.conversationState.context.messages = global.conversationState.context.messages.slice(-15);
      console.log('[generate-ai-code-stream] Trimmed conversation history to prevent context overflow');
    }
    
    // Clean up old edits
    if (global.conversationState.context.edits.length > 10) {
      global.conversationState.context.edits = global.conversationState.context.edits.slice(-8);
    }
    
    // Debug: Show a sample of actual file content
    if (context?.currentFiles && Object.keys(context.currentFiles).length > 0) {
      const firstFile = Object.entries(context.currentFiles)[0];
      console.log('[generate-ai-code-stream] - sample file:', firstFile[0]);
      console.log('[generate-ai-code-stream] - sample content preview:', 
        typeof firstFile[1] === 'string' ? firstFile[1].substring(0, 100) + '...' : 'not a string');
    }
    
    if (!prompt) {
      return NextResponse.json({ 
        success: false, 
        error: 'Prompt is required' 
      }, { status: 400 });
    }
    
    // Create a stream for real-time updates
    const encoder = new TextEncoder();
    const stream = new TransformStream();
    const writer = stream.writable.getWriter();
    
    // Function to send progress updates with flushing
    const sendProgress = async (data: any) => {
      const message = `data: ${JSON.stringify(data)}\n\n`;
      try {
        await writer.write(encoder.encode(message));
        // Force flush by writing a keep-alive comment
        if (data.type === 'stream' || data.type === 'conversation') {
          await writer.write(encoder.encode(': keepalive\n\n'));
        }
      } catch (error) {
        console.error('[generate-ai-code-stream] Error writing to stream:', error);
      }
    };
    
    // Start processing in background
    (async () => {
      try {
        // Send initial status
        await sendProgress({ type: 'status', message: 'Initializing AI...' });
        
        // No keep-alive needed - sandbox provisioned for 10 minutes
        
        // Check if we have a file manifest for edit mode (skip for error fixes,  they're fast path)
        let editContext = null;
        let enhancedSystemPrompt = '';
        
        if (isEdit && !isErrorFix) {
          console.log('[generate-ai-code-stream] Edit mode detected - starting agentic search workflow');
          console.log('[generate-ai-code-stream] Has fileCache:', !!global.sandboxState?.fileCache);
          console.log('[generate-ai-code-stream] Has manifest:', !!global.sandboxState?.fileCache?.manifest);
          
          const manifest: FileManifest | undefined = global.sandboxState?.fileCache?.manifest;
          
          if (manifest) {
            await sendProgress({ type: 'status', message: '🔍 Creating search plan...' });
            
            const fileContents = global.sandboxState.fileCache?.files || {};
            console.log('[generate-ai-code-stream] Files available for search:', Object.keys(fileContents).length);
            
            // STEP 1: Get search plan from AI
            try {
              const intentResponse = await fetch(`${process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000'}/api/analyze-edit-intent`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt, manifest, model })
              });
              
              if (intentResponse.ok) {
                const { searchPlan } = await intentResponse.json();
                console.log('[generate-ai-code-stream] Search plan received:', searchPlan);
                
                await sendProgress({ 
                  type: 'status', 
                  message: `🔎 Searching for: "${searchPlan.searchTerms.join('", "')}"`
                });
                
                // STEP 2: Execute the search plan
                const searchExecution = executeSearchPlan(searchPlan, 
                  Object.fromEntries(
                    Object.entries(fileContents).map(([path, data]) => [
                      path.startsWith('/') ? path : `/home/user/app/${path}`,
                      data.content
                    ])
                  )
                );
                
                console.log('[generate-ai-code-stream] Search execution:', {
                  success: searchExecution.success,
                  resultsCount: searchExecution.results.length,
                  filesSearched: searchExecution.filesSearched,
                  time: searchExecution.executionTime + 'ms'
                });
                
                if (searchExecution.success && searchExecution.results.length > 0) {
                  // STEP 3: Select the best target file
                  const target = selectTargetFile(searchExecution.results, searchPlan.editType);
                  
                  if (target) {
                    await sendProgress({ 
                      type: 'status', 
                      message: `✅ Found code in ${target.filePath.split('/').pop()} at line ${target.lineNumber}`
                    });
                    
                    console.log('[generate-ai-code-stream] Target selected:', target);
                    
                    // Create surgical edit context with exact location
                    // normalizedPath would be: target.filePath.replace('/home/user/app/', '');
                    // fileContent available but not used in current implementation
                    // const fileContent = fileContents[normalizedPath]?.content || '';
                    
                    // Build enhanced context with search results
                    enhancedSystemPrompt = `
${formatSearchResultsForAI(searchExecution.results)}

SURGICAL EDIT INSTRUCTIONS:
You have been given the EXACT location of the code to edit.
- File: ${target.filePath}
- Line: ${target.lineNumber}
- Reason: ${target.reason}

Make ONLY the change requested by the user. Do not modify any other code.
User request: "${prompt}"`;
                    
                    // Set up edit context with just this one file
                    editContext = {
                      primaryFiles: [target.filePath],
                      contextFiles: [],
                      systemPrompt: enhancedSystemPrompt,
                      editIntent: {
                        type: searchPlan.editType,
                        description: searchPlan.reasoning,
                        targetFiles: [target.filePath],
                        confidence: 0.95, // High confidence since we found exact location
                        searchTerms: searchPlan.searchTerms
                      }
                    };
                    
                    console.log('[generate-ai-code-stream] Surgical edit context created');
                  }
                } else {
                  // Search failed - fall back to old behavior but inform user
                  console.warn('[generate-ai-code-stream] Search found no results, falling back to broader context');
                  await sendProgress({ 
                    type: 'status', 
                    message: '⚠️ Could not find exact match, using broader search...'
                  });
                }
              } else {
                console.error('[generate-ai-code-stream] Failed to get search plan');
              }
            } catch (error) {
              console.error('[generate-ai-code-stream] Error in agentic search workflow:', error);
              await sendProgress({ 
                type: 'status', 
                message: '⚠️ Search workflow error, falling back to keyword method...'
              });
              // Fall back to old method on any error if we have a manifest
              if (manifest) {
                editContext = selectFilesForEdit(prompt, manifest);
              }
            }
          } else {
            // Fall back to old method if AI analysis fails
            console.warn('[generate-ai-code-stream] AI intent analysis failed, falling back to keyword method');
            if (manifest) {
              editContext = selectFilesForEdit(prompt, manifest);
            } else {
              console.log('[generate-ai-code-stream] No manifest available for fallback');
              await sendProgress({ 
                type: 'status', 
                message: '⚠️ No file manifest available, will use broad context'
              });
            }
          }
          
          // If we got an edit context from any method, use its system prompt
          if (editContext) {
            enhancedSystemPrompt = editContext.systemPrompt;
            
            await sendProgress({ 
              type: 'status', 
              message: `Identified edit type: ${editContext.editIntent?.description || 'Code modification'}`
            });
          } else if (!manifest) {
            console.log('[generate-ai-code-stream] WARNING: No manifest available for edit mode!');
            
            // Try to fetch files from sandbox if we have one
            if (global.activeSandbox) {
              await sendProgress({ type: 'status', message: 'Fetching current files from sandbox...' });
              
              try {
                // Fetch files directly from sandbox
                const filesResponse = await fetch(`${process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000'}/api/get-sandbox-files`, {
                  method: 'GET',
                  headers: { 'Content-Type': 'application/json' }
                });
                
                if (filesResponse.ok) {
                  const filesData = await filesResponse.json();
                  
                  if (filesData.success && filesData.manifest) {
                    console.log('[generate-ai-code-stream] Successfully fetched manifest from sandbox');
                    const manifest = filesData.manifest;
                    
                    // Now try to analyze edit intent with the fetched manifest
                    try {
                      const intentResponse = await fetch(`${process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000'}/api/analyze-edit-intent`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ prompt, manifest, model })
                      });
                      
                      if (intentResponse.ok) {
                        const { searchPlan } = await intentResponse.json();
                        console.log('[generate-ai-code-stream] Search plan received (after fetch):', searchPlan);
                        
                        // For now, fall back to keyword search since we don't have file contents for search execution
                        // This path happens when no manifest was initially available
                        let targetFiles: any[] = [];
                        if (!searchPlan || searchPlan.searchTerms.length === 0) {
                          console.warn('[generate-ai-code-stream] No target files after fetch, searching for relevant files');
                          
                          const promptLower = prompt.toLowerCase();
                          const allFilePaths = Object.keys(manifest.files);
                          
                          // Look for component names mentioned in the prompt
                          if (promptLower.includes('hero')) {
                            targetFiles = allFilePaths.filter(p => p.toLowerCase().includes('hero'));
                          } else if (promptLower.includes('header')) {
                            targetFiles = allFilePaths.filter(p => p.toLowerCase().includes('header'));
                          } else if (promptLower.includes('footer')) {
                            targetFiles = allFilePaths.filter(p => p.toLowerCase().includes('footer'));
                          } else if (promptLower.includes('nav')) {
                            targetFiles = allFilePaths.filter(p => p.toLowerCase().includes('nav'));
                          } else if (promptLower.includes('button')) {
                            targetFiles = allFilePaths.filter(p => p.toLowerCase().includes('button'));
                          }
                          
                          if (targetFiles.length > 0) {
                            console.log('[generate-ai-code-stream] Found target files by keyword search after fetch:', targetFiles);
                          }
                        }
                        
                        const allFiles = Object.keys(manifest.files)
                          .filter(path => !targetFiles.includes(path));
                        
                        editContext = {
                          primaryFiles: targetFiles,
                          contextFiles: allFiles,
                          systemPrompt: `
You are an expert senior software engineer performing a surgical, context-aware code modification. Your primary directive is **precision and preservation**.

Think of yourself as a surgeon making a precise incision, not a construction worker demolishing a wall.

## Search-Based Edit
Search Terms: ${searchPlan?.searchTerms?.join(', ') || 'keyword-based'}
Edit Type: ${searchPlan?.editType || 'UPDATE_COMPONENT'}
Reasoning: ${searchPlan?.reasoning || 'Modifying based on user request'}

Files to Edit: ${targetFiles.join(', ') || 'To be determined'}
User Request: "${prompt}"

## Your Mandatory Thought Process (Execute Internally):
Before writing ANY code, you MUST follow these steps:

1. **Understand Intent:**
   - What is the user's core goal? (adding feature, fixing bug, changing style?)
   - Does the conversation history provide extra clues?

2. **Locate the Code:**
   - First examine the Primary Files provided
   - Check the "ALL PROJECT FILES" list to find the EXACT file name
   - "nav" might be Navigation.tsx, NavBar.tsx, Nav.tsx, or Header.tsx
   - DO NOT create a new file if a similar one exists!

3. **Plan the Changes (Mental Diff):**
   - What is the *minimal* set of changes required?
   - Which exact lines need to be added, modified, or deleted?
   - Will this require new packages?

4. **Verify Preservation:**
   - What existing code, props, state, and logic must NOT be touched?
   - How can I make my change without disrupting surrounding code?

5. **Construct the Final Code:**
   - Only after completing steps above, generate the final code
   - Provide the ENTIRE file content with modifications integrated

## Critical Rules & Constraints:

**PRESERVATION IS KEY:** You MUST NOT rewrite entire components or files. Integrate your changes into the existing code. Preserve all existing logic, props, state, and comments not directly related to the user's request.

**MINIMALISM:** Only output files you have actually changed. If a file doesn't need modification, don't include it.

**COMPLETENESS:** Each file must be COMPLETE from first line to last:
- NEVER TRUNCATE - Include EVERY line
- NO ellipsis (...) to skip content
- ALL imports, functions, JSX, and closing tags must be present
- The file MUST be runnable

**SURGICAL PRECISION:**
- Change ONLY what's explicitly requested
- If user says "change background to green", change ONLY the background class
- 99% of the original code should remain untouched
- NO refactoring, reformatting, or "improvements" unless requested

**NO CONVERSATION:** Your output must contain ONLY the code. No explanations or apologies.

## EXAMPLES:

### CORRECT APPROACH for "change hero background to blue":
<thinking>
I need to change the background color of the Hero component. Looking at the file, I see the main div has 'bg-gray-900'. I will change ONLY this to 'bg-blue-500' and leave everything else exactly as is.
</thinking>

Then return the EXACT same file with only 'bg-gray-900' changed to 'bg-blue-500'.

### WRONG APPROACH (DO NOT DO THIS):
- Rewriting the Hero component from scratch
- Changing the structure or reorganizing imports
- Adding or removing unrelated code
- Reformatting or "cleaning up" the code

Remember: You are a SURGEON making a precise incision, not an artist repainting the canvas!`,
                          editIntent: {
                            type: searchPlan?.editType || 'UPDATE_COMPONENT',
                            targetFiles: targetFiles,
                            confidence: searchPlan ? 0.85 : 0.6,
                            description: searchPlan?.reasoning || 'Keyword-based file selection',
                            suggestedContext: []
                          }
                        };
                        
                        enhancedSystemPrompt = editContext.systemPrompt;
                        
                        await sendProgress({ 
                          type: 'status', 
                          message: `Identified edit type: ${editContext.editIntent.description}`
                        });
                      }
                    } catch (error) {
                      console.error('[generate-ai-code-stream] Error analyzing intent after fetch:', error);
                    }
                  } else {
                    console.error('[generate-ai-code-stream] Failed to get manifest from sandbox files');
                  }
                } else {
                  console.error('[generate-ai-code-stream] Failed to fetch sandbox files:', filesResponse.status);
                }
              } catch (error) {
                console.error('[generate-ai-code-stream] Error fetching sandbox files:', error);
                await sendProgress({ 
                  type: 'warning', 
                  message: 'Could not analyze existing files for targeted edits. Proceeding with general edit mode.'
                });
              }
            } else {
              console.log('[generate-ai-code-stream] No active sandbox to fetch files from');
              await sendProgress({ 
                type: 'warning', 
                message: 'No existing files found. Consider generating initial code first.'
              });
            }
          }
        }
        
        // Build conversation context for system prompt
        let conversationContext = '';
        if (global.conversationState && global.conversationState.context.messages.length > 1) {
          console.log('[generate-ai-code-stream] Building conversation context');
          console.log('[generate-ai-code-stream] Total messages:', global.conversationState.context.messages.length);
          console.log('[generate-ai-code-stream] Total edits:', global.conversationState.context.edits.length);
          
          conversationContext = `\n\n## Conversation History (Recent)\n`;
          
          // Include only the last 3 edits to save context
          const recentEdits = global.conversationState.context.edits.slice(-3);
          if (recentEdits.length > 0) {
            console.log('[generate-ai-code-stream] Including', recentEdits.length, 'recent edits in context');
            conversationContext += `\n### Recent Edits:\n`;
            recentEdits.forEach(edit => {
              conversationContext += `- "${edit.userRequest}" → ${edit.editType} (${edit.targetFiles.map(f => f.split('/').pop()).join(', ')})\n`;
            });
          }
          
          // Include recently created files - CRITICAL for preventing duplicates
          const recentMsgs = global.conversationState.context.messages.slice(-5);
          const recentlyCreatedFiles: string[] = [];
          recentMsgs.forEach(msg => {
            if (msg.metadata?.editedFiles) {
              recentlyCreatedFiles.push(...msg.metadata.editedFiles);
            }
          });
          
          if (recentlyCreatedFiles.length > 0) {
            const uniqueFiles = [...new Set(recentlyCreatedFiles)];
            conversationContext += `\n### 🚨 RECENTLY CREATED/EDITED FILES (DO NOT RECREATE THESE):\n`;
            uniqueFiles.forEach(file => {
              conversationContext += `- ${file}\n`;
            });
            conversationContext += `\nIf the user mentions any of these components, UPDATE the existing file!\n`;
          }
          
          // Include only last 5 messages for context (reduced from 10)
          const recentMessages = recentMsgs;
          if (recentMessages.length > 2) { // More than just current message
            conversationContext += `\n### Recent Messages:\n`;
            recentMessages.slice(0, -1).forEach(msg => { // Exclude current message
              if (msg.role === 'user') {
                const truncatedContent = msg.content.length > 100 ? msg.content.substring(0, 100) + '...' : msg.content;
                conversationContext += `- "${truncatedContent}"\n`;
              }
            });
          }
          
          // Include only last 2 major changes
          const majorChanges = global.conversationState.context.projectEvolution.majorChanges.slice(-2);
          if (majorChanges.length > 0) {
            conversationContext += `\n### Recent Changes:\n`;
            majorChanges.forEach(change => {
              conversationContext += `- ${change.description}\n`;
            });
          }
          
          // Keep user preferences - they're concise
          const userPrefs = analyzeUserPreferences(global.conversationState.context.messages);
          if (userPrefs.commonPatterns.length > 0) {
            conversationContext += `\n### User Preferences:\n`;
            conversationContext += `- Edit style: ${userPrefs.preferredEditStyle}\n`;
          }
          
          // Limit total conversation context length
          if (conversationContext.length > 2000) {
            conversationContext = conversationContext.substring(0, 2000) + '\n[Context truncated to prevent length errors]';
          }
        }
        
        // Build system prompt with conversation awareness and template context
        const activeTemplate = getTemplate(global.activeTemplateId || DEFAULT_TEMPLATE);
        const templateContext = `PROJECT TEMPLATE: ${activeTemplate.name} (${activeTemplate.id})
${activeTemplate.systemPromptAddition}

${activeTemplate.fileFormatInstructions}
`;

        // Detect if this is creation mode (no scraped websites) vs clone mode
        const hasScrapedContent = context?.conversationContext?.scrapedWebsites?.length > 0;
        const isCreationMode = !hasScrapedContent && !isEdit;
        
        // Template-aware creation mode instructions
        let creationModeInstructions = '';
        if (isCreationMode) {
          if (activeTemplate.id === 'react-spa') {
            creationModeInstructions = `
CREATION MODE ACTIVE - You are CUSTOMIZING an existing React SPA template.
- The sandbox ALREADY has working components: Header, Hero, Features, Footer, App.jsx
- DO NOT create these from scratch. MODIFY the existing files to match the user's request.
- Change the content, colors, and text to match what the user wants.
- Output ONLY the files you changed. If Header doesn't need changes, don't output it.
- Use real, relevant content that matches the user's request (not lorem ipsum).
- Focus on modern design, smooth animations, and excellent UX.
- You may add NEW component files if the user's request needs sections beyond Header/Hero/Features/Footer.
- If adding a new component, also output the modified App.jsx with the new import.
`;
          } else if (activeTemplate.id === 'static-site') {
            creationModeInstructions = `
CREATION MODE ACTIVE - You are building a STATIC website with vanilla HTML, CSS, and JavaScript.
- NO React. NO JSX. NO frameworks. Pure HTML, CSS, and JS.
- The sandbox already has index.html, styles/main.css, and js/main.js.
- MODIFY these existing files to match the user's request.
- The entry point is index.html,  put ALL your HTML structure there.
- Put ALL styles in styles/main.css using modern CSS (flexbox, grid, custom properties, animations).
- Put interactivity in js/main.js using vanilla ES6+ JavaScript.
- Use <file path="index.html">, <file path="styles/main.css">, <file path="js/main.js">.
- You may add additional CSS/JS files if needed, but keep it simple.
- NEVER output .jsx files. NEVER import React.
`;
          } else if (activeTemplate.id === 'fullstack') {
            creationModeInstructions = `
CREATION MODE ACTIVE - You are building a FULL-STACK app with React frontend and Express.js backend.
- Frontend in src/ with Vite + React + Tailwind CSS.
- Backend in server/index.js with Express.js.
- Modify existing files to match the user's request.
`;
          } else if (activeTemplate.id === 'node-api') {
            creationModeInstructions = `
CREATION MODE ACTIVE - You are building a backend-only REST API with Express.js.
- NO frontend. NO React. NO HTML pages.
- Main entry: index.js with Express.js routes.
- Output only .js files for the API.
`;
          } else if (activeTemplate.id === 'nextjs') {
            creationModeInstructions = `
CREATION MODE ACTIVE - You are building a Next.js app with App Router.
- Use app/ directory for routes.
- app/layout.js is the root layout (REQUIRED).
- app/page.js is the home page.
- Use Tailwind CSS for styling.
`;
          } else {
            creationModeInstructions = `
CREATION MODE ACTIVE - Build the application using the template context above.
- Use real, relevant content (not lorem ipsum).
- Focus on modern design and excellent UX.
`;
          }
        }
        
        let systemPrompt = `You are a code generator. You ONLY output code in <file> XML tags. You NEVER output explanations, greetings, or conversational text.

ACTIVE TEMPLATE: ${activeTemplate.name}
${templateContext}
${creationModeInstructions}
${conversationContext}

OUTPUT FORMAT (MANDATORY):
You MUST output ONLY <file> tags. Nothing else. No text before, between, or after the files.

Example of CORRECT output:
<file path="src/index.css">
@tailwind base;
@tailwind components;
@tailwind utilities;
</file>
<file path="src/App.jsx">
import React from 'react';
export default function App() { return <div>Hello</div>; }
</file>

Example of WRONG output (NEVER DO THIS):
"Sure! Here's your code:" <-- WRONG, no conversation
"I'll create a React app for you" <-- WRONG, no explanations
"Let me know if you need changes" <-- WRONG, no follow-up text

RULES:
1. Output ONLY <file path="...">...code...</file> tags
2. Use standard Tailwind classes (bg-white, text-black, bg-blue-500) NOT (bg-background, text-foreground)
3. Each file must be COMPLETE - no truncation, no ellipsis
4. Do exactly what is asked - nothing more, nothing less
5. For edits: only output files that changed
6. NEVER output markdown code fences (\`\`\`)
7. NEVER create tailwind.config.js, vite.config.js, or package.json

COMPONENT RELATIONSHIPS (CHECK THESE FIRST):
- Navigation usually lives INSIDE Header.jsx, not separate Nav.jsx
- Logo is typically in Header, not standalone
- Footer often contains nav links already
- Menu/Hamburger is part of Header, not separate

PACKAGE USAGE RULES:
- DO NOT use react-router-dom unless user explicitly asks for routing
- For simple nav links in a single-page app, use scroll-to-section or href="#"
- Only add routing if building a multi-page application
- Common packages are auto-installed from your imports
- ONLY import packages that ACTUALLY EXIST on npm. If unsure, use inline code instead.
- For icons: use @heroicons/react (NOT hero-icons-react, NOT heroicons-react, NOT heroicons)
  Example: import { HeartIcon } from '@heroicons/react/24/solid'
  Alternative: use lucide-react,  import { Heart } from 'lucide-react'
  Alternative: use react-icons,  import { FaHeart } from 'react-icons/fa'
- For animations: use framer-motion (NOT @react-native-particles or other non-existent packages)
- NEVER import packages with "react-native" in the name,  this is a web project
- If you need particles/confetti: use canvas-confetti or tsparticles
- When in doubt, use inline SVGs or CSS animations instead of importing an icon/animation library

WEBSITE CLONING REQUIREMENTS:
When recreating/cloning a website, you MUST include:
1. **Header with Navigation** - Usually Header.jsx containing nav
2. **Hero Section** - The main landing area (Hero.jsx)
3. **Main Content Sections** - Features, Services, About, etc.
4. **Footer** - Contact info, links, copyright (Footer.jsx)
5. **App.jsx** - Main app component that imports and uses all components

${isEdit ? `THIS IS AN EDIT. Rules:
- ONLY output <file> tags for files that need changes
- NEVER recreate config files (tailwind.config.js, vite.config.js, package.json)
- Each file must be COMPLETE (all imports, all code, all closing tags)
- Minimal changes: if user says "change color", change ONLY the color
- Do NOT regenerate the entire app
${editContext ? `
TARGETED EDIT:
- Files to Edit: ${editContext.primaryFiles.join(', ')}
- Edit Type: ${editContext.editIntent?.type || 'modification'}
- Output ONLY the files listed above. No extras.` : ''}

ERROR FIX MODE:
If the user message contains a build/compile error (SyntaxError, plugin:vite, Missing semicolon, Identifier directly after number, etc.):
1. READ THE ERROR to identify the EXACT file and line
2. Output ONLY that ONE broken file with the fix,  nothing else
3. Do NOT output any other files. Do NOT regenerate the app.
4. Fix the syntax while preserving ALL existing functionality
5. Common fixes: escape apostrophes in JSX, fix unclosed tags, fix invalid SVG paths, remove TS syntax from .jsx
6. For broken SVG paths: use a simple valid path or remove the icon` : ''}
${activeTemplate.id === 'static-site' ? `
STYLING: Use custom CSS in styles/main.css. Use modern CSS features (flexbox, grid, custom properties, animations, transitions).
NO Tailwind. NO CSS frameworks. Pure handwritten CSS.
Responsive design with @media queries.` : activeTemplate.id === 'node-api' ? `
STYLING: Not applicable,  this is a backend-only API project.` : `
STYLING: Use only Tailwind CSS classes. No inline styles, no CSS-in-JS, no component CSS files.
Only create src/index.css with @tailwind base/components/utilities directives.
Use standard Tailwind: bg-white, text-gray-900, border-gray-200 (NOT bg-background, text-foreground, border-border).
Escape apostrophes in strings. Use straight quotes only.
Responsive design: use sm:, md:, lg: breakpoints.`}

CODE QUALITY (MUST FOLLOW):
${activeTemplate.id === 'static-site' ? `- Files are .html, .css, .js,  NO React, NO JSX, NO frameworks
- Use semantic HTML5 elements
- Use modern CSS (flexbox, grid, custom properties, keyframes)
- Use vanilla ES6+ JavaScript` : activeTemplate.id === 'node-api' ? `- Files are .js,  backend Node.js/Express code
- No frontend files, no HTML, no CSS` : `- Files are .jsx NOT .tsx,  NEVER use TypeScript syntax (no type annotations, no !, no as, no interface, no : type)
- NEVER use document.getElementById('x')!.innerText,  use useRef or useState instead
- Escape all apostrophes in JSX text: use &apos; or {"'"}
- SVG paths must be valid,  if copying a complex icon path, use Heroicons or Lucide imports instead
- Every JSX component must have matching opening/closing tags`}
- Every function must have matching { }
- ALWAYS test mentally: would this file parse without errors?

CRITICAL,  NEVER TRUNCATE:
- You MUST finish every <file> tag you open with a complete </file> closing tag.
- If you run low on output space, close the current file properly, then STOP. Do NOT start a new file you cannot finish.
- NEVER leave code mid-line, mid-function, or mid-component.
- Complete EVERY file fully,  all imports, all functions, all closing braces, all closing tags.

WHEN GENERATING CODE:
${activeTemplate.id === 'react-spa' ? `1. The sandbox ALREADY has Header.jsx, Hero.jsx, Features.jsx, Footer.jsx, App.jsx, index.css
2. Output ONLY files you are MODIFYING,  skip unchanged files
3. If you need a new section, create it as src/components/NewSection.jsx and update App.jsx` : activeTemplate.id === 'static-site' ? `1. The sandbox already has index.html, styles/main.css, and js/main.js
2. Output ONLY the files you need to modify
3. For additional pages, create new .html files` : activeTemplate.id === 'fullstack' ? `1. Frontend files go in src/ (React + Vite)
2. Backend files go in server/ (Express.js)
3. Output ONLY the files you need` : activeTemplate.id === 'node-api' ? `1. Main API entry is index.js
2. Add routes in routes/ directory
3. Output ONLY backend .js files` : activeTemplate.id === 'nextjs' ? `1. Pages go in app/ directory (App Router)
2. app/layout.js is required
3. Output ONLY files you need` : `1. Follow the template structure above
2. Output ONLY the files you need`}
4. NEVER say "I'll continue" - generate ALL changed files in one response
5. Complete every file - no truncation

DEPLOYMENT HELP:
When the user asks to deploy, publish, or host their app, provide step-by-step instructions for their chosen platform. Common options:
- **Vercel**: Run \`npm i -g vercel && vercel\` in the project directory. Follow the prompts.
- **Netlify**: Run \`npm run build\`, then drag the dist/ folder to netlify.com/drop
- **GitHub Pages**: Push to GitHub, enable Pages in repo settings, set source to gh-pages branch
- **Railway/Render**: Connect GitHub repo, set build command to \`npm run build\`, publish dir to \`dist\`
- **VPS/Server**: Run \`npm run build\`, serve the dist/ folder with nginx or any static file server
- **Docker**: Create a Dockerfile with nginx serving the dist/ folder
Always ask what platform they prefer if not specified. Provide the exact commands they need to run.`;

        // ERROR FIX OVERRIDE: when isErrorFix is true, replace the system prompt with a laser-focused one
        if (isErrorFix) {
          // Extract the broken file name from the prompt
          const brokenFileMatch = prompt.match(/(?:broken file|output the file)[:\s]*"?([A-Za-z0-9_.-]+\.(?:jsx|tsx|js|ts|css))"?/i);
          const brokenFile = brokenFileMatch ? brokenFileMatch[1] : null;
          
          // Try to read the broken file content from the sandbox file cache
          let brokenFileContent = '';
          if (brokenFile && global.sandboxState?.fileCache?.files) {
            const possiblePaths = [
              `src/components/${brokenFile}`,
              `src/${brokenFile}`,
              brokenFile
            ];
            for (const p of possiblePaths) {
              if (global.sandboxState.fileCache.files[p]) {
                const cached = global.sandboxState.fileCache.files[p];
                brokenFileContent = typeof cached === 'string' ? cached : cached.content || '';
                break;
              }
            }
          }
          
          systemPrompt = `You are a code fixer. You fix ONE broken file.

RULES:
1. Read the error message below
2. Identify the EXACT syntax error
3. Output ONLY ONE <file> tag with the COMPLETE fixed file
4. Do NOT output any other files
5. Do NOT output explanations or text
6. Do NOT create new components
7. Preserve ALL existing functionality,  only fix the broken syntax
${brokenFile ? `\nTHE BROKEN FILE IS: ${brokenFile}\nOutput: <file path="src/components/${brokenFile}">...fixed code...</file>` : ''}
${brokenFileContent ? `\nCURRENT CONTENT OF THE BROKEN FILE:\n${brokenFileContent}` : ''}

Common fixes:
- Unclosed JSX tags: add the missing closing tag
- Unescaped apostrophes: use &apos; or {"'"}
- Missing closing quote in className: add the missing "
- TypeScript in .jsx: remove type annotations  
- Unterminated string: close the string
- Invalid SVG path: use a simpler valid path`;
        }

        // Morph Fast Apply override for edit mode
        const morphFastApplyEnabled = Boolean(isEdit && process.env.MORPH_API_KEY);
        if (morphFastApplyEnabled) {
          systemPrompt += `\nMORPH FAST APPLY MODE: Output <edit> blocks for existing files, <file> blocks for new files only.`;
        }

        // Build full prompt with context
        let fullPrompt = prompt;
        if (isErrorFix) {
          // Fast path for error fixes,  skip all heavy context, just pass the error
          fullPrompt = prompt;
          console.log('[generate-ai-code-stream] Error fix fast path,  skipping context building');
        } else if (context) {
          const contextParts = [];
          
          if (context.sandboxId) {
            contextParts.push(`Current sandbox ID: ${context.sandboxId}`);
          }
          
          if (context.structure) {
            contextParts.push(`Current file structure:\n${context.structure}`);
          }
          
          // Use backend file cache instead of frontend-provided files
          let backendFiles = global.sandboxState?.fileCache?.files || {};
          let hasBackendFiles = Object.keys(backendFiles).length > 0;
          
          console.log('[generate-ai-code-stream] Backend file cache status:');
          console.log('[generate-ai-code-stream] - Has sandboxState:', !!global.sandboxState);
          console.log('[generate-ai-code-stream] - Has fileCache:', !!global.sandboxState?.fileCache);
          console.log('[generate-ai-code-stream] - File count:', Object.keys(backendFiles).length);
          console.log('[generate-ai-code-stream] - Has manifest:', !!global.sandboxState?.fileCache?.manifest);
          
          // If no backend files and we're in edit mode, try to fetch from sandbox
          if (!hasBackendFiles && isEdit && (global.activeSandbox || context?.sandboxId)) {
            console.log('[generate-ai-code-stream] No backend files, attempting to fetch from sandbox...');
            
            try {
              const filesResponse = await fetch(`${process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000'}/api/get-sandbox-files`, {
                method: 'GET',
                headers: { 'Content-Type': 'application/json' }
              });
              
              if (filesResponse.ok) {
                const filesData = await filesResponse.json();
                if (filesData.success && filesData.files) {
                  console.log('[generate-ai-code-stream] Successfully fetched', Object.keys(filesData.files).length, 'files from sandbox');
                  
                  // Initialize sandboxState if needed
                  if (!global.sandboxState) {
                    global.sandboxState = {
                      fileCache: {
                        files: {},
                        lastSync: Date.now(),
                        sandboxId: context?.sandboxId || 'unknown'
                      }
                    } as any;
                  } else if (!global.sandboxState.fileCache) {
                    global.sandboxState.fileCache = {
                      files: {},
                      lastSync: Date.now(),
                      sandboxId: context?.sandboxId || 'unknown'
                    };
                  }
                  
                  // Store files in cache
                  for (const [path, content] of Object.entries(filesData.files)) {
                    const normalizedPath = path.replace('/home/user/app/', '');
                    if (global.sandboxState.fileCache) {
                      global.sandboxState.fileCache.files[normalizedPath] = {
                        content: content as string,
                        lastModified: Date.now()
                      };
                    }
                  }
                  
                  if (filesData.manifest && global.sandboxState.fileCache) {
                    global.sandboxState.fileCache.manifest = filesData.manifest;
                    
                    // Now try to analyze edit intent with the fetched manifest
                    if (!editContext) {
                      console.log('[generate-ai-code-stream] Analyzing edit intent with fetched manifest');
                      try {
                        const intentResponse = await fetch(`${process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000'}/api/analyze-edit-intent`, {
                          method: 'POST',
                          headers: { 'Content-Type': 'application/json' },
                          body: JSON.stringify({ prompt, manifest: filesData.manifest, model })
                        });
                        
                        if (intentResponse.ok) {
                          const { searchPlan } = await intentResponse.json();
                          console.log('[generate-ai-code-stream] Search plan received:', searchPlan);
                          
                          // Create edit context from AI analysis
                          // Note: We can't execute search here without file contents, so fall back to keyword method
                          const fileContext = selectFilesForEdit(prompt, filesData.manifest);
                          editContext = fileContext;
                          enhancedSystemPrompt = fileContext.systemPrompt;
                          
                          console.log('[generate-ai-code-stream] Edit context created with', editContext.primaryFiles.length, 'primary files');
                        }
                      } catch (error) {
                        console.error('[generate-ai-code-stream] Failed to analyze edit intent:', error);
                      }
                    }
                  }
                  
                  // Update variables
                  backendFiles = global.sandboxState.fileCache?.files || {};
                  hasBackendFiles = Object.keys(backendFiles).length > 0;
                  console.log('[generate-ai-code-stream] Updated backend cache with fetched files');
                }
              }
            } catch (error) {
              console.error('[generate-ai-code-stream] Failed to fetch sandbox files:', error);
            }
          }
          
          // Include current file contents from backend cache
          if (hasBackendFiles) {
            // If we have edit context, use intelligent file selection
            if (editContext && editContext.primaryFiles.length > 0) {
              contextParts.push('\nEXISTING APPLICATION - TARGETED EDIT MODE');
              contextParts.push(`\n${editContext.systemPrompt || enhancedSystemPrompt}\n`);
              
              // Get contents of primary and context files
              const primaryFileContents = await getFileContents(editContext.primaryFiles, global.sandboxState!.fileCache!.manifest!);
              const contextFileContents = await getFileContents(editContext.contextFiles, global.sandboxState!.fileCache!.manifest!);
              
              // Format files for AI
              const formattedFiles = formatFilesForAI(primaryFileContents, contextFileContents);
              contextParts.push(formattedFiles);
              
              contextParts.push('\nIMPORTANT: Only modify the files listed under "Files to Edit". The context files are provided for reference only.');
            } else {
              // Fallback to showing all files if no edit context
              console.log('[generate-ai-code-stream] WARNING: Using fallback mode - no edit context available');
              contextParts.push('\nEXISTING APPLICATION - TARGETED EDIT REQUIRED');
              contextParts.push('\nYou MUST analyze the user request and determine which specific file(s) to edit.');
              contextParts.push('\nCurrent project files (DO NOT regenerate all of these):');
              
              const fileEntries = Object.entries(backendFiles);
              console.log(`[generate-ai-code-stream] Using backend cache: ${fileEntries.length} files`);
              
              // Show file list first for reference
              contextParts.push('\n### File List:');
              for (const [path] of fileEntries) {
                contextParts.push(`- ${path}`);
              }
              
              // Include ALL files as context in fallback mode
              contextParts.push('\n### File Contents (ALL FILES FOR CONTEXT):');
              for (const [path, fileData] of fileEntries) {
                const content = fileData.content;
                if (typeof content === 'string') {
                  contextParts.push(`\n<file path="${path}">\n${content}\n</file>`);
                }
              }
              
              contextParts.push('\n🚨 CRITICAL INSTRUCTIONS - VIOLATION = FAILURE 🚨');
              contextParts.push('1. Analyze the user request: "' + prompt + '"');
              contextParts.push('2. Identify the MINIMUM number of files that need editing (usually just ONE)');
              contextParts.push('3. PRESERVE ALL EXISTING CONTENT in those files');
              contextParts.push('4. ONLY ADD/MODIFY the specific part requested');
              contextParts.push('5. DO NOT regenerate entire components from scratch');
              contextParts.push('6. DO NOT change unrelated parts of any file');
              contextParts.push('7. Generate ONLY the files that MUST be changed - NO EXTRAS');
              contextParts.push('\n⚠️ FILE COUNT RULE:');
              contextParts.push('- Simple change (color, text, spacing) = 1 file ONLY');
              contextParts.push('- Adding new component = 2 files MAX (new component + parent that imports it)');
              contextParts.push('- DO NOT exceed these limits unless absolutely necessary');
              contextParts.push('\nEXAMPLES OF CORRECT BEHAVIOR:');
              contextParts.push('✅ "add a chart to the hero" → Edit ONLY Hero.jsx, ADD the chart, KEEP everything else');
              contextParts.push('✅ "change header to black" → Edit ONLY Header.jsx, change ONLY the color');
              contextParts.push('✅ "fix spacing in footer" → Edit ONLY Footer.jsx, adjust ONLY spacing');
              contextParts.push('\nEXAMPLES OF FAILURES:');
              contextParts.push('❌ "change header color" → You edit Header, Footer, and App "for consistency"');
              contextParts.push('❌ "add chart to hero" → You regenerate the entire Hero component');
              contextParts.push('❌ "fix button" → You update 5 different component files');
              contextParts.push('\n⚠️ FINAL WARNING:');
              contextParts.push('If you generate MORE files than necessary, you have FAILED');
              contextParts.push('If you DELETE or REWRITE existing functionality, you have FAILED');
              contextParts.push('ONLY change what was EXPLICITLY requested - NOTHING MORE');
            }
          } else if (context.currentFiles && Object.keys(context.currentFiles).length > 0) {
            // Fallback to frontend-provided files if backend cache is empty
            console.log('[generate-ai-code-stream] Warning: Backend cache empty, using frontend files');
            contextParts.push('\nEXISTING APPLICATION - DO NOT REGENERATE FROM SCRATCH');
            contextParts.push('Current project files (modify these, do not recreate):');
            
            const fileEntries = Object.entries(context.currentFiles);
            for (const [path, content] of fileEntries) {
              if (typeof content === 'string') {
                contextParts.push(`\n<file path="${path}">\n${content}\n</file>`);
              }
            }
            contextParts.push('\nThe above files already exist. When the user asks to modify something (like "change the header color to black"), find the relevant file above and generate ONLY that file with the requested changes.');
          }
          
          // Add explicit edit mode indicator
          if (isEdit) {
            contextParts.push('\nEDIT MODE ACTIVE');
            contextParts.push('This is an incremental update to an existing application.');
            contextParts.push('DO NOT regenerate App.jsx, index.css, or other core files unless explicitly requested.');
            contextParts.push('ONLY create or modify the specific files needed for the user\'s request.');
            contextParts.push('\n⚠️ CRITICAL FILE OUTPUT FORMAT - VIOLATION = FAILURE:');
            contextParts.push('YOU MUST OUTPUT EVERY FILE IN THIS EXACT XML FORMAT:');
            contextParts.push('<file path="src/components/ComponentName.jsx">');
            contextParts.push('// Complete file content here');
            contextParts.push('</file>');
            contextParts.push('<file path="src/index.css">');
            contextParts.push('/* CSS content here */');
            contextParts.push('</file>');
            contextParts.push('\n❌ NEVER OUTPUT: "Generated Files: index.css, App.jsx"');
            contextParts.push('❌ NEVER LIST FILE NAMES WITHOUT CONTENT');
            contextParts.push('✅ ALWAYS: One <file> tag per file with COMPLETE content');
            contextParts.push('✅ ALWAYS: Include EVERY file you modified');
          } else if (!hasBackendFiles) {
            // First generation mode - make it beautiful!
            contextParts.push('\n🎨 FIRST GENERATION MODE - CREATE SOMETHING BEAUTIFUL!');
            contextParts.push('\nThis is the user\'s FIRST experience. Make it impressive:');
            contextParts.push('1. **USE TAILWIND PROPERLY** - Use standard Tailwind color classes');
            contextParts.push('2. **NO PLACEHOLDERS** - Use real content, not lorem ipsum');
            contextParts.push('3. **COMPLETE COMPONENTS** - Header, Hero, Features, Footer minimum');
            contextParts.push('4. **VISUAL POLISH** - Shadows, hover states, transitions');
            contextParts.push('5. **STANDARD CLASSES** - bg-white, text-gray-900, bg-blue-500, NOT bg-background');
            contextParts.push('6. **BUILD FROM PROMPT** - Create what the user described in their message, NOT a website clone');
            contextParts.push('\nCreate a polished, professional application that works perfectly on first load.');
            contextParts.push('The user typed a DESCRIPTION of what they want - build that. Do NOT scrape or clone anything.');
            contextParts.push('\n⚠️ OUTPUT FORMAT:');
            contextParts.push('Use <file path="...">content</file> tags for EVERY file');
            contextParts.push('NEVER output "Generated Files:" as plain text');
          }
          
          // Add conversation context (scraped websites, etc)
          if (context.conversationContext) {
            if (context.conversationContext.scrapedWebsites?.length > 0) {
              contextParts.push('\nScraped Websites in Context:');
              context.conversationContext.scrapedWebsites.forEach((site: any) => {
                contextParts.push(`\nURL: ${site.url}`);
                contextParts.push(`Scraped: ${new Date(site.timestamp).toLocaleString()}`);
                if (site.content) {
                  // Include a summary of the scraped content
                  const contentPreview = typeof site.content === 'string' 
                    ? site.content.substring(0, 1000) 
                    : JSON.stringify(site.content).substring(0, 1000);
                  contextParts.push(`Content Preview: ${contentPreview}...`);
                }
              });
            }
            
            if (context.conversationContext.currentProject) {
              contextParts.push(`\nCurrent Project: ${context.conversationContext.currentProject}`);
            }
          }
          
          if (contextParts.length > 0) {
            if (morphFastApplyEnabled) {
              contextParts.push('\nOUTPUT FORMAT (REQUIRED IN MORPH MODE):');
              contextParts.push('<edit target_file="src/components/Component.jsx">');
              contextParts.push('<instructions>Minimal, precise instruction.</instructions>');
              contextParts.push('<update>// Smallest necessary snippet</update>');
              contextParts.push('</edit>');
              contextParts.push('\nIf you need to create a NEW file, then and only then output a full file:');
              contextParts.push('<file path="src/components/NewComponent.jsx">');
              contextParts.push('// Full file content when creating new files');
              contextParts.push('</file>');
            }
            fullPrompt = `CONTEXT:\n${contextParts.join('\n')}\n\nUSER REQUEST:\n${prompt}`;
          }
        }
        
        await sendProgress({ type: 'status', message: 'Planning application structure...' });
        
        console.log('\n[generate-ai-code-stream] Starting streaming response...\n');
        
        // Track packages that need to be installed
        const packagesToInstall: string[] = [];
        
        // ─── Fast Mode: override model if enabled ──────────
        let effectiveModel = model;
        if (global.sableFastModeState && isFastModeEnabled(global.sableFastModeState)) {
          effectiveModel = getFastModeModel(global.sableFastModeState);
          console.log(`[generate-ai-code-stream] Fast mode active: ${model} → ${effectiveModel}`);
        }
        
        // Determine which provider to use based on model
        const isAnthropic = effectiveModel.startsWith('anthropic/');
        const isGoogle = effectiveModel.startsWith('google/');
        const isOpenAI = effectiveModel.startsWith('openai/');
        const isOllama = effectiveModel.startsWith('ollama/');
        const isOpenWebUI = effectiveModel.startsWith('openwebui/');
        const isKimiGroq = effectiveModel === 'moonshotai/kimi-k2-instruct-0905';
        const modelProvider = isOllama ? ollamaProvider :
                              (isOpenWebUI ? openWebUIProvider :
                              (isAnthropic ? anthropic : 
                              (isOpenAI ? openai : 
                              (isGoogle ? googleGenerativeAI : 
                              (isKimiGroq ? groq : groq)))));
        
        // Fix model name transformation for different providers
        let actualModel: string;
        if (isOllama) {
          actualModel = effectiveModel.replace('ollama/', '');
        } else if (isOpenWebUI) {
          actualModel = effectiveModel.replace('openwebui/', '');
        } else if (isAnthropic) {
          actualModel = effectiveModel.replace('anthropic/', '');
        } else if (isOpenAI) {
          actualModel = effectiveModel.replace('openai/', '');
        } else if (isKimiGroq) {
          // Kimi on Groq - use full model string
          actualModel = 'moonshotai/kimi-k2-instruct-0905';
        } else if (isGoogle) {
          // Google uses specific model names - convert our naming to theirs  
          actualModel = effectiveModel.replace('google/', '');
        } else {
          actualModel = effectiveModel;
        }

        const providerName = isOllama ? 'Ollama' : isOpenWebUI ? 'OpenWebUI' : isAnthropic ? 'Anthropic' : isGoogle ? 'Google' : isOpenAI ? 'OpenAI' : 'Groq';
        console.log(`[generate-ai-code-stream] Using provider: ${providerName}, model: ${actualModel}`);
        console.log(`[generate-ai-code-stream] AI Gateway enabled: ${isUsingAIGateway}`);
        console.log(`[generate-ai-code-stream] Model string: ${effectiveModel}`);

        // Make streaming API call with appropriate provider
        // AI SDK v5: default provider() uses Responses API (/v1/responses)
        // Ollama/OpenWebUI only support Chat Completions API, so use .chat()
        const modelInstance = (isOllama || isOpenWebUI)
          ? (modelProvider as any).chat(actualModel)
          : modelProvider(actualModel);
        
        // Build tool context from active sandbox for agentic editing
        const toolContext = buildToolContext(global.activeSandboxProvider);
        
        // Filter tools by user intent (reduces prompt size for small models)
        let aiTools: Record<string, any> | undefined;
        if (toolContext && isEdit) {
          const intentTools = selectToolsForIntent(fullPrompt);
          // Build AI SDK tools from full set (intent info is logged for debugging)
          aiTools = buildAISDKTools(toolContext);
          console.log(`[generate-ai-code-stream] Intent filter: ${intentTools.map(t => t.name).join(', ')}`);
        }
        const useTools = isEdit && aiTools && Object.keys(aiTools).length > 0;
        
        // Gather context pipeline data (git, recent edits, deps) for edit mode
        let pipelineContextStr = '';
        if (isEdit && toolContext) {
          try {
            const pipelineCtx = await runContextPipeline({
              readFile: toolContext.readFile,
              runCommand: toolContext.runCommand,
            });
            pipelineContextStr = formatContextForPrompt(pipelineCtx);
            if (pipelineContextStr) {
              console.log(`[generate-ai-code-stream] Context pipeline: ${pipelineContextStr.substring(0, 100)}...`);
            }
          } catch (e) {
            console.warn('[generate-ai-code-stream] Context pipeline failed (non-fatal):', e);
          }
        }
        
        if (useTools) {
          console.log(`[generate-ai-code-stream] Agentic mode: ${Object.keys(aiTools!).length} tools available`);
          // Add tool descriptions to system prompt for edit mode
          systemPrompt += `\n\n${getToolSummaryForPrompt()}`;
        }
        
        // Add context pipeline data to system prompt
        if (pipelineContextStr) {
          systemPrompt += `\n\n${pipelineContextStr}`;
        }
        
        // ─── Session Memory: inject remembered context ────────
        const memoryPrompt = formatMemoryForPrompt(global.sableSessionMemoryState!);
        if (memoryPrompt) {
          systemPrompt += `\n\n${memoryPrompt}`;
        }
        
        // ─── Effort Mode: resolve and apply ───────────────────
        const autoEffort = resolveEffortFromPrompt(prompt);
        const effortLevel = autoEffort || global.sableEffortLevel || 'medium';
        const effortConfig = getEffortConfig(effortLevel);
        if (effortConfig.systemPromptSuffix) {
          systemPrompt += `\n\n${effortConfig.systemPromptSuffix}`;
        }
        
        // ─── Token Budget: check before sending ──────────────
        const conversationMessages = (global.conversationState?.context?.messages || [])
          .map((m: any) => ({ role: m.role, content: m.content }));
        const currentTokens = estimateTotalTokens(conversationMessages);
        const budgetTracker = createBudgetTracker(model);
        const budgetDecision = checkTokenBudget(budgetTracker, currentTokens);
        
        if (budgetDecision.action === 'compact') {
          console.log(`[generate-ai-code-stream] Token budget recommends compaction (${currentTokens} tokens)`);
          // Auto-compact if threshold is hit
          if (shouldAutoCompact(conversationMessages, model, global.sableAutoCompactState!)) {
            try {
              const compactResult = await compactConversation(conversationMessages, model);
              // Replace conversation messages with compacted version
              if (global.conversationState) {
                global.conversationState.context.messages = compactResult.messages.map((m: any, i: number) => ({
                  id: `msg-compact-${i}`,
                  role: m.role,
                  content: m.content,
                  timestamp: Date.now(),
                  metadata: {}
                }));
              }
              console.log(`[generate-ai-code-stream] Auto-compacted: ${conversationMessages.length} → ${compactResult.messages.length} messages`);
              await sendProgress({ type: 'status', message: 'Context compacted to save tokens.' });
            } catch (e) {
              console.warn('[generate-ai-code-stream] Auto-compact failed (non-fatal):', e);
            }
          }
        }
        
        // ─── Micro-compact old tool outputs ──────────────────
        const turnNumber = global.sableAutoCompactState?.turnCounter || 0;
        global.sableAutoCompactState = advanceTurn(global.sableAutoCompactState!);
        
        // Apply micro-compact to trim old tool results from conversation
        const microCompactResult = microCompact(
          (global.conversationState?.context?.messages || []).map((m: any) => ({
            role: m.role,
            content: m.content,
            turnIndex: undefined,
          })),
          turnNumber,
        );
        // Update conversation messages with micro-compacted versions
        if (global.conversationState && microCompactResult.messages.length > 0) {
          for (let i = 0; i < Math.min(microCompactResult.messages.length, global.conversationState.context.messages.length); i++) {
            global.conversationState.context.messages[i].content = microCompactResult.messages[i].content;
          }
          if (microCompactResult.tokensFreed > 0) {
            console.log(`[generate-ai-code-stream] Micro-compact freed ~${microCompactResult.tokensFreed} tokens`);
          }
        }
        
        // ─── Plan Mode: suggest plan for complex tasks ───────
        if (isEdit && shouldSuggestPlan(prompt) && !global.sableActivePlan) {
          const planPromptSuffix = getPlanModePrompt();
          systemPrompt += `\n\n${planPromptSuffix}`;
          console.log('[generate-ai-code-stream] Plan mode suggested for complex task');
        }
        
        // ─── File Change Detection: check for external edits ─
        if (global.sableFileTrackingState && context?.currentFiles) {
          const currentFileEntries = Object.entries(context.currentFiles)
            .map(([path, content]) => ({ path, content: typeof content === 'string' ? content : (content as any)?.content || null }));
          const changes = currentFileEntries
            .map(f => detectFileChange(global.sableFileTrackingState!, f.path, f.content))
            .filter((c): c is NonNullable<typeof c> => c !== null);
          if (changes.length > 0) {
            const changePrompt = formatChangesForPrompt(changes);
            systemPrompt += `\n\n${changePrompt}`;
            global.sableFileTrackingState = advanceTurnWithChanges(global.sableFileTrackingState, changes);
            console.log(`[generate-ai-code-stream] Detected ${changes.length} file changes since last turn`);
          }
          // Snapshot current files for next comparison
          for (const f of currentFileEntries) {
            if (f.content) {
              global.sableFileTrackingState = snapshotFile(global.sableFileTrackingState, f.path, f.content);
            }
          }
        }
        
        // Track generation start time for cost tracking
        const generationStartTime = Date.now();
        
        // Conversation recovery: detect if we're resuming from an interruption
        let recoveryPrefix = '';
        if (global.conversationState?.context?.messages && global.conversationState.context.messages.length > 0) {
          const interruptState = detectInterruption(
            global.conversationState.context.messages.map((m: any) => ({ role: m.role, content: m.content })),
            false // wasStreaming — we don't know from server side, assume clean
          );
          const recovery = buildRecoveryMessage(interruptState);
          if (recovery) {
            recoveryPrefix = recovery + '\n\n';
            console.log(`[generate-ai-code-stream] Conversation recovery: ${interruptState.type}`);
          }
        }
        
        const streamOptions: any = {
          model: modelInstance,
          messages: [
            { 
              role: 'system', 
              content: systemPrompt
            },
            { 
              role: 'user', 
              content: recoveryPrefix + (useTools
                ? fullPrompt + `\n\nYou may use tools to read and understand the codebase before making changes. After exploring, output your changes as <file path="...">...complete code...</file> tags. If making small edits, prefer the file_edit tool over full file rewrites.`
                : fullPrompt + `\n\nREMINDER: Output ONLY <file path="...">...complete code...</file> tags. NO explanations. NO markdown. NO conversational text. Start your response immediately with <file.`)
            }
          ],
          maxTokens: effortConfig.maxTokens,
          stopSequences: [],
          // Enable multi-step tool calling for edit mode
          ...(useTools ? { tools: aiTools, maxSteps: effortConfig.maxSteps } : {}),
        };
        
        // Add temperature from effort mode for non-reasoning models
        if (!model.startsWith('openai/gpt-5')) {
          streamOptions.temperature = effortConfig.temperature;
        }
        
        // Add reasoning effort for GPT-5 models
        if (isOpenAI) {
          streamOptions.experimental_providerMetadata = {
            openai: {
              reasoningEffort: 'high'
            }
          };
        }
        
        let result;
        let retryCount = 0;
        const maxRetries = 2;
        
        while (retryCount <= maxRetries) {
          try {
            result = await streamText(streamOptions);
            break; // Success, exit retry loop
          } catch (streamError: any) {
            console.error(`[generate-ai-code-stream] Error calling streamText (attempt ${retryCount + 1}/${maxRetries + 1}):`, streamError);
            
            // Use error-recovery classifier for smarter retry decisions
            const errorCategory = classifyError(streamError);
            console.log(`[generate-ai-code-stream] Error classified as: ${errorCategory}`);
            
            // Permanent errors — don't retry
            if (errorCategory === 'permanent') {
              await sendProgress({ 
                type: 'error', 
                message: `Model error: ${streamError.message}` 
              });
              throw streamError;
            }
            
            // Context overflow — don't retry (need to reduce context)
            if (errorCategory === 'context_overflow') {
              await sendProgress({ 
                type: 'error', 
                message: `Context too large for model. Try a shorter prompt or fewer files.` 
              });
              throw streamError;
            }
            
            // Check if this is a Groq service unavailable error
            const isGroqServiceError = isKimiGroq && streamError.message?.includes('Service unavailable');
            
            if (retryCount < maxRetries && (errorCategory === 'retryable' || errorCategory === 'rate_limit')) {
              retryCount++;
              
              // Rate limit: respect retry-after header
              let delay = retryCount * 2000;
              if (errorCategory === 'rate_limit' && streamError?.headers?.['retry-after']) {
                const retryAfter = parseInt(streamError.headers['retry-after'], 10);
                if (!isNaN(retryAfter)) delay = retryAfter * 1000;
              }
              
              console.log(`[generate-ai-code-stream] Retrying in ${delay}ms (${errorCategory})...`);
              
              // Send progress update about retry
              await sendProgress({ 
                type: 'info', 
                message: `${errorCategory === 'rate_limit' ? 'Rate limited' : 'Service temporarily unavailable'}, retrying (attempt ${retryCount + 1}/${maxRetries + 1})...` 
              });
              
              // Wait before retry with exponential backoff
              await new Promise(resolve => setTimeout(resolve, delay));
              
              // If Groq fails, try switching to a fallback model
              if (isGroqServiceError && retryCount === maxRetries) {
                console.log('[generate-ai-code-stream] Groq service unavailable, falling back to GPT-4');
                streamOptions.model = openai('gpt-4-turbo');
                actualModel = 'gpt-4-turbo';
              }
            } else {
              // Final error, send to user
              await sendProgress({ 
                type: 'error', 
                message: `Failed to initialize ${isGoogle ? 'Gemini' : isAnthropic ? 'Claude' : isOpenAI ? 'GPT-5' : isKimiGroq ? 'Kimi (Groq)' : 'Groq'} streaming: ${streamError.message}` 
              });
              
              // If this is a Google model error, provide helpful info
              if (isGoogle) {
                await sendProgress({ 
                  type: 'info', 
                  message: 'Tip: Make sure your GEMINI_API_KEY is set correctly and has proper permissions.' 
                });
              }
              
              throw streamError;
            }
          }
        }
        
        // Stream the response and parse in real-time
        let generatedCode = '';
        let currentFile = '';
        let currentFilePath = '';
        let componentCount = 0;
        let isInFile = false;
        let isInTag = false;
        let conversationalBuffer = '';
        
        // Buffer for incomplete tags
        let tagBuffer = '';
        
        // Track tool calls for UI feedback
        let toolCallCount = 0;
        
        // Use fullStream when tools are enabled to capture tool events,
        // otherwise fall back to textStream for simpler parsing
        if (useTools) {
          for await (const part of (result as any)?.fullStream || []) {
            if (part.type === 'text-delta') {
              const text = part.textDelta || '';
              generatedCode += text;
              currentFile += text;
              
              const searchText = tagBuffer + text;
              process.stdout.write(text);
              
              const hasOpenTag = /<(file|package|packages|explanation|command|structure|template)\b/.test(text);
              const hasCloseTag = /<\/(file|package|packages|explanation|command|structure|template)>/.test(text);
              
              if (hasOpenTag) {
                if (conversationalBuffer.trim() && !isInTag) {
                  await sendProgress({ type: 'conversation', text: conversationalBuffer.trim() });
                  conversationalBuffer = '';
                }
                isInTag = true;
              }
              if (hasCloseTag) isInTag = false;
              if (!isInTag && !hasOpenTag) conversationalBuffer += text;
              
              await sendProgress({ type: 'stream', text, raw: true });
              
              if (text.includes('<file path="')) {
                const pathMatch = text.match(/<file path="([^"]+)"/);
                if (pathMatch) {
                  currentFilePath = pathMatch[1];
                  isInFile = true;
                  currentFile = text;
                }
              }
              
              if (isInFile && currentFile.includes('</file>')) {
                isInFile = false;
                if (currentFilePath.includes('components/')) {
                  componentCount++;
                  const componentName = currentFilePath.split('/').pop()?.replace('.jsx', '') || 'Component';
                  await sendProgress({ type: 'component', name: componentName, path: currentFilePath, index: componentCount });
                } else if (currentFilePath.includes('App.jsx')) {
                  await sendProgress({ type: 'app', message: 'Generated main App.jsx', path: currentFilePath });
                }
                currentFile = '';
                currentFilePath = '';
              }
              
              tagBuffer = searchText.substring(Math.max(0, searchText.length - 50));
            } else if (part.type === 'tool-call') {
              toolCallCount++;
              console.log(`[generate-ai-code-stream] Tool call #${toolCallCount}: ${part.toolName}(${JSON.stringify(part.args).substring(0, 100)})`);
              await sendProgress({
                type: 'tool-call',
                name: part.toolName,
                args: part.args,
                index: toolCallCount,
              });
            } else if (part.type === 'tool-result') {
              const output = typeof part.result === 'string' ? part.result : JSON.stringify(part.result);
              const truncated = output.length > 500 ? output.substring(0, 500) + '...' : output;
              console.log(`[generate-ai-code-stream] Tool result for ${part.toolName}: ${truncated.substring(0, 100)}`);
              await sendProgress({
                type: 'tool-result',
                name: part.toolName,
                output: truncated,
                index: toolCallCount,
              });
              
              // File change detection: snapshot files modified by tools
              if (part.toolName === 'file_write' || part.toolName === 'file_edit') {
                const filePath = (part as any).args?.path || (part as any).args?.filePath;
                if (filePath && global.sableFileTrackingState && global.activeSandboxProvider) {
                  try {
                    const content = await global.activeSandboxProvider.readFile(filePath);
                    global.sableFileTrackingState = snapshotFile(global.sableFileTrackingState, filePath, content);
                  } catch { /* file read failed, skip snapshot */ }
                }
              }
            }
          }
        } else {
        // Stream the response and parse for packages in real-time
        for await (const textPart of result?.textStream || []) {
          const text = textPart || '';
          generatedCode += text;
          currentFile += text;
          
          // Combine with buffer for tag detection
          const searchText = tagBuffer + text;
          
          // Log streaming chunks to console
          process.stdout.write(text);
          
          // Check if we're entering or leaving a tag
          const hasOpenTag = /<(file|package|packages|explanation|command|structure|template)\b/.test(text);
          const hasCloseTag = /<\/(file|package|packages|explanation|command|structure|template)>/.test(text);
          
          if (hasOpenTag) {
            // Send any buffered conversational text before the tag
            if (conversationalBuffer.trim() && !isInTag) {
              await sendProgress({ 
                type: 'conversation', 
                text: conversationalBuffer.trim()
              });
              conversationalBuffer = '';
            }
            isInTag = true;
          }
          
          if (hasCloseTag) {
            isInTag = false;
          }
          
          // If we're not in a tag, buffer as conversational text
          if (!isInTag && !hasOpenTag) {
            conversationalBuffer += text;
          }
          
          // Stream the raw text for live preview
          await sendProgress({ 
            type: 'stream', 
            text: text,
            raw: true 
          });
          
          // Debug: Log every 100 characters streamed
          if (generatedCode.length % 100 < text.length) {
            console.log(`[generate-ai-code-stream] Streamed ${generatedCode.length} chars`);
          }
          
          // Check for package tags in buffered text (ONLY for edits, not initial generation)
          let lastIndex = 0;
          if (isEdit) {
            const packageRegex = /<package>([^<]+)<\/package>/g;
            let packageMatch;
            
            while ((packageMatch = packageRegex.exec(searchText)) !== null) {
              const packageName = packageMatch[1].trim();
              if (packageName && !packagesToInstall.includes(packageName)) {
                packagesToInstall.push(packageName);
                console.log(`[generate-ai-code-stream] Package detected: ${packageName}`);
                await sendProgress({ 
                  type: 'package', 
                  name: packageName,
                  message: `Package detected: ${packageName}`
                });
              }
              lastIndex = packageMatch.index + packageMatch[0].length;
            }
          }
          
          // Keep unmatched portion in buffer for next iteration
          tagBuffer = searchText.substring(Math.max(0, lastIndex - 50)); // Keep last 50 chars
          
          // Check for file boundaries
          if (text.includes('<file path="')) {
            const pathMatch = text.match(/<file path="([^"]+)"/);
            if (pathMatch) {
              currentFilePath = pathMatch[1];
              isInFile = true;
              currentFile = text;
            }
          }
          
          // Check for file end
          if (isInFile && currentFile.includes('</file>')) {
            isInFile = false;
            
            // Send component progress update
            if (currentFilePath.includes('components/')) {
              componentCount++;
              const componentName = currentFilePath.split('/').pop()?.replace('.jsx', '') || 'Component';
              await sendProgress({ 
                type: 'component', 
                name: componentName,
                path: currentFilePath,
                index: componentCount
              });
            } else if (currentFilePath.includes('App.jsx')) {
              await sendProgress({ 
                type: 'app', 
                message: 'Generated main App.jsx',
                path: currentFilePath
              });
            }
            
            currentFile = '';
            currentFilePath = '';
          }
        }
        } // end else (non-tool mode)
        
        if (toolCallCount > 0) {
          console.log(`[generate-ai-code-stream] Agentic mode completed: ${toolCallCount} tool calls`);
          await sendProgress({ type: 'info', message: `Used ${toolCallCount} tool calls to analyze and edit the project` });
        }
        
        console.log('\n\n[generate-ai-code-stream] Streaming complete.');
        console.log('[generate-ai-code-stream] Total generated:', generatedCode.length, 'chars');
        
        // Send any remaining conversational text
        if (conversationalBuffer.trim()) {
          await sendProgress({ 
            type: 'conversation', 
            text: conversationalBuffer.trim()
          });
        }

        // RECOVERY: If the LLM didn't use <file> tags, try to extract code
        const hasFileTags = /<file path="[^"]+">/.test(generatedCode);
        if (!hasFileTags && generatedCode.length > 50) {
          console.warn('[generate-ai-code-stream] NO <file> tags found! Attempting code recovery...');
          await sendProgress({ type: 'status', message: 'Extracting code from response...' });
          
          // Try to extract code from markdown fences
          const fenceRegex = /```(?:jsx?|tsx?|css|html|json)?\n([\s\S]*?)```/g;
          let fenceMatch;
          let recoveredFiles: Array<{ path: string; content: string }> = [];
          let fenceIdx = 0;
          
          while ((fenceMatch = fenceRegex.exec(generatedCode)) !== null) {
            const code = fenceMatch[1].trim();
            if (code.length < 20) continue;
            
            // Try to infer filename from content
            let inferredPath = 'src/App.jsx';
            if (code.includes('@tailwind')) {
              inferredPath = 'src/index.css';
            } else if (code.includes('export default function App') || code.includes('function App(')) {
              inferredPath = 'src/App.jsx';
            } else if (code.includes('export default function') || code.includes('export function')) {
              const fnMatch = code.match(/export\s+(?:default\s+)?function\s+(\w+)/);
              const name = fnMatch ? fnMatch[1] : `Component${fenceIdx}`;
              inferredPath = `src/components/${name}.jsx`;
            } else if (fenceIdx > 0) {
              inferredPath = `src/components/Component${fenceIdx}.jsx`;
            }
            
            recoveredFiles.push({ path: inferredPath, content: code });
            fenceIdx++;
          }
          
          // If we found fenced code, wrap it in <file> tags and add to generatedCode
          if (recoveredFiles.length > 0) {
            console.log('[generate-ai-code-stream] Recovered', recoveredFiles.length, 'files from markdown fences');
            let recoveredCode = '';
            for (const rf of recoveredFiles) {
              recoveredCode += `<file path="${rf.path}">\n${rf.content}\n</file>\n`;
            }
            generatedCode = recoveredCode;
            
            await sendProgress({ type: 'info', message: `Recovered ${recoveredFiles.length} file(s) from AI response` });
          } else {
            // Last resort: if the response looks like JSX/React code, wrap it as App.jsx
            const looksLikeCode = generatedCode.includes('import ') || 
                                  generatedCode.includes('function ') || 
                                  generatedCode.includes('const ') ||
                                  generatedCode.includes('export ') ||
                                  generatedCode.includes('className=');
            if (looksLikeCode) {
              console.log('[generate-ai-code-stream] Response looks like raw code, wrapping as App.jsx');
              // Strip any leading conversational text before the first import/function
              const codeStart = generatedCode.search(/^(import |const |function |export )/m);
              const rawCode = codeStart >= 0 ? generatedCode.slice(codeStart) : generatedCode;
              generatedCode = `<file path="src/App.jsx">\n${rawCode.trim()}\n</file>`;
              await sendProgress({ type: 'info', message: 'Wrapped raw code as App.jsx' });
            } else {
              // The LLM just returned text, no code at all
              console.error('[generate-ai-code-stream] LLM returned only text, no code found');
              await sendProgress({ 
                type: 'error', 
                error: 'The AI model did not return any code. It returned conversational text instead. Try again or switch to a different model.'
              });
            }
          }
        }
        
        // Also parse <packages> tag for multiple packages - ONLY for edits
        if (isEdit) {
          const packagesRegex = /<packages>([\s\S]*?)<\/packages>/g;
          let packagesMatch;
          while ((packagesMatch = packagesRegex.exec(generatedCode)) !== null) {
            const packagesContent = packagesMatch[1].trim();
            const packagesList = packagesContent.split(/[\n,]+/)
              .map(pkg => pkg.trim())
              .filter(pkg => pkg.length > 0);
            
            for (const packageName of packagesList) {
              if (!packagesToInstall.includes(packageName)) {
                packagesToInstall.push(packageName);
                console.log(`[generate-ai-code-stream] Package from <packages> tag: ${packageName}`);
                await sendProgress({ 
                  type: 'package', 
                  name: packageName,
                  message: `Package detected: ${packageName}`
                });
              }
            }
          }
        }
        
        // Function to extract packages from import statements
        function extractPackagesFromCode(content: string): string[] {
          const packages: string[] = [];
          // Match ES6 imports
          const importRegex = /import\s+(?:(?:\{[^}]*\}|\*\s+as\s+\w+|\w+)(?:\s*,\s*(?:\{[^}]*\}|\*\s+as\s+\w+|\w+))*\s+from\s+)?['"]([^'"]+)['"]/g;
          let importMatch;
          
          while ((importMatch = importRegex.exec(content)) !== null) {
            const importPath = importMatch[1];
            // Skip relative imports and built-in React
            if (!importPath.startsWith('.') && !importPath.startsWith('/') && 
                importPath !== 'react' && importPath !== 'react-dom' &&
                !importPath.startsWith('@/')) {
              // Extract package name (handle scoped packages like @heroicons/react)
              const packageName = importPath.startsWith('@') 
                ? importPath.split('/').slice(0, 2).join('/')
                : importPath.split('/')[0];
              
              if (!packages.includes(packageName)) {
                packages.push(packageName);
              }
            }
          }
          
          return packages;
        }
        
        // Parse files and send progress for each
        const fileRegex = /<file path="([^"]+)">([\s\S]*?)<\/file>/g;
        const files = [];
        let match;
        
        while ((match = fileRegex.exec(generatedCode)) !== null) {
          const filePath = match[1];
          const content = match[2].trim();
          files.push({ path: filePath, content });
          
          // Extract packages from file content - ONLY for edits
          if (isEdit) {
            const filePackages = extractPackagesFromCode(content);
            for (const pkg of filePackages) {
              if (!packagesToInstall.includes(pkg)) {
                packagesToInstall.push(pkg);
                console.log(`[generate-ai-code-stream] Package detected from imports: ${pkg}`);
                await sendProgress({ 
                  type: 'package', 
                  name: pkg,
                  message: `Package detected from imports: ${pkg}`
                });
              }
            }
          }
          
          // Send progress for each file (reusing componentCount from streaming)
          if (filePath.includes('components/')) {
            const componentName = filePath.split('/').pop()?.replace('.jsx', '') || 'Component';
            await sendProgress({ 
              type: 'component', 
              name: componentName,
              path: filePath,
              index: componentCount
            });
          } else if (filePath.includes('App.jsx')) {
            await sendProgress({ 
              type: 'app', 
              message: 'Generated main App.jsx',
              path: filePath
            });
          }
        }
        
        // Extract explanation
        const explanationMatch = generatedCode.match(/<explanation>([\s\S]*?)<\/explanation>/);
        const explanation = explanationMatch ? explanationMatch[1].trim() : 'Code generated successfully!';
        
        // Validate generated code for truncation issues
        const truncationWarnings: string[] = [];
        
        // Skip ellipsis checking entirely - too many false positives with spread operators, loading text, etc.
        
        // Check for unclosed file tags
        const fileOpenCount = (generatedCode.match(/<file path="/g) || []).length;
        const fileCloseCount = (generatedCode.match(/<\/file>/g) || []).length;
        if (fileOpenCount !== fileCloseCount) {
          truncationWarnings.push(`Unclosed file tags detected: ${fileOpenCount} open, ${fileCloseCount} closed`);
        }
        
        // Check for files that seem truncated (very short or ending abruptly)
        const truncationCheckRegex = /<file path="([^"]+)">([\s\S]*?)(?:<\/file>|$)/g;
        let truncationMatch;
        while ((truncationMatch = truncationCheckRegex.exec(generatedCode)) !== null) {
          const filePath = truncationMatch[1];
          const content = truncationMatch[2];
          
          // Only check for really obvious HTML truncation - file ends with opening tag
          if (content.trim().endsWith('<') || content.trim().endsWith('</')) {
            truncationWarnings.push(`File ${filePath} appears to have incomplete HTML tags`);
          }
          
          // Skip "..." check - too many false positives with loading text, etc.
          
          // Only check for SEVERE truncation issues
          if (filePath.match(/\.(jsx?|tsx?)$/)) {
            // Only check for severely unmatched brackets (more than 3 difference)
            const openBraces = (content.match(/{/g) || []).length;
            const closeBraces = (content.match(/}/g) || []).length;
            const braceDiff = Math.abs(openBraces - closeBraces);
            if (braceDiff > 3) { // Only flag severe mismatches
              truncationWarnings.push(`File ${filePath} has severely unmatched braces (${openBraces} open, ${closeBraces} closed)`);
            }
            
            // Check if file is extremely short and looks incomplete
            if (content.length < 20 && content.includes('function') && !content.includes('}')) {
              truncationWarnings.push(`File ${filePath} appears severely truncated`);
            }
          }
        }
        
        // Handle truncation with automatic retry (if enabled in config)
        if (truncationWarnings.length > 0 && appConfig.codeApplication.enableTruncationRecovery) {
          console.warn('[generate-ai-code-stream] Truncation detected, attempting to fix:', truncationWarnings);
          
          await sendProgress({
            type: 'warning',
            message: 'Detected incomplete code generation. Attempting to complete...',
            warnings: truncationWarnings
          });
          
          // Try to fix truncated files automatically
          const truncatedFiles: string[] = [];
          const fileRegex = /<file path="([^"]+)">([\s\S]*?)(?:<\/file>|$)/g;
          let match;
          
          while ((match = fileRegex.exec(generatedCode)) !== null) {
            const filePath = match[1];
            const content = match[2];
            
            // Check if this file appears truncated - be more selective
            const hasEllipsis = content.includes('...') && 
                               !content.includes('...rest') && 
                               !content.includes('...props') &&
                               !content.includes('spread');
                               
            const endsAbruptly = content.trim().endsWith('...') || 
                                 content.trim().endsWith(',') ||
                                 content.trim().endsWith('(');
                                 
            const hasUnclosedTags = content.includes('</') && 
                                    !content.match(/<\/[a-zA-Z0-9]+>/) &&
                                    content.includes('<');
                                    
            const tooShort = content.length < 50 && filePath.match(/\.(jsx?|tsx?)$/);
            
            // Check for unmatched braces specifically
            const openBraceCount = (content.match(/{/g) || []).length;
            const closeBraceCount = (content.match(/}/g) || []).length;
            const hasUnmatchedBraces = Math.abs(openBraceCount - closeBraceCount) > 1;
            
            const isTruncated = (hasEllipsis && endsAbruptly) || 
                               hasUnclosedTags || 
                               (tooShort && !content.includes('export')) ||
                               hasUnmatchedBraces;
            
            if (isTruncated) {
              truncatedFiles.push(filePath);
            }
          }
          
          // If we have truncated files, try to regenerate them
          if (truncatedFiles.length > 0) {
            console.log('[generate-ai-code-stream] Attempting to regenerate truncated files:', truncatedFiles);
            
            for (const filePath of truncatedFiles) {
              await sendProgress({
                type: 'info',
                message: `Completing ${filePath}...`
              });
              
              try {
                // Create a focused prompt to complete just this file
                const completionPrompt = `Complete the following file that was truncated. Provide the FULL file content.
                
File: ${filePath}
Original request: ${prompt}
                
Provide the complete file content without any truncation. Include all necessary imports, complete all functions, and close all tags properly.`;
                
                // Make a focused API call to complete this specific file
                // Create a new client for the completion based on the provider
                let completionClient: any;
                let useChat = false;
                if (isOllama) {
                  completionClient = ollamaProvider;
                  useChat = true;
                } else if (isOpenWebUI) {
                  completionClient = openWebUIProvider;
                  useChat = true;
                } else if (model.includes('gpt') || model.includes('openai')) {
                  completionClient = openai;
                } else if (model.includes('claude')) {
                  completionClient = anthropic;
                } else if (model === 'moonshotai/kimi-k2-instruct-0905') {
                  completionClient = groq;
                } else {
                  completionClient = groq;
                }
                
                // Determine the correct model name for the completion
                let completionModelName: string;
                if (model === 'moonshotai/kimi-k2-instruct-0905') {
                  completionModelName = 'moonshotai/kimi-k2-instruct-0905';
                } else if (model.includes('openai')) {
                  completionModelName = model.replace('openai/', '');
                } else if (model.includes('anthropic')) {
                  completionModelName = model.replace('anthropic/', '');
                } else if (model.includes('google')) {
                  completionModelName = model.replace('google/', '');
                } else {
                  completionModelName = model;
                }
                
                const completionResult = await streamText({
                  model: useChat ? completionClient.chat(completionModelName) : completionClient(completionModelName),
                  messages: [
                    { 
                      role: 'system', 
                      content: 'You are completing a truncated file. Provide the complete, working file content.'
                    },
                    { role: 'user', content: completionPrompt }
                  ],
                  temperature: model.startsWith('openai/gpt-5') ? undefined : appConfig.ai.defaultTemperature
                });
                
                // Get the full text from the stream
                let completedContent = '';
                for await (const chunk of completionResult.textStream) {
                  completedContent += chunk;
                }
                
                // Replace the truncated file in the generatedCode
                const filePattern = new RegExp(
                  `<file path="${filePath.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}">[\\s\\S]*?(?:</file>|$)`,
                  'g'
                );
                
                // Extract just the code content (remove any markdown or explanation)
                let cleanContent = completedContent;
                if (cleanContent.includes('```')) {
                  const codeMatch = cleanContent.match(/```[\w]*\n([\s\S]*?)```/);
                  if (codeMatch) {
                    cleanContent = codeMatch[1];
                  }
                }
                
                generatedCode = generatedCode.replace(
                  filePattern,
                  `<file path="${filePath}">\n${cleanContent}\n</file>`
                );
                
                console.log(`[generate-ai-code-stream] Successfully completed ${filePath}`);
                
              } catch (completionError) {
                console.error(`[generate-ai-code-stream] Failed to complete ${filePath}:`, completionError);
                await sendProgress({
                  type: 'warning',
                  message: `Could not auto-complete ${filePath}. Manual review may be needed.`
                });
              }
            }
            
            // Clear the warnings after attempting fixes
            truncationWarnings.length = 0;
            await sendProgress({
              type: 'info',
              message: 'Truncation recovery complete'
            });
          }
        }
        
        // Send completion with packages info
        await sendProgress({ 
          type: 'complete', 
          generatedCode,
          explanation,
          files: files.length,
          components: componentCount,
          model,
          packagesToInstall: packagesToInstall.length > 0 ? packagesToInstall : undefined,
          warnings: truncationWarnings.length > 0 ? truncationWarnings : undefined
        });
        
        // Track edit in conversation history
        if (isEdit && editContext && global.conversationState) {
          const editRecord: ConversationEdit = {
            timestamp: Date.now(),
            userRequest: prompt,
            editType: editContext.editIntent.type,
            targetFiles: editContext.primaryFiles,
            confidence: editContext.editIntent.confidence,
            outcome: 'success' // Assuming success if we got here
          };
          
          global.conversationState.context.edits.push(editRecord);
          
          // Track major changes
          if (editContext.editIntent.type === 'ADD_FEATURE' || files.length > 3) {
            global.conversationState.context.projectEvolution.majorChanges.push({
              timestamp: Date.now(),
              description: editContext.editIntent.description,
              filesAffected: editContext.primaryFiles
            });
          }
          
          // Update last updated timestamp
          global.conversationState.lastUpdated = Date.now();
          
          // ─── Cost Tracking: record this turn's usage ─────────
          const generationDuration = Date.now() - generationStartTime;
          const estimatedPromptTokens = estimateTotalTokens([{ role: 'system', content: systemPrompt }, { role: 'user', content: fullPrompt }]);
          const estimatedCompletionTokens = Math.round(generatedCode.length / 3.5);
          global.sableCostState = trackUsage(
            global.sableCostState!,
            model,
            { promptTokens: estimatedPromptTokens, completionTokens: estimatedCompletionTokens },
            toolCallCount,
            generationDuration,
          );
          console.log(`[generate-ai-code-stream] Cost: ${getCompactCostSummary(global.sableCostState!)}`);
          
          // ─── Session Memory: trigger extraction if threshold met ─
          const totalTokensNow = estimateTotalTokens(
            (global.conversationState?.context?.messages || []).map((m: any) => ({ role: m.role, content: m.content }))
          );
          if (shouldExtractMemory(global.sableSessionMemoryState!, totalTokensNow)) {
            // Fire and forget — non-blocking background extraction
            extractSessionMemory(
              (global.conversationState?.context?.messages || []).map((m: any) => ({ role: m.role, content: m.content })),
              global.sableSessionMemoryState!,
            ).then(newState => {
              global.sableSessionMemoryState = newState;
              console.log(`[generate-ai-code-stream] Session memory extracted (${newState.entries.length} entries)`);
            }).catch(e => {
              console.warn('[generate-ai-code-stream] Session memory extraction failed:', e);
            });
          }
          
          // Record tool calls for session memory threshold
          if (toolCallCount > 0 && global.sableSessionMemoryState) {
            for (let i = 0; i < toolCallCount; i++) {
              global.sableSessionMemoryState = recordMemoryToolCall(global.sableSessionMemoryState);
            }
          }
          
          // Persist session to disk for restart resilience
          saveSession();
          
          console.log('[generate-ai-code-stream] Updated conversation history with edit:', editRecord);
        }
        
      } catch (error) {
        console.error('[generate-ai-code-stream] Stream processing error:', error);
        
        // Check if it's a tool validation error
        if ((error as any).message?.includes('tool call validation failed')) {
          console.error('[generate-ai-code-stream] Tool call validation error - this may be due to the AI model sending incorrect parameters');
          await sendProgress({ 
            type: 'warning', 
            message: 'Package installation tool encountered an issue. Packages will be detected from imports instead.'
          });
          // Continue processing - packages can still be detected from the code
        } else {
          await sendProgress({ 
            type: 'error', 
            error: (error as Error).message 
          });
        }
      } finally {
        try { await writer.close(); } catch { /* stream already closed */ }
      }
    })();
    
    // Return the stream with proper headers for streaming support
    return new Response(stream.readable, {
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Transfer-Encoding': 'chunked',
        'Content-Encoding': 'none', // Prevent compression that can break streaming
        'X-Accel-Buffering': 'no', // Disable nginx buffering
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type, Authorization',
      },
    });
    
  } catch (error) {
    console.error('[generate-ai-code-stream] Error:', error);
    return NextResponse.json({ 
      success: false, 
      error: (error as Error).message 
    }, { status: 500 });
  }
}