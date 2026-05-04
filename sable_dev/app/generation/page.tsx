'use client';

import { useState, useEffect, useRef, useCallback, Suspense } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';
import Image from 'next/image';
import { appConfig } from '@/config/app.config';
import { getTemplate, DEFAULT_TEMPLATE, type TemplateId } from '@/lib/sandbox/templates';
import HeroInput from '@/components/HeroInput';
import SidebarInput from '@/components/app/generation/SidebarInput';
import HeaderBrandKit from '@/components/shared/header/BrandKit/BrandKit';
import { HeaderProvider } from '@/components/shared/header/HeaderContext';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
// Import icons from centralized module to avoid Turbopack chunk issues
import { 
  FiFile, 
  FiChevronRight, 
  FiChevronDown,
  FiGithub,
  BsFolderFill, 
  BsFolder2Open,
  SiJavascript, 
  SiReact, 
  SiCss3, 
  SiJson 
} from '@/lib/icons';
import { motion } from 'framer-motion';
import CodeApplicationProgress, { type CodeApplicationState } from '@/components/CodeApplicationProgress';
// New feature components
import CodeEditor from '@/components/CodeEditor';
import DevicePreviewToolbar, { type DeviceMode, getDeviceWidth } from '@/components/DevicePreviewToolbar';
import ConsolePanel from '@/components/ConsolePanel';
import VersionTimeline from '@/components/VersionTimeline';
import ImageUpload, { type ImageData } from '@/components/ImageUpload';
import ImportProjectModal from '@/components/ImportProjectModal';
import { useSnapshots } from '@/hooks/useSnapshots';

interface SandboxData {
  sandboxId: string;
  url: string;
  [key: string]: any;
}

interface ChatMessage {
  content: string;
  type: 'user' | 'ai' | 'system' | 'file-update' | 'command' | 'error';
  timestamp: Date;
  metadata?: {
    scrapedUrl?: string;
    scrapedContent?: any;
    generatedCode?: string;
    appliedFiles?: string[];
    commandType?: 'input' | 'output' | 'error' | 'success';
    brandingData?: any;
    sourceUrl?: string;
  };
}

interface ScrapeData {
  success: boolean;
  content?: string;
  url?: string;
  title?: string;
  source?: string;
  screenshot?: string;
  structured?: any;
  metadata?: any;
  message?: string;
  error?: string;
}

function AISandboxPage() {
  // Resolve sandbox URLs for remote access,  replace localhost with the browser's hostname
  const resolveSandboxUrl = (url: string): string => {
    if (typeof window === 'undefined') return url;
    const browserHost = window.location.hostname;
    if (browserHost && browserHost !== 'localhost' && browserHost !== '127.0.0.1') {
      return url.replace(/localhost|127\.0\.0\.1/, browserHost);
    }
    return url;
  };

  const [sandboxData, setSandboxData] = useState<SandboxData | null>(null);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState({ text: 'Not connected', active: false });
  const [responseArea, setResponseArea] = useState<string[]>([]);
  const [structureContent, setStructureContent] = useState('No sandbox created yet');
  const [promptInput, setPromptInput] = useState('');
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([
    {
      content: 'Welcome! I can help you generate code with full context of your sandbox files and structure. Just start chatting - I\'ll automatically create a sandbox for you if needed!\n\nTip: If you see package errors like "react-router-dom not found", just type "npm install" or "check packages" to automatically install missing packages.',
      type: 'system',
      timestamp: new Date()
    }
  ]);
  const [aiChatInput, setAiChatInput] = useState('');
  const [aiEnabled] = useState(true);
  const searchParams = useSearchParams();
  const router = useRouter();
  const [aiModel, setAiModel] = useState(() => {
    const modelParam = searchParams.get('model');
    // Accept static models, or any model with a known provider prefix
    if (modelParam && (
      appConfig.ai.availableModels.includes(modelParam) ||
      modelParam.startsWith('ollama/') ||
      modelParam.startsWith('openwebui/')
    )) {
      return modelParam;
    }
    return appConfig.ai.defaultModel;
  });
  const [dynamicModels, setDynamicModels] = useState<Array<{ id: string; name: string; provider: string }>>([]);
  const [urlOverlayVisible, setUrlOverlayVisible] = useState(false);
  const [urlInput, setUrlInput] = useState('');
  const [urlStatus, setUrlStatus] = useState<string[]>([]);
  const [showHomeScreen, setShowHomeScreen] = useState(true);
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(new Set(['root', 'app', 'src', 'src/components']));
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [homeScreenFading, setHomeScreenFading] = useState(false);
  const [homeUrlInput, setHomeUrlInput] = useState('');
  const [homeContextInput, setHomeContextInput] = useState('');
  const [activeTab, setActiveTab] = useState<'generation' | 'preview'>('preview');
  const [showStyleSelector, setShowStyleSelector] = useState(false);
  const [selectedStyle, setSelectedStyle] = useState<string | null>(null);
  const [selectedTemplate, setSelectedTemplate] = useState<TemplateId>(DEFAULT_TEMPLATE);
  const [showLoadingBackground, setShowLoadingBackground] = useState(false);
  const [urlScreenshot, setUrlScreenshot] = useState<string | null>(null);
  const [isScreenshotLoaded, setIsScreenshotLoaded] = useState(false);
  const [isCapturingScreenshot, setIsCapturingScreenshot] = useState(false);
  const [screenshotError, setScreenshotError] = useState<string | null>(null);
  const [isPreparingDesign, setIsPreparingDesign] = useState(false);
  const [targetUrl, setTargetUrl] = useState<string>('');
  const [sidebarScrolled, setSidebarScrolled] = useState(false);
  const [screenshotCollapsed, setScreenshotCollapsed] = useState(false);
  const [loadingStage, setLoadingStage] = useState<'gathering' | 'planning' | 'generating' | null>(null);
  const [isStartingNewGeneration, setIsStartingNewGeneration] = useState(false);
  const [sandboxFiles, setSandboxFiles] = useState<Record<string, string>>({});
  const [hasInitialSubmission, setHasInitialSubmission] = useState<boolean>(false);
  const [fileStructure, setFileStructure] = useState<string>('');
  
  const [conversationContext, setConversationContext] = useState<{
    scrapedWebsites: Array<{ url: string; content: any; timestamp: Date }>;
    generatedComponents: Array<{ name: string; path: string; content: string }>;
    appliedCode: Array<{ files: string[]; timestamp: Date }>;
    currentProject: string;
    lastGeneratedCode?: string;
  }>({
    scrapedWebsites: [],
    generatedComponents: [],
    appliedCode: [],
    currentProject: '',
    lastGeneratedCode: undefined
  });
  
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const chatMessagesRef = useRef<HTMLDivElement>(null);
  const codeDisplayRef = useRef<HTMLDivElement>(null);
  
  // New feature state
  const [deviceMode, setDeviceMode] = useState<DeviceMode>('desktop');
  const [consoleOpen, setConsoleOpen] = useState(false);
  const [showVersionHistory, setShowVersionHistory] = useState(false);
  const [showImportModal, setShowImportModal] = useState(false);
  const [attachedImages, setAttachedImages] = useState<ImageData[]>([]);
  const [splitView, setSplitView] = useState(false);
  
  // Snapshot/undo-redo system
  const {
    snapshots,
    currentIndex: snapshotIndex,
    takeSnapshot,
    undo,
    redo,
    restoreTo,
    canUndo,
    canRedo,
  } = useSnapshots({ maxHistory: 50 });
  
  const [codeApplicationState, setCodeApplicationState] = useState<CodeApplicationState>({
    stage: null
  });
  
  const [generationProgress, setGenerationProgress] = useState<{
    isGenerating: boolean;
    status: string;
    components: Array<{ name: string; path: string; completed: boolean }>;
    currentComponent: number;
    streamedCode: string;
    isStreaming: boolean;
    isThinking: boolean;
    thinkingText?: string;
    thinkingDuration?: number;
    currentFile?: { path: string; content: string; type: string };
    files: Array<{ path: string; content: string; type: string; completed: boolean; edited?: boolean }>;
    lastProcessedPosition: number;
    isEdit?: boolean;
    toolCalls?: Array<{ name: string; args: any; index: number }>;
    toolResults?: Array<{ name: string; output: string; index: number }>;
    reasoningSteps?: Array<{ thought: string; next_step?: string; index: number }>;
  }>({
    isGenerating: false,
    status: '',
    components: [],
    currentComponent: 0,
    streamedCode: '',
    isStreaming: false,
    isThinking: false,
    files: [],
    lastProcessedPosition: 0
  });

  // Store flag to trigger generation after component mounts
  const [shouldAutoGenerate, setShouldAutoGenerate] = useState(false);
  const [pendingCreationPrompt, setPendingCreationPrompt] = useState<string | null>(null);
  
  // Ref-based sendChatMessage so initializePage can call it after sandbox creation
  const sendChatMessageRef = useRef<((msg?: string) => Promise<void>) | null>(null);

  // Agent system state
  const [activeAgents, setActiveAgents] = useState<Array<{
    id: string;
    role: string;
    model: string;
    status: 'idle' | 'spawning' | 'working' | 'completed' | 'error';
    task: string;
    result?: string;
    error?: string;
    name: string;
    icon: string;
    durationMs?: number;
    tokensUsed?: number;
  }>>([]);
  const [agentPanelOpen, setAgentPanelOpen] = useState(false);
  const agentPollRef = useRef<NodeJS.Timeout | null>(null);
  const autoFixRetryRef = useRef<number>(0); // max retries for auto-error-fix loop
  
  // File diffs from code application
  const [fileDiffs, setFileDiffs] = useState<Array<{
    path: string;
    type: 'added' | 'modified' | 'deleted';
    hunks: Array<{
      oldStart: number; oldLines: number; newStart: number; newLines: number;
      lines: Array<{ type: 'add' | 'remove' | 'context'; content: string; oldLineNumber?: number; newLineNumber?: number }>;
    }>;
    additions: number;
    deletions: number;
  }>>([]);
  const [showDiffPanel, setShowDiffPanel] = useState(false);

  // Fetch dynamic models from Ollama & OpenWebUI on mount
  useEffect(() => {
    let cancelled = false;
    const fetchModels = async () => {
      try {
        const res = await fetch('/api/available-models');
        if (!res.ok) return;
        const data = await res.json();
        if (!cancelled && data.models) {
          setDynamicModels(data.models);
          // If current model is default and not in available but we have dynamic models, keep it
          // Also allow selecting a model from URL that is dynamic (ollama/openwebui)
          const modelParam = searchParams.get('model');
          if (modelParam && data.models.some((m: { id: string }) => m.id === modelParam)) {
            setAiModel(modelParam);
          }
        }
      } catch (err) {
        console.log('[generation] Failed to fetch available models:', err);
      }
    };
    fetchModels();
    return () => { cancelled = true; };
  }, [searchParams]);

  // Clear old conversation data on component mount and create/restore sandbox
  // Keyboard shortcut: Ctrl+Z undo, Ctrl+Shift+Z / Ctrl+Y redo (project-level)
  // NOTE: moved to after handleUndo/handleRedo declarations below

  useEffect(() => {
    let isMounted = true;
    let sandboxCreated = false; // Track if sandbox was created in this effect

    const initializePage = async () => {
      // Prevent double execution in React StrictMode
      if (sandboxCreated) return;
      
      // Detect if this is a brand-new project (prompt from home page)
      const isNewProject = !!(searchParams.get('prompt') || sessionStorage.getItem('creationPrompt'));
      
      // If starting a new project, clear chat completely and persist the wipe
      if (isNewProject) {
        console.log('[initializePage] New project detected,  clearing chat history');
        setChatMessages([{
          content: 'Welcome! I can help you generate code with full context of your sandbox files and structure. Just start chatting!\n\nTip: If you see package errors, just type "npm install" to fix them.',
          type: 'system',
          timestamp: new Date()
        }]);
        try {
          await fetch('/api/persistence', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: 'clear' }),
          });
        } catch (_) { /* best effort */ }
      } else {
        // Only restore persisted chat for existing/resumed sessions
        try {
          const persistRes = await fetch('/api/persistence?type=chat-history');
          const persistData = await persistRes.json();
          if (persistData.hasData && persistData.messages?.length > 0 && isMounted) {
            const restored = persistData.messages.map((m: any) => ({
              ...m,
              timestamp: new Date(m.timestamp),
            }));
            setChatMessages(prev => {
              // Only restore if we still have the default welcome message
              if (prev.length === 1 && prev[0].type === 'system') {
                // Filter out any duplicate welcome messages from restored data
                const filtered = restored.filter((m: any) =>
                  !(m.type === 'system' && typeof m.content === 'string' && m.content.startsWith('Welcome! I can help'))
                );
                return [...prev, ...filtered];
              }
              return prev;
            });
            console.log('[initializePage] Restored', restored.length, 'persisted chat messages');
          }
        } catch (e) {
          console.log('[initializePage] No persisted chat history:', e);
        }
      }
      
      // Also try to restore server-side session
      try {
        await fetch('/api/persistence', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ action: 'restore' }),
        });
      } catch (e) {
        console.log('[initializePage] No persisted session:', e);
      }
      
      // First check URL parameters (from home page navigation)
      const urlParam = searchParams.get('url');
      const templateParam = searchParams.get('template');
      const detailsParam = searchParams.get('details');
      
      // Read template selection
      const storedTemplate = templateParam || sessionStorage.getItem('selectedTemplate') || DEFAULT_TEMPLATE;
      setSelectedTemplate(storedTemplate as TemplateId);
      sessionStorage.removeItem('selectedTemplate');

      // Then check session storage as fallback
      const storedUrl = urlParam || sessionStorage.getItem('targetUrl');
      const storedStyle = sessionStorage.getItem('selectedStyle');
      const storedModel = sessionStorage.getItem('selectedModel');
      const storedInstructions = sessionStorage.getItem('additionalInstructions');
      
      if (storedUrl) {
        // Mark that we have an initial submission since we're loading with a URL
        setHasInitialSubmission(true);
        
        // Clear sessionStorage after reading  
        sessionStorage.removeItem('targetUrl');
        sessionStorage.removeItem('selectedStyle');
        sessionStorage.removeItem('selectedModel');
        sessionStorage.removeItem('additionalInstructions');
        // Note: Don't clear siteMarkdown here, it will be cleared when used
        
        // Set the values in the component state
        setHomeUrlInput(storedUrl);
        setSelectedStyle(storedStyle || 'modern');
        
        // Add details to context if provided
        if (detailsParam) {
          setHomeContextInput(detailsParam);
        } else if (storedStyle && !urlParam) {
          // Only apply stored style if no screenshot URL is provided
          // This prevents unwanted style inheritance when using screenshot search
          const styleNames: Record<string, string> = {
            '1': 'Glassmorphism',
            '2': 'Neumorphism',
            '3': 'Brutalism',
            '4': 'Minimalist',
            '5': 'Dark Mode',
            '6': 'Gradient Rich',
            '7': '3D Depth',
            '8': 'Retro Wave',
            'modern': 'Modern clean and minimalist',
            'playful': 'Fun colorful and playful',
            'professional': 'Corporate professional and sleek',
            'artistic': 'Creative artistic and unique'
          };
          const styleName = styleNames[storedStyle] || storedStyle;
          let contextString = `${styleName} style design`;
          
          // Add additional instructions if provided
          if (storedInstructions) {
            contextString += `. ${storedInstructions}`;
          }
          
          setHomeContextInput(contextString);
        } else if (storedInstructions && !urlParam) {
          // Apply only instructions if no style but instructions are provided
          // and no screenshot URL is provided
          setHomeContextInput(storedInstructions);
        }
        
        if (storedModel) {
          setAiModel(storedModel);
        }
        
        // Skip the home screen and go directly to builder
        setShowHomeScreen(false);
        setHomeScreenFading(false);
        
        // Set flag to auto-trigger generation after component updates
        setShouldAutoGenerate(true);
        
        // Also set autoStart flag for the effect
        sessionStorage.setItem('autoStart', 'true');
      }

      // Handle creation-from-prompt mode (no URL, just a text prompt)
      const promptParam = searchParams.get('prompt');
      const creationPrompt = promptParam || sessionStorage.getItem('creationPrompt');
      let isCreationMode = false;
      if (creationPrompt && !storedUrl) {
        isCreationMode = true;
        setHasInitialSubmission(true);
        sessionStorage.removeItem('creationPrompt');
        sessionStorage.removeItem('selectedModel');
        sessionStorage.removeItem('selectedTemplate');
        
        if (storedModel) {
          setAiModel(storedModel);
        }
        
        // Set the prompt as the chat input so sendChatMessage can use it
        setAiChatInput(creationPrompt);
        
        // Skip home screen
        setShowHomeScreen(false);
        setHomeScreenFading(false);
        
        // We'll trigger sendChatMessage after sandbox is ready
        setShouldAutoGenerate(true);
        sessionStorage.setItem('autoStart', 'true');
        sessionStorage.setItem('creationPromptPending', creationPrompt);
      }
      
      // Clear old conversation
      try {
        await fetch('/api/conversation-state', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ action: 'clear-old' })
        });
        console.log('[home] Cleared old conversation data on mount');
      } catch (error) {
        console.error('[ai-sandbox] Failed to clear old conversation:', error);
        if (isMounted) {
          addChatMessage('Failed to clear old conversation data.', 'error');
        }
      }
      
      if (!isMounted) return;

      // Check if sandbox ID is in URL
      const sandboxIdParam = searchParams.get('sandbox');
      let restoreSandboxId = sessionStorage.getItem('restoreSandboxId');
      sessionStorage.removeItem('restoreSandboxId');
      
      // Auto-restore: if no sandbox ID from URL or sessionStorage,
      // check server-side persisted session for last active sandbox
      if (!sandboxIdParam && !restoreSandboxId && !isNewProject) {
        try {
          const sessionRes = await fetch('/api/persistence?type=session');
          const sessionData = await sessionRes.json();
          if (sessionData.hasSession && sessionData.session?.sandboxData?.sandboxId) {
            restoreSandboxId = sessionData.session.sandboxData.sandboxId;
            console.log('[home] Auto-restoring previous session sandbox:', restoreSandboxId);
          }
        } catch (e) {
          console.log('[home] Could not check persisted session:', e);
        }
      }
      
      setLoading(true);
      try {
        if (sandboxIdParam || restoreSandboxId) {
          const targetSandbox = restoreSandboxId || sandboxIdParam || '';
          console.log('[home] Attempting to restore sandbox:', targetSandbox);
          sandboxCreated = true;
          
          // Try to restore from persistent project store first
          try {
            const restoreRes = await fetch('/api/restore-project', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ sandboxId: targetSandbox }),
            });
            const restoreData = await restoreRes.json();
            
            if (restoreData.success && isMounted) {
              console.log(`[home] Restored project with ${restoreData.restoredFiles} files`);
              setSandboxData({
                sandboxId: restoreData.sandboxId,
                url: restoreData.url,
                provider: 'local',
              });
              setStructureContent(`Project restored: ${restoreData.restoredFiles} files`);
              addChatMessage(`Project restored with ${restoreData.restoredFiles} files from your previous session.`, 'system');
            } else {
              // Restore failed,  fall back to creating a new sandbox
              console.log('[home] Project restore not available, creating fresh sandbox');
              await createSandbox(true, storedTemplate);
            }
          } catch (e) {
            console.log('[home] Project restore error, creating fresh sandbox:', e);
            await createSandbox(true, storedTemplate);
          }
        } else {
          console.log('[home] No sandbox in URL, creating new sandbox automatically...');
          sandboxCreated = true;
          await createSandbox(true, storedTemplate);
        }
        
        // DIRECT TRIGGER: If creation mode, send the prompt directly after sandbox is ready
        // This bypasses the useEffect chain which can fail due to stale closures
        if (isCreationMode && creationPrompt && isMounted) {
          console.log('[initializePage] Sandbox ready, directly sending creation prompt:', creationPrompt);
          // Small delay to let React process the sandboxData state update
          setTimeout(() => {
            if (sendChatMessageRef.current) {
              console.log('[initializePage] Calling sendChatMessage via ref');
              sendChatMessageRef.current(creationPrompt);
            } else {
              console.log('[initializePage] sendChatMessageRef not ready, setting pending prompt');
              setPendingCreationPrompt(creationPrompt);
            }
            // Clean up sessionStorage
            sessionStorage.removeItem('creationPromptPending');
            sessionStorage.removeItem('autoStart');
          }, 800);
        }
        
        // If we have a URL from the home page, mark for automatic start
        if (storedUrl && isMounted) {
          sessionStorage.setItem('autoStart', 'true');
        }
      } catch (error) {
        console.error('[ai-sandbox] Failed to create or restore sandbox:', error);
        if (isMounted) {
          addChatMessage('Failed to create or restore sandbox.', 'error');
        }
      } finally {
        if (isMounted) {
          setLoading(false);
        }
      }
    };
    
    initializePage();

    return () => {
      isMounted = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Run only on mount
  
  useEffect(() => {
    // Handle Escape key for home screen
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && showHomeScreen) {
        setHomeScreenFading(true);
        setTimeout(() => {
          setShowHomeScreen(false);
          setHomeScreenFading(false);
        }, 500);
      }
    };
    
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [showHomeScreen]);
  
  // Start capturing screenshot if URL is provided on mount (from home screen)
  useEffect(() => {
    if (!showHomeScreen && homeUrlInput && !urlScreenshot && !isCapturingScreenshot) {
      let screenshotUrl = homeUrlInput.trim();
      if (!screenshotUrl.match(/^https?:\/\//i)) {
        screenshotUrl = 'https://' + screenshotUrl;
      }
      captureUrlScreenshot(screenshotUrl);
    }
  }, [showHomeScreen, homeUrlInput]); // eslint-disable-line react-hooks/exhaustive-deps

  // Auto-start generation if flagged
  useEffect(() => {
    const autoStart = sessionStorage.getItem('autoStart');
    if (autoStart === 'true' && !showHomeScreen && homeUrlInput) {
      sessionStorage.removeItem('autoStart');
      // Small delay to ensure everything is ready
      setTimeout(() => {
        console.log('[generation] Auto-starting generation for URL:', homeUrlInput);
        startGeneration();
      }, 1000);
    }
  }, [showHomeScreen, homeUrlInput]); // eslint-disable-line react-hooks/exhaustive-deps


  useEffect(() => {
    // Only check sandbox status on mount if we don't already have sandboxData
    // AND we're not auto-starting a new generation (which would create a new sandbox)
    const autoStart = sessionStorage.getItem('autoStart');
    if (!sandboxData && autoStart !== 'true') {
      checkSandboxStatus();
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (chatMessagesRef.current) {
      chatMessagesRef.current.scrollTop = chatMessagesRef.current.scrollHeight;
    }
  }, [chatMessages]);
  
  // Persist chat history to disk (debounced)
  const persistTimerRef = useRef<NodeJS.Timeout | null>(null);
  useEffect(() => {
    // Only persist if we have user/AI messages (not just the welcome message)
    if (chatMessages.length <= 1) return;
    
    if (persistTimerRef.current) clearTimeout(persistTimerRef.current);
    persistTimerRef.current = setTimeout(() => {
      const serialized = chatMessages.map(m => ({
        content: m.content,
        type: m.type,
        timestamp: m.timestamp.toISOString(),
        metadata: m.metadata,
      }));
      fetch('/api/persistence', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'save-chat', messages: serialized }),
      }).catch(() => {}); // Silent fail
    }, 3000); // Save every 3s at most
    
    return () => {
      if (persistTimerRef.current) clearTimeout(persistTimerRef.current);
    };
  }, [chatMessages]);

  // Auto-trigger generation when flag is set (from home page navigation)
  useEffect(() => {
    if (shouldAutoGenerate && !showHomeScreen) {
      // Check if this is a creation-from-prompt (no URL)
      const pendingPrompt = sessionStorage.getItem('creationPromptPending');
      
      if (pendingPrompt && !homeUrlInput) {
        // Creation mode,  the direct trigger in initializePage handles this now
        // This effect is just a safety fallback
        setShouldAutoGenerate(false);
        // Only set pendingCreationPrompt if the direct trigger hasn't already handled it
        if (!sendChatMessageRef.current) {
          sessionStorage.removeItem('creationPromptPending');
          setAiChatInput(pendingPrompt);
          setPendingCreationPrompt(pendingPrompt);
        }
        return;
      } else if (homeUrlInput) {
        // URL clone mode,  original behavior
        setShouldAutoGenerate(false);
        
        const timer = setTimeout(() => {
          console.log('[generation] Auto-triggering generation from URL params');
          startGeneration();
        }, 1000);
        
        return () => clearTimeout(timer);
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [shouldAutoGenerate, homeUrlInput, showHomeScreen, sandboxData]);

  const updateStatus = (text: string, active: boolean) => {
    setStatus({ text, active });
  };

  const log = (message: string, type: 'info' | 'error' | 'command' = 'info') => {
    setResponseArea(prev => [...prev, `[${type}] ${message}`]);
  };

  const addChatMessage = (content: string | unknown, type: ChatMessage['type'], metadata?: ChatMessage['metadata']) => {
    // Defensive: ensure content is always a string
    const safeContent = typeof content === 'string' ? content : String(content ?? '');
    if (!safeContent || safeContent === '[object Object]') {
      console.warn('[addChatMessage] Skipping invalid content:', content);
      return;
    }
    setChatMessages(prev => {
      // Skip duplicate consecutive system messages
      if (type === 'system' && prev.length > 0) {
        const lastMessage = prev[prev.length - 1];
        if (lastMessage.type === 'system' && lastMessage.content === safeContent) {
          return prev; // Skip duplicate
        }
      }
      return [...prev, { content: safeContent, type, timestamp: new Date(), metadata }];
    });
  };

  // ---- Project History ----
  const [projectHistory, setProjectHistory] = useState<Array<{
    id: string; name: string; prompt: string; template: string;
    model: string; createdAt: number; updatedAt: number; fileCount: number;
  }>>([]);
  const [showProjectHistory, setShowProjectHistory] = useState(false);

  const fetchProjectHistory = async () => {
    try {
      const res = await fetch('/api/project-history');
      if (res.ok) {
        const data = await res.json();
        if (data.success) setProjectHistory(data.projects || []);
      }
    } catch { /* silent */ }
  };

  const saveCurrentProjectToHistory = async () => {
    if (!sandboxData) return;
    const firstPrompt = chatMessages.find(m => m.type === 'user')?.content || 'Untitled Project';
    const projectEntry = {
      id: sandboxData.sandboxId,
      name: firstPrompt.slice(0, 60),
      prompt: firstPrompt,
      template: selectedTemplate,
      model: aiModel,
      createdAt: Date.now(),
      updatedAt: Date.now(),
      fileCount: generationProgress.files.length + Object.keys(sandboxFiles).length,
      sandboxId: sandboxData.sandboxId,
    };
    try {
      await fetch('/api/project-history', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'add', project: projectEntry }),
      });
    } catch { /* silent */ }
  };

  // ---- Reset / New Project ----
  const resetProject = async () => {
    // Save current project to history first if there's meaningful content
    if (sandboxData && chatMessages.some(m => m.type === 'user')) {
      await saveCurrentProjectToHistory();
    }

    try {
      await fetch('/api/reset-project', { method: 'POST' });
    } catch (e) {
      console.error('[resetProject] Failed to reset server:', e);
    }

    // Clear persisted chat history on disk so it won't be reloaded
    try {
      await fetch('/api/persistence', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'clear' }),
      });
    } catch (e) {
      console.error('[resetProject] Failed to clear persisted chat:', e);
    }

    // Clear server-side conversation state
    try {
      await fetch('/api/conversation-state', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'clear-old' }),
      });
    } catch (_) { /* best effort */ }

    // Reset all client state
    setSandboxData(null);
    setChatMessages([{
      content: 'Project reset. Describe what you want to build!',
      type: 'system',
      timestamp: new Date()
    }]);
    setAiChatInput('');
    setConversationContext({
      scrapedWebsites: [],
      generatedComponents: [],
      appliedCode: [],
      currentProject: '',
      lastGeneratedCode: undefined
    });
    setGenerationProgress({
      isGenerating: false,
      status: '',
      components: [],
      currentComponent: 0,
      streamedCode: '',
      isStreaming: false,
      isThinking: false,
      files: [],
      lastProcessedPosition: 0
    });
    setSandboxFiles({});
    setFileStructure('');
    setSelectedFile(null);
    setActiveAgents([]);
    setAgentPanelOpen(false);
    setActiveTab('preview');
    setCodeApplicationState({ stage: null });
    setUrlScreenshot(null);
    setIsScreenshotLoaded(false);
    setTargetUrl('');
    setHomeUrlInput('');
    setHomeContextInput('');
    setHasInitialSubmission(false);
    setLoading(false);
    setStructureContent('No sandbox created yet');

    // Create a fresh sandbox
    try {
      await createSandbox(true);
    } catch { /* will be created on first prompt */ }
  };

  // Fetch history on mount
  useEffect(() => { fetchProjectHistory(); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Close project history dropdown on outside click
  useEffect(() => {
    if (!showProjectHistory) return;
    const handleClick = (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      if (!target.closest('[data-project-history]')) setShowProjectHistory(false);
    };
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, [showProjectHistory]);
  
  const checkAndInstallPackages = async () => {
    // This function is only called when user explicitly requests it
    // Don't show error if no sandbox - it's likely being created
    if (!sandboxData) {
      console.log('[checkAndInstallPackages] No sandbox data available yet');
      return;
    }
    
    // Vite error checking removed - handled by template setup
    addChatMessage('Checking packages... Sandbox is ready with Vite configuration.', 'system');
  };
  
  const handleSurfaceError = (_errors: any[]) => {
    // Function kept for compatibility but Vite errors are now handled by template
    
    // Focus the input
    const textarea = document.querySelector('textarea') as HTMLTextAreaElement;
    if (textarea) {
      textarea.focus();
    }
  };
  
  const installPackages = async (packages: string[]) => {
    if (!sandboxData) {
      addChatMessage('No active sandbox. Create a sandbox first!', 'system');
      return;
    }
    
    try {
      const response = await fetch('/api/install-packages', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ packages })
      });
      
      if (!response.ok) {
        throw new Error(`Failed to install packages: ${response.statusText}`);
      }
      
      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      
      while (reader) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              
              switch (data.type) {
                case 'command':
                  // Don't show npm install commands - they're handled by info messages
                  if (!data.command.includes('npm install')) {
                    addChatMessage(data.command, 'command', { commandType: 'input' });
                  }
                  break;
                case 'output':
                  addChatMessage(data.message, 'command', { commandType: 'output' });
                  break;
                case 'error':
                  if (data.message && data.message !== 'undefined') {
                    addChatMessage(data.message, 'command', { commandType: 'error' });
                  }
                  break;
                case 'warning':
                  addChatMessage(data.message, 'command', { commandType: 'output' });
                  break;
                case 'success':
                  addChatMessage(`${data.message}`, 'system');
                  break;
                case 'status':
                  addChatMessage(data.message, 'system');
                  break;
              }
            } catch (e) {
              console.error('Failed to parse SSE data:', e);
            }
          }
        }
      }
    } catch (error: any) {
      addChatMessage(`Failed to install packages: ${error.message}`, 'system');
    }
  };

  const checkSandboxStatus = async () => {
    try {
      const response = await fetch('/api/sandbox-status');
      const data = await response.json();
      
      if (data.active && data.healthy && data.sandboxData) {
        console.log('[checkSandboxStatus] Setting sandboxData from API:', data.sandboxData);
        setSandboxData(data.sandboxData);
        updateStatus('Sandbox active', true);
      } else if (data.active && !data.healthy) {
        // Sandbox exists but not responding
        updateStatus('Sandbox not responding', false);
        // Keep existing sandboxData if we have it - don't clear it
      } else {
        // Only clear sandboxData if we don't already have it or if we're explicitly checking from a fresh state
        // This prevents clearing sandboxData during normal operation when it should persist
        if (!sandboxData) {
          console.log('[checkSandboxStatus] No existing sandboxData, clearing state');
          setSandboxData(null);
          updateStatus('No sandbox', false);
        } else {
          // Keep existing sandboxData and just update status
          console.log('[checkSandboxStatus] Keeping existing sandboxData, sandbox inactive but data preserved');
          updateStatus('Sandbox status unknown', false);
        }
      }
    } catch (error) {
      console.error('Failed to check sandbox status:', error);
      // Only clear on error if we don't have existing sandboxData
      if (!sandboxData) {
        setSandboxData(null);
        updateStatus('Error', false);
      } else {
        updateStatus('Status check failed', false);
      }
    }
  };

  const sandboxCreationRef = useRef<boolean>(false);
  
  const createSandbox = async (fromHomeScreen = false, templateOverride?: string) => {
    // Prevent duplicate sandbox creation
    if (sandboxCreationRef.current) {
      console.log('[createSandbox] Sandbox creation already in progress, skipping...');
      return null;
    }
    
    sandboxCreationRef.current = true;
    const templateToUse = templateOverride || selectedTemplate;
    console.log('[createSandbox] Starting sandbox creation with template:', templateToUse);
    setLoading(true);
    setShowLoadingBackground(true);
    updateStatus('Creating sandbox...', false);
    setResponseArea([]);
    setScreenshotError(null);
    
    try {
      const response = await fetch('/api/create-ai-sandbox-v2', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ template: templateToUse })
      });
      
      const data = await response.json();
      console.log('[createSandbox] Response data:', data);
      
      if (data.success) {
        sandboxCreationRef.current = false; // Reset the ref on success
        console.log('[createSandbox] Setting sandboxData from creation:', data);
        // Resolve sandbox URL for remote access
        const resolvedData = { ...data, url: resolveSandboxUrl(data.url) };
        setSandboxData(resolvedData);
        updateStatus('Sandbox active', true);
        log('Sandbox created successfully!');
        log(`Sandbox ID: ${resolvedData.sandboxId}`);
        log(`URL: ${resolvedData.url}`);
        
        // Update URL with sandbox ID
        const newParams = new URLSearchParams(searchParams.toString());
        newParams.set('sandbox', data.sandboxId);
        newParams.set('model', aiModel);
        router.push(`/generation?${newParams.toString()}`, { scroll: false });
        
        // Fade out loading background after sandbox loads
        setTimeout(() => {
          setShowLoadingBackground(false);
        }, 3000);
        
        if (data.structure) {
          displayStructure(data.structure);
        }
        
        // Fetch sandbox files after creation
        setTimeout(fetchSandboxFiles, 1000);
        
        // For Vercel sandboxes, Vite is already started during setupViteApp
        // No need to restart it immediately after creation
        // Only restart if there's an actual issue later
        console.log('[createSandbox] Sandbox ready with Vite server running');
        
        // Only add welcome message if not coming from home screen
        if (!fromHomeScreen) {
          addChatMessage(`Sandbox created! ID: ${data.sandboxId}. I now have context of your sandbox and can help you build your app. Just ask me to create components and I'll automatically apply them!

Tip: I automatically detect and install npm packages from your code imports (like react-router-dom, axios, etc.)`, 'system');
        }
        
        setTimeout(() => {
          if (iframeRef.current) {
            iframeRef.current.src = resolvedData.url;
          }
        }, 100);
        
        // Return the sandbox data so it can be used immediately
        return resolvedData;
      } else {
        throw new Error(data.error || 'Unknown error');
      }
    } catch (error: any) {
      console.error('[createSandbox] Error:', error);
      updateStatus('Error', false);
      log(`Failed to create sandbox: ${error.message}`, 'error');
      addChatMessage(`Failed to create sandbox: ${error.message}`, 'system');
      throw error;
    } finally {
      setLoading(false);
      sandboxCreationRef.current = false; // Reset the ref
    }
  };

  const displayStructure = (structure: any) => {
    if (typeof structure === 'object') {
      setStructureContent(JSON.stringify(structure, null, 2));
    } else {
      setStructureContent(structure || 'No structure available');
    }
  };

  const applyGeneratedCode = async (code: string, isEdit: boolean = false, overrideSandboxData?: SandboxData) => {
    setLoading(true);
    log('Applying AI-generated code...');
    
    try {
      // Take snapshot before applying (for undo)
      if (Object.keys(sandboxFiles).length > 0) {
        takeSnapshot(sandboxFiles, isEdit ? 'Before edit' : 'Before generation', aiChatInput?.slice(0, 80) || undefined);
      }
      
      // Show progress component instead of individual messages
      setCodeApplicationState({ stage: 'analyzing' });
      
      // Get pending packages from tool calls
      const pendingPackages = ((window as any).pendingPackages || []).filter((pkg: any) => pkg && typeof pkg === 'string');
      if (pendingPackages.length > 0) {
        console.log('[applyGeneratedCode] Sending packages from tool calls:', pendingPackages);
        // Clear pending packages after use
        (window as any).pendingPackages = [];
      }
      
      // Use streaming endpoint for real-time feedback
      const effectiveSandboxData = overrideSandboxData || sandboxData;
      const response = await fetch('/api/apply-ai-code-stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          response: code,
          isEdit: isEdit,
          packages: pendingPackages,
          sandboxId: effectiveSandboxData?.sandboxId // Pass the sandbox ID to ensure proper connection
        })
      });
      
      if (!response.ok) {
        throw new Error(`Failed to apply code: ${response.statusText}`);
      }
      
      // Handle streaming response
      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let finalData: any = null;
      
      while (reader) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              
              switch (data.type) {
                case 'start':
                  // Don't add as chat message, just update state
                  setCodeApplicationState({ stage: 'analyzing' });
                  break;
                  
                case 'step':
                  // Update progress state based on step
                  if (data.message.includes('Installing') && data.packages) {
                    setCodeApplicationState({ 
                      stage: 'installing', 
                      packages: data.packages 
                    });
                  } else if (data.message.includes('Creating files') || data.message.includes('Applying')) {
                    setCodeApplicationState({ 
                      stage: 'applying',
                      filesGenerated: [] // Files will be populated when complete
                    });
                  }
                  break;
                  
                case 'package-progress':
                  // Handle package installation progress
                  if (data.installedPackages) {
                    setCodeApplicationState(prev => ({ 
                      ...prev,
                      installedPackages: data.installedPackages 
                    }));
                  }
                  break;
                  
                case 'command':
                  // Don't show npm install commands - they're handled by info messages
                  if (data.command && !data.command.includes('npm install')) {
                    addChatMessage(data.command, 'command', { commandType: 'input' });
                  }
                  break;
                  
                case 'success':
                  if (data.installedPackages) {
                    setCodeApplicationState(prev => ({ 
                      ...prev,
                      installedPackages: data.installedPackages 
                    }));
                  }
                  break;
                  
                case 'file-progress':
                  // Skip file progress messages, they're noisy
                  break;
                  
                case 'file-complete':
                  // Store diff information for updated files
                  if (data.action === 'updated' && data.diff) {
                    setFileDiffs(prev => [...prev, data.diff]);
                  }
                  break;
                  
                case 'command-progress':
                  addChatMessage(`${data.action} command: ${data.command}`, 'command', { commandType: 'input' });
                  break;
                  
                case 'command-output':
                  addChatMessage(data.output, 'command', { 
                    commandType: data.stream === 'stderr' ? 'error' : 'output' 
                  });
                  break;
                  
                case 'command-complete':
                  if (data.success) {
                    addChatMessage(`Command completed successfully`, 'system');
                  } else {
                    addChatMessage(`Command failed with exit code ${data.exitCode}`, 'system');
                  }
                  break;
                  
                case 'complete':
                  finalData = data;
                  setCodeApplicationState({ stage: 'complete' });
                  // Clear the state after a delay
                  setTimeout(() => {
                    setCodeApplicationState({ stage: null });
                  }, 3000);
                  // Reset loading state when complete
                  setLoading(false);
                  break;
                  
                case 'error':
                  addChatMessage(`Error: ${data.message || data.error || 'Unknown error'}`, 'system');
                  // Reset loading state on error
                  setLoading(false);
                  break;
                  
                case 'warning':
                  addChatMessage(`${data.message}`, 'system');
                  break;
                  
                case 'info':
                  // Show info messages, especially for package installation
                  if (data.message) {
                    addChatMessage(data.message, 'system');
                  }
                  break;
              }
            } catch {
              // Ignore parse errors
            }
          }
        }
      }
      
      // Process final data
      if (finalData && finalData.type === 'complete') {
        const data: any = {
          success: true,
          results: finalData.results,
          explanation: finalData.explanation,
          structure: finalData.structure,
          message: finalData.message,
          autoCompleted: finalData.autoCompleted,
          autoCompletedComponents: finalData.autoCompletedComponents,
          warning: finalData.warning,
          missingImports: finalData.missingImports,
          debug: finalData.debug
        };
        
        if (data.success) {
          const { results } = data;
        
        // Log package installation results without duplicate messages
        if (results.packagesInstalled?.length > 0) {
          log(`Packages installed: ${results.packagesInstalled.join(', ')}`);
        }
        
        if (results.filesCreated?.length > 0) {
          log('Files created:');
          results.filesCreated.forEach((file: string) => {
            log(`  ${file}`, 'command');
          });
          
          // Verify files were actually created by refreshing the sandbox if needed
          if (sandboxData?.sandboxId && results.filesCreated.length > 0) {
            // Small delay to ensure files are written
            setTimeout(() => {
              // Force refresh the iframe to show new files
              if (iframeRef.current) {
                iframeRef.current.src = iframeRef.current.src;
              }
            }, 1000);
          }
        }
        
        if (results.filesUpdated?.length > 0) {
          log('Files updated:');
          results.filesUpdated.forEach((file: string) => {
            log(`  ${file}`, 'command');
          });
        }
        
        // Update conversation context with applied code
        setConversationContext(prev => ({
          ...prev,
          appliedCode: [...prev.appliedCode, {
            files: [...(results.filesCreated || []), ...(results.filesUpdated || [])],
            timestamp: new Date()
          }]
        }));
        
        if (results.commandsExecuted?.length > 0) {
          log('Commands executed:');
          results.commandsExecuted.forEach((cmd: string) => {
            log(`  $ ${cmd}`, 'command');
          });
        }
        
        if (results.errors?.length > 0) {
          results.errors.forEach((err: string) => {
            log(err, 'error');
          });
        }
        
        if (data.structure) {
          displayStructure(data.structure);
        }
        
        if (data.explanation) {
          log(data.explanation);
        }
        
        if (data.autoCompleted) {
          log('Auto-generating missing components...', 'command');
          
          if (data.autoCompletedComponents) {
            setTimeout(() => {
              log('Auto-generated missing components:', 'info');
              data.autoCompletedComponents.forEach((comp: string) => {
                log(`  ${comp}`, 'command');
              });
            }, 1000);
          }
        } else if (data.warning) {
          log(data.warning, 'error');
          
          if (data.missingImports && data.missingImports.length > 0) {
            const missingList = data.missingImports.join(', ');
            addChatMessage(
              `Ask me to "create the missing components: ${missingList}" to fix these import errors.`,
              'system'
            );
          }
        }
        
        log('Code applied successfully!');
        
        // Take snapshot after successful application (for redo/history)
        const allAppliedFiles = [...(results.filesCreated || []), ...(results.filesUpdated || [])];
        // Fetch updated files to create accurate snapshot
        fetchSandboxFiles().then(() => {
          // sandboxFiles will be updated by fetchSandboxFiles
          setTimeout(() => {
            takeSnapshot(sandboxFiles, `Applied ${allAppliedFiles.length} files`);
          }, 500);
        }).catch(() => {});
        console.log('[applyGeneratedCode] Response data:', data);
        console.log('[applyGeneratedCode] Debug info:', data.debug);
        console.log('[applyGeneratedCode] Current sandboxData:', sandboxData);
        console.log('[applyGeneratedCode] Current iframe element:', iframeRef.current);
        console.log('[applyGeneratedCode] Current iframe src:', iframeRef.current?.src);
        
        // Set applying code state for edits to show loading overlay
        // Removed overlay - changes apply directly
        
        if (results.filesCreated?.length > 0) {
          setConversationContext(prev => ({
            ...prev,
            appliedCode: [...prev.appliedCode, {
              files: results.filesCreated,
              timestamp: new Date()
            }]
          }));
          
          // Update the chat message to show success
          // Only show file list if not in edit mode
          if (isEdit) {
            const diffCount = fileDiffs.length;
            addChatMessage(
              diffCount > 0
                ? `Edit applied successfully! (${diffCount} file${diffCount > 1 ? 's' : ''} changed)`
                : `Edit applied successfully!`,
              'system'
            );
            if (diffCount > 0) {
              setShowDiffPanel(true);
            }
          } else {
            // Check if this is part of a generation flow (has recent AI recreation message)
            const recentMessages = chatMessages.slice(-5);
            const isPartOfGeneration = recentMessages.some(m => 
              m.content.includes('AI recreation generated') || 
              m.content.includes('Code generated')
            );
            
            // Don't show files if part of generation flow to avoid duplication
            if (isPartOfGeneration) {
              addChatMessage(`Applied ${results.filesCreated.length} files successfully!`, 'system');
            } else {
              addChatMessage(`Applied ${results.filesCreated.length} files successfully!`, 'system', {
                appliedFiles: results.filesCreated
              });
            }
          }
          
          // If there are failed packages, add a message about checking for errors
          if (results.packagesFailed?.length > 0) {
            addChatMessage(`⚠️ Some packages failed to install. Check the error banner above for details.`, 'system');
          }
          
          // Fetch updated file structure
          await fetchSandboxFiles();
          
          // Auto-save project to history after successful code application
          saveCurrentProjectToHistory();
          
          // Skip automatic package check - it's not needed here and can cause false "no sandbox" messages
          // Packages are already installed during the apply-ai-code-stream process
          
          // Test build to ensure everything compiles correctly
          // Skip build test for now - it's causing errors with undefined activeSandbox
          // The build test was trying to access global.activeSandbox from the frontend,
          // but that's only available in the backend API routes
          console.log('[build-test] Skipping build test - would need API endpoint');
          
          // Force iframe refresh after applying code
          const refreshDelay = appConfig.codeApplication.defaultRefreshDelay; // Allow Vite to process changes
          
          setTimeout(() => {
            const currentSandboxData = effectiveSandboxData;
            if (iframeRef.current && currentSandboxData?.url) {
              console.log('[home] Refreshing iframe after code application...');
              
              // Method 1: Change src with timestamp
              const urlWithTimestamp = `${currentSandboxData.url}?t=${Date.now()}&applied=true`;
              iframeRef.current.src = urlWithTimestamp;
              
              // Method 2: Force reload after a short delay
              setTimeout(() => {
                try {
                  if (iframeRef.current?.contentWindow) {
                    iframeRef.current.contentWindow.location.reload();
                    console.log('[home] Force reloaded iframe content');
                  }
                } catch (e) {
                  console.log('[home] Could not reload iframe (cross-origin):', e);
                }
                // Reload completed
              }, 1000);
            }
          }, refreshDelay);
          
          // Vite error checking removed - handled by template setup
        }
        
          // Give Vite HMR a moment to detect changes, then ensure refresh
          const currentSandboxData = effectiveSandboxData;
          if (iframeRef.current && currentSandboxData?.url) {
            // Wait for Vite to process the file changes
            // If packages were installed, wait longer for Vite to restart
            const packagesInstalled = results?.packagesInstalled?.length > 0 || data.results?.packagesInstalled?.length > 0;
            const refreshDelay = packagesInstalled ? appConfig.codeApplication.packageInstallRefreshDelay : appConfig.codeApplication.defaultRefreshDelay;
            console.log(`[applyGeneratedCode] Packages installed: ${packagesInstalled}, refresh delay: ${refreshDelay}ms`);
            
            setTimeout(async () => {
            if (iframeRef.current && currentSandboxData?.url) {
              console.log('[applyGeneratedCode] Starting iframe refresh sequence...');
              console.log('[applyGeneratedCode] Current iframe src:', iframeRef.current.src);
              console.log('[applyGeneratedCode] Sandbox URL:', currentSandboxData.url);
              
              // Method 1: Try direct navigation first
              try {
                const urlWithTimestamp = `${currentSandboxData.url}?t=${Date.now()}&force=true`;
                console.log('[applyGeneratedCode] Attempting direct navigation to:', urlWithTimestamp);
                
                // Remove any existing onload handler
                iframeRef.current.onload = null;
                
                // Navigate directly
                iframeRef.current.src = urlWithTimestamp;
                
                // Wait a bit and check if it loaded
                await new Promise(resolve => setTimeout(resolve, 2000));
                
                // Try to access the iframe content to verify it loaded
                try {
                  const iframeDoc = iframeRef.current.contentDocument || iframeRef.current.contentWindow?.document;
                  if (iframeDoc && iframeDoc.readyState === 'complete') {
                    console.log('[applyGeneratedCode] Iframe loaded successfully');
                    return;
                  }
                } catch {
                  console.log('[applyGeneratedCode] Cannot access iframe content (CORS), assuming loaded');
                  return;
                }
              } catch (e) {
                console.error('[applyGeneratedCode] Direct navigation failed:', e);
              }
              
              // Method 2: Force complete iframe recreation if direct navigation failed
              console.log('[applyGeneratedCode] Falling back to iframe recreation...');
              const parent = iframeRef.current.parentElement;
              const newIframe = document.createElement('iframe');
              
              // Copy attributes
              newIframe.className = iframeRef.current.className;
              newIframe.title = iframeRef.current.title;
              newIframe.allow = iframeRef.current.allow;
              // Copy sandbox attributes
              const sandboxValue = iframeRef.current.getAttribute('sandbox');
              if (sandboxValue) {
                newIframe.setAttribute('sandbox', sandboxValue);
              }
              
              // Remove old iframe
              iframeRef.current.remove();
              
              // Add new iframe
              newIframe.src = `${currentSandboxData.url}?t=${Date.now()}&recreated=true`;
              parent?.appendChild(newIframe);
              
              // Update ref
              (iframeRef as any).current = newIframe;
              
              console.log('[applyGeneratedCode] Iframe recreated with new content');
            } else {
              console.error('[applyGeneratedCode] No iframe or sandbox URL available for refresh');
            }
          }, refreshDelay); // Dynamic delay based on whether packages were installed
        }
        
        } else {
          throw new Error(finalData?.error || 'Failed to apply code');
        }
      } else {
        // If no final data was received, still close loading
        addChatMessage('Code application may have partially succeeded. Check the preview.', 'system');
      }
    } catch (error: any) {
      log(`Failed to apply code: ${error.message}`, 'error');
    } finally {
      setLoading(false);
      // Clear isEdit flag after applying code
      setGenerationProgress(prev => ({
        ...prev,
        isEdit: false
      }));
    }
  };

  const fetchSandboxFiles = async () => {
    if (!sandboxData) return;
    
    try {
      const response = await fetch('/api/get-sandbox-files', {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        }
      });
      
      if (response.ok) {
        const data = await response.json();
        if (data.success) {
          setSandboxFiles(data.files || {});
          setFileStructure(data.structure || '');
          console.log('[fetchSandboxFiles] Updated file list:', Object.keys(data.files || {}).length, 'files');
        }
      }
    } catch (error) {
      console.error('[fetchSandboxFiles] Error fetching files:', error);
    }
  };
  
//   const restartViteServer = async () => {
//     try {
//       addChatMessage('Restarting Vite dev server...', 'system');
//       
//       const response = await fetch('/api/restart-vite', {
//         method: 'POST',
//         headers: { 'Content-Type': 'application/json' }
//       });
//       
//       if (response.ok) {
//         const data = await response.json();
//         if (data.success) {
//           addChatMessage('✓ Vite dev server restarted successfully!', 'system');
//           
//           // Refresh the iframe after a short delay
//           setTimeout(() => {
//             if (iframeRef.current && sandboxData?.url) {
//               iframeRef.current.src = `${sandboxData.url}?t=${Date.now()}`;
//             }
//           }, 2000);
//         } else {
//           addChatMessage(`Failed to restart Vite: ${data.error}`, 'error');
//         }
//       } else {
//         addChatMessage('Failed to restart Vite server', 'error');
//       }
//     } catch (error) {
//       console.error('[restartViteServer] Error:', error);
//       addChatMessage(`Error restarting Vite: ${error instanceof Error ? error.message : 'Unknown error'}`, 'error');
//     }
//   };

//   const applyCode = async () => {
//     const code = promptInput.trim();
//     if (!code) {
//       log('Please enter some code first', 'error');
//       addChatMessage('No code to apply. Please generate code first.', 'system');
//       return;
//     }
//     
//     // Prevent double clicks
//     if (loading) {
//       console.log('[applyCode] Already loading, skipping...');
//       return;
//     }
//     
//     // Determine if this is an edit based on whether we have applied code before
//     const isEdit = conversationContext.appliedCode.length > 0;
//     await applyGeneratedCode(code, isEdit);
//   };

  // Build a unified file tree from sandbox files + generation progress
  const buildFileTree = (): { [dir: string]: Array<{ name: string; fullPath: string; edited?: boolean; generated?: boolean }> } => {
    const tree: { [dir: string]: Array<{ name: string; fullPath: string; edited?: boolean; generated?: boolean }> } = {};
    const seen = new Set<string>();

    // 1) Files from generation progress (recently generated / edited)
    generationProgress.files.forEach(file => {
      const parts = file.path.split('/');
      const dir = parts.length > 1 ? parts.slice(0, -1).join('/') : '';
      const fileName = parts[parts.length - 1];
      const fullPath = file.path;
      if (!tree[dir]) tree[dir] = [];
      tree[dir].push({ name: fileName, fullPath, edited: file.edited || false, generated: true });
      seen.add(fullPath);
    });

    // 2) Files fetched from the actual sandbox (everything on disk)
    Object.keys(sandboxFiles).forEach(rawPath => {
      // Normalize: strip leading ./ or /
      const filePath = rawPath.replace(/^\.[\/]/, '').replace(/^\//, '');
      if (seen.has(filePath)) return; // already shown via generation progress
      const parts = filePath.split('/');
      const dir = parts.length > 1 ? parts.slice(0, -1).join('/') : '';
      const fileName = parts[parts.length - 1];
      if (!tree[dir]) tree[dir] = [];
      tree[dir].push({ name: fileName, fullPath: filePath });
      seen.add(filePath);
    });

    return tree;
  };

  const renderFileExplorer = () => {
    const fileTree = buildFileTree();
    const hasSandboxFiles = Object.keys(sandboxFiles).length > 0;
    const hasGenFiles = generationProgress.files.length > 0;
    const hasFiles = hasSandboxFiles || hasGenFiles;

    return (
      <div className="w-[250px] border-r border-[#2a2a4a] bg-[#0a0a1a] flex flex-col flex-shrink-0">
        <div className="p-4 bg-[#12122a] text-gray-200 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <BsFolderFill style={{ width: '16px', height: '16px' }} />
            <span className="text-sm font-medium">Explorer</span>
          </div>
          {sandboxData && (
            <button
              onClick={() => fetchSandboxFiles()}
              className="p-1 rounded hover:bg-[#1e1e3a] text-gray-400 hover:text-gray-200 transition-colors"
              title="Refresh files"
            >
              <svg width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
            </button>
          )}
        </div>

        <div className="flex-1 overflow-y-auto p-4 scrollbar-hide">
          {!hasFiles ? (
            <div className="flex flex-col items-center justify-center h-full text-gray-500 text-xs">
              <BsFolderFill style={{ width: '24px', height: '24px' }} className="mb-2 opacity-30" />
              <p>No files yet</p>
              <p className="text-[10px] mt-1">Files will appear after generation</p>
            </div>
          ) : (
            <div className="text-sm">
              {/* Root project folder */}
              <div
                className="flex items-center gap-2 py-0.5 px-3 hover:bg-[#1e1e3a] rounded cursor-pointer text-gray-300"
                onClick={() => toggleFolder('root')}
              >
                {expandedFolders.has('root') ? (
                  <FiChevronDown style={{ width: '16px', height: '16px' }} className="text-gray-400" />
                ) : (
                  <FiChevronRight style={{ width: '16px', height: '16px' }} className="text-gray-400" />
                )}
                {expandedFolders.has('root') ? (
                  <BsFolder2Open style={{ width: '16px', height: '16px' }} className="text-blue-500" />
                ) : (
                  <BsFolderFill style={{ width: '16px', height: '16px' }} className="text-blue-500" />
                )}
                <span className="font-medium text-gray-200">/</span>
              </div>

              {expandedFolders.has('root') && (
                <div className="ml-6">
                  {Object.entries(fileTree)
                    .sort(([a], [b]) => a.localeCompare(b))
                    .map(([dir, files]) => (
                    <div key={dir || '__root__'} className="mb-1">
                      {dir && (
                        <div
                          className="flex items-center gap-2 py-0.5 px-3 hover:bg-[#1e1e3a] rounded cursor-pointer text-gray-300"
                          onClick={() => toggleFolder(dir)}
                        >
                          {expandedFolders.has(dir) ? (
                            <FiChevronDown style={{ width: '16px', height: '16px' }} className="text-gray-400" />
                          ) : (
                            <FiChevronRight style={{ width: '16px', height: '16px' }} className="text-gray-400" />
                          )}
                          {expandedFolders.has(dir) ? (
                            <BsFolder2Open style={{ width: '16px', height: '16px' }} className="text-yellow-600" />
                          ) : (
                            <BsFolderFill style={{ width: '16px', height: '16px' }} className="text-yellow-600" />
                          )}
                          <span className="text-gray-300">{dir}</span>
                        </div>
                      )}
                      {(!dir || expandedFolders.has(dir)) && (
                        <div className={dir ? 'ml-8' : ''}>
                          {files.sort((a, b) => a.name.localeCompare(b.name)).map(fileInfo => {
                            const isSelected = selectedFile === fileInfo.fullPath;
                            return (
                              <div
                                key={fileInfo.fullPath}
                                className={`flex items-center gap-2 py-0.5 px-3 rounded cursor-pointer transition-all ${
                                  isSelected
                                    ? 'bg-blue-500 text-white'
                                    : 'text-gray-300 hover:bg-[#1e1e3a]'
                                }`}
                                onClick={() => handleFileClick(fileInfo.fullPath)}
                              >
                                {getFileIcon(fileInfo.name)}
                                <span className={`text-xs flex items-center gap-1 ${isSelected ? 'font-medium' : ''}`}>
                                  {fileInfo.name}
                                  {fileInfo.edited && (
                                    <span className={`text-[10px] px-1 rounded ${isSelected ? 'bg-blue-400' : 'bg-orange-500 text-white'}`}>✓</span>
                                  )}
                                  {fileInfo.generated && !fileInfo.edited && (
                                    <span className={`text-[10px] px-1 rounded ${isSelected ? 'bg-blue-400' : 'bg-green-600 text-white'}`}>new</span>
                                  )}
                                </span>
                              </div>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    );
  };

  const renderMainContent = () => {
    if (activeTab === 'generation') {
      // Show Code tab with file explorer + code viewer
      const hasGenContent = generationProgress.isGenerating || generationProgress.files.length > 0;
      return (
        /* Generation Tab Content */
        <div className="absolute inset-0 flex overflow-hidden">
          {/* File Explorer - always visible in Code tab (hide during edits) */}
          {!generationProgress.isEdit && renderFileExplorer()}
          
          {/* Code Content */}
          <div className="flex-1 flex flex-col overflow-hidden">
            {/* Thinking Mode Display - Only show during active generation */}
            {generationProgress.isGenerating && (generationProgress.isThinking || generationProgress.thinkingText) && (
              <div className="px-6 pb-6">
                <div className="flex items-center gap-2 mb-2">
                  <div className="text-purple-600 font-medium flex items-center gap-2">
                    {generationProgress.isThinking ? (
                      <>
                        <div className="w-3 h-3 bg-purple-600 rounded-full animate-pulse" />
                        AI is thinking...
                      </>
                    ) : (
                      <>
                        <span className="text-purple-600">✓</span>
                        Thought for {generationProgress.thinkingDuration || 0} seconds
                      </>
                    )}
                  </div>
                </div>
                {generationProgress.thinkingText && (
                  <div className="bg-purple-950 border border-purple-700 rounded-lg p-4 max-h-48 overflow-y-auto scrollbar-hide">
                    <pre className="text-xs font-mono text-purple-300 whitespace-pre-wrap">
                      {generationProgress.thinkingText}
                    </pre>
                  </div>
                )}
              </div>
            )}
            
            {/* Live Code Display */}
            <div className="flex-1 rounded-lg p-6 flex flex-col min-h-0 overflow-hidden">
              <div className="flex-1 overflow-y-auto min-h-0 scrollbar-hide" ref={codeDisplayRef}>
                {/* Show selected file if one is selected */}
                {selectedFile ? (
                  <div className="animate-in fade-in slide-in-from-top-2 duration-300 h-full">
                    <div className="bg-black border border-[#2a2a4a] rounded-lg overflow-hidden shadow-sm h-full flex flex-col">
                      <div className="px-4 py-2 bg-[#12122a] text-white flex items-center justify-between flex-shrink-0">
                        <div className="flex items-center gap-2">
                          {getFileIcon(selectedFile)}
                          <span className="font-mono text-sm">{selectedFile}</span>
                        </div>
                        <button
                          onClick={() => setSelectedFile(null)}
                          className="hover:bg-[#2a2a4a] p-1 rounded transition-colors"
                        >
                          <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                          </svg>
                        </button>
                      </div>
                      <div className="flex-1 min-h-0">
                        <CodeEditor
                          content={getFileContent(selectedFile)}
                          filePath={selectedFile}
                          onSave={(path, content) => handleFileSave(path, content)}
                        />
                      </div>
                    </div>
                  </div>
                ) : /* If no files parsed yet, show loading or raw stream */
                generationProgress.files.length === 0 && !generationProgress.currentFile ? (
                  generationProgress.isThinking ? (
                    // Beautiful loading state while thinking
                    <div className="flex items-center justify-center h-full">
                      <div className="text-center">
                        <div className="mb-8 relative">
                          <div className="w-48 h-48 mx-auto">
                            <div className="absolute inset-0 border-8 border-gray-800 rounded-full"></div>
                            <div className="absolute inset-0 border-8 border-green-500 rounded-full animate-spin border-t-transparent"></div>
                          </div>
                        </div>
                        <h3 className="text-xl font-medium text-white mb-2">AI is analyzing your request</h3>
                        <p className="text-gray-400 text-sm">{generationProgress.status || 'Preparing to generate code...'}</p>
                        {/* Tool call progress */}
                        {generationProgress.toolCalls && generationProgress.toolCalls.length > 0 && (
                          <div className="mt-4 text-left max-w-md mx-auto">
                            <div className="text-xs text-gray-500 uppercase tracking-wide mb-2">Tool Activity</div>
                            <div className="space-y-1 max-h-32 overflow-y-auto">
                              {generationProgress.toolCalls.map((tc, i) => {
                                const hasResult = generationProgress.toolResults?.some(tr => tr.index === tc.index);
                                return (
                                  <div key={i} className="flex items-center gap-2 text-xs">
                                    {hasResult ? (
                                      <span className="text-green-400">✓</span>
                                    ) : (
                                      <span className="text-yellow-400 animate-pulse">●</span>
                                    )}
                                    <span className="text-gray-400 font-mono">{tc.name}</span>
                                    <span className="text-gray-600 truncate">{tc.args?.path || tc.args?.pattern || tc.args?.command?.substring(0, 30) || ''}</span>
                                  </div>
                                );
                              })}
                            </div>
                          </div>
                        )}
                        {/* ReAct Reasoning Steps */}
                        {generationProgress.reasoningSteps && generationProgress.reasoningSteps.length > 0 && (
                          <div className="mt-4 text-left max-w-md mx-auto">
                            <div className="text-xs text-purple-400 uppercase tracking-wide mb-2">Reasoning</div>
                            <div className="space-y-2 max-h-48 overflow-y-auto">
                              {generationProgress.reasoningSteps.map((rs, i) => (
                                <div key={i} className="bg-purple-950/50 border border-purple-800/40 rounded-md p-2">
                                  <p className="text-xs text-purple-300 italic">{rs.thought}</p>
                                  {rs.next_step && (
                                    <p className="text-xs text-purple-400 mt-1 font-medium">Next: {rs.next_step}</p>
                                  )}
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  ) : (
                    <div className="bg-[#0a0a1a] border border-[#2a2a4a] rounded-lg overflow-hidden">
                      <div className="px-4 py-2 bg-[#12122a] text-gray-200 flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <div className="w-16 h-16 border-2 border-orange-500 border-t-transparent rounded-full animate-spin" />
                          <span className="font-mono text-sm">Streaming code...</span>
                        </div>
                      </div>
                      <div className="p-4 bg-gray-900 rounded">
                        <SyntaxHighlighter
                          language="jsx"
                          style={vscDarkPlus}
                          customStyle={{
                            margin: 0,
                            padding: '1rem',
                            fontSize: '0.875rem',
                            background: 'transparent',
                          }}
                          showLineNumbers={true}
                        >
                          {generationProgress.streamedCode || 'Starting code generation...'}
                        </SyntaxHighlighter>
                        <span className="inline-block w-3 h-5 bg-orange-400 ml-1 animate-pulse" />
                      </div>
                    </div>
                  )
                ) : (
                  <div className="space-y-4">
                    {/* Show current file being generated */}
                    {generationProgress.currentFile && (
                      <div className="bg-black border-2 border-gray-400 rounded-lg overflow-hidden shadow-sm">
                        <div className="px-4 py-2 bg-[#36322F] text-white flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <div className="w-16 h-16 border-2 border-white border-t-transparent rounded-full animate-spin" />
                            <span className="font-mono text-sm">{generationProgress.currentFile.path}</span>
                            <span className={`px-2 py-0.5 text-xs rounded ${
                              generationProgress.currentFile.type === 'css' ? 'bg-blue-600 text-white' :
                              generationProgress.currentFile.type === 'javascript' ? 'bg-yellow-600 text-white' :
                              generationProgress.currentFile.type === 'json' ? 'bg-green-600 text-white' :
                              'bg-[#2a2a4a] text-gray-300'
                            }`}>
                              {generationProgress.currentFile.type === 'javascript' ? 'JSX' : generationProgress.currentFile.type.toUpperCase()}
                            </span>
                          </div>
                        </div>
                        <div className="bg-gray-900 border border-gray-700 rounded">
                          <SyntaxHighlighter
                            language={
                              generationProgress.currentFile.type === 'css' ? 'css' :
                              generationProgress.currentFile.type === 'json' ? 'json' :
                              generationProgress.currentFile.type === 'html' ? 'html' :
                              'jsx'
                            }
                            style={vscDarkPlus}
                            customStyle={{
                              margin: 0,
                              padding: '1rem',
                              fontSize: '0.75rem',
                              background: 'transparent',
                            }}
                            showLineNumbers={true}
                          >
                            {generationProgress.currentFile.content}
                          </SyntaxHighlighter>
                          <span className="inline-block w-3 h-4 bg-orange-400 ml-4 mb-4 animate-pulse" />
                        </div>
                      </div>
                    )}
                    
                    {/* Show completed files */}
                    {generationProgress.files.map((file, idx) => (
                      <div key={idx} className="bg-[#0a0a1a] border border-[#2a2a4a] rounded-lg overflow-hidden">
                        <div className="px-4 py-2 bg-[#36322F] text-white flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <span className="text-green-500">✓</span>
                            <span className="font-mono text-sm">{file.path}</span>
                          </div>
                          <span className={`px-2 py-0.5 text-xs rounded ${
                            file.type === 'css' ? 'bg-blue-600 text-white' :
                            file.type === 'javascript' ? 'bg-yellow-600 text-white' :
                            file.type === 'json' ? 'bg-green-600 text-white' :
                            'bg-[#2a2a4a] text-gray-300'
                          }`}>
                            {file.type === 'javascript' ? 'JSX' : file.type.toUpperCase()}
                          </span>
                        </div>
                        <div className="bg-gray-900 border border-gray-700  max-h-48 overflow-y-auto scrollbar-hide">
                          <SyntaxHighlighter
                            language={
                              file.type === 'css' ? 'css' :
                              file.type === 'json' ? 'json' :
                              file.type === 'html' ? 'html' :
                              'jsx'
                            }
                            style={vscDarkPlus}
                            customStyle={{
                              margin: 0,
                              padding: '1rem',
                              fontSize: '0.75rem',
                              background: 'transparent',
                            }}
                            showLineNumbers={true}
                            wrapLongLines={true}
                          >
                            {file.content}
                          </SyntaxHighlighter>
                        </div>
                      </div>
                    ))}
                    
                    {/* Show remaining raw stream if there's content after the last file */}
                    {!generationProgress.currentFile && generationProgress.streamedCode.length > 0 && (
                      <div className="bg-black border border-gray-200 rounded-lg overflow-hidden">
                        <div className="px-4 py-2 bg-[#36322F] text-white flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <div className="w-16 h-16 border-2 border-gray-400 border-t-transparent rounded-full animate-spin" />
                            <span className="font-mono text-sm">Processing...</span>
                          </div>
                        </div>
                        <div className="bg-gray-900 border border-gray-700 rounded">
                          <SyntaxHighlighter
                            language="jsx"
                            style={vscDarkPlus}
                            customStyle={{
                              margin: 0,
                              padding: '1rem',
                              fontSize: '0.75rem',
                              background: 'transparent',
                            }}
                            showLineNumbers={false}
                          >
                            {(() => {
                              // Show only the tail of the stream after the last file
                              const lastFileEnd = generationProgress.files.length > 0 
                                ? generationProgress.streamedCode.lastIndexOf('</file>') + 7
                                : 0;
                              let remainingContent = generationProgress.streamedCode.slice(lastFileEnd).trim();
                              
                              // Remove explanation tags and content
                              remainingContent = remainingContent.replace(/<explanation>[\s\S]*?<\/explanation>/g, '').trim();

                              // If only whitespace or nothing left, show loading message
                              // Use "Loading sandbox..." instead of "Waiting for next file..." for better UX
                              return remainingContent || 'Loading sandbox...';
                            })()}
                          </SyntaxHighlighter>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
            
            {/* Progress indicator */}
            {generationProgress.components.length > 0 && (
              <div className="mx-6 mb-6">
                <div className="h-2 bg-[#2a2a4a] rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-gradient-to-r from-orange-500 to-orange-400 transition-all duration-300"
                    style={{
                      width: `${(generationProgress.currentComponent / Math.max(generationProgress.components.length, 1)) * 100}%`
                    }}
                  />
                </div>
              </div>
            )}

            {/* Empty state when no generation content but sandbox exists */}
            {!hasGenContent && (
              <div className="flex-1 flex flex-col overflow-hidden">
                <div className="flex-1 rounded-lg flex flex-col min-h-0 overflow-hidden">
                  <div className="flex-1 min-h-0">
                    {selectedFile ? (
                      <div className="h-full animate-in fade-in slide-in-from-top-2 duration-300">
                        <div className="bg-black border border-[#2a2a4a] rounded-lg overflow-hidden shadow-sm h-full flex flex-col">
                          <div className="px-4 py-2 bg-[#12122a] text-white flex items-center justify-between flex-shrink-0">
                            <div className="flex items-center gap-2">
                              {getFileIcon(selectedFile)}
                              <span className="font-mono text-sm">{selectedFile}</span>
                            </div>
                            <button
                              onClick={() => setSelectedFile(null)}
                              className="hover:bg-[#2a2a4a] p-1 rounded transition-colors"
                            >
                              <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                              </svg>
                            </button>
                          </div>
                          <div className="flex-1 min-h-0">
                            <CodeEditor
                              content={getFileContent(selectedFile)}
                              filePath={selectedFile}
                              onSave={(path, content) => handleFileSave(path, content)}
                            />
                          </div>
                        </div>
                      </div>
                    ) : (
                      <div className="flex items-center justify-center h-full text-gray-500">
                        <div className="text-center">
                          <svg width="48" height="48" fill="none" viewBox="0 0 24 24" stroke="currentColor" className="mx-auto mb-3 opacity-30">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
                          </svg>
                          <p className="text-sm">Select a file from the explorer</p>
                          <p className="text-xs mt-1 text-gray-600">Or switch to View tab for the live preview</p>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      );
    } else if (activeTab === 'preview') {
      // Show loading state for initial generation or when starting a new generation with existing sandbox
      const isInitialGeneration = !sandboxData?.url && (urlScreenshot || isCapturingScreenshot || isPreparingDesign || loadingStage);
      const isNewGenerationWithSandbox = isStartingNewGeneration && sandboxData?.url;
      const shouldShowLoadingOverlay = (isInitialGeneration || isNewGenerationWithSandbox) && 
        (loading || generationProgress.isGenerating || isPreparingDesign || loadingStage || isCapturingScreenshot || isStartingNewGeneration);
      
      if (isInitialGeneration || isNewGenerationWithSandbox) {
        return (
          <div className="relative w-full h-full bg-gray-900">
            {/* Screenshot as background when available */}
            {urlScreenshot && (
              /* eslint-disable-next-line @next/next/no-img-element */
              <img 
                src={urlScreenshot} 
                alt="Website preview" 
                className="absolute inset-0 w-full h-full object-cover transition-opacity duration-700"
                style={{ 
                  opacity: isScreenshotLoaded ? 1 : 0,
                  willChange: 'opacity'
                }}
                onLoad={() => setIsScreenshotLoaded(true)}
                loading="eager"
              />
            )}
            
            {/* Loading overlay - only show when actively processing initial generation */}
            {shouldShowLoadingOverlay && (
              <div className="absolute inset-0 bg-black/70 flex flex-col items-center justify-center backdrop-blur-sm">
                {/* Loading animation with skeleton */}
                <div className="text-center max-w-md">
                  {/* Animated skeleton lines */}
                  <div className="mb-6 space-y-3">
                    <div className="h-2 bg-gradient-to-r from-transparent via-white/20 to-transparent rounded animate-pulse" 
                         style={{ animationDuration: '1.5s', animationDelay: '0s' }} />
                    <div className="h-2 bg-gradient-to-r from-transparent via-white/20 to-transparent rounded animate-pulse w-4/5 mx-auto" 
                         style={{ animationDuration: '1.5s', animationDelay: '0.2s' }} />
                    <div className="h-2 bg-gradient-to-r from-transparent via-white/20 to-transparent rounded animate-pulse w-3/5 mx-auto" 
                         style={{ animationDuration: '1.5s', animationDelay: '0.4s' }} />
                  </div>
                  
                  {/* Status text */}
                  <p className="text-white text-lg font-medium">
                    {isCapturingScreenshot ? 'Analyzing website...' :
                     isPreparingDesign ? 'Preparing design...' :
                     generationProgress.isGenerating ? 'Generating code...' :
                     'Loading...'}
                  </p>
                  
                  {/* Subtle progress hint */}
                  <p className="text-white/60 text-sm mt-2">
                    {isCapturingScreenshot ? 'Taking a screenshot of the site' :
                     isPreparingDesign ? 'Understanding the layout and structure' :
                     generationProgress.isGenerating ? 'Writing React components' :
                     'Please wait...'}
                  </p>
                </div>
              </div>
            )}
          </div>
        );
      }
      
      // Show sandbox iframe - keep showing during edits, only hide during initial loading
      if (sandboxData?.url) {
        return (
          <div className="relative w-full h-full">
            <iframe
              ref={iframeRef}
              src={sandboxData.url}
              className="w-full h-full border-none"
              title="Sable Dev Sandbox"
              allow="clipboard-write"
              sandbox="allow-scripts allow-same-origin allow-forms allow-popups allow-modals"
            />
            
            {/* Package installation overlay - shows when installing packages or applying code */}
            {codeApplicationState.stage && codeApplicationState.stage !== 'complete' && (
              <div className="absolute inset-0 bg-[#0a0a1a]/95 backdrop-blur-sm flex items-center justify-center z-10">
                <div className="text-center max-w-md">
                  <div className="mb-6">
                    {/* Animated icon based on stage */}
                    {codeApplicationState.stage === 'installing' ? (
                      <div className="w-16 h-16 mx-auto">
                        <svg className="w-full h-full animate-spin" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                      </div>
                    ) : null}
                  </div>
                  
                  <h3 className="text-lg font-semibold text-gray-100 mb-2">
                    {codeApplicationState.stage === 'analyzing' && 'Analyzing code...'}
                    {codeApplicationState.stage === 'installing' && 'Installing packages...'}
                    {codeApplicationState.stage === 'applying' && 'Applying changes...'}
                  </h3>
                  
                  {/* Package list during installation */}
                  {codeApplicationState.stage === 'installing' && codeApplicationState.packages && (
                    <div className="mb-4">
                      <div className="flex flex-wrap gap-2 justify-center">
                        {codeApplicationState.packages.map((pkg, index) => (
                          <span 
                            key={index}
                            className={`px-2 py-1 text-xs rounded-full transition-all ${
                              codeApplicationState.installedPackages?.includes(pkg)
                                ? 'bg-green-900/50 text-green-400'
                                : 'bg-[#1e1e3a] text-gray-400'
                            }`}
                          >
                            {pkg}
                            {codeApplicationState.installedPackages?.includes(pkg) && (
                              <span className="ml-1">✓</span>
                            )}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  {/* Files being generated */}
                  {codeApplicationState.stage === 'applying' && codeApplicationState.filesGenerated && (
                    <div className="text-sm text-gray-400">
                      Creating {codeApplicationState.filesGenerated.length} files...
                    </div>
                  )}
                  
                  <p className="text-sm text-gray-500 mt-2">
                    {codeApplicationState.stage === 'analyzing' && 'Parsing generated code and detecting dependencies...'}
                    {codeApplicationState.stage === 'installing' && 'This may take a moment while npm installs the required packages...'}
                    {codeApplicationState.stage === 'applying' && 'Writing files to your sandbox environment...'}
                  </p>
                </div>
              </div>
            )}
            
            {/* Show a subtle indicator when code is being edited/generated */}
            {generationProgress.isGenerating && generationProgress.isEdit && !codeApplicationState.stage && (
              <div className="absolute top-4 right-4 inline-flex items-center gap-2 px-3 py-1.5 bg-black/80 backdrop-blur-sm rounded-lg">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                <span className="text-white text-xs font-medium">Generating code...</span>
              </div>
            )}
            
            {/* Refresh button */}
            <button
              onClick={() => {
                if (iframeRef.current && sandboxData?.url) {
                  console.log('[Manual Refresh] Forcing iframe reload...');
                  const newSrc = `${sandboxData.url}?t=${Date.now()}&manual=true`;
                  iframeRef.current.src = newSrc;
                }
              }}
              className="absolute bottom-4 right-4 bg-[#1e1e3a]/90 hover:bg-[#2a2a4a] text-gray-300 p-2 rounded-lg shadow-lg transition-all duration-200 hover:scale-105"
              title="Refresh sandbox"
            >
              <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
            </button>
          </div>
        );
      }
      
      // Default state when no sandbox and no screenshot
      return (
        <div className="flex items-center justify-center h-full bg-[#0a0a1a] text-gray-400 text-lg">
          {screenshotError ? (
            <div className="text-center">
              <p className="mb-2">Failed to capture screenshot</p>
              <p className="text-sm text-gray-500">{screenshotError}</p>
            </div>
          ) : sandboxData ? (
            <div className="text-gray-500">
              <div className="w-16 h-16 border-2 border-[#3a3a5a] border-t-transparent rounded-full animate-spin mx-auto mb-2" />
              <p className="text-sm">Loading preview...</p>
            </div>
          ) : (
            <div className="text-gray-500 text-center">
              <p className="text-sm">Start chatting to create your first app</p>
            </div>
          )}
        </div>
      );
    }
    return null;
  };

  // Agent spawning and polling
  const spawnAgents = async (prompt: string, isEdit: boolean, generatedCode: string = '') => {
    try {
      const res = await fetch('/api/agents/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt,
          primaryModel: aiModel,
          isEdit,
          hasErrors: chatMessages.some(m => m.type === 'error'),
          generatedCode,
          availableModels: dynamicModels,
          codeContext: Object.values(sandboxFiles).join('\n').substring(0, 3000),
        }),
      });
      if (!res.ok) return;
      const data = await res.json();
      if (data.agents?.length > 0) {
        setActiveAgents(data.agents);
        setAgentPanelOpen(true);
        // Start polling
        startAgentPolling();
      }
    } catch (err) {
      console.log('[agents] Failed to spawn agents:', err);
    }
  };

  const startAgentPolling = () => {
    if (agentPollRef.current) clearInterval(agentPollRef.current);
    agentPollRef.current = setInterval(async () => {
      try {
        const res = await fetch('/api/agents/status');
        if (!res.ok) return;
        const data = await res.json();
        if (data.agents) {
          setActiveAgents(data.agents);
          // Stop polling when all agents are done
          const allDone = data.agents.every((a: any) => a.status === 'completed' || a.status === 'error');
          if (allDone && agentPollRef.current) {
            clearInterval(agentPollRef.current);
            agentPollRef.current = null;
            // Auto-collapse after 10s
            setTimeout(() => setAgentPanelOpen(false), 10000);
          }
        }
      } catch {
        // Silently fail
      }
    }, 1500);
  };

  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      if (agentPollRef.current) clearInterval(agentPollRef.current);
    };
  }, []);

  const sendChatMessage = async (overrideMessage?: string) => {
    // Guard against receiving non-string args (e.g. React FormEvent from onSubmit)
    const rawOverride = (typeof overrideMessage === 'string') ? overrideMessage : undefined;
    const message = (rawOverride || aiChatInput || '').trim();
    if (!message) return;
    
    if (!aiEnabled) {
      addChatMessage('AI is disabled. Please enable it first.', 'system');
      return;
    }
    
    // Auto-fix/auto-continue messages start with these prefixes,  don't show them as user messages
    const isAutoMessage = message.startsWith('FIX THIS ERROR') || message.startsWith('Continue customizing');
    if (!isAutoMessage) {
      addChatMessage(message, 'user');
      // Reset auto-fix retry counter on genuine user input
      autoFixRetryRef.current = 0;
    }
    setAiChatInput('');
    
    // Check for special commands
    const lowerMessage = message.toLowerCase().trim();
    if (lowerMessage === 'check packages' || lowerMessage === 'install packages' || lowerMessage === 'npm install') {
      if (!sandboxData) {
        // More helpful message - user might be trying to run this too early
        addChatMessage('The sandbox is still being set up. Please wait for the generation to complete, then try again.', 'system');
        return;
      }
      await checkAndInstallPackages();
      return;
    }
    
    // Detect if the user is pasting a Vite/babel/build error and auto-wrap it as a fix request
    const isViteError = /\[plugin:vite|SyntaxError|Unexpected token|Missing semicolon|Identifier directly after|Cannot find module|Failed to resolve import|TypeError:|ReferenceError:|parse5 error|Unable to parse HTML|missing-whitespace/i.test(message);
    let processedMessage = message;
    
    if (isViteError && conversationContext.appliedCode.length > 0) {
      // Extract the file path from the error message
      const fileMatch = message.match(/\/([^\s:]+\.(jsx|tsx|js|ts|css|html))/);
      const fileName = fileMatch ? fileMatch[1].split('/').pop() : null;
      
      // Wrap the error in a focused fix instruction
      processedMessage = `FIX THIS ERROR - Output ONLY the broken file, nothing else:\n\n${message}\n\n${fileName ? `ONLY fix and output the file: ${fileName}. Do NOT regenerate any other files.` : 'Fix ONLY the file that has the error. Do NOT regenerate other files.'}`;
      
      addChatMessage(`Detected build error${fileName ? ` in ${fileName}` : ''}. Fixing...`, 'system');
    }
    
    // Start sandbox creation in parallel if needed
    let sandboxPromise: Promise<void> | null = null;
    let sandboxCreating = false;
    
    if (!sandboxData) {
      sandboxCreating = true;
      addChatMessage('Creating sandbox while I plan your app...', 'system');
      sandboxPromise = createSandbox(true).catch((error: any) => {
        addChatMessage(`Failed to create sandbox: ${error.message}`, 'system');
        throw error;
      });
    }
    
    // Determine if this is an edit
    const isEdit = conversationContext.appliedCode.length > 0;
    
    try {
      // Generation tab is already active from scraping phase
      setGenerationProgress(prev => ({
        ...prev,  // Preserve all existing state
        isGenerating: true,
        status: 'Starting AI generation...',
        components: [],
        currentComponent: 0,
        streamedCode: '',
        isStreaming: false,
        isThinking: true,
        thinkingText: 'Analyzing your request...',
        thinkingDuration: undefined,
        currentFile: undefined,
        lastProcessedPosition: 0,
        // Add isEdit flag to generation progress
        isEdit: isEdit,
        // Keep existing files for edits - we'll mark edited ones differently
        files: prev.files
      }));
      
      // Backend now manages file state - no need to fetch from frontend
      console.log('[chat] Using backend file cache for context');
      
      const fullContext = {
        sandboxId: sandboxData?.sandboxId || (sandboxCreating ? 'pending' : null),
        structure: structureContent,
        recentMessages: chatMessages.slice(-20),
        conversationContext: conversationContext,
        currentCode: promptInput,
        sandboxUrl: sandboxData?.url,
        sandboxCreating: sandboxCreating
      };
      
      // Debug what we're sending
      console.log('[chat] Sending context to AI:');
      console.log('[chat] - sandboxId:', fullContext.sandboxId);
      console.log('[chat] - isEdit:', conversationContext.appliedCode.length > 0);
      
      // Detect if this is specifically an error-fix request (auto or manual)
      const isErrorFix = processedMessage.startsWith('FIX THIS ERROR') || 
        (/\[plugin:vite|SyntaxError|Unexpected token/i.test(processedMessage) && conversationContext.appliedCode.length > 0);

      const response = await fetch('/api/generate-ai-code-stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: processedMessage,
          model: aiModel,
          context: fullContext,
          isEdit: conversationContext.appliedCode.length > 0,
          isErrorFix: isErrorFix,
          template: selectedTemplate
        })
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      // Spawn secondary agents in parallel (fire and forget)
      // spawnAgents(message, isEdit); // DISABLED: agent system hidden for now
      
      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let generatedCode = '';
      let explanation = '';
      let buffer = ''; // Buffer for incomplete lines
      
      if (reader) {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          
          const chunk = decoder.decode(value, { stream: true });
          console.log('[chat] Received chunk:', chunk.length, 'bytes');
          buffer += chunk;
          const lines = buffer.split('\n');
          
          // Keep the last line in buffer if it's incomplete
          buffer = lines.pop() || '';
          
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6));
                
                if (data.type === 'status') {
                  setGenerationProgress(prev => ({ ...prev, status: data.message }));
                } else if (data.type === 'thinking') {
                  setGenerationProgress(prev => ({ 
                    ...prev, 
                    isThinking: true,
                    thinkingText: (prev.thinkingText || '') + data.text
                  }));
                } else if (data.type === 'thinking_complete') {
                  setGenerationProgress(prev => ({ 
                    ...prev, 
                    isThinking: false,
                    thinkingDuration: data.duration
                  }));
                } else if (data.type === 'conversation') {
                  // Add conversational text to chat only if it's not code
                  let text = data.text || '';
                  
                  // Remove package tags from the text
                  text = text.replace(/<package>[^<]*<\/package>/g, '');
                  text = text.replace(/<packages>[^<]*<\/packages>/g, '');
                  
                  // Filter out any XML tags and file content that slipped through
                  if (!text.includes('<file') && !text.includes('import React') && 
                      !text.includes('export default') && !text.includes('className=') &&
                      text.trim().length > 0) {
                    addChatMessage(text.trim(), 'ai');
                  }
                } else if (data.type === 'stream' && data.raw) {
                  setGenerationProgress(prev => {
                    const newStreamedCode = prev.streamedCode + data.text;
                    
                    // Tab is already switched after scraping
                    
                    const updatedState = { 
                      ...prev, 
                      streamedCode: newStreamedCode,
                      isStreaming: true,
                      isThinking: false,
                      status: 'Generating code...'
                    };
                    
                    // Process complete files from the accumulated stream
                    const fileRegex = /<file path="([^"]+)">([^]*?)<\/file>/g;
                    let match;
                    const processedFiles = new Set(prev.files.map(f => f.path));
                    
                    while ((match = fileRegex.exec(newStreamedCode)) !== null) {
                      const filePath = match[1];
                      const fileContent = match[2];
                      
                      // Only add if we haven't processed this file yet
                      if (!processedFiles.has(filePath)) {
                        const fileExt = filePath.split('.').pop() || '';
                        const fileType = fileExt === 'jsx' || fileExt === 'js' ? 'javascript' :
                                        fileExt === 'css' ? 'css' :
                                        fileExt === 'json' ? 'json' :
                                        fileExt === 'html' ? 'html' : 'text';
                        
                        // Check if file already exists
                        const existingFileIndex = updatedState.files.findIndex(f => f.path === filePath);
                        
                        if (existingFileIndex >= 0) {
                          // Update existing file and mark as edited
                          updatedState.files = [
                            ...updatedState.files.slice(0, existingFileIndex),
                            {
                              ...updatedState.files[existingFileIndex],
                              content: fileContent.trim(),
                              type: fileType,
                              completed: true,
                              edited: true
                            },
                            ...updatedState.files.slice(existingFileIndex + 1)
                          ];
                        } else {
                          // Add new file
                          updatedState.files = [...updatedState.files, {
                            path: filePath,
                            content: fileContent.trim(),
                            type: fileType,
                            completed: true,
                            edited: false
                          }];
                        }
                        
                        // Only show file status if not in edit mode
                        if (!prev.isEdit) {
                          updatedState.status = `Completed ${filePath}`;
                        }
                        processedFiles.add(filePath);
                      }
                    }
                    
                    // Check for current file being generated (incomplete file at the end)
                    const lastFileMatch = newStreamedCode.match(/<file path="([^"]+)">([^]*?)$/);
                    if (lastFileMatch && !lastFileMatch[0].includes('</file>')) {
                      const filePath = lastFileMatch[1];
                      const partialContent = lastFileMatch[2];
                      
                      if (!processedFiles.has(filePath)) {
                        const fileExt = filePath.split('.').pop() || '';
                        const fileType = fileExt === 'jsx' || fileExt === 'js' ? 'javascript' :
                                        fileExt === 'css' ? 'css' :
                                        fileExt === 'json' ? 'json' :
                                        fileExt === 'html' ? 'html' : 'text';
                        
                        updatedState.currentFile = { 
                          path: filePath, 
                          content: partialContent, 
                          type: fileType 
                        };
                        // Only show file status if not in edit mode
                        if (!prev.isEdit) {
                          updatedState.status = `Generating ${filePath}`;
                        }
                      }
                    } else {
                      updatedState.currentFile = undefined;
                    }
                    
                    return updatedState;
                  });
                } else if (data.type === 'app') {
                  setGenerationProgress(prev => ({ 
                    ...prev, 
                    status: 'Generated App.jsx structure'
                  }));
                } else if (data.type === 'component') {
                  setGenerationProgress(prev => ({
                    ...prev,
                    status: `Generated ${data.name}`,
                    components: [...prev.components, { 
                      name: data.name, 
                      path: data.path, 
                      completed: true 
                    }],
                    currentComponent: data.index
                  }));
                } else if (data.type === 'package') {
                  // Handle package installation from tool calls
                  setGenerationProgress(prev => ({
                    ...prev,
                    status: data.message || `Installing ${data.name}`
                  }));
                } else if (data.type === 'tool-call') {
                  // AI is calling a tool to explore/modify the project
                  const args = data.args || {};
                  const toolLabel = data.name === 'file_read' ? `Reading ${args.path || 'file'}` :
                                    data.name === 'file_edit' ? `Editing ${args.path || 'file'}` :
                                    data.name === 'file_write' ? `Writing ${args.path || 'file'}` :
                                    data.name === 'grep' ? `Searching for "${String(args.pattern || '').substring(0, 30)}"` :
                                    data.name === 'glob' ? `Finding files: ${args.pattern || ''}` :
                                    data.name === 'list_files' ? `Listing ${args.path || 'files'}` :
                                    data.name === 'bash' ? `Running: ${String(args.command || '').substring(0, 40)}` :
                                    `Tool: ${data.name}`;
                  setGenerationProgress(prev => ({
                    ...prev,
                    status: toolLabel,
                    toolCalls: [...(prev.toolCalls || []), { name: data.name, args: data.args, index: data.index }],
                  }));
                } else if (data.type === 'tool-result') {
                  // Tool call completed
                  setGenerationProgress(prev => ({
                    ...prev,
                    status: `Tool ${data.name} completed`,
                    toolResults: [...(prev.toolResults || []), { name: data.name, output: data.output, index: data.index }],
                  }));
                } else if (data.type === 'complete') {
                  generatedCode = data.generatedCode;
                  explanation = data.explanation;
                  
                  // Save the last generated code
                  setConversationContext(prev => ({
                    ...prev,
                    lastGeneratedCode: generatedCode
                  }));
                  
                  // Clear thinking state when generation completes
                  setGenerationProgress(prev => ({
                    ...prev,
                    isThinking: false,
                    thinkingText: undefined,
                    thinkingDuration: undefined
                  }));
                  
                  // Store packages to install from tool calls
                  if (data.packagesToInstall && data.packagesToInstall.length > 0) {
                    console.log('[generate-code] Packages to install from tools:', data.packagesToInstall);
                    // Store packages globally for later installation
                    (window as any).pendingPackages = data.packagesToInstall;
                  }
                  
                  // Parse all files from the completed code if not already done
                  const fileRegex = /<file path="([^"]+)">([^]*?)<\/file>/g;
                  const parsedFiles: Array<{path: string; content: string; type: string; completed: boolean}> = [];
                  let fileMatch;
                  
                  while ((fileMatch = fileRegex.exec(data.generatedCode)) !== null) {
                    const filePath = fileMatch[1];
                    const fileContent = fileMatch[2];
                    const fileExt = filePath.split('.').pop() || '';
                    const fileType = fileExt === 'jsx' || fileExt === 'js' ? 'javascript' :
                                    fileExt === 'css' ? 'css' :
                                    fileExt === 'json' ? 'json' :
                                    fileExt === 'html' ? 'html' : 'text';
                    
                    parsedFiles.push({
                      path: filePath,
                      content: fileContent.trim(),
                      type: fileType,
                      completed: true
                    });
                  }
                  
                  setGenerationProgress(prev => ({
                    ...prev,
                    status: `Generated ${parsedFiles.length > 0 ? parsedFiles.length : prev.files.length} file${(parsedFiles.length > 0 ? parsedFiles.length : prev.files.length) !== 1 ? 's' : ''}!`,
                    isGenerating: false,
                    isStreaming: false,
                    isEdit: prev.isEdit,
                    // Keep the files that were already parsed during streaming
                    files: prev.files.length > 0 ? prev.files : parsedFiles
                  }));
                } else if (data.type === 'command') {
                  // Handle slash command actions from the backend
                  if (data.action === 'undo') {
                    handleUndo();
                  } else if (data.action === 'redo') {
                    handleRedo();
                  } else if (data.action === 'download') {
                    // Trigger project download via download-zip API
                    try {
                      const zipResp = await fetch('/api/download-zip', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ files: sandboxFiles }),
                      });
                      if (zipResp.ok) {
                        const blob = await zipResp.blob();
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = 'project.zip';
                        document.body.appendChild(a);
                        a.click();
                        document.body.removeChild(a);
                        URL.revokeObjectURL(url);
                        addChatMessage('Project downloaded as project.zip', 'system');
                      } else {
                        addChatMessage('Failed to create zip file', 'error');
                      }
                    } catch (e) {
                      addChatMessage('Failed to download project', 'error');
                    }
                  } else if (data.action === 'review') {
                    // /review returns handled: false, so the AI gets the review prompt,  nothing extra needed here
                  }
                } else if (data.type === 'error') {
                  throw new Error(data.error || 'Unknown stream error');
                }
              } catch (e) {
                console.error('Failed to parse SSE data:', e);
              }
            }
          }
        }
      }
      
      if (generatedCode) {
        // Parse files from generated code for metadata
        const fileRegex = /<file path="([^"]+)">([^]*?)<\/file>/g;
        const generatedFiles = [];
        let match;
        while ((match = fileRegex.exec(generatedCode)) !== null) {
          generatedFiles.push(match[1]);
        }
        
        // Show appropriate message based on edit mode
        if (isEdit && generatedFiles.length > 0) {
          // For edits, show which file(s) were edited
          const editedFileNames = generatedFiles.map(f => f.split('/').pop()).join(', ');
          addChatMessage(
            explanation || `Updated ${editedFileNames}`,
            'ai',
            {
              appliedFiles: [generatedFiles[0]] // Only show the first edited file
            }
          );
        } else {
          // For new generation, show all files
          addChatMessage(explanation || 'Code generated!', 'ai', {
            appliedFiles: generatedFiles
          });
        }
        
        setPromptInput(generatedCode);
        // Don't show the Generated Code panel by default
        // setLeftPanelVisible(true);
        
        // Wait for sandbox creation if it's still in progress
        let activeSandboxData = sandboxData;
        if (sandboxPromise) {
          addChatMessage('Waiting for sandbox to be ready...', 'system');
          try {
            const newSandboxData = await sandboxPromise;
            if (newSandboxData != null) {
              activeSandboxData = newSandboxData;
              // Also update the state for future use
              setSandboxData(newSandboxData);
            }
            // Remove the waiting message
            setChatMessages(prev => prev.filter(msg => msg.content !== 'Waiting for sandbox to be ready...'));
          } catch {
            addChatMessage('Sandbox creation failed. Cannot apply code.', 'system');
            return;
          }
        }
        
        if (activeSandboxData && generatedCode) {
          // For new sandbox creations (especially Vercel), add a delay to ensure Vite is ready
          if (sandboxCreating) {
            console.log('[startGeneration] New sandbox created, waiting for services to be ready...');
            await new Promise(resolve => setTimeout(resolve, 2000));
          }
          
          // Use isEdit flag that was determined at the start
          // Pass the sandbox data from the promise if it's different from the state
          await applyGeneratedCode(generatedCode, isEdit, activeSandboxData !== sandboxData ? activeSandboxData : undefined);

          // ── AUTO-ERROR-FIX: poll Vite for compile errors after applying ──
          const MAX_AUTO_FIX_RETRIES = 3;
          if (autoFixRetryRef.current < MAX_AUTO_FIX_RETRIES) {
            // Give Vite 3 seconds to recompile then check
            await new Promise(r => setTimeout(r, 3000));
            try {
              const errRes = await fetch('/api/check-vite-errors');
              const errData = await errRes.json();
              if (errData.hasError && errData.error) {
                const errorText = errData.error as string;
                const brokenFile = errData.file || 'unknown';

                // Check if error is a missing package import (npm 404 / not installed)
                const missingImportMatch = errorText.match(/Failed to resolve import "([^"]+)"/i)
                  || errorText.match(/Module not found.*['"]([^'"]+)['"]/i);

                if (missingImportMatch) {
                  const rawPkg = missingImportMatch[1];
                  // Extract base package name
                  const pkg = rawPkg.startsWith('@')
                    ? rawPkg.split('/').slice(0, 2).join('/')
                    : rawPkg.split('/')[0];

                  // Correct commonly hallucinated package names
                  const PACKAGE_CORRECTIONS: Record<string, string> = {
                    'hero-icons-react': '@heroicons/react',
                    'heroicons-react': '@heroicons/react',
                    'heroicons': '@heroicons/react',
                    '@react-native-particles/particle-generator': '', // doesn't exist, remove
                    'react-icons/hi': 'react-icons',
                    'react-icons/fa': 'react-icons',
                    'react-icons/fi': 'react-icons',
                    'react-icons/ai': 'react-icons',
                    'lucide': 'lucide-react',
                    '@lucide/react': 'lucide-react',
                  };

                  const correctedPkg = PACKAGE_CORRECTIONS[pkg] ?? pkg;

                  if (!correctedPkg) {
                    // Package doesn't exist at all,  ask AI to remove the import
                    autoFixRetryRef.current += 1;
                    addChatMessage(`Package "${pkg}" does not exist on npm. Asking AI to remove the import (attempt ${autoFixRetryRef.current}/${MAX_AUTO_FIX_RETRIES})...`, 'system');
                    await fetch('/api/check-vite-errors', { method: 'POST' });
                    const fixPrompt = `FIX THIS ERROR - The package "${pkg}" does NOT exist on npm. REMOVE the import and replace the functionality with inline code or a valid alternative. Output ONLY the file "${brokenFile}":\n\n${errorText}\n\nDo NOT use the package "${pkg}". Use inline SVGs, emoji, or plain HTML instead.`;
                    setTimeout(() => {
                      if (sendChatMessageRef.current) sendChatMessageRef.current(fixPrompt);
                    }, 500);
                    return;
                  }

                  // Try installing the (possibly corrected) package first
                  addChatMessage(`Missing package detected: "${pkg}"${correctedPkg !== pkg ? ` → corrected to "${correctedPkg}"` : ''}. Installing...`, 'system');
                  await fetch('/api/check-vite-errors', { method: 'POST' });
                  try {
                    await installPackages([correctedPkg]);
                    // Don't count this as a fix attempt,  it was a package install
                    // Vite will auto-reload; skip to next iteration
                    return;
                  } catch {
                    // Install failed,  ask AI to remove the import
                    autoFixRetryRef.current += 1;
                    addChatMessage(`Failed to install "${correctedPkg}". Asking AI to remove the import...`, 'system');
                    const fixPrompt = `FIX THIS ERROR - The package "${pkg}" failed to install. REMOVE the import and replace icons/components with inline SVGs or plain HTML/CSS. Output ONLY the file "${brokenFile}":\n\n${errorText}`;
                    setTimeout(() => {
                      if (sendChatMessageRef.current) sendChatMessageRef.current(fixPrompt);
                    }, 500);
                    return;
                  }
                }

                // Regular compile error,  ask AI to fix
                autoFixRetryRef.current += 1;
                const fixPrompt = `FIX THIS ERROR - Output ONLY the broken file "${brokenFile}", nothing else:\n\n${errorText}\n\nONLY fix and output the file: ${brokenFile}. Do NOT regenerate any other files.`;
                addChatMessage(`Auto-detected compile error in ${brokenFile} (attempt ${autoFixRetryRef.current}/${MAX_AUTO_FIX_RETRIES}). Fixing...`, 'system');
                // Clear the error so we detect fresh ones
                await fetch('/api/check-vite-errors', { method: 'POST' });
                // Recursive call to fix the error
                setTimeout(() => {
                  if (sendChatMessageRef.current) {
                    sendChatMessageRef.current(fixPrompt);
                  }
                }, 500);
                return; // exit early,  the recursive call handles completion
              }
            } catch (e) {
              console.log('[auto-fix] Error checking vite errors:', e);
            }
            // No errors,  reset retry counter
            autoFixRetryRef.current = 0;
          } else {
            // Max retries reached,  reset and let user handle it
            autoFixRetryRef.current = 0;
            addChatMessage('Auto-fix reached max retries. Check the preview for remaining errors.', 'system');
          }

          // ── AUTO-CONTINUE: detect truncated/incomplete output ──
          if (generatedCode) {
            // Check if any file tag was left unclosed (AI cut off mid-file)
            const openTags = (generatedCode.match(/<file path="/g) || []).length;
            const closeTags = (generatedCode.match(/<\/file>/g) || []).length;
            if (openTags > closeTags) {
              addChatMessage(`Generation was cut off mid-file. Auto-continuing...`, 'system');
              const continuePrompt = `Your previous output was TRUNCATED (cut off mid-file). Continue EXACTLY where you left off. Output the remaining <file> tags to complete the generation. Do NOT repeat files you already output.`;
              setTimeout(() => {
                if (sendChatMessageRef.current) {
                  sendChatMessageRef.current(continuePrompt);
                }
              }, 500);
              return;
            }

            // For react-spa template: check if expected components are missing
            if (!isEdit && selectedTemplate === 'react-spa') {
              const outputFileNames = new Set<string>();
              const fRegex = /<file path="([^"]+)">/g;
              let fm;
              while ((fm = fRegex.exec(generatedCode)) !== null) {
                outputFileNames.add(fm[1].split('/').pop() || '');
              }
              const expectedComponents = ['Header.jsx', 'Hero.jsx', 'Features.jsx', 'Footer.jsx', 'App.jsx'];
              const missingComponents = expectedComponents.filter(c => !outputFileNames.has(c));
              if (missingComponents.length > 0 && missingComponents.length < expectedComponents.length) {
                addChatMessage(`Continuing generation,  still need: ${missingComponents.join(', ')}...`, 'system');
                const continuePrompt = `Continue customizing the remaining component files to match the user request. Output ONLY these files: ${missingComponents.join(', ')}. Keep the same theme, colors, and content style as the files you already generated.`;
                setTimeout(() => {
                  if (sendChatMessageRef.current) {
                    sendChatMessageRef.current(continuePrompt);
                  }
                }, 500);
                return;
              }
            }
          }
        }
      }
      
      // Show completion status briefly then switch to preview
      setGenerationProgress(prev => ({
        ...prev,
        isGenerating: false,
        isStreaming: false,
        status: 'Generation complete!',
        isEdit: prev.isEdit,
        // Clear thinking state on completion
        isThinking: false,
        thinkingText: undefined,
        thinkingDuration: undefined
      }));
      
      setTimeout(() => {
        // Switch to preview but keep files for display
        setActiveTab('preview');
      }, 1000); // Reduced from 3000ms to 1000ms
    } catch (error: any) {
      setChatMessages(prev => prev.filter(msg => msg.content !== 'Thinking...'));
      addChatMessage(`Error: ${error.message}`, 'system');
      // Reset generation progress and switch back to preview on error
      setGenerationProgress({
        isGenerating: false,
        status: '',
        components: [],
        currentComponent: 0,
        streamedCode: '',
        isStreaming: false,
        isThinking: false,
        thinkingText: undefined,
        thinkingDuration: undefined,
        files: [],
        currentFile: undefined,
        lastProcessedPosition: 0
      });
      setActiveTab('preview');
    }
  };

  // Auto-trigger sendChatMessage when a creation prompt is pending and sandbox is ready
  // Keep sendChatMessage ref always up to date
  sendChatMessageRef.current = sendChatMessage;
  
  useEffect(() => {
    if (pendingCreationPrompt && sandboxData) {
      const prompt = pendingCreationPrompt;
      setPendingCreationPrompt(null);
      
      const timer = setTimeout(() => {
        console.log('[generation] Auto-triggering sendChatMessage for creation prompt (effect):', prompt);
        // Use ref to get the latest sendChatMessage with fresh closures
        if (sendChatMessageRef.current) {
          sendChatMessageRef.current(prompt);
        } else {
          sendChatMessage(prompt);
        }
      }, 500);
      
      return () => clearTimeout(timer);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pendingCreationPrompt, sandboxData]);

  const downloadZip = async () => {
    if (!sandboxData && Object.keys(sandboxFiles).length === 0) {
      addChatMessage('Please wait for the sandbox to be created before downloading.', 'system');
      return;
    }
    
    setLoading(true);
    log('Creating zip file...');
    addChatMessage('Creating ZIP file of your project...', 'system');
    
    try {
      const response = await fetch('/api/download-zip', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ files: sandboxFiles }),
      });
      
      if (response.ok) {
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = 'project.zip';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
        
        addChatMessage(
          'Your project has been downloaded! To run it locally:\n' +
          '1. Unzip the file\n' +
          '2. Run: npm install\n' +
          '3. Run: npm run dev\n' +
          '4. Open http://localhost:5173',
          'system'
        );
      } else {
        const data = await response.json();
        throw new Error(data.error || 'Failed to create zip');
      }
    } catch (error: any) {
      log(`Failed to create zip: ${error.message}`, 'error');
      addChatMessage(`Failed to create ZIP: ${error.message}`, 'system');
    } finally {
      setLoading(false);
    }
  };

  const reapplyLastGeneration = async () => {
    if (!conversationContext.lastGeneratedCode) {
      addChatMessage('No previous generation to re-apply', 'system');
      return;
    }
    
    if (!sandboxData) {
      addChatMessage('Please create a sandbox first', 'system');
      return;
    }
    
    addChatMessage('Re-applying last generation...', 'system');
    const isEdit = conversationContext.appliedCode.length > 0;
    await applyGeneratedCode(conversationContext.lastGeneratedCode, isEdit);
  };

  // Auto-scroll code display to bottom when streaming
  useEffect(() => {
    if (codeDisplayRef.current && generationProgress.isStreaming) {
      codeDisplayRef.current.scrollTop = codeDisplayRef.current.scrollHeight;
    }
  }, [generationProgress.streamedCode, generationProgress.isStreaming]);

  const toggleFolder = (folderPath: string) => {
    const newExpanded = new Set(expandedFolders);
    if (newExpanded.has(folderPath)) {
      newExpanded.delete(folderPath);
    } else {
      newExpanded.add(folderPath);
    }
    setExpandedFolders(newExpanded);
  };

  const handleFileClick = async (filePath: string) => {
    setSelectedFile(filePath);
    // Switch to code tab and ensure split view shows editor
    if (activeTab === 'preview' && !splitView) {
      setActiveTab('generation');
    }
  };

  // Save file from Monaco editor
  const handleFileSave = useCallback(async (filePath: string, content: string) => {
    // Update local state
    setSandboxFiles(prev => ({ ...prev, [filePath]: content }));
    // Also update generation files state for display
    setGenerationProgress(prev => ({
      ...prev,
      files: prev.files.map(f => f.path === filePath ? { ...f, content, edited: true } : f)
    }));
    // Save to sandbox
    try {
      await fetch('/api/save-file', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: filePath, content }),
      });
      // Refresh preview
      if (iframeRef.current) {
        iframeRef.current.src = iframeRef.current.src;
      }
    } catch (e) {
      console.error('Failed to save file:', e);
    }
  }, []);

  // Write snapshot files back to the sandbox on disk so the Vite preview stays in sync
  const syncSnapshotToSandbox = useCallback(async (files: Record<string, string>) => {
    setSandboxFiles(files);
    // Write each file back to the sandbox provider
    const writes = Object.entries(files).map(([path, content]) =>
      fetch('/api/save-file', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path, content }),
      }).catch(() => {/* best-effort */})
    );
    await Promise.all(writes);
    // Refresh preview after files are written
    setTimeout(() => {
      if (iframeRef.current) iframeRef.current.src = iframeRef.current.src;
    }, 400);
  }, []);

  // Undo handler
  const handleUndo = useCallback(() => {
    const snapshot = undo();
    if (snapshot) {
      syncSnapshotToSandbox(snapshot.files);
    }
  }, [undo, syncSnapshotToSandbox]);

  // Redo handler
  const handleRedo = useCallback(() => {
    const snapshot = redo();
    if (snapshot) {
      syncSnapshotToSandbox(snapshot.files);
    }
  }, [redo, syncSnapshotToSandbox]);

  // Keyboard shortcut: Ctrl+Z undo, Ctrl+Shift+Z / Ctrl+Y redo (project-level)
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      if (target.closest('.monaco-editor')) return;
      
      if ((e.ctrlKey || e.metaKey) && e.key === 'z' && !e.shiftKey) {
        e.preventDefault();
        handleUndo();
      } else if ((e.ctrlKey || e.metaKey) && (e.key === 'y' || (e.key === 'z' && e.shiftKey))) {
        e.preventDefault();
        handleRedo();
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [handleUndo, handleRedo]);

  // Restore version from timeline
  const handleRestoreVersion = useCallback((id: string) => {
    const snapshot = restoreTo(id);
    if (snapshot) {
      setShowVersionHistory(false);
      syncSnapshotToSandbox(snapshot.files);
    }
  }, [restoreTo, syncSnapshotToSandbox]);

  // Import project handler
  const handleImportProject = useCallback((files: Record<string, string>) => {
    setSandboxFiles(files);
    setShowImportModal(false);
    // Refresh preview
    setTimeout(() => {
      if (iframeRef.current) iframeRef.current.src = iframeRef.current.src;
    }, 500);
  }, []);

  // Get content for a file path
  const getFileContent = useCallback((filePath: string): string => {
    // Try generation files first
    const genFile = generationProgress.files.find(f => f.path === filePath);
    if (genFile?.content) return genFile.content;
    // Then sandbox files with various path formats
    return sandboxFiles[filePath] || sandboxFiles['./' + filePath] || sandboxFiles['/' + filePath] || '';
  }, [generationProgress.files, sandboxFiles]);

  const getFileIcon = (fileName: string) => {
    const ext = fileName.split('.').pop()?.toLowerCase();
    
    if (ext === 'jsx' || ext === 'js') {
      return <SiJavascript style={{ width: '16px', height: '16px' }} className="text-yellow-500" />;
    } else if (ext === 'tsx' || ext === 'ts') {
      return <SiReact style={{ width: '16px', height: '16px' }} className="text-blue-500" />;
    } else if (ext === 'css') {
      return <SiCss3 style={{ width: '16px', height: '16px' }} className="text-blue-500" />;
    } else if (ext === 'json') {
      return <SiJson style={{ width: '16px', height: '16px' }} className="text-gray-400" />;
    } else {
      return <FiFile style={{ width: '16px', height: '16px' }} className="text-gray-400" />;
    }
  };

//   const clearChatHistory = () => {
//     setChatMessages([{
//       content: 'Chat history cleared. How can I help you?',
//       type: 'system',
//       timestamp: new Date()
//     }]);
//   };
// 

//   const cloneWebsite = async () => {
//     let url = urlInput.trim();
//     if (!url) {
//       setUrlStatus(prev => [...prev, 'Please enter a URL']);
//       return;
//     }
//     
//     if (!url.match(/^https?:\/\//i)) {
//       url = 'https://' + url;
//     }
//     
//     setUrlStatus([`Using: ${url}`, 'Starting to scrape...']);
//     
//     setUrlOverlayVisible(false);
//     
//     // Remove protocol for cleaner display
//     const cleanUrl = url.replace(/^https?:\/\//i, '');
//     addChatMessage(`Starting to clone ${cleanUrl}...`, 'system');
//     
//     // Capture screenshot immediately and switch to preview tab
//     captureUrlScreenshot(url);
//     
//     try {
//       addChatMessage('Scraping website content...', 'system');
//       const scrapeResponse = await fetch('/api/scrape-url-enhanced', {
//         method: 'POST',
//         headers: { 'Content-Type': 'application/json' },
//         body: JSON.stringify({ url })
//       });
//       
//       if (!scrapeResponse.ok) {
//         throw new Error(`Scraping failed: ${scrapeResponse.status}`);
//       }
//       
//       const scrapeData = await scrapeResponse.json();
//       
//       if (!scrapeData.success) {
//         throw new Error(scrapeData.error || 'Failed to scrape website');
//       }
//       
//       addChatMessage(`Scraped ${scrapeData.content.length} characters from ${url}`, 'system');
//       
//       // Clear preparing design state and switch to generation tab
//       setIsPreparingDesign(false);
//       setActiveTab('generation');
//       
//       setConversationContext(prev => ({
//         ...prev,
//         scrapedWebsites: [...prev.scrapedWebsites, {
//           url,
//           content: scrapeData,
//           timestamp: new Date()
//         }],
//         currentProject: `Clone of ${url}`
//       }));
//       
//       // Start sandbox creation in parallel with code generation
//       let sandboxPromise: Promise<any> | null = null;
//       if (!sandboxData) {
//         addChatMessage('Creating sandbox while generating your React app...', 'system');
//         sandboxPromise = createSandbox(true);
//       }
//       
//       addChatMessage('Analyzing and generating React recreation...', 'system');
//       
//       const recreatePrompt = `I scraped this website and want you to recreate it as a modern React application.
// 
// URL: ${url}
// 
// SCRAPED CONTENT:
// ${scrapeData.content}
// 
// ${homeContextInput ? `ADDITIONAL CONTEXT/REQUIREMENTS FROM USER:
// ${homeContextInput}
// 
// Please incorporate these requirements into the design and implementation.` : ''}
// 
// REQUIREMENTS:
// 1. Create a COMPLETE React application with App.jsx as the main component
// 2. App.jsx MUST import and render all other components
// 3. Recreate the main sections and layout from the scraped content
// 4. ${homeContextInput ? `Apply the user's context/theme: "${homeContextInput}"` : `Use a modern dark theme with excellent contrast:
//    - Background: #0a0a0a
//    - Text: #ffffff
//    - Links: #60a5fa
//    - Accent: #3b82f6`}
// 5. Make it fully responsive
// 6. Include hover effects and smooth transitions
// 7. Create separate components for major sections (Header, Hero, Features, etc.)
// 8. Use semantic HTML5 elements
// 
// IMPORTANT CONSTRAINTS:
// - DO NOT use React Router or any routing libraries
// - Use regular <a> tags with href="#section" for navigation, NOT Link or NavLink components
// - This is a single-page application, no routing needed
// - ALWAYS create src/App.jsx that imports ALL components
// - Each component should be in src/components/
// - Use Tailwind CSS for ALL styling (no custom CSS files)
// - Make sure the app actually renders visible content
// - Create ALL components that you reference in imports
// 
// IMAGE HANDLING RULES:
// - When the scraped content includes images, USE THE ORIGINAL IMAGE URLS whenever appropriate
// - Keep existing images from the scraped site (logos, product images, hero images, icons, etc.)
// - Use the actual image URLs provided in the scraped content, not placeholders
// - Only use placeholder images or generic services when no real images are available
// - For company logos and brand images, ALWAYS use the original URLs to maintain brand identity
// - If scraped data contains image URLs, include them in your img tags
// - Example: If you see "https://example.com/logo.png" in the scraped content, use that exact URL
// 
// Focus on the key sections and content, making it clean and modern while preserving visual assets.`;
//       
//       setGenerationProgress(prev => ({
//         isGenerating: true,
//         status: 'Initializing AI...',
//         components: [],
//         currentComponent: 0,
//         streamedCode: '',
//         isStreaming: true,
//         isThinking: false,
//         thinkingText: undefined,
//         thinkingDuration: undefined,
//         // Keep previous files until new ones are generated
//         files: prev.files || [],
//         currentFile: undefined,
//         lastProcessedPosition: 0
//       }));
//       
//       // Switch to generation tab when starting
//       setActiveTab('generation');
//       
//       const aiResponse = await fetch('/api/generate-ai-code-stream', {
//         method: 'POST',
//         headers: { 'Content-Type': 'application/json' },
//         body: JSON.stringify({
//           prompt: recreatePrompt,
//           model: aiModel,
//           context: {
//             sandboxId: sandboxData?.id,
//             structure: structureContent,
//             conversationContext: conversationContext
//           }
//         })
//       });
//       
//       if (!aiResponse.ok) {
//         throw new Error(`AI generation failed: ${aiResponse.status}`);
//       }
//       
//       const reader = aiResponse.body?.getReader();
//       const decoder = new TextDecoder();
//       let generatedCode = '';
//       let explanation = '';
//       
//       if (reader) {
//         while (true) {
//           const { done, value } = await reader.read();
//           if (done) break;
//           
//           const chunk = decoder.decode(value);
//           const lines = chunk.split('\n');
//           
//           for (const line of lines) {
//             if (line.startsWith('data: ')) {
//               try {
//                 const data = JSON.parse(line.slice(6));
//                 
//                 if (data.type === 'status') {
//                   setGenerationProgress(prev => ({ ...prev, status: data.message }));
//                 } else if (data.type === 'thinking') {
//                   setGenerationProgress(prev => ({ 
//                     ...prev, 
//                     isThinking: true,
//                     thinkingText: (prev.thinkingText || '') + data.text
//                   }));
//                 } else if (data.type === 'thinking_complete') {
//                   setGenerationProgress(prev => ({ 
//                     ...prev, 
//                     isThinking: false,
//                     thinkingDuration: data.duration
//                   }));
//                 } else if (data.type === 'conversation') {
//                   // Add conversational text to chat only if it's not code
//                   let text = data.text || '';
//                   
//                   // Remove package tags from the text
//                   text = text.replace(/<package>[^<]*<\/package>/g, '');
//                   text = text.replace(/<packages>[^<]*<\/packages>/g, '');
//                   
//                   // Filter out any XML tags and file content that slipped through
//                   if (!text.includes('<file') && !text.includes('import React') && 
//                       !text.includes('export default') && !text.includes('className=') &&
//                       text.trim().length > 0) {
//                     addChatMessage(text.trim(), 'ai');
//                   }
//                 } else if (data.type === 'stream' && data.raw) {
//                   setGenerationProgress(prev => ({ 
//                     ...prev, 
//                     streamedCode: prev.streamedCode + data.text,
//                     lastProcessedPosition: prev.lastProcessedPosition || 0
//                   }));
//                 } else if (data.type === 'component') {
//                   setGenerationProgress(prev => ({
//                     ...prev,
//                     status: `Generated ${data.name}`,
//                     components: [...prev.components, { 
//                       name: data.name,
//                       path: data.path,
//                       completed: true
//                     }],
//                     currentComponent: prev.currentComponent + 1
//                   }));
//                 } else if (data.type === 'complete') {
//                   generatedCode = data.generatedCode;
//                   explanation = data.explanation;
//                   
//                   // Save the last generated code
//                   setConversationContext(prev => ({
//                     ...prev,
//                     lastGeneratedCode: generatedCode
//                   }));
//                 }
//               } catch (e) {
//                 console.error('Error parsing streaming data:', e);
//               }
//             }
//           }
//         }
//       }
//       
//       setGenerationProgress(prev => ({
//         ...prev,
//         isGenerating: false,
//         isStreaming: false,
//         status: 'Generation complete!',
//         isEdit: prev.isEdit
//       }));
//       
//       if (generatedCode) {
//         addChatMessage('AI recreation generated!', 'system');
//         
//         // Add the explanation to chat if available
//         if (explanation && explanation.trim()) {
//           addChatMessage(explanation, 'ai');
//         }
//         
//         setPromptInput(generatedCode);
//         // Don't show the Generated Code panel by default
//         // setLeftPanelVisible(true);
//         
//         // Wait for sandbox creation if it's still in progress
//         let activeSandboxData = sandboxData;
//         if (sandboxPromise) {
//           addChatMessage('Waiting for sandbox to be ready...', 'system');
//           try {
//             const newSandboxData = await sandboxPromise;
//             if (newSandboxData) {
//               activeSandboxData = newSandboxData;
//             }
//             // Remove the waiting message
//             setChatMessages(prev => prev.filter(msg => msg.content !== 'Waiting for sandbox to be ready...'));
//           } catch (error: any) {
//             addChatMessage('Sandbox creation failed. Cannot apply code.', 'system');
//             throw error;
//           }
//         }
//         
//         // Only apply code if we have sandbox data
//         if (activeSandboxData) {
//           // First application for cloned site should not be in edit mode
//           await applyGeneratedCode(generatedCode, false);
//         }
//         
//         addChatMessage(
//           `Successfully recreated ${url} as a modern React app${homeContextInput ? ` with your requested context: "${homeContextInput}"` : ''}! The scraped content is now in my context, so you can ask me to modify specific sections or add features based on the original site.`, 
//           'ai',
//           {
//             scrapedUrl: url,
//             scrapedContent: scrapeData,
//             generatedCode: generatedCode
//           }
//         );
//         
//         setUrlInput('');
//         setUrlStatus([]);
//         setHomeContextInput('');
//         
//         // Clear generation progress and all screenshot/design states
//         setGenerationProgress(prev => ({
//           ...prev,
//           isGenerating: false,
//           isStreaming: false,
//           status: 'Generation complete!'
//         }));
//         
//         // Clear screenshot and preparing design states to prevent them from showing on next run
//         setUrlScreenshot(null);
//         setIsPreparingDesign(false);
//         setTargetUrl('');
//         setScreenshotError(null);
//         setLoadingStage(null); // Clear loading stage
//         setShowLoadingBackground(false); // Clear loading background
//         
//         setTimeout(() => {
//           // Switch back to preview tab but keep files
//           setActiveTab('preview');
//         }, 1000); // Show completion briefly then switch
//       } else {
//         throw new Error('Failed to generate recreation');
//       }
//       
//     } catch (error: any) {
//       addChatMessage(`Failed to clone website: ${error.message}`, 'system');
//       setUrlStatus([]);
//       setIsPreparingDesign(false);
//       // Clear all states on error
//       setUrlScreenshot(null);
//       setTargetUrl('');
//       setScreenshotError(null);
//       setLoadingStage(null);
//       setGenerationProgress(prev => ({
//         ...prev,
//         isGenerating: false,
//         isStreaming: false,
//         status: '',
//         // Keep files to display in sidebar
//         files: prev.files
//       }));
//       setActiveTab('preview');
//     }
//   };

  const captureUrlScreenshot = async (url: string) => {
    setIsCapturingScreenshot(true);
    setScreenshotError(null);
    try {
      const response = await fetch('/api/scrape-screenshot', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url })
      });
      
      const data = await response.json();
      if (data.success && data.screenshot) {
        setIsScreenshotLoaded(false); // Reset loaded state for new screenshot
        setUrlScreenshot(data.screenshot);
        // Set preparing design state
        setIsPreparingDesign(true);
        // Store the clean URL for display
        const cleanUrl = url.replace(/^https?:\/\//i, '');
        setTargetUrl(cleanUrl);
        // Switch to preview tab to show the screenshot
        if (activeTab !== 'preview') {
          setActiveTab('preview');
        }
      } else {
        setScreenshotError(data.error || 'Failed to capture screenshot');
      }
    } catch (error) {
      console.error('Failed to capture screenshot:', error);
      setScreenshotError('Network error while capturing screenshot');
    } finally {
      setIsCapturingScreenshot(false);
    }
  };

  const handleHomeScreenSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    await startGeneration();
  };

  const startGeneration = async () => {
    if (!homeUrlInput.trim()) return;
    
    setHomeScreenFading(true);
    
    // Set immediate loading state for better UX
    setIsStartingNewGeneration(true);
    setLoadingStage('gathering');
    
    // Immediately switch to preview tab to show loading
    setActiveTab('preview');
    
    // Set loading background to ensure proper visual feedback
    setShowLoadingBackground(true);
    
    // Clear messages and immediately show the initial message
    setChatMessages([]);
    let displayUrl = homeUrlInput.trim();
    if (!displayUrl.match(/^https?:\/\//i)) {
      displayUrl = 'https://' + displayUrl;
    }
    // Remove protocol for cleaner display
    const cleanUrl = displayUrl.replace(/^https?:\/\//i, '');

    // Check if we're in brand extension mode
    const brandExtensionMode = sessionStorage.getItem('brandExtensionMode') === 'true';

    addChatMessage(
      brandExtensionMode
        ? `Analyzing brand from ${cleanUrl}...`
        : `Starting to clone ${cleanUrl}...`,
      'system'
    );
    
    // Start creating sandbox and capturing screenshot immediately in parallel
    const sandboxPromise = !sandboxData ? createSandbox(true) : Promise.resolve(null);
    
    // Set loading stage immediately before hiding home screen
    setLoadingStage('gathering');
    // Also ensure we're on preview tab to show the loading overlay
    setActiveTab('preview');
    
    // Always capture screenshot for new URLs, even if sandbox exists
    // This ensures the loading screen shows properly
    captureUrlScreenshot(displayUrl);
    
    setTimeout(async () => {
      setShowHomeScreen(false);
      setHomeScreenFading(false);
      
      // Clear the starting flag after transition
      setTimeout(() => {
        setIsStartingNewGeneration(false);
      }, 1000);
      
      // Wait for sandbox to be ready (if it's still creating)
      const createdSandbox = await sandboxPromise;
      
      // Now start the clone process which will stream the generation
      setUrlInput(homeUrlInput);
      setUrlOverlayVisible(false); // Make sure overlay is closed
      setUrlStatus(['Scraping website content...']);
      
      try {
        // Scrape the website
        let url = homeUrlInput.trim();
        if (!url.match(/^https?:\/\//i)) {
          url = 'https://' + url;
        }

        // Check if we're in brand extension mode
        const brandExtensionMode = sessionStorage.getItem('brandExtensionMode') === 'true';
        const brandExtensionPrompt = sessionStorage.getItem('brandExtensionPrompt') || '';

        // Screenshot is already being captured in parallel above

        let scrapeData: ScrapeData | undefined;
        let brandGuidelines: any;

        if (brandExtensionMode) {
          // === BRAND EXTENSION MODE ===
          addChatMessage('Extracting brand styles from the website...', 'system');

          // Call the brand extraction endpoint
          const extractResponse = await fetch('/api/extract-brand-styles', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              url,
              prompt: brandExtensionPrompt
            })
          });

          if (!extractResponse.ok) {
            throw new Error('Failed to extract brand styles');
          }

          brandGuidelines = await extractResponse.json();

          if (!brandGuidelines.success) {
            throw new Error(brandGuidelines.error || 'Failed to extract brand styles');
          }

          // Display branding summary with visual UI
          addChatMessage(`Acquired branding format from ${cleanUrl}`, 'system', {
            brandingData: brandGuidelines.guidelines,
            sourceUrl: cleanUrl
          });
          addChatMessage(`Building your custom component using these brand guidelines...`, 'system');

          // Clear the flags after use
          sessionStorage.removeItem('brandExtensionMode');
          sessionStorage.removeItem('brandExtensionPrompt');

        } else {
          // === NORMAL CLONE MODE ===
          // Check if we have pre-scraped markdown content from search results
          const storedMarkdown = sessionStorage.getItem('siteMarkdown');
        if (storedMarkdown) {
          // Use the pre-scraped content
          scrapeData = {
            success: true,
            content: storedMarkdown,
            title: new URL(url).hostname,
            source: 'search-result'
          };
          sessionStorage.removeItem('siteMarkdown'); // Clear after use
          addChatMessage('Using cached content from search results...', 'system');
        } else {
          // Perform fresh scraping
          const scrapeResponse = await fetch('/api/scrape-url-enhanced', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url })
          });
          
          if (!scrapeResponse.ok) {
            throw new Error('Failed to scrape website');
          }
          
          scrapeData = await scrapeResponse.json() as ScrapeData;
          
          if (!scrapeData.success) {
            throw new Error(scrapeData.error || 'Failed to scrape website');
          }
        }
        }

        setUrlStatus(brandExtensionMode ? ['Brand styles extracted!', 'Building your component...'] : ['Website scraped successfully!', 'Generating React app...']);

        // Clear preparing design state and switch to generation tab
        setIsPreparingDesign(false);
        setIsScreenshotLoaded(false); // Reset loaded state
        setUrlScreenshot(null); // Clear screenshot when starting generation
        setTargetUrl(''); // Clear target URL

        // Update loading stage to planning
        setLoadingStage('planning');

        // Brief pause before switching to generation tab
        setTimeout(() => {
          setLoadingStage('generating');
          setActiveTab('generation');
        }, 1500);

        // Build the appropriate prompt based on mode
        let prompt;

        if (brandExtensionMode && brandGuidelines) {
          // === BRAND EXTENSION PROMPT ===
          // Store brand guidelines in conversation context
          setConversationContext(prev => ({
            ...prev,
            scrapedWebsites: [...prev.scrapedWebsites, {
              url: url,
              content: { brandGuidelines },
              timestamp: new Date()
            }],
            currentProject: `Custom build using ${url} brand`
          }));

          // Extract comprehensive brand data
          const branding = brandGuidelines.guidelines;

          // Build detailed brand instruction string
          const brandInstructions = `
BRAND GUIDELINES FROM ${url}:

COLOR SYSTEM:
- Color Scheme: ${branding.colorScheme || 'light'} mode
- Primary Color: ${branding.colors?.primary || 'not specified'}
- Accent Color: ${branding.colors?.accent || 'not specified'}
- Background: ${branding.colors?.background || 'not specified'}
- Text Primary: ${branding.colors?.textPrimary || 'not specified'}
- Link Color: ${branding.colors?.link || 'not specified'}

TYPOGRAPHY:
- Primary Font: ${branding.typography?.fontFamilies?.primary || 'system default'}
- Heading Font: ${branding.typography?.fontFamilies?.heading || 'system default'}
- Font Stack (Body): ${branding.typography?.fontStacks?.body?.join(', ') || 'system-ui, sans-serif'}
- Font Stack (Heading): ${branding.typography?.fontStacks?.heading?.join(', ') || 'system-ui, sans-serif'}
- H1 Size: ${branding.typography?.fontSizes?.h1 || '36px'}
- H2 Size: ${branding.typography?.fontSizes?.h2 || '30px'}
- Body Size: ${branding.typography?.fontSizes?.body || '16px'}

SPACING & LAYOUT:
- Base Spacing Unit: ${branding.spacing?.baseUnit || '4'}px
- Border Radius: ${branding.spacing?.borderRadius || '6px'}

BUTTON STYLES:
Primary Button:
  - Background: ${branding.components?.buttonPrimary?.background || branding.colors?.primary}
  - Text Color: ${branding.components?.buttonPrimary?.textColor || '#FFFFFF'}
  - Border Radius: ${branding.components?.buttonPrimary?.borderRadius || branding.spacing?.borderRadius || '8px'}
  - Shadow: ${branding.components?.buttonPrimary?.shadow || 'none'}

Secondary Button:
  - Background: ${branding.components?.buttonSecondary?.background || '#F9F9F9'}
  - Text Color: ${branding.components?.buttonSecondary?.textColor || branding.colors?.textPrimary}
  - Border Radius: ${branding.components?.buttonSecondary?.borderRadius || branding.spacing?.borderRadius || '8px'}
  - Shadow: ${branding.components?.buttonSecondary?.shadow || 'none'}

INPUT FIELDS:
- Border Color: ${branding.components?.input?.borderColor || '#CCCCCC'}
- Border Radius: ${branding.components?.input?.borderRadius || branding.spacing?.borderRadius || '6px'}

BRAND PERSONALITY:
- Tone: ${branding.personality?.tone || 'professional'}
- Energy: ${branding.personality?.energy || 'medium'}
- Target Audience: ${branding.personality?.targetAudience || 'general'}

DESIGN SYSTEM:
- Framework: ${branding.designSystem?.framework || 'tailwind'}
- Component Library: ${branding.designSystem?.componentLibrary || 'custom'}

ASSETS:
${branding.images?.logo ? `- Logo Available: Yes (use carefully if needed)` : '- Logo: Not available'}
${branding.images?.favicon ? `- Favicon: ${branding.images.favicon}` : ''}`;

          prompt = `I want you to build a NEW React component/application based on these brand guidelines and the user's requirements.

<branding-format source="${url}">
${brandInstructions}

RAW BRAND DATA (for reference):
${JSON.stringify(branding, null, 2)}
</branding-format>

USER'S REQUEST:
${brandExtensionPrompt || 'Build a modern web component using these brand guidelines'}

IMPORTANT: The content above in the <branding-format> tags contains the extracted brand guidelines from ${url}.
Use these guidelines (colors, fonts, spacing, design patterns) to build what the user requested.

CRITICAL REQUIREMENTS:
- DO NOT recreate the original website at ${url}
- DO create a COMPLETELY NEW component that fulfills the user's request
- The user wants: "${brandExtensionPrompt}"
- Build ONLY what the user requested - nothing more
- App.jsx should render ONLY the requested component - no extra Header/Footer/Hero unless specifically requested
- Make it a minimal, focused implementation of the user's request

STYLING REQUIREMENTS:
- Apply the EXACT colors from the brand palette (primary, accent, background, text colors)
- Use the EXACT typography (font families, font sizes for h1, h2, body)
- Apply the spacing system (base unit: ${branding.spacing?.baseUnit || '4'}px)
- Use the specified border radius (${branding.spacing?.borderRadius || '6px'}) consistently
- Implement button styles EXACTLY as specified (colors, shadows, border radius)
- Style input fields with the exact border color and border radius
- Match the brand's ${branding.colorScheme || 'light'} color scheme
- Apply the brand personality: ${branding.personality?.tone || 'professional'} tone with ${branding.personality?.energy || 'medium'} energy
- Use Tailwind CSS with inline color values matching the brand palette EXACTLY
- If fonts need to be imported, add @import or @font-face rules to index.css
- Create custom CSS classes in index.css for complex shadows/effects that can't be done with Tailwind

FONT SETUP:
${branding.typography?.fontFamilies?.primary ? `
- Add font family "${branding.typography.fontFamilies.primary}" to your CSS
- Use font stack: ${branding.typography?.fontStacks?.body?.join(', ') || 'system-ui, sans-serif'}
- Set body font size to ${branding.typography?.fontSizes?.body || '16px'}` : '- Use system fonts'}

COMPONENT STRUCTURE:
- src/index.css - Include brand fonts, custom shadows/effects, and base styling
- src/App.jsx - Should ONLY render the requested component (e.g., just <PricingPage /> if user wants pricing)
- src/components/[RequestedComponent].jsx - The actual component fulfilling the user's request

TECHNICAL REQUIREMENTS:
- Create a WORKING, self-contained application
- DO NOT import components that don't exist
- Make sure the app renders immediately with visible content
- All colors must match the brand palette EXACTLY
- All spacing must use the ${branding.spacing?.baseUnit || '4'}px base unit
- Buttons must have the exact styling specified in the guidelines

Focus on building something NEW, minimal, and functional that perfectly matches the ${brandGuidelines.styleName || 'brand'} aesthetic and design system.`;

        } else {
          // === NORMAL CLONE MODE PROMPT ===
          // Store scraped data in conversation context
          if (!scrapeData) {
            throw new Error('Scrape data is missing');
          }
          setConversationContext(prev => ({
            ...prev,
            scrapedWebsites: [...prev.scrapedWebsites, {
              url: url,
              content: scrapeData,
              timestamp: new Date()
            }],
            currentProject: `${url} Clone`
          }));

          // Filter out style-related context when using screenshot/URL-based generation
          // Only keep user's explicit instructions, not inherited styles
          let filteredContext = homeContextInput;
          if (homeUrlInput && homeContextInput) {
            // Check if the context contains default style names that shouldn't be inherited
            const stylePatterns = [
              'Glassmorphism style design',
              'Neumorphism style design',
              'Brutalism style design',
              'Minimalist style design',
              'Dark Mode style design',
              'Gradient Rich style design',
              '3D Depth style design',
              'Retro Wave style design',
              'Modern clean and minimalist style design',
              'Fun colorful and playful style design',
              'Corporate professional and sleek style design',
              'Creative artistic and unique style design'
            ];

            // If the context exactly matches or starts with a style pattern, filter it out
            const startsWithStyle = stylePatterns.some(pattern =>
              homeContextInput.trim().startsWith(pattern)
            );

            if (startsWithStyle) {
              // Extract only the additional instructions part after the style
              const additionalMatch = homeContextInput.match(/\. (.+)$/);
              filteredContext = additionalMatch ? additionalMatch[1] : '';
            }
          }

          prompt = `I want to recreate the ${url} website as a complete React application based on the scraped content below.

${JSON.stringify(scrapeData, null, 2)}

${filteredContext ? `ADDITIONAL CONTEXT/REQUIREMENTS FROM USER:
${filteredContext}

Please incorporate these requirements into the design and implementation.` : ''}

IMPORTANT INSTRUCTIONS:
- Create a COMPLETE, working React application
- Implement ALL sections and features from the original site
- Use Tailwind CSS for all styling (no custom CSS files)
- Make it responsive and modern
- Ensure all text content matches the original
- Create proper component structure
- Make sure the app actually renders visible content
- Create ALL components that you reference in imports
${filteredContext ? '- Apply the user\'s context/theme requirements throughout the application' : ''}

Focus on the key sections and content, making it clean and modern.`;
        }

        setGenerationProgress(prev => ({
          isGenerating: true,
          status: 'Initializing AI...',
          components: [],
          currentComponent: 0,
          streamedCode: '',
          isStreaming: true,
          isThinking: false,
          thinkingText: undefined,
          thinkingDuration: undefined,
          // Keep previous files until new ones are generated
          files: prev.files || [],
          currentFile: undefined,
          lastProcessedPosition: 0
        }));
        
        const aiResponse = await fetch('/api/generate-ai-code-stream', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ 
            prompt,
            model: aiModel,
            template: selectedTemplate,
            context: {
              sandboxId: sandboxData?.sandboxId,
              structure: structureContent,
              conversationContext: conversationContext
            }
          })
        });
        
        if (!aiResponse.ok || !aiResponse.body) {
          throw new Error('Failed to generate code');
        }
        
        const reader = aiResponse.body.getReader();
        const decoder = new TextDecoder();
        let generatedCode = '';
        let explanation = '';
        
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          
          const chunk = decoder.decode(value);
          const lines = chunk.split('\n');
          
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6));
                
                if (data.type === 'status') {
                  setGenerationProgress(prev => ({ ...prev, status: data.message }));
                } else if (data.type === 'thinking') {
                  setGenerationProgress(prev => ({ 
                    ...prev, 
                    isThinking: true,
                    thinkingText: (prev.thinkingText || '') + data.text
                  }));
                } else if (data.type === 'thinking_complete') {
                  setGenerationProgress(prev => ({ 
                    ...prev, 
                    isThinking: false,
                    thinkingDuration: data.duration
                  }));
                } else if (data.type === 'conversation') {
                  // Add conversational text to chat only if it's not code
                  let text = data.text || '';
                  
                  // Remove package tags from the text
                  text = text.replace(/<package>[^<]*<\/package>/g, '');
                  text = text.replace(/<packages>[^<]*<\/packages>/g, '');
                  
                  // Filter out any XML tags and file content that slipped through
                  if (!text.includes('<file') && !text.includes('import React') && 
                      !text.includes('export default') && !text.includes('className=') &&
                      text.trim().length > 0) {
                    addChatMessage(text.trim(), 'ai');
                  }
                } else if (data.type === 'stream' && data.raw) {
                  setGenerationProgress(prev => {
                    const newStreamedCode = prev.streamedCode + data.text;
                    
                    // Tab is already switched after scraping
                    
                    const updatedState = { 
                      ...prev, 
                      streamedCode: newStreamedCode,
                      isStreaming: true,
                      isThinking: false,
                      status: 'Generating code...'
                    };
                    
                    // Process complete files from the accumulated stream
                    const fileRegex = /<file path="([^"]+)">([^]*?)<\/file>/g;
                    let match;
                    const processedFiles = new Set(prev.files.map(f => f.path));
                    
                    while ((match = fileRegex.exec(newStreamedCode)) !== null) {
                      const filePath = match[1];
                      const fileContent = match[2];
                      
                      // Only add if we haven't processed this file yet
                      if (!processedFiles.has(filePath)) {
                        const fileExt = filePath.split('.').pop() || '';
                        const fileType = fileExt === 'jsx' || fileExt === 'js' ? 'javascript' :
                                        fileExt === 'css' ? 'css' :
                                        fileExt === 'json' ? 'json' :
                                        fileExt === 'html' ? 'html' : 'text';
                        
                        // Check if file already exists
                        const existingFileIndex = updatedState.files.findIndex(f => f.path === filePath);
                        
                        if (existingFileIndex >= 0) {
                          // Update existing file and mark as edited
                          updatedState.files = [
                            ...updatedState.files.slice(0, existingFileIndex),
                            {
                              ...updatedState.files[existingFileIndex],
                              content: fileContent.trim(),
                              type: fileType,
                              completed: true,
                              edited: true
                            },
                            ...updatedState.files.slice(existingFileIndex + 1)
                          ];
                        } else {
                          // Add new file
                          updatedState.files = [...updatedState.files, {
                            path: filePath,
                            content: fileContent.trim(),
                            type: fileType,
                            completed: true,
                            edited: false
                          }];
                        }
                        
                        // Only show file status if not in edit mode
                        if (!prev.isEdit) {
                          updatedState.status = `Completed ${filePath}`;
                        }
                        processedFiles.add(filePath);
                      }
                    }
                    
                    // Check for current file being generated (incomplete file at the end)
                    const lastFileMatch = newStreamedCode.match(/<file path="([^"]+)">([^]*?)$/);
                    if (lastFileMatch && !lastFileMatch[0].includes('</file>')) {
                      const filePath = lastFileMatch[1];
                      const partialContent = lastFileMatch[2];
                      
                      if (!processedFiles.has(filePath)) {
                        const fileExt = filePath.split('.').pop() || '';
                        const fileType = fileExt === 'jsx' || fileExt === 'js' ? 'javascript' :
                                        fileExt === 'css' ? 'css' :
                                        fileExt === 'json' ? 'json' :
                                        fileExt === 'html' ? 'html' : 'text';
                        
                        updatedState.currentFile = { 
                          path: filePath, 
                          content: partialContent, 
                          type: fileType 
                        };
                        // Only show file status if not in edit mode
                        if (!prev.isEdit) {
                          updatedState.status = `Generating ${filePath}`;
                        }
                      }
                    } else {
                      updatedState.currentFile = undefined;
                    }
                    
                    return updatedState;
                  });
                } else if (data.type === 'complete') {
                  generatedCode = data.generatedCode;
                  explanation = data.explanation;
                  
                  // Save the last generated code
                  setConversationContext(prev => ({
                    ...prev,
                    lastGeneratedCode: generatedCode
                  }));
                } else if (data.type === 'tool-call') {
                  setGenerationProgress(prev => ({
                    ...prev,
                    status: `Tool: ${data.name} ${data.args?.path || ''}`.trim(),
                    toolCalls: [...(prev.toolCalls || []), { name: data.name, args: data.args, index: data.index }],
                  }));
                } else if (data.type === 'tool-result') {
                  setGenerationProgress(prev => ({
                    ...prev,
                    toolResults: [...(prev.toolResults || []), { name: data.name, output: data.output, index: data.index }],
                  }));
                } else if (data.type === 'reasoning') {
                  setGenerationProgress(prev => ({
                    ...prev,
                    status: `Reasoning: ${(data.thought as string)?.substring(0, 60) || 'thinking...'}`,
                    reasoningSteps: [...(prev.reasoningSteps || []), { thought: data.thought, next_step: data.next_step, index: data.index }],
                  }));
                }
              } catch (e) {
                console.error('Failed to parse SSE data:', e);
              }
            }
          }
        }
        
        setGenerationProgress(prev => ({
          ...prev,
          isGenerating: false,
          isStreaming: false,
          status: 'Generation complete!'
        }));
        
        if (generatedCode) {
          addChatMessage('AI recreation generated!', 'system');
          
          // Add the explanation to chat if available
          if (explanation && explanation.trim()) {
            addChatMessage(explanation, 'ai');
          }
          
          setPromptInput(generatedCode);

          // Apply the code (first time is not edit mode)
          await applyGeneratedCode(generatedCode, false);

          addChatMessage(
            brandExtensionMode
              ? `Successfully built your custom component using ${cleanUrl}'s brand guidelines! You can now ask me to modify it or add more features.`
              : `Successfully recreated ${url} as a modern React app${homeContextInput ? ` with your requested context: "${homeContextInput}"` : ''}! The scraped content is now in my context, so you can ask me to modify specific sections or add features based on the original site.`,
            'ai',
            {
              scrapedUrl: url,
              scrapedContent: brandExtensionMode ? { brandGuidelines } : scrapeData,
              generatedCode: generatedCode
            }
          );
          
          setConversationContext(prev => ({
            ...prev,
            generatedComponents: [],
            appliedCode: [...prev.appliedCode, {
              files: [],
              timestamp: new Date()
            }]
          }));
        } else {
          throw new Error('Failed to generate recreation');
        }
        
        setUrlInput('');
        setUrlStatus([]);
        setHomeContextInput('');
        
        // Clear generation progress and all screenshot/design states
        setGenerationProgress(prev => ({
          ...prev,
          isGenerating: false,
          isStreaming: false,
          status: 'Generation complete!'
        }));
        
        // Clear screenshot and preparing design states to prevent them from showing on next run
        setIsScreenshotLoaded(false); // Reset loaded state
        setUrlScreenshot(null);
        setIsPreparingDesign(false);
        setTargetUrl('');
        setScreenshotError(null);
        setLoadingStage(null); // Clear loading stage
        setIsStartingNewGeneration(false); // Clear new generation flag
        setShowLoadingBackground(false); // Clear loading background
        
        setTimeout(() => {
          // Switch back to preview tab but keep files
          setActiveTab('preview');
        }, 1000); // Show completion briefly then switch
      } catch (error: any) {
        addChatMessage(`Failed to clone website: ${error.message}`, 'system');
        setUrlStatus([]);
        setIsPreparingDesign(false);
        setIsStartingNewGeneration(false); // Clear new generation flag on error
        setLoadingStage(null);
        // Also clear generation progress on error
        setGenerationProgress(prev => ({
          ...prev,
          isGenerating: false,
          isStreaming: false,
          status: '',
          // Keep files to display in sidebar
          files: prev.files
        }));
      }
    }, 500);
  };

  return (
    <HeaderProvider>
      <div className="font-sans bg-background text-foreground h-screen flex flex-col">
      <div className="bg-[#0a0a1a] py-[15px] py-[8px] border-b border-[#2a2a4a] flex items-center justify-between shadow-sm">
        <HeaderBrandKit />
        <div className="flex items-center gap-2">
          {/* Model Selector - Left side */}
          <select
            value={aiModel}
            onChange={(e) => {
              const newModel = e.target.value;
              setAiModel(newModel);
              const params = new URLSearchParams(searchParams);
              params.set('model', newModel);
              if (sandboxData?.sandboxId) {
                params.set('sandbox', sandboxData.sandboxId);
              }
              router.push(`/generation?${params.toString()}`);
            }}
            className="px-3 py-1.5 text-sm text-gray-200 bg-[#12122a] border border-[#2a2a4a] rounded-lg focus:outline-none focus:border-[#3a3a5a] transition-colors"
          >
            {/* Cloud AI Providers */}
            <optgroup label="☁️ Cloud AI">
              {(dynamicModels.length > 0 ? dynamicModels : appConfig.ai.availableModels.map(id => ({ id, name: appConfig.ai.modelDisplayNames?.[id] || id, provider: 'cloud' })))
                .filter(m => !['ollama', 'openwebui'].includes(m.provider))
                .map(model => (
                  <option key={model.id} value={model.id}>
                    {model.name}
                  </option>
                ))}
            </optgroup>
            {/* Ollama Models */}
            {dynamicModels.some(m => m.provider === 'ollama') && (
              <optgroup label="🦙 Ollama (Local)">
                {dynamicModels
                  .filter(m => m.provider === 'ollama')
                  .map(model => (
                    <option key={model.id} value={model.id}>
                      {model.name}
                    </option>
                  ))}
              </optgroup>
            )}
            {/* OpenWebUI Models */}
            {dynamicModels.some(m => m.provider === 'openwebui') && (
              <optgroup label="🌐 OpenWebUI">
                {dynamicModels
                  .filter(m => m.provider === 'openwebui')
                  .map(model => (
                    <option key={model.id} value={model.id}>
                      {model.name}
                    </option>
                  ))}
              </optgroup>
            )}
          </select>
          {/* Template Badge */}
          <span className="px-3 py-1.5 text-xs text-orange-300 bg-[#12122a] border border-orange-500/30 rounded-lg" title={`Template: ${getTemplate(selectedTemplate)?.name || selectedTemplate}`}>
            {getTemplate(selectedTemplate)?.icon} {getTemplate(selectedTemplate)?.name || selectedTemplate}
          </span>
          <button 
            onClick={() => createSandbox()}
            className="p-8 rounded-lg transition-colors bg-[#12122a] border border-[#2a2a4a] text-gray-300 hover:bg-[#1e1e3a]"
            title="Create new sandbox"
          >
            <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
          </button>
          <button 
            onClick={reapplyLastGeneration}
            className="p-8 rounded-lg transition-colors bg-[#12122a] border border-[#2a2a4a] text-gray-300 hover:bg-[#1e1e3a] disabled:opacity-50 disabled:cursor-not-allowed"
            title="Re-apply last generation"
            disabled={!conversationContext.lastGeneratedCode || !sandboxData}
          >
            <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
          </button>
          <button 
            onClick={downloadZip}
            disabled={!sandboxData}
            className="p-8 rounded-lg transition-colors bg-[#12122a] border border-[#2a2a4a] text-gray-300 hover:bg-[#1e1e3a] disabled:opacity-50 disabled:cursor-not-allowed"
            title="Download your Vite app as ZIP"
          >
            <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M9 19l3 3m0 0l3-3m-3 3V10" />
            </svg>
          </button>
          {/* Import Project */}
          <button
            onClick={() => setShowImportModal(true)}
            className="p-8 rounded-lg transition-colors bg-[#12122a] border border-[#2a2a4a] text-gray-300 hover:bg-[#1e1e3a]"
            title="Import project (ZIP or GitHub)"
          >
            <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
            </svg>
          </button>

          {/* Divider */}
          <div className="w-px h-6 bg-[#2a2a4a]" />

          {/* Undo */}
          <button
            onClick={handleUndo}
            disabled={!canUndo}
            className="p-8 rounded-lg transition-colors bg-[#12122a] border border-[#2a2a4a] text-gray-300 hover:bg-[#1e1e3a] disabled:opacity-30 disabled:cursor-not-allowed"
            title="Undo (Ctrl+Z)"
          >
            <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 10h10a5 5 0 015 5v2M3 10l4 4M3 10l4-4" />
            </svg>
          </button>
          {/* Redo */}
          <button
            onClick={handleRedo}
            disabled={!canRedo}
            className="p-8 rounded-lg transition-colors bg-[#12122a] border border-[#2a2a4a] text-gray-300 hover:bg-[#1e1e3a] disabled:opacity-30 disabled:cursor-not-allowed"
            title="Redo (Ctrl+Y)"
          >
            <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 10H11a5 5 0 00-5 5v2M21 10l-4 4M21 10l-4-4" />
            </svg>
          </button>
          {/* Version History */}
          <button
            onClick={() => setShowVersionHistory(true)}
            disabled={snapshots.length === 0}
            className="p-8 rounded-lg transition-colors bg-[#12122a] border border-[#2a2a4a] text-gray-300 hover:bg-[#1e1e3a] disabled:opacity-30 disabled:cursor-not-allowed"
            title="Version history"
          >
            <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </button>
          {/* Split View Toggle */}
          <button
            onClick={() => setSplitView(v => !v)}
            className={`p-8 rounded-lg transition-colors border ${
              splitView
                ? 'bg-purple-600/20 border-purple-500/40 text-purple-300'
                : 'bg-[#12122a] border-[#2a2a4a] text-gray-300 hover:bg-[#1e1e3a]'
            }`}
            title={splitView ? 'Exit split view' : 'Split view (Code + Preview)'}
          >
            <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7" />
            </svg>
          </button>

          {/* Divider */}
          <div className="w-px h-6 bg-[#2a2a4a]" />

          {/* New Project / Reset */}
          <button
            onClick={() => {
              if (sandboxData && chatMessages.some(m => m.type === 'user')) {
                if (confirm('Start a new project? Current project will be saved to history.')) {
                  resetProject();
                }
              } else {
                resetProject();
              }
            }}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg transition-colors bg-[#12122a] border border-[#2a2a4a] text-gray-300 hover:bg-[#1e1e3a] hover:text-white text-xs font-medium"
            title="Start a new project (saves current to history)"
          >
            <svg width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
            New
          </button>

          {/* Project History */}
          <div className="relative" data-project-history>
            <button
              onClick={() => setShowProjectHistory(!showProjectHistory)}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg transition-colors bg-[#12122a] border border-[#2a2a4a] text-gray-300 hover:bg-[#1e1e3a] hover:text-white text-xs font-medium"
              title="Project history"
            >
              <svg width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              History
              {projectHistory.length > 0 && (
                <span className="ml-1 px-1.5 py-0.5 rounded-full bg-[#2a2a4a] text-[10px] text-gray-400">{projectHistory.length}</span>
              )}
            </button>

            {showProjectHistory && (
              <div className="absolute right-0 top-full mt-2 w-80 max-h-96 overflow-y-auto bg-[#0a0a1a] border border-[#2a2a4a] rounded-xl shadow-2xl z-50">
                <div className="p-3 border-b border-[#2a2a4a] flex items-center justify-between">
                  <span className="text-sm font-medium text-white">Project History</span>
                  <button onClick={() => setShowProjectHistory(false)} className="text-gray-500 hover:text-white text-xs">✕</button>
                </div>
                {projectHistory.length === 0 ? (
                  <div className="p-4 text-center text-gray-500 text-xs">No projects yet</div>
                ) : (
                  projectHistory.map((proj) => (
                    <div
                      key={proj.id}
                      className="p-3 border-b border-[#1a1a2e] hover:bg-[#12122a] cursor-pointer transition-colors group"
                      onClick={async () => {
                        setShowProjectHistory(false);
                        addChatMessage(`Restoring project...`, 'system');
                        setLoading(true);
                        try {
                          const res = await fetch('/api/restore-project', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ sandboxId: proj.id }),
                          });
                          const data = await res.json();
                          if (data.success) {
                            setSandboxData({ sandboxId: data.sandboxId, url: data.url, provider: 'local' });
                            setAiModel(proj.model);
                            addChatMessage(`Project restored with ${data.restoredFiles} files.`, 'system');
                          } else {
                            addChatMessage(`Could not restore: ${data.error}`, 'error');
                          }
                        } catch (e) {
                          addChatMessage(`Restore failed: ${(e as Error).message}`, 'error');
                        } finally {
                          setLoading(false);
                        }
                      }}
                    >
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-medium text-gray-200 truncate max-w-[200px]">{proj.name}</span>
                        <span className="text-[10px] text-gray-500">{new Date(proj.updatedAt).toLocaleDateString()}</span>
                      </div>
                      <div className="flex items-center gap-2 mt-1">
                        <span className="text-[10px] text-gray-500 bg-[#1a1a2e] px-1.5 py-0.5 rounded">{proj.template}</span>
                        <span className="text-[10px] text-gray-500">{proj.fileCount} files</span>
                      </div>
                    </div>
                  ))
                )}
              </div>
            )}
          </div>
       
        </div>
      </div>

      {/* Agent Status Bar - Below header, grid layout (HIDDEN) */}
      {false && activeAgents.length > 0 && (
        <div className="bg-[#0a0a1a] border-b border-[#2a2a4a] px-3 py-2">
          <div className="flex items-center gap-2 mb-1.5">
            <span className="text-xs font-medium text-gray-300 flex items-center gap-1.5">
              🤖 Agents
              <span className="text-[10px] text-gray-500">
                ({activeAgents.filter(a => a.status === 'completed').length}/{activeAgents.length} done)
              </span>
            </span>
            {activeAgents.some(a => a.status === 'working' || a.status === 'spawning') && (
              <span className="w-1.5 h-1.5 rounded-full bg-orange-400 animate-pulse" />
            )}
            <button
              onClick={() => setAgentPanelOpen(!agentPanelOpen)}
              className="ml-auto text-[10px] text-gray-500 hover:text-gray-300 transition-colors"
            >
              {agentPanelOpen ? '▲ Collapse' : '▼ Expand'}
            </button>
          </div>
          
          {/* Agent grid - always shows summary row */}
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-1.5">
            {activeAgents.map((agent) => (
              <div
                key={agent.id}
                className={`rounded-lg px-2.5 py-1.5 border transition-colors ${
                  agent.status === 'working' || agent.status === 'spawning'
                    ? 'bg-orange-500/5 border-orange-500/20'
                    : agent.status === 'completed'
                    ? 'bg-green-500/5 border-green-500/20'
                    : agent.status === 'error'
                    ? 'bg-red-500/5 border-red-500/20'
                    : 'bg-[#12122a] border-[#1e1e3a]'
                }`}
              >
                <div className="flex items-center gap-1.5">
                  <span className="text-xs">{agent.icon}</span>
                  <span className="text-[11px] font-medium text-gray-300 truncate">{agent.name}</span>
                  {agent.status === 'working' || agent.status === 'spawning' ? (
                    <span className="ml-auto w-1.5 h-1.5 rounded-full bg-orange-400 animate-pulse flex-shrink-0" />
                  ) : agent.status === 'completed' ? (
                    <span className="ml-auto text-[10px] text-green-400 flex-shrink-0">✓</span>
                  ) : agent.status === 'error' ? (
                    <span className="ml-auto text-[10px] text-red-400 flex-shrink-0">✗</span>
                  ) : null}
                </div>
                <div className="text-[9px] text-gray-500 truncate mt-0.5">
                  {agent.model.split('/').pop()}
                  {agent.durationMs ? ` · ${(agent.durationMs / 1000).toFixed(1)}s` : ''}
                </div>
              </div>
            ))}
          </div>

          {/* Expanded results */}
          {agentPanelOpen && (
            <div className="mt-2 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
              {activeAgents.filter(a => a.result || a.error).map((agent) => (
                <div
                  key={`${agent.id}-result`}
                  className="rounded-lg px-3 py-2 bg-[#12122a]/60 border border-[#1e1e3a]"
                >
                  <div className="flex items-center gap-1.5 mb-1">
                    <span className="text-xs">{agent.icon}</span>
                    <span className="text-[11px] font-medium text-gray-300">{agent.name}</span>
                    {agent.tokensUsed && (
                      <span className="ml-auto text-[9px] text-gray-600">{agent.tokensUsed} tokens</span>
                    )}
                  </div>
                  {agent.result && (
                    <pre className="text-[10px] text-gray-400 leading-relaxed max-h-32 overflow-y-auto whitespace-pre-wrap font-mono">
                      {agent.result.substring(0, 600)}{agent.result.length > 600 ? '...' : ''}
                    </pre>
                  )}
                  {agent.error && (
                    <p className="text-[10px] text-red-400/80">{agent.error}</p>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      <div className="flex-1 flex overflow-hidden">
        {/* Center Panel - AI Chat (1/3 of remaining width) */}
        <div className="flex-1 max-w-[400px] flex flex-col border-r border-border bg-background">
          {/* Sidebar Input Component */}
          {!hasInitialSubmission ? (
            <div className="p-4 border-b border-border">
              <SidebarInput
                onSubmit={(url, style, model, instructions) => {
                  // Mark that we've had an initial submission
                  setHasInitialSubmission(true);
                  
                  // Store the configuration in sessionStorage (same as home page)
                  sessionStorage.setItem('targetUrl', url);
                  sessionStorage.setItem('selectedStyle', style);
                  sessionStorage.setItem('selectedModel', model);
                  if (instructions) {
                    sessionStorage.setItem('additionalInstructions', instructions);
                  }
                  sessionStorage.setItem('autoStart', 'true');
                  
                  // Start generation using the existing logic
                  setHomeUrlInput(url);
                  setHomeContextInput(instructions || '');
                  startGeneration();
                }}
                disabled={loading || generationProgress.isGenerating}
              />
            </div>
          ) : null}

          {conversationContext.scrapedWebsites.length > 0 && (
            <div className="p-4 bg-card border-b border-gray-200">
              <div className="flex flex-col gap-4">
                {conversationContext.scrapedWebsites.map((site, idx) => {
                  // Extract favicon and site info from the scraped data
                  const metadata = site.content?.metadata || {};
                  const sourceURL = metadata.sourceURL || site.url;
                  const favicon = metadata.favicon || `https://www.google.com/s2/favicons?domain=${new URL(sourceURL).hostname}&sz=128`;
                  const siteName = metadata.ogSiteName || metadata.title || new URL(sourceURL).hostname;
                  const screenshot = site.content?.screenshot || sessionStorage.getItem('websiteScreenshot');
                  
                  return (
                    <div key={idx} className="flex flex-col gap-3">
                      {/* Site info with favicon */}
                      <div className="flex items-center gap-4 text-sm">
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img 
                          src={favicon} 
                          alt={siteName}
                          className="w-16 h-16 rounded"
                          onError={(e) => {
                            e.currentTarget.src = `https://www.google.com/s2/favicons?domain=${new URL(sourceURL).hostname}&sz=128`;
                          }}
                        />
                        <a 
                          href={sourceURL} 
                          target="_blank" 
                          rel="noopener noreferrer"
                          className="text-gray-200 hover:text-gray-100 truncate max-w-[250px] font-medium"
                          title={sourceURL}
                        >
                          {siteName}
                        </a>
                      </div>
                      
                      {/* Pinned screenshot */}
                      {screenshot && (
                        <div className="w-full">
                          <div className="flex items-center justify-between mb-2">
                            <span className="text-xs font-medium text-gray-400">Screenshot Preview</span>
                            <button
                              onClick={() => setScreenshotCollapsed(!screenshotCollapsed)}
                              className="text-gray-500 hover:text-gray-300 transition-colors p-1"
                              aria-label={screenshotCollapsed ? 'Expand screenshot' : 'Collapse screenshot'}
                            >
                              <svg
                                width="16"
                                height="16"
                                viewBox="0 0 16 16"
                                fill="none"
                                xmlns="http://www.w3.org/2000/svg"
                                className={`transition-transform duration-300 ${screenshotCollapsed ? 'rotate-180' : ''}`}
                              >
                                <path d="M4 6L8 10L12 6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                              </svg>
                            </button>
                          </div>
                          <div
                            className="w-full rounded-lg overflow-hidden border border-[#2a2a4a] transition-all duration-300"
                            style={{
                              opacity: screenshotCollapsed ? 0 : 1,
                              transform: screenshotCollapsed ? 'translateY(-20px)' : 'translateY(0)',
                              pointerEvents: screenshotCollapsed ? 'none' : 'auto',
                              maxHeight: screenshotCollapsed ? '0' : '200px'
                            }}
                          >
                            {/* eslint-disable-next-line @next/next/no-img-element */}
                            <img
                              src={screenshot}
                              alt={`${siteName} preview`}
                              className="w-full h-auto object-cover"
                              style={{ maxHeight: '200px' }}
                            />
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          <div
            className="flex-1 overflow-y-auto p-6 flex flex-col gap-4 scrollbar-hide"
            ref={chatMessagesRef}>
            {chatMessages.map((msg, idx) => {
              // Check if this message is from a successful generation
              const isGenerationComplete = msg.content.includes('Successfully recreated') || 
                                         msg.content.includes('AI recreation generated!') ||
                                         msg.content.includes('Code generated!');
              
              // Get the files from metadata if this is a completion message
              // const completedFiles = msg.metadata?.appliedFiles || [];
              
              return (
                <div key={idx} className="block">
                  <div className={`flex ${msg.type === 'user' ? 'justify-end' : 'justify-start'}`}>
                    <div className="block">
                      <div className={`block rounded-[10px] px-14 py-8 ${
                        msg.type === 'user' ? 'bg-[#36322F] text-white ml-auto max-w-[80%]' :
                        msg.type === 'ai' ? 'bg-[#1e1e3a] text-gray-200 mr-auto max-w-[80%]' :
                        msg.type === 'system' ? 'bg-[#36322F] text-white text-sm' :
                        msg.type === 'command' ? 'bg-[#36322F] text-white font-mono text-sm' :
                        msg.type === 'error' ? 'bg-red-900 text-red-100 text-sm border border-red-700' :
                        'bg-[#36322F] text-white text-sm'
                      }`}>
                    {msg.type === 'command' ? (
                      <div className="flex items-start gap-2">
                        <span className={`text-xs ${
                          msg.metadata?.commandType === 'input' ? 'text-blue-400' :
                          msg.metadata?.commandType === 'error' ? 'text-red-400' :
                          msg.metadata?.commandType === 'success' ? 'text-green-400' :
                          'text-gray-400'
                        }`}>
                          {msg.metadata?.commandType === 'input' ? '$' : '>'}
                        </span>
                        <span className="flex-1 whitespace-pre-wrap text-white">{msg.content}</span>
                      </div>
                    ) : msg.type === 'error' ? (
                      <div className="flex items-start gap-3">
                        <div className="flex-shrink-0">
                          <div className="w-8 h-8 bg-red-800 rounded-full flex items-center justify-center">
                            <svg className="w-6 h-6 text-red-200" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                            </svg>
                          </div>
                        </div>
                        <div className="flex-1">
                          <div className="font-semibold mb-1">Build Errors Detected</div>
                          <div className="whitespace-pre-wrap text-sm">{msg.content}</div>
                          <div className="mt-2 text-xs opacity-70">Press 'F' or click the Fix button above to resolve</div>
                        </div>
                      </div>
                    ) : (
                      <span className="text-sm">{msg.content}</span>
                    )}
                      </div>
                  
                      {/* Show branding data if this is a brand extraction message */}
                      {msg.metadata?.brandingData && (
                        <div className="mt-3 bg-gradient-to-br from-gray-50 to-white border-2 border-gray-200 rounded-xl overflow-hidden max-w-[500px] shadow-sm">
                          <div className="bg-[#36322F] px-16 py-12">
                            <div className="flex items-center gap-8">
                              <Image
                                src={`https://www.google.com/s2/favicons?domain=${msg.metadata.sourceUrl}&sz=32`}
                                alt=""
                                width={64}
                                height={64}
                                className="w-16 h-16"
                              />
                              <div className="text-sm font-semibold text-white">
                                Brand Guidelines
                              </div>
                            </div>
                          </div>

                          <div className="p-16">
                            {/* Color Scheme Mode */}
                            {msg.metadata.brandingData.colorScheme && (
                              <div className="mb-16">
                                <div className="text-sm">
                                  <span className="text-gray-400 font-medium">Mode:</span>{' '}
                                  <span className="font-semibold text-gray-100 capitalize">{msg.metadata.brandingData.colorScheme}</span>
                                </div>
                              </div>
                            )}

                            {/* Colors */}
                            {msg.metadata.brandingData.colors && (
                              <div className="mb-16">
                                <div className="text-sm font-semibold text-gray-100 mb-8">Colors</div>
                                <div className="flex flex-wrap gap-12">
                                  {msg.metadata.brandingData.colors.primary && (
                                    <div className="flex items-center gap-8">
                                      <div className="w-32 h-32 rounded border border-[#3a3a5a]" style={{ backgroundColor: msg.metadata.brandingData.colors.primary }} />
                                      <div className="text-sm">
                                        <div className="font-semibold text-gray-100">Primary</div>
                                        <div className="text-gray-400 font-mono text-xs">{msg.metadata.brandingData.colors.primary}</div>
                                      </div>
                                    </div>
                                  )}
                                  {msg.metadata.brandingData.colors.accent && (
                                    <div className="flex items-center gap-8">
                                      <div className="w-32 h-32 rounded border border-[#3a3a5a]" style={{ backgroundColor: msg.metadata.brandingData.colors.accent }} />
                                      <div className="text-sm">
                                        <div className="font-semibold text-gray-100">Accent</div>
                                        <div className="text-gray-400 font-mono text-xs">{msg.metadata.brandingData.colors.accent}</div>
                                      </div>
                                    </div>
                                  )}
                                  {msg.metadata.brandingData.colors.background && (
                                    <div className="flex items-center gap-8">
                                      <div className="w-32 h-32 rounded border border-[#3a3a5a]" style={{ backgroundColor: msg.metadata.brandingData.colors.background }} />
                                      <div className="text-sm">
                                        <div className="font-semibold text-gray-100">Background</div>
                                        <div className="text-gray-400 font-mono text-xs">{msg.metadata.brandingData.colors.background}</div>
                                      </div>
                                    </div>
                                  )}
                                  {msg.metadata.brandingData.colors.textPrimary && (
                                    <div className="flex items-center gap-8">
                                      <div className="w-32 h-32 rounded border border-[#3a3a5a]" style={{ backgroundColor: msg.metadata.brandingData.colors.textPrimary }} />
                                      <div className="text-sm">
                                        <div className="font-semibold text-gray-100">Text</div>
                                        <div className="text-gray-400 font-mono text-xs">{msg.metadata.brandingData.colors.textPrimary}</div>
                                      </div>
                                    </div>
                                  )}
                                </div>
                              </div>
                            )}

                            {/* Typography */}
                            {msg.metadata.brandingData.typography && (
                              <div className="mb-16">
                                <div className="text-sm font-semibold text-gray-100 mb-8">Typography</div>
                                <div className="grid grid-cols-2 gap-12 text-sm">
                                  {msg.metadata.brandingData.typography.fontFamilies?.primary && (
                                    <div>
                                      <span className="text-gray-400 font-medium">Primary:</span>{' '}
                                      <span className="font-semibold text-gray-100">{msg.metadata.brandingData.typography.fontFamilies.primary}</span>
                                    </div>
                                  )}
                                  {msg.metadata.brandingData.typography.fontFamilies?.heading && (
                                    <div>
                                      <span className="text-gray-400 font-medium">Heading:</span>{' '}
                                      <span className="font-semibold text-gray-100">{msg.metadata.brandingData.typography.fontFamilies.heading}</span>
                                    </div>
                                  )}
                                  {msg.metadata.brandingData.typography.fontSizes?.h1 && (
                                    <div>
                                      <span className="text-gray-400 font-medium">H1 Size:</span>{' '}
                                      <span className="font-semibold text-gray-100">{msg.metadata.brandingData.typography.fontSizes.h1}</span>
                                    </div>
                                  )}
                                  {msg.metadata.brandingData.typography.fontSizes?.h2 && (
                                    <div>
                                      <span className="text-gray-400 font-medium">H2 Size:</span>{' '}
                                      <span className="font-semibold text-gray-100">{msg.metadata.brandingData.typography.fontSizes.h2}</span>
                                    </div>
                                  )}
                                  {msg.metadata.brandingData.typography.fontSizes?.body && (
                                    <div>
                                      <span className="text-gray-400 font-medium">Body Size:</span>{' '}
                                      <span className="font-semibold text-gray-100">{msg.metadata.brandingData.typography.fontSizes.body}</span>
                                    </div>
                                  )}
                                </div>
                              </div>
                            )}

                            {/* Spacing */}
                            {msg.metadata.brandingData.spacing && (
                              <div className="mb-16">
                                <div className="text-sm font-semibold text-gray-100 mb-8">Spacing</div>
                                <div className="flex flex-wrap gap-16 text-sm">
                                  {msg.metadata.brandingData.spacing.baseUnit && (
                                    <div>
                                      <span className="text-gray-400 font-medium">Base Unit:</span>{' '}
                                      <span className="font-semibold text-gray-100">{msg.metadata.brandingData.spacing.baseUnit}px</span>
                                    </div>
                                  )}
                                  {msg.metadata.brandingData.spacing.borderRadius && (
                                    <div>
                                      <span className="text-gray-400 font-medium">Border Radius:</span>{' '}
                                      <span className="font-semibold text-gray-100">{msg.metadata.brandingData.spacing.borderRadius}</span>
                                    </div>
                                  )}
                                </div>
                              </div>
                            )}

                            {/* Button Styles */}
                            {msg.metadata.brandingData.components?.buttonPrimary && (
                              <div className="mb-16">
                                <div className="text-sm font-semibold text-gray-100 mb-8">Button Styles</div>
                                <div className="flex flex-wrap gap-12">
                                  <div>
                                    <div className="text-xs text-gray-400 mb-6 font-medium">Primary Button</div>
                                    <button
                                      className="px-16 py-8 text-sm font-medium"
                                      style={{
                                        backgroundColor: msg.metadata.brandingData.components.buttonPrimary.background,
                                        color: msg.metadata.brandingData.components.buttonPrimary.textColor,
                                        borderRadius: msg.metadata.brandingData.components.buttonPrimary.borderRadius,
                                        boxShadow: msg.metadata.brandingData.components.buttonPrimary.shadow
                                      }}
                                    >
                                      Sample Button
                                    </button>
                                  </div>
                                  {msg.metadata.brandingData.components?.buttonSecondary && (
                                    <div>
                                      <div className="text-xs text-gray-400 mb-6 font-medium">Secondary Button</div>
                                      <button
                                        className="px-16 py-8 text-sm font-medium"
                                        style={{
                                          backgroundColor: msg.metadata.brandingData.components.buttonSecondary.background,
                                          color: msg.metadata.brandingData.components.buttonSecondary.textColor,
                                          borderRadius: msg.metadata.brandingData.components.buttonSecondary.borderRadius,
                                          boxShadow: msg.metadata.brandingData.components.buttonSecondary.shadow
                                        }}
                                      >
                                        Sample Button
                                      </button>
                                    </div>
                                  )}
                                </div>
                              </div>
                            )}

                            {/* Personality */}
                            {msg.metadata.brandingData.personality && (
                              <div className="text-sm">
                                <span className="text-gray-400 font-medium">Personality:</span>{' '}
                                <span className="font-semibold text-gray-100 capitalize">
                                  {msg.metadata.brandingData.personality.tone} tone, {msg.metadata.brandingData.personality.energy} energy
                                </span>
                              </div>
                            )}

                            {/* Target Audience */}
                            {msg.metadata.brandingData.personality?.targetAudience && (
                              <div className="text-sm mt-8">
                                <span className="text-gray-400 font-medium">Target:</span>{' '}
                                <span className="text-gray-100">{msg.metadata.brandingData.personality.targetAudience}</span>
                              </div>
                            )}
                          </div>
                        </div>
                      )}

                      {/* Show applied files if this is an apply success message */}
                      {msg.metadata?.appliedFiles && msg.metadata.appliedFiles.length > 0 && (
                    <div className="mt-3 inline-block bg-[#1e1e3a] rounded-[10px] p-5">
                      <div className="text-sm font-medium mb-3 text-gray-300">
                        {msg.content.includes('Applied') ? 'Files Updated:' : 'Generated Files:'}
                      </div>
                      <div className="flex flex-wrap items-start gap-2">
                        {msg.metadata.appliedFiles.map((filePath, fileIdx) => {
                          const fileName = filePath.split('/').pop() || filePath;
                          const fileExt = fileName.split('.').pop() || '';
                          const fileType = fileExt === 'jsx' || fileExt === 'js' ? 'javascript' :
                                          fileExt === 'css' ? 'css' :
                                          fileExt === 'json' ? 'json' : 'text';

                          return (
                            <div
                              key={`applied-${fileIdx}`}
                              className="inline-flex items-center gap-1.5 px-6 py-1.5 bg-[#36322F] text-white rounded-[10px] text-sm animate-fade-in-up"
                              style={{ animationDelay: `${fileIdx * 30}ms` }}
                            >
                              <span className={`inline-block w-1.5 h-1.5 rounded-full ${
                                fileType === 'css' ? 'bg-blue-400' :
                                fileType === 'javascript' ? 'bg-yellow-400' :
                                fileType === 'json' ? 'bg-green-400' :
                                'bg-gray-400'
                              }`} />
                              {fileName}
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}
                  
                      {/* Show generated files for completion messages - but only if no appliedFiles already shown */}
                      {isGenerationComplete && generationProgress.files.length > 0 && idx === chatMessages.length - 1 && !msg.metadata?.appliedFiles && !chatMessages.some(m => m.metadata?.appliedFiles) && (
                    <div className="mt-2 inline-block bg-[#1e1e3a] rounded-[10px] p-3">
                      <div className="text-xs font-medium mb-1 text-gray-300">Generated Files:</div>
                      <div className="flex flex-wrap items-start gap-1">
                        {generationProgress.files.map((file, fileIdx) => (
                          <div
                            key={`complete-${fileIdx}`}
                            className="inline-flex items-center gap-1.5 px-6 py-1.5 bg-[#36322F] text-white rounded-[10px] text-xs animate-fade-in-up"
                            style={{ animationDelay: `${fileIdx * 30}ms` }}
                          >
                            <span className={`inline-block w-1.5 h-1.5 rounded-full ${
                              file.type === 'css' ? 'bg-blue-400' :
                              file.type === 'javascript' ? 'bg-yellow-400' :
                              file.type === 'json' ? 'bg-green-400' :
                              'bg-gray-400'
                            }`} />
                            {file.path.split('/').pop()}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                    </div>
                    </div>
                  </div>
              );
            })}
            
            {/* Code application progress */}
            {codeApplicationState.stage && (
              <CodeApplicationProgress state={codeApplicationState} />
            )}
            
            {/* File generation progress - inline display (during generation) */}
            {generationProgress.isGenerating && (
              <div className="inline-block bg-[#1e1e3a] rounded-lg p-3">
                <div className="text-sm font-medium mb-2 text-gray-300">
                  {generationProgress.status}
                </div>
                <div className="flex flex-wrap items-start gap-1">
                  {/* Show completed files */}
                  {generationProgress.files.map((file, idx) => (
                    <div
                      key={`file-${idx}`}
                      className="inline-flex items-center gap-1.5 px-6 py-1.5 bg-[#36322F] text-white rounded-[10px] text-xs animate-fade-in-up"
                      style={{ animationDelay: `${idx * 30}ms` }}
                    >
                      <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                      </svg>
                      {file.path.split('/').pop()}
                    </div>
                  ))}
                  
                  {/* Show current file being generated */}
                  {generationProgress.currentFile && (
                    <div className="flex items-center gap-1 px-2 py-1 bg-[#36322F]/70 text-white rounded-[10px] text-sm animate-pulse"
                      style={{ animationDelay: `${generationProgress.files.length * 30}ms` }}>
                      <div className="w-16 h-16 border-2 border-white border-t-transparent rounded-full animate-spin" />
                      {generationProgress.currentFile.path.split('/').pop()}
                    </div>
                  )}
                </div>
                
                {/* Live streaming response display */}
                {generationProgress.streamedCode && (
                  <motion.div 
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    transition={{ duration: 0.3 }}
                    className="mt-3 border-t border-[#3a3a5a] pt-3"
                  >
                    <div className="flex items-center gap-2 mb-2">
                      <div className="flex items-center gap-1">
                        <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse" />
                        <span className="text-xs font-medium text-gray-400">AI Response Stream</span>
                      </div>
                      <div className="flex-1 h-px bg-gradient-to-r from-gray-300 to-transparent" />
                    </div>
                    <div className="bg-gray-900 border border-gray-700 rounded max-h-128 overflow-y-auto scrollbar-hide">
                      <SyntaxHighlighter
                        language="jsx"
                        style={vscDarkPlus}
                        customStyle={{
                          margin: 0,
                          padding: '0.75rem',
                          fontSize: '11px',
                          lineHeight: '1.5',
                          background: 'transparent',
                          maxHeight: '8rem',
                          overflow: 'hidden'
                        }}
                      >
                        {(() => {
                          const lastContent = generationProgress.streamedCode.slice(-1000);
                          // Show the last part of the stream, starting from a complete tag if possible
                          const startIndex = lastContent.indexOf('<');
                          return startIndex !== -1 ? lastContent.slice(startIndex) : lastContent;
                        })()}
                      </SyntaxHighlighter>
                      <span className="inline-block w-3 h-4 bg-orange-400 ml-3 mb-3 animate-pulse" />
                    </div>
                  </motion.div>
                )}
              </div>
            )}
          </div>

          <div className="p-4 border-t border-border bg-background-base">
            <div className="flex items-center gap-2">
              <ImageUpload
                onImageSelect={(imgs) => setAttachedImages(prev => [...prev, ...imgs])}
                disabled={loading || generationProgress.isGenerating}
              />
              <div className="flex-1">
                <HeroInput
                  value={aiChatInput}
                  onChange={setAiChatInput}
                  onSubmit={sendChatMessage}
                  placeholder="Describe what you want to build..."
                  showSearchFeatures={false}
                />
              </div>
            </div>
            {/* Show attached images */}
            {attachedImages.length > 0 && (
              <div className="flex gap-2 mt-2 flex-wrap">
                {attachedImages.map((img, i) => (
                  <div key={i} className="relative group">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={`data:${img.mimeType};base64,${img.base64}`} alt={img.name} className="w-12 h-12 rounded-lg object-cover border border-[#2a2a4a]" />
                    <button
                      onClick={() => setAttachedImages(prev => prev.filter((_, idx) => idx !== i))}
                      className="absolute -top-1 -right-1 w-4 h-4 bg-red-500 rounded-full text-white text-[10px] flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
                    >
                      ×
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Right Panel - Split View or Single Panel */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* Content toolbar with tabs and device controls */}
          <div className="px-3 pt-3 pb-3 bg-[#0a0a1a] border-b border-[#2a2a4a] flex justify-between items-center">
            <div className="flex items-center gap-2">
              {/* Toggle-style Code/View switcher */}
              {!splitView && (
                <div className="inline-flex bg-[#12122a] border border-[#2a2a4a] rounded-md p-0.5">
                  <button
                    onClick={() => {
                      setActiveTab('generation');
                      if (sandboxData && Object.keys(sandboxFiles).length === 0) fetchSandboxFiles();
                    }}
                    className={`px-3 py-1 rounded transition-all text-xs font-medium ${
                      activeTab === 'generation' ? 'bg-[#2a2a4a] text-gray-100 shadow-sm' : 'bg-transparent text-gray-400 hover:text-gray-200'
                    }`}
                  >
                    <div className="flex items-center gap-1.5">
                      <svg width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
                      </svg>
                      <span>Code</span>
                    </div>
                  </button>
                  <button
                    onClick={() => setActiveTab('preview')}
                    className={`px-3 py-1 rounded transition-all text-xs font-medium ${
                      activeTab === 'preview' ? 'bg-[#2a2a4a] text-gray-100 shadow-sm' : 'bg-transparent text-gray-400 hover:text-gray-200'
                    }`}
                  >
                    <div className="flex items-center gap-1.5">
                      <svg width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                      </svg>
                      <span>View</span>
                    </div>
                  </button>
                </div>
              )}
              {splitView && <span className="text-xs text-gray-400 font-medium">Split View</span>}
              {/* Device preview controls in preview or split mode */}
              {(activeTab === 'preview' || splitView) && (
                <DevicePreviewToolbar
                  deviceMode={deviceMode}
                  onDeviceChange={setDeviceMode}
                  sandboxUrl={sandboxData?.url ? resolveSandboxUrl(sandboxData.url) : undefined}
                  onRefresh={() => { if (iframeRef.current) iframeRef.current.src = iframeRef.current.src; }}
                />
              )}
            </div>
            <div className="flex gap-2 items-center">
              {activeTab === 'generation' && !generationProgress.isEdit && generationProgress.files.length > 0 && (
                <div className="text-gray-500 text-xs font-medium">{generationProgress.files.length} files</div>
              )}
              {generationProgress.isGenerating && (
                <div className="inline-flex items-center gap-1.5 px-2 py-0.5 bg-[#1e1e3a] border border-[#2a2a4a] rounded text-xs font-medium text-green-400">
                  <div className="w-1.5 h-1.5 bg-green-500 rounded-full animate-pulse" />
                  {generationProgress.isEdit ? 'Editing' : 'Generating'}
                </div>
              )}
              {sandboxData && (
                <div className="inline-flex items-center gap-1.5 px-2 py-0.5 bg-[#1e1e3a] border border-[#2a2a4a] rounded text-xs font-medium text-gray-400">
                  <div className="w-1.5 h-1.5 bg-green-500 rounded-full" />
                  Active
                </div>
              )}
              {sandboxData && (
                <a href={resolveSandboxUrl(sandboxData.url)} target="_blank" rel="noopener noreferrer" title="Open in new tab"
                  className="p-1 rounded hover:bg-[#2a2a4a] text-gray-400 hover:text-gray-200 transition-colors">
                  <svg width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                  </svg>
                </a>
              )}
            </div>
          </div>
          
          {/* Main content area */}
          <div className="flex-1 relative overflow-hidden flex">
            {splitView ? (
              <>
                {/* Split: Code on left */}
                <div className="w-1/2 flex flex-col overflow-hidden border-r border-[#2a2a4a]">
                  <div className="flex-1 overflow-hidden flex">
                    {!generationProgress.isEdit && renderFileExplorer()}
                    <div className="flex-1 flex flex-col min-h-0">
                      {selectedFile ? (
                        <div className="flex-1 flex flex-col min-h-0">
                          <div className="px-3 py-1.5 bg-[#12122a] border-b border-[#2a2a4a] flex items-center gap-2 flex-shrink-0">
                            {getFileIcon(selectedFile)}
                            <span className="font-mono text-xs text-gray-300">{selectedFile}</span>
                          </div>
                          <div className="flex-1 min-h-0">
                            <CodeEditor
                              content={getFileContent(selectedFile)}
                              filePath={selectedFile}
                              onSave={(path, content) => handleFileSave(path, content)}
                            />
                          </div>
                        </div>
                      ) : (
                        <div className="flex-1 flex items-center justify-center text-gray-500">
                          <div className="text-center">
                            <svg width="32" height="32" fill="none" viewBox="0 0 24 24" stroke="currentColor" className="mx-auto mb-2 opacity-30">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
                            </svg>
                            <p className="text-xs">Select a file to edit</p>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
                {/* Split: Preview on right */}
                <div className="w-1/2 flex flex-col overflow-hidden">
                  <div className="flex-1 flex items-center justify-center overflow-hidden bg-[#0a0a0a]">
                    <div style={{ width: getDeviceWidth(deviceMode), maxWidth: '100%', height: '100%' }}
                      className="transition-all duration-300">
                      {sandboxData?.url ? (
                        <iframe
                          ref={iframeRef}
                          src={resolveSandboxUrl(sandboxData.url)}
                          className="w-full h-full border-0 bg-white"
                          title="Preview"
                        />
                      ) : (
                        <div className="flex items-center justify-center h-full text-gray-500 text-sm">
                          Preview will appear here
                        </div>
                      )}
                    </div>
                  </div>
                  <ConsolePanel isOpen={consoleOpen} onToggle={() => setConsoleOpen(v => !v)} iframeRef={iframeRef} />
                </div>
              </>
            ) : (
              /* Single panel mode */
              <div className="flex-1 flex flex-col overflow-hidden">
                {activeTab === 'preview' ? (
                  <>
                    <div className="flex-1 flex items-center justify-center overflow-hidden bg-[#0a0a0a]">
                      <div style={{ width: getDeviceWidth(deviceMode), maxWidth: '100%', height: '100%' }}
                        className="transition-all duration-300">
                        {renderMainContent()}
                      </div>
                    </div>
                    <ConsolePanel isOpen={consoleOpen} onToggle={() => setConsoleOpen(v => !v)} iframeRef={iframeRef} />
                  </>
                ) : (
                  <div className="flex-1 relative overflow-hidden">
                    {renderMainContent()}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Modals */}
      {/* Diff Panel,  shows file changes after edits */}
      {showDiffPanel && fileDiffs.length > 0 && (
        <div className="fixed inset-y-0 right-0 w-[480px] bg-[#0d0d1a] border-l border-[#2a2a4a] z-50 flex flex-col shadow-2xl">
          <div className="flex items-center justify-between px-4 py-3 border-b border-[#2a2a4a] bg-[#12122a]">
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium text-white">Changes</span>
              <span className="text-xs text-gray-500">
                {fileDiffs.reduce((a, d) => a + d.additions, 0)} additions, {fileDiffs.reduce((a, d) => a + d.deletions, 0)} deletions
              </span>
            </div>
            <button
              onClick={() => { setShowDiffPanel(false); setFileDiffs([]); }}
              className="text-gray-400 hover:text-white text-lg"
            >×</button>
          </div>
          <div className="flex-1 overflow-y-auto">
            {fileDiffs.map((diff, di) => (
              <div key={di} className="border-b border-[#1a1a3a]">
                <div className="px-4 py-2 bg-[#0a0a18] flex items-center gap-2 text-xs">
                  <span className={diff.type === 'added' ? 'text-green-400' : diff.type === 'deleted' ? 'text-red-400' : 'text-yellow-400'}>
                    {diff.type === 'added' ? '+' : diff.type === 'deleted' ? '-' : '~'}
                  </span>
                  <span className="text-gray-300 font-mono">{diff.path}</span>
                  <span className="ml-auto text-green-400">+{diff.additions}</span>
                  <span className="text-red-400">-{diff.deletions}</span>
                </div>
                {diff.hunks.map((hunk, hi) => (
                  <div key={hi} className="font-mono text-[11px] leading-5">
                    <div className="px-4 py-0.5 bg-[#1a1a3a] text-gray-500">
                      @@ -{hunk.oldStart},{hunk.oldLines} +{hunk.newStart},{hunk.newLines} @@
                    </div>
                    {hunk.lines.map((line, li) => (
                      <div
                        key={li}
                        className={`px-4 ${
                          line.type === 'add' ? 'bg-green-900/20 text-green-300' :
                          line.type === 'remove' ? 'bg-red-900/20 text-red-300' :
                          'text-gray-500'
                        }`}
                      >
                        <span className="inline-block w-4 text-center select-none opacity-50">
                          {line.type === 'add' ? '+' : line.type === 'remove' ? '-' : ' '}
                        </span>
                        {line.content}
                      </div>
                    ))}
                  </div>
                ))}
              </div>
            ))}
          </div>
        </div>
      )}

      <VersionTimeline
        isOpen={showVersionHistory}
        onClose={() => setShowVersionHistory(false)}
        snapshots={snapshots}
        currentIndex={snapshotIndex}
        onRestore={handleRestoreVersion}
      />
      <ImportProjectModal
        isOpen={showImportModal}
        onClose={() => setShowImportModal(false)}
        onImport={handleImportProject}
      />

    </div>
    </HeaderProvider>
  );
}

export default function Page() {
  return (
    <Suspense fallback={<div className="flex items-center justify-center min-h-screen">Loading...</div>}>
      <AISandboxPage />
    </Suspense>
  );
}