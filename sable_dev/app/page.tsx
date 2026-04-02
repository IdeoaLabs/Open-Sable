"use client";

import Link from "next/link";
import Image from "next/image";
import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { appConfig } from '@/config/app.config';
import { toast } from "sonner";
import { templateList, DEFAULT_TEMPLATE, type TemplateId } from '@/lib/sandbox/templates';

// Import shared components
import { Connector } from "@/components/shared/layout/curvy-rect";
import HeroFlame from "@/components/shared/effects/flame/hero-flame";
import AsciiExplosion from "@/components/shared/effects/flame/ascii-explosion";
import { HeaderProvider } from "@/components/shared/header/HeaderContext";

// Import hero section components
import HomeHeroBackground from "@/components/app/(home)/sections/hero/Background/Background";
import { BackgroundOuterPiece } from "@/components/app/(home)/sections/hero/Background/BackgroundOuterPiece";
import HomeHeroBadge from "@/components/app/(home)/sections/hero/Badge/Badge";
import HomeHeroPixi from "@/components/app/(home)/sections/hero/Pixi/Pixi";
import HomeHeroTitle from "@/components/app/(home)/sections/hero/Title/Title";
import HeroInputSubmitButton from "@/components/app/(home)/sections/hero-input/Button/Button";
// import Globe from "@/components/app/(home)/sections/hero-input/_svg/Globe";

// Import header components
import HeaderBrandKit from "@/components/shared/header/BrandKit/BrandKit";
import HeaderWrapper from "@/components/shared/header/Wrapper/Wrapper";
import HeaderDropdownWrapper from "@/components/shared/header/Dropdown/Wrapper/Wrapper";
import GithubIcon from "@/components/shared/header/Github/_svg/GithubIcon";
import ButtonUI from "@/components/ui/shadcn/button"

interface SearchResult {
  url: string;
  title: string;
  description: string;
  screenshot: string | null;
  markdown: string;
}

export default function HomePage() {
  const [url, setUrl] = useState<string>("");
  const selectedStyle = "5"; // Dark Mode always
  const [selectedModel, setSelectedModel] = useState<string>(appConfig.ai.defaultModel);
  const [isValidUrl, setIsValidUrl] = useState<boolean>(false);
  const [showSearchTiles, setShowSearchTiles] = useState<boolean>(false);
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [isSearching, setIsSearching] = useState<boolean>(false);
  const [hasSearched, setHasSearched] = useState<boolean>(false);
  const [isFadingOut, setIsFadingOut] = useState<boolean>(false);
  const [showSelectMessage, setShowSelectMessage] = useState<boolean>(false);
  const [showInstructionsForIndex, setShowInstructionsForIndex] = useState<number | null>(null);
  const [additionalInstructions, setAdditionalInstructions] = useState<string>('');
  const [extendBrandStyles, setExtendBrandStyles] = useState<boolean>(false);
  const [selectedTemplate, setSelectedTemplate] = useState<TemplateId>(DEFAULT_TEMPLATE);
  const router = useRouter();
  
  // Simple URL validation
  const validateUrl = (urlString: string) => {
    if (!urlString) return false;
    // Basic URL pattern - accepts domains with or without protocol
    const urlPattern = /^(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)*\/?$/;
    return urlPattern.test(urlString.toLowerCase());
  };

  // Check if input is a URL (contains a dot)
  const isURL = (str: string): boolean => {
    const urlPattern = /^(https?:\/\/)?([a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(\/.*)?$/;
    return urlPattern.test(str.trim());
  };

  const [dynamicModels, setDynamicModels] = useState<Array<{ id: string; name: string; provider: string }>>([]);
  const [recentProjects, setRecentProjects] = useState<Array<{
    id: string; name: string; prompt: string; template: string;
    model: string; createdAt: number; updatedAt: number; fileCount: number; sandboxId?: string;
  }>>([]);

  // Fetch recent projects on mount
  useEffect(() => {
    const fetchProjects = async () => {
      try {
        const res = await fetch('/api/project-history');
        if (res.ok) {
          const data = await res.json();
          if (data.success) setRecentProjects(data.projects || []);
        }
      } catch { /* silent */ }
    };
    fetchProjects();
  }, []);

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
        }
      } catch (err) {
        console.log('[home] Failed to fetch available models:', err);
      }
    };
    fetchModels();
    return () => { cancelled = true; };
  }, []);

  const models = dynamicModels.length > 0
    ? dynamicModels
    : appConfig.ai.availableModels.map(model => ({
        id: model,
        name: appConfig.ai.modelDisplayNames[model] || model,
        provider: 'cloud',
      }));

  const handleSubmit = async (selectedResult?: SearchResult) => {
    const inputValue = url.trim();

    if (!inputValue) {
      toast.error("Please enter a URL or search term");
      return;
    }

    // Validate brand extension mode requirements
    if (extendBrandStyles && isURL(inputValue) && !additionalInstructions.trim()) {
      toast.error("Please describe what you want to build with this brand's styles");
      return;
    }
    
    // If it's a search result being selected, fade out and redirect
    if (selectedResult) {
      setIsFadingOut(true);
      
      // Wait for fade animation
      setTimeout(() => {
        sessionStorage.setItem('targetUrl', selectedResult.url);
        sessionStorage.setItem('selectedStyle', selectedStyle);
        sessionStorage.setItem('selectedModel', selectedModel);
        sessionStorage.setItem('selectedTemplate', selectedTemplate);
        sessionStorage.setItem('autoStart', 'true');
        if (selectedResult.markdown) {
          sessionStorage.setItem('siteMarkdown', selectedResult.markdown);
        }
        router.push('/generation');
      }, 500);
      return;
    }
    
    // If it's a URL, check if we're extending brand styles or cloning
    if (isURL(inputValue)) {
      if (extendBrandStyles) {
        // Brand extension mode - extract brand styles and use them with the prompt
        sessionStorage.setItem('targetUrl', inputValue);
        sessionStorage.setItem('selectedModel', selectedModel);
        sessionStorage.setItem('selectedTemplate', selectedTemplate);
        sessionStorage.setItem('autoStart', 'true');
        sessionStorage.setItem('brandExtensionMode', 'true');
        sessionStorage.setItem('brandExtensionPrompt', additionalInstructions || '');
        router.push('/generation');
      } else {
        // Normal clone mode
        sessionStorage.setItem('targetUrl', inputValue);
        sessionStorage.setItem('selectedStyle', selectedStyle);
        sessionStorage.setItem('selectedModel', selectedModel);
        sessionStorage.setItem('selectedTemplate', selectedTemplate);
        sessionStorage.setItem('autoStart', 'true');
        router.push('/generation');
      }
    } else {
      // It's a prompt,  go directly to generation/creation mode
      sessionStorage.setItem('selectedModel', selectedModel);
      sessionStorage.setItem('selectedTemplate', selectedTemplate);
      sessionStorage.setItem('creationPrompt', inputValue);
      sessionStorage.setItem('autoStart', 'true');
      router.push(`/generation?prompt=${encodeURIComponent(inputValue)}&model=${encodeURIComponent(selectedModel)}&template=${encodeURIComponent(selectedTemplate)}`);
    }
  };

  // Perform search when user types
  const performSearch = async (searchQuery: string) => {
    if (!searchQuery.trim() || isURL(searchQuery)) {
      setSearchResults([]);
      setShowSearchTiles(false);
      return;
    }

    setIsSearching(true);
    setShowSearchTiles(true);
    try {
      const response = await fetch('/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: searchQuery }),
      });

      if (response.ok) {
        const data = await response.json();
        setSearchResults(data.results || []);
        setShowSearchTiles(true);
      }
    } catch (error) {
      console.error('Search error:', error);
    } finally {
      setIsSearching(false);
    }
  };

  return (
    <HeaderProvider>
      <div className="min-h-screen bg-background-base">
        {/* Header/Navigation Section */}
        <HeaderDropdownWrapper />

        <div className="sticky top-0 left-0 w-full z-[101] bg-background-base header">
          <div className="absolute top-0 cmw-container border-x border-border-faint h-full pointer-events-none" />
          <div className="h-1 bg-border-faint w-full left-0 -bottom-1 absolute" />
          <div className="cmw-container absolute h-full pointer-events-none top-0">
            <Connector className="absolute -left-[10.5px] -bottom-11" />
            <Connector className="absolute -right-[10.5px] -bottom-11" />
          </div>

          <HeaderWrapper>
            <div className="max-w-[900px] mx-auto w-full flex justify-between items-center">
              <div className="flex gap-24 items-center">
                <HeaderBrandKit />
              </div>
              <div className="flex gap-8">
              </div>
            </div>
          </HeaderWrapper>
        </div>

        {/* Hero Section */}
        <section className="overflow-x-clip" id="home-hero">
          <div className="pt-28 lg:pt-254 lg:-mt-100 pb-115 relative" id="hero-content">
            <HomeHeroPixi />
            <HeroFlame />
            <BackgroundOuterPiece />
            <HomeHeroBackground />

            <div className="relative container px-16">
              <HomeHeroBadge />
              <HomeHeroTitle />
              <p className="text-center text-body-large">
                Describe what you want to build, or paste a URL to clone.
              </p>
            </div>
          </div>

          {/* Mini Playground Input */}
          <div className="container lg:contents !p-16 relative -mt-90">
            <div className="absolute top-0 left-[calc(50%-50vw)] w-screen h-1 bg-border-faint lg:hidden" />
            <div className="absolute bottom-0 left-[calc(50%-50vw)] w-screen h-1 bg-border-faint lg:hidden" />
            <Connector className="-top-10 -left-[10.5px] lg:hidden" />
            <Connector className="-top-10 -right-[10.5px] lg:hidden" />
            <Connector className="-bottom-10 -left-[10.5px] lg:hidden" />
            <Connector className="-bottom-10 -right-[10.5px] lg:hidden" />

            {/* Template Selector */}
            <div className="max-w-552 mx-auto z-[12] relative mb-4">
              <div className="flex flex-wrap justify-center gap-2">
                {templateList.map((template) => (
                  <button
                    key={template.id}
                    onClick={() => setSelectedTemplate(template.id)}
                    className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all border ${
                      selectedTemplate === template.id
                        ? 'bg-orange-500/20 border-orange-500/50 text-orange-300'
                        : 'bg-[#12122a] border-[#2a2a4a] text-gray-400 hover:border-[#3a3a5a] hover:text-gray-300'
                    }`}
                  >
                    <span className="mr-1.5">{template.icon}</span>
                    {template.name}
                  </button>
                ))}
              </div>
              <p className="text-center text-xs text-gray-500 mt-2">
                {templateList.find(t => t.id === selectedTemplate)?.description}
              </p>
            </div>

            {/* Model Selector - Always visible */}
            <div className="max-w-552 mx-auto z-[12] relative mb-4">
              <div className="flex justify-center">
                <select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  className="px-4 py-2 text-xs font-medium text-gray-200 bg-[#12122a] rounded-lg border border-[#2a2a4a] focus:border-orange-500 focus:outline-none focus:ring-1 focus:ring-orange-500"
                >
                  <optgroup label="☁️ Cloud AI">
                    {models.filter(m => !['ollama', 'openwebui'].includes(m.provider)).map((model) => (
                      <option key={model.id} value={model.id}>{model.name}</option>
                    ))}
                  </optgroup>
                  {models.some(m => m.provider === 'ollama') && (
                    <optgroup label="🦙 Ollama (Local)">
                      {models.filter(m => m.provider === 'ollama').map((model) => (
                        <option key={model.id} value={model.id}>{model.name}</option>
                      ))}
                    </optgroup>
                  )}
                  {models.some(m => m.provider === 'openwebui') && (
                    <optgroup label="🌐 OpenWebUI">
                      {models.filter(m => m.provider === 'openwebui').map((model) => (
                        <option key={model.id} value={model.id}>{model.name}</option>
                      ))}
                    </optgroup>
                  )}
                </select>
              </div>
            </div>

            {/* Hero Input Component */}
            <div className="max-w-552 mx-auto z-[11] lg:z-[2]">
              <div className="rounded-20 -mt-30 lg:-mt-30">
                <div
                  className="bg-[#1a1a2e] rounded-20 relative z-10 border border-[#2a2a4a]"
                  style={{
                    boxShadow:
                      "0px 0px 44px 0px rgba(0, 0, 0, 0.15), 0px 88px 56px -20px rgba(0, 0, 0, 0.12), 0px 0px 0px 1px rgba(255, 255, 255, 0.05), 0px 0px 0px 10px rgba(15, 15, 30, 0.8)",
                  }}
                >

                <div className="p-[28px] flex gap-12 items-center w-full relative bg-[#1a1a2e] rounded-20">
                  {/* Show different UI when search results are displayed */}
                  {hasSearched && searchResults.length > 0 && !isFadingOut ? (
                    <>
                      {/* Selection mode icon */}
                      <svg 
                        width="20" 
                        height="20" 
                        viewBox="0 0 20 20" 
                        fill="none" 
                        xmlns="http://www.w3.org/2000/svg"
                        className="opacity-40 flex-shrink-0"
                      >
                        <rect x="2" y="4" width="7" height="5" rx="1" stroke="currentColor" strokeWidth="1.5"/>
                        <rect x="11" y="4" width="7" height="5" rx="1" stroke="currentColor" strokeWidth="1.5"/>
                        <rect x="2" y="11" width="7" height="5" rx="1" stroke="currentColor" strokeWidth="1.5"/>
                        <rect x="11" y="11" width="7" height="5" rx="1" stroke="currentColor" strokeWidth="1.5"/>
                      </svg>
                      
                      {/* Selection message */}
                      <div className="flex-1 text-body-input text-accent-black">
                        Select which site to clone from the results below
                      </div>
                      
                      {/* Search again button */}
                      <button
                        onClick={(e) => {
                          e.preventDefault();
                          setIsFadingOut(true);
                          setTimeout(() => {
                            setSearchResults([]);
                            setHasSearched(false);
                            setShowSearchTiles(false);
                            setIsFadingOut(false);
                            setUrl('');
                          }, 500);
                        }}
                        className="button relative rounded-10 px-12 py-8 text-label-medium font-medium flex items-center justify-center gap-6 bg-[#1e1e3a] hover:bg-[#2a2a4a] text-gray-300 active:scale-[0.995] transition-all"
                      >
                        <svg 
                          width="16" 
                          height="16" 
                          viewBox="0 0 16 16" 
                          fill="none" 
                          xmlns="http://www.w3.org/2000/svg"
                          className="opacity-60"
                        >
                          <path d="M14 14L10 10M11 6.5C11 9 9 11 6.5 11C4 11 2 9 2 6.5C2 4 4 2 6.5 2C9 2 11 4 11 6.5Z" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
                        </svg>
                        <span>Search Again</span>
                      </button>
                    </>
                  ) : (
                    <>
                      {isURL(url) ? (
                        // Scrape icon for URLs
                        <svg 
                          width="20" 
                          height="20" 
                          viewBox="0 0 20 20" 
                          fill="none" 
                          xmlns="http://www.w3.org/2000/svg"
                          className="opacity-40 flex-shrink-0"
                        >
                          <rect x="3" y="3" width="14" height="14" rx="2" stroke="currentColor" strokeWidth="1.5"/>
                          <path d="M7 10L9 12L13 8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                        </svg>
                      ) : (
                        // Search icon for search terms
                        <svg 
                          width="20" 
                          height="20" 
                          viewBox="0 0 20 20" 
                          fill="none" 
                          xmlns="http://www.w3.org/2000/svg"
                          className="opacity-40 flex-shrink-0"
                        >
                          <circle cx="8.5" cy="8.5" r="5.5" stroke="currentColor" strokeWidth="1.5"/>
                          <path d="M12.5 12.5L16.5 16.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
                        </svg>
                      )}
                      <input
                        className="flex-1 bg-transparent text-body-input text-white placeholder:text-gray-500 focus:outline-none focus:ring-0 focus:border-transparent"
                        placeholder="Describe what you want to build, or paste a URL..."
                        type="text"
                        value={url}
                        disabled={isSearching}
                        onChange={(e) => {
                          const value = e.target.value;
                          setUrl(value);
                          setIsValidUrl(validateUrl(value));
                          // Reset search state when input changes
                          if (value.trim() === "") {
                            setShowSearchTiles(false);
                            setHasSearched(false);
                            setSearchResults([]);
                          }
                        }}
                        onKeyDown={(e) => {
                          if (e.key === "Enter" && !isSearching) {
                            e.preventDefault();
                            handleSubmit();
                          }
                        }}
                        onFocus={() => {
                          if (url.trim() && !isURL(url) && searchResults.length > 0) {
                            setShowSearchTiles(true);
                          }
                        }}
                      />
                      <div
                        onClick={(e) => {
                          e.preventDefault();
                          if (!isSearching) {
                            handleSubmit();
                          }
                        }}
                        className={isSearching ? 'pointer-events-none' : ''}
                      >
                        <HeroInputSubmitButton 
                          dirty={url.length > 0} 
                          buttonText={isURL(url) ? 'Scrape Site' : 'Create'} 
                          disabled={isSearching}
                        />
                      </div>
                    </>
                  )}
                </div>

                {/* Options Section - Only show when valid URL */}
                <div className={`overflow-hidden transition-all duration-500 ease-in-out ${
                  isValidUrl ? (extendBrandStyles ? 'max-h-[400px]' : 'max-h-[300px]') + ' opacity-100' : 'max-h-0 opacity-0'
                }`}>
                  <div className="px-[28px] pt-0 pb-[28px]">
                    <div className="border-t border-[#2a2a4a] bg-transparent">
                      {/* Extend Brand Styles Toggle */}
                      <div className={`transition-all duration-300 transform ${
                        isValidUrl ? 'translate-y-0 opacity-100' : '-translate-y-2 opacity-0'
                      }`} style={{ transitionDelay: '50ms' }}>
                        <div className="py-8 grid grid-cols-2 items-center gap-12 group cursor-pointer" onClick={() => setExtendBrandStyles(!extendBrandStyles)}>
                          <div className="flex select-none">
                            <div className="flex lg-max:flex-col whitespace-nowrap flex-wrap min-w-0 gap-8 lg:justify-between flex-1">
                              <div className="text-xs font-medium text-black-alpha-72 transition-all group-hover:text-accent-black relative">
                                Extend brand styles
                              </div>
                            </div>
                          </div>
                          <div className="flex justify-end">
                            <button
                              className="transition-all relative rounded-full group bg-black-alpha-10"
                              type="button"
                              onClick={(e) => {
                                e.stopPropagation();
                                setExtendBrandStyles(!extendBrandStyles);
                              }}
                              style={{
                                width: '50px',
                                height: '20px',
                                boxShadow: 'rgba(0, 0, 0, 0.02) 0px 6px 12px 0px inset, rgba(0, 0, 0, 0.02) 0px 0.75px 0.75px 0px inset, rgba(0, 0, 0, 0.04) 0px 0.25px 0.25px 0px inset'
                              }}
                            >
                              <div
                                className={`overlay transition-opacity ${extendBrandStyles ? 'opacity-100' : 'opacity-0'}`}
                                style={{ background: 'color(display-p3 0.9059 0.3294 0.0784)', backgroundColor: '#FA4500' }}
                              />
                              <div
                                className="top-[2px] left-[2px] transition-all absolute rounded-full bg-accent-white"
                                style={{
                                  width: '28px',
                                  height: '16px',
                                  boxShadow: 'rgba(0, 0, 0, 0.06) 0px 6px 12px -3px, rgba(0, 0, 0, 0.06) 0px 3px 6px -1px, rgba(0, 0, 0, 0.04) 0px 1px 2px 0px, rgba(0, 0, 0, 0.08) 0px 0.5px 0.5px 0px',
                                  transform: extendBrandStyles ? 'translateX(16px)' : 'none'
                                }}
                              />
                            </button>
                          </div>
                        </div>
                      </div>

                      {/* Brand Extension Prompt - Show when toggle is enabled */}
                      {extendBrandStyles && (
                        <div className="pb-10 transition-all duration-300 opacity-100">
                          <textarea
                            value={additionalInstructions}
                            onChange={(e) => setAdditionalInstructions(e.target.value)}
                            placeholder="Describe the new functionality you want to build using this brand's styles..."
                            className="w-full px-4 py-10 text-xs font-medium text-gray-200 bg-[#12122a] rounded border border-[#2a2a4a] focus:border-orange-500 focus:outline-none focus:ring-1 focus:ring-orange-500 placeholder:text-gray-500 min-h-[80px] resize-none"
                          />
                        </div>
                      )}

                      {/* Style locked to Dark Mode */}

                      {/* Additional Instructions */}
                      <div className={`flex items-center gap-3 mt-2 pb-4 transition-all duration-300 transform ${
                        isValidUrl ? 'translate-y-0 opacity-100' : '-translate-y-2 opacity-0'
                      }`} style={{ transitionDelay: '400ms' }}>
                        {!extendBrandStyles && (
                          <input
                            type="text"
                            className="flex-1 px-3 py-2.5 text-xs font-medium text-gray-200 bg-[#12122a] rounded border border-[#2a2a4a] focus:border-orange-500 focus:outline-none focus:ring-1 focus:ring-orange-500 placeholder:text-gray-500"
                            placeholder="Additional instructions (optional)"
                            onChange={(e) => sessionStorage.setItem('additionalInstructions', e.target.value)}
                          />
                        )}
                      </div>
                    </div>
                  </div>
                </div>

                </div>

                <div className="h-248 top-84 cw-768 pointer-events-none absolute overflow-clip -z-10">
                  <AsciiExplosion className="-top-200" />
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Full-width oval carousel section */}
        {showSearchTiles && hasSearched && (
          <section className={`carousel-section relative w-full overflow-hidden mt-32 mb-32 transition-opacity duration-500 ${
            isFadingOut ? 'opacity-0' : 'opacity-100'
          }`}>
            <div className="absolute inset-0 bg-gradient-to-b from-gray-50/50 to-white rounded-[50%] transform scale-x-150 -translate-y-24" />
            
            {isSearching ? (
              // Loading state with animated scrolling skeletons
              <div className="relative h-[250px] overflow-hidden">
                {/* Edge fade overlays */}
                <div className="absolute left-0 top-0 bottom-0 w-[120px] z-20 pointer-events-none" style={{background: 'linear-gradient(to right, white 0%, white 20%, transparent 100%)'}} />
                <div className="absolute right-0 top-0 bottom-0 w-[120px] z-20 pointer-events-none" style={{background: 'linear-gradient(to left, white 0%, white 20%, transparent 100%)'}} />
                
                <div className="carousel-container absolute left-0 flex gap-12 py-4">
                  {/* Duplicate skeleton tiles for continuous scroll */}
                  {[...Array(10), ...Array(10)].map((_, index) => (
                    <div
                      key={`loading-${index}`}
                      className="flex-shrink-0 w-[400px] h-[240px] rounded-lg overflow-hidden border-2 border-[#2a2a4a]/30 bg-[#12122a] relative"
                    >
                      <div className="absolute inset-0 skeleton-shimmer">
                        <div className="absolute inset-0 bg-gradient-to-r from-gray-100 via-gray-50 to-gray-100 skeleton-gradient" />
                      </div>
                      
                      {/* Fake browser UI - 5x bigger */}
                      <div className="absolute top-0 left-0 right-0 h-40 bg-[#1e1e3a] border-b border-[#2a2a4a]/50 flex items-center px-6 gap-4">
                        <div className="flex gap-3">
                          <div className="w-5 h-5 rounded-full bg-[#2a2a4a] animate-pulse" />
                          <div className="w-5 h-5 rounded-full bg-[#2a2a4a] animate-pulse" style={{ animationDelay: '0.1s' }} />
                          <div className="w-5 h-5 rounded-full bg-[#2a2a4a] animate-pulse" style={{ animationDelay: '0.2s' }} />
                        </div>
                        <div className="flex-1 h-8 bg-[#2a2a4a] rounded-md mx-6 animate-pulse" />
                      </div>
                      
                      {/* Content skeleton - positioned just below nav bar */}
                      <div className="absolute top-44 left-4 right-4">
                        <div className="h-3 bg-[#2a2a4a] rounded w-3/4 mb-2 animate-pulse" />
                        <div className="h-3 bg-[#1e1e3a] rounded w-1/2 mb-2 animate-pulse" style={{ animationDelay: '0.2s' }} />
                        <div className="h-3 bg-[#1e1e3a] rounded w-2/3 animate-pulse" style={{ animationDelay: '0.3s' }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ) : searchResults.length > 0 ? (
              // Actual results
              <div className="relative h-[250px] overflow-hidden">
                {/* Edge fade overlays */}
                <div className="absolute left-0 top-0 bottom-0 w-[120px] z-20 pointer-events-none" style={{background: 'linear-gradient(to right, white 0%, white 20%, transparent 100%)'}} />
                <div className="absolute right-0 top-0 bottom-0 w-[120px] z-20 pointer-events-none" style={{background: 'linear-gradient(to left, white 0%, white 20%, transparent 100%)'}} />
                
                <div className="carousel-container absolute left-0 flex gap-12 py-4">
                  {/* Duplicate results for infinite scroll */}
                  {[...searchResults, ...searchResults].map((result, index) => (
                    <div
                      key={`${result.url}-${index}`}
                      className="group flex-shrink-0 w-[400px] h-[240px] rounded-lg overflow-hidden border-2 border-[#2a2a4a]/50 transition-all duration-300 hover:shadow-2xl bg-[#12122a] relative"
                      onMouseLeave={() => {
                        if (showInstructionsForIndex === index) {
                          setShowInstructionsForIndex(null);
                          setAdditionalInstructions('');
                        }
                      }}
                    >
                      {/* Hover overlay with clone buttons or instructions input */}
                      <div className="absolute inset-0 bg-gradient-to-t from-black/90 via-black/50 to-black/30 opacity-0 group-hover:opacity-100 transition-opacity duration-300 z-10 flex flex-col items-center justify-center p-6">
                        {showInstructionsForIndex === index ? (
                          /* Instructions input view - matching main input style exactly */
                          <div className="w-full max-w-[380px]">
                            <div className="bg-[#0a0a1a] rounded-20" style={{
                              boxShadow: "0px 0px 44px 0px rgba(0, 0, 0, 0.02), 0px 88px 56px -20px rgba(0, 0, 0, 0.03), 0px 56px 56px -20px rgba(0, 0, 0, 0.02), 0px 32px 32px -20px rgba(0, 0, 0, 0.03), 0px 16px 24px -12px rgba(0, 0, 0, 0.03), 0px 0px 0px 1px rgba(0, 0, 0, 0.05)"
                            }}>
                              {/* Input area matching main search */}
                              <div className="p-16 flex gap-12 items-start w-full relative">
                                {/* Instructions icon */}
                                <div className="mt-2 flex-shrink-0">
                                  <svg 
                                    width="20" 
                                    height="20" 
                                    viewBox="0 0 20 20" 
                                    fill="none" 
                                    xmlns="http://www.w3.org/2000/svg"
                                    className="opacity-40"
                                  >
                                    <path d="M5 5H15M5 10H15M5 15H10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
                                  </svg>
                                </div>
                                
                                <textarea
                                  value={additionalInstructions}
                                  onChange={(e) => setAdditionalInstructions(e.target.value)}
                                  placeholder="Describe your customizations..."
                                  className="flex-1 bg-transparent text-body-input text-accent-black placeholder:text-black-alpha-48 focus:outline-none focus:ring-0 focus:border-transparent resize-none min-h-[60px]"
                                  autoFocus
                                  onClick={(e) => e.stopPropagation()}
                                  onKeyDown={(e) => {
                                    if (e.key === 'Escape') {
                                      e.stopPropagation();
                                      setShowInstructionsForIndex(null);
                                      setAdditionalInstructions('');
                                    }
                                  }}
                                />
                              </div>
                              
                              {/* Divider */}
                              <div className="border-t border-black-alpha-5" />
                              
                              {/* Buttons area matching main style */}
                              <div className="p-10 flex justify-between items-center">
                                <button
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    setShowInstructionsForIndex(null);
                                    setAdditionalInstructions('');
                                  }}
                                  className="button relative rounded-10 px-8 py-8 text-label-medium font-medium flex items-center justify-center bg-black-alpha-4 hover:bg-black-alpha-6 text-black-alpha-48 active:scale-[0.995] transition-all"
                                >
                                  <svg 
                                    width="20" 
                                    height="20" 
                                    viewBox="0 0 20 20" 
                                    fill="none" 
                                    xmlns="http://www.w3.org/2000/svg"
                                  >
                                    <path d="M12 5L7 10L12 15" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                                  </svg>
                                </button>
                                
                                <button
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    if (additionalInstructions.trim()) {
                                      sessionStorage.setItem('additionalInstructions', additionalInstructions);
                                      handleSubmit(result);
                                    }
                                  }}
                                  disabled={!additionalInstructions.trim()}
                                  className={`
                                    button relative rounded-10 px-8 py-8 text-label-medium font-medium
                                    flex items-center justify-center gap-6
                                    ${additionalInstructions.trim() 
                                      ? 'button-primary text-accent-white active:scale-[0.995]' 
                                      : 'bg-black-alpha-4 text-black-alpha-24 cursor-not-allowed'
                                    }
                                  `}
                                >
                                  {additionalInstructions.trim() && <div className="button-background absolute inset-0 rounded-10 pointer-events-none" />}
                                  <span className="px-6 relative">Apply & Clone</span>
                                  <svg 
                                    width="20" 
                                    height="20" 
                                    viewBox="0 0 20 20" 
                                    fill="none" 
                                    xmlns="http://www.w3.org/2000/svg"
                                    className="relative"
                                  >
                                    <path d="M11.6667 4.79163L16.875 9.99994M16.875 9.99994L11.6667 15.2083M16.875 9.99994H3.125" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5"/>
                                  </svg>
                                </button>
                              </div>
                            </div>
                          </div>
                        ) : (
                          /* Default buttons view */
                          <>
                            <div className="text-white text-center mb-3">
                              <p className="text-base font-semibold mb-0.5">{result.title}</p>
                              <p className="text-[11px] opacity-80">Choose how to clone this site</p>
                            </div>
                            
                            <div className="flex gap-3 justify-center">
                              {/* Instant Clone Button - Orange/Heat style */}
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleSubmit(result);
                                }}
                                className="bg-orange-500 hover:bg-orange-600 flex items-center justify-center button relative text-label-medium button-primary group/button rounded-10 p-8 gap-2 text-white active:scale-[0.995]"
                              >
                                <div className="button-background absolute inset-0 rounded-10 pointer-events-none" />
                                <svg 
                                  width="20" 
                                  height="20" 
                                  viewBox="0 0 20 20" 
                                  fill="none" 
                                  xmlns="http://www.w3.org/2000/svg"
                                  className="relative"
                                >
                                  <path d="M11.6667 4.79163L16.875 9.99994M16.875 9.99994L11.6667 15.2083M16.875 9.99994H3.125" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5"/>
                                </svg>
                                <span className="px-6 relative">Instant Clone</span>
                              </button>
                              
                              {/* Instructions Button - Gray style */}
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  setShowInstructionsForIndex(index);
                                  setAdditionalInstructions('');
                                }}
                                className="bg-[#1e1e3a] hover:bg-[#2a2a4a] flex items-center justify-center button relative text-label-medium rounded-10 p-8 gap-2 text-gray-300 active:scale-[0.995]"
                              >
                                <svg 
                                  width="20" 
                                  height="20" 
                                  viewBox="0 0 20 20" 
                                  fill="none" 
                                  xmlns="http://www.w3.org/2000/svg"
                                  className="opacity-60"
                                >
                                  <path d="M5 5H15M5 10H15M5 15H10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
                                  <path d="M14 14L16 16L14 18" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                                </svg>
                                <span className="px-6">Add Instructions</span>
                              </button>
                            </div>
                          </>
                        )}
                      </div>
                      
                      {result.screenshot ? (
                        <div className="relative w-full h-full">
                          <Image 
                            src={result.screenshot} 
                            alt={result.title}
                            fill
                            className="object-cover object-top"
                            loading="lazy"
                          />
                        </div>
                      ) : (
                        <div className="w-full h-full bg-gradient-to-br from-gray-100 to-gray-50 flex items-center justify-center">
                          <div className="text-center">
                            <div className="w-16 h-16 rounded-full bg-[#2a2a4a] mx-auto mb-3 flex items-center justify-center">
                              <svg 
                                width="32" 
                                height="32" 
                                viewBox="0 0 24 24" 
                                fill="none" 
                                xmlns="http://www.w3.org/2000/svg"
                                className="text-gray-500"
                              >
                                <rect x="3" y="3" width="18" height="18" rx="2" stroke="currentColor" strokeWidth="1.5"/>
                                <path d="M3 9H21" stroke="currentColor" strokeWidth="1.5"/>
                                <circle cx="6" cy="6" r="1" fill="currentColor"/>
                                <circle cx="9" cy="6" r="1" fill="currentColor"/>
                                <circle cx="12" cy="6" r="1" fill="currentColor"/>
                              </svg>
                            </div>
                            <p className="text-gray-500 text-sm font-medium">{result.title}</p>
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              // No results state
              <div className="relative h-[250px] flex items-center justify-center">
                <div className="text-center">
                  <div className="mb-4">
                    <svg className="w-16 h-16 mx-auto text-gray-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                    </svg>
                  </div>
                  <p className="text-gray-500 text-lg">No results found</p>
                  <p className="text-gray-400 text-sm mt-1">Try a different search term</p>
                </div>
              </div>
            )}
          </section>
        )}

        {/* Recent Projects */}
        {recentProjects.length > 0 && !showSearchTiles && (
          <section className="container max-w-3xl mx-auto mt-16 mb-20 px-6">
            <h2 className="text-sm font-medium text-gray-400 mb-4 flex items-center gap-2">
              <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" className="text-gray-500">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              Recent Projects
            </h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
              {recentProjects.slice(0, 6).map((proj) => (
                <div
                  key={proj.id}
                  className="group p-4 rounded-xl bg-[#12122a] border border-[#2a2a4a] hover:border-orange-500/40 hover:bg-[#1a1a3a] cursor-pointer transition-all duration-200"
                  onClick={() => {
                    sessionStorage.setItem('selectedModel', proj.model);
                    sessionStorage.setItem('selectedTemplate', proj.template);
                    if (proj.sandboxId) sessionStorage.setItem('restoreSandboxId', proj.sandboxId);
                    router.push(`/generation?sandbox=${encodeURIComponent(proj.sandboxId || proj.id)}`);
                  }}
                >
                  <div className="flex items-start justify-between gap-2 mb-2">
                    <span className="text-xs font-medium text-gray-200 truncate flex-1">
                      {proj.name === '[object Object]' ? 'Untitled Project' : proj.name}
                    </span>
                    <svg width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="currentColor" className="text-gray-600 group-hover:text-orange-400 transition-colors flex-shrink-0 mt-0.5">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="text-[10px] text-gray-500 bg-[#1a1a2e] px-1.5 py-0.5 rounded">{proj.template}</span>
                    <span className="text-[10px] text-gray-500">{proj.fileCount} files</span>
                    <span className="text-[10px] text-gray-600 ml-auto">{new Date(proj.updatedAt).toLocaleDateString()}</span>
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}

      </div>

      <style jsx>{`
        @keyframes infiniteScroll {
          from {
            transform: translateX(0);
          }
          to {
            transform: translateX(-50%);
          }
        }

        @keyframes shimmer {
          0% {
            transform: translateX(-100%);
          }
          100% {
            transform: translateX(100%);
          }
        }

        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        .carousel-container {
          animation: infiniteScroll 30s linear infinite;
        }

        .carousel-container:hover {
          animation-play-state: paused;
        }

        .skeleton-shimmer {
          position: relative;
          overflow: hidden;
        }

        .skeleton-gradient {
          animation: shimmer 2s infinite;
        }
      `}</style>
    </HeaderProvider>
  );
}