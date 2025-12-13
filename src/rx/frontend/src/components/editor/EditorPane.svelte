<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import type { OpenFile } from '$lib/types';
  import { files, settings } from '$lib/stores';
  import Spinner from '../common/Spinner.svelte';
  import FileBadges from '../common/FileBadges.svelte';
  import { detectLanguage, highlightLine } from '$lib/utils/highlighter';
  import { updateUrlState, debounce } from '$lib/utils/urlState';
  import { applyRegexFilter } from '$lib/utils/regexFilter';
  import Prism from 'prismjs';
  import 'prismjs/components/prism-regex';

  export let file: OpenFile;
  export let hideHeader: boolean = false;
  export let isActive: boolean = false;

  let contentEl: HTMLDivElement;
  let paneEl: HTMLDivElement;

  // Search state
  let searchVisible = false;
  let searchQuery = '';
  let searchMatches: number[] = []; // Line numbers with matches
  let currentMatchIndex = -1;
  let searchInputEl: HTMLInputElement;

  // Goto line state (consolidated - used by both : and dash click)
  let dashGotoVisible = false;
  let dashGotoLineNumber = '';
  let dashGotoInputEl: HTMLInputElement;

  // Regex filter state
  let filterPanelVisible = false;
  let filterPattern = '';
  let filterMode: 'hide' | 'show' | 'highlight' = 'highlight';
  let regexInputEl: HTMLDivElement;
  let highlightedRegex = '';
  let highlightTimeout: number | null = null;

  // Track scroll state for preserving position when loading at top
  let previousStartLine = file.startLine;
  let previousScrollHeight = 0;

  // Prevent auto-loading when scrolling to a target line
  let isScrollingToTarget = false;

  $: fontSize = $settings.editorFontSize;
  $: showLineNumbers = $settings.showLineNumbers;
  $: syntaxHighlighting = file.syntaxHighlighting;
  $: fileMatches = $files.matches.get(file.path) || [];
  $: matchLineNumbers = new Set(fileMatches.map(m => m.lineNumber));
  $: detectedLanguage = detectLanguage(file.name);

  function toggleSyntaxHighlighting() {
    files.toggleSyntaxHighlighting(file.path);
    // Update URL immediately when syntax highlighting is toggled
    if (isActive) {
      const centerLine = getCurrentViewportCenterLine();
      updateUrlState({
        path: file.path,
        line: centerLine,
        syntaxHighlighting: !file.syntaxHighlighting, // Use the new state
      });
    }
  }

  function toggleFilterPanel() {
    filterPanelVisible = !filterPanelVisible;
    if (filterPanelVisible && !file.regexFilter) {
      // Initialize filter when opening panel
      files.toggleRegexFilter(file.path);
    }
  }

  function toggleInvisibleChars() {
    files.toggleInvisibleChars(file.path);
  }

  function applyFilter() {
    files.updateRegexFilter(file.path, filterPattern, filterMode);
  }

  function clearFilter() {
    files.clearRegexFilter(file.path);
    filterPattern = '';
    filterMode = 'highlight';
    filterPanelVisible = false;
  }

  function handleFilterKeyDown(e: KeyboardEvent) {
    if (e.key === 'Enter') {
      e.preventDefault();
      applyFilter();
    } else if (e.key === 'Escape') {
      e.preventDefault();
      filterPanelVisible = false;
    }
  }

  // Highlight regex pattern using Prism.js
  function highlightRegexPattern(pattern: string): string {
    if (!pattern) return '';
    try {
      return Prism.highlight(pattern, Prism.languages.regex, 'regex');
    } catch (e) {
      // If highlighting fails, return escaped HTML
      return pattern.replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }
  }

  // Get absolute cursor position in text content
  function getCursorPosition(element: HTMLElement): number {
    const selection = window.getSelection();
    if (!selection || selection.rangeCount === 0) return 0;

    const range = selection.getRangeAt(0);
    const preCaretRange = range.cloneRange();
    preCaretRange.selectNodeContents(element);
    preCaretRange.setEnd(range.endContainer, range.endOffset);
    return preCaretRange.toString().length;
  }

  // Set cursor position in contenteditable
  function setCursorPosition(element: HTMLElement, position: number) {
    const selection = window.getSelection();
    if (!selection) return;

    let currentPos = 0;
    const walker = document.createTreeWalker(
      element,
      NodeFilter.SHOW_TEXT,
      null
    );

    let node: Node | null;
    while ((node = walker.nextNode())) {
      const textLength = node.textContent?.length || 0;
      if (currentPos + textLength >= position) {
        const range = document.createRange();
        range.setStart(node, position - currentPos);
        range.collapse(true);
        selection.removeAllRanges();
        selection.addRange(range);
        return;
      }
      currentPos += textLength;
    }

    // If we couldn't find the position, place cursor at the end
    if (element.lastChild) {
      const range = document.createRange();
      range.selectNodeContents(element);
      range.collapse(false);
      selection.removeAllRanges();
      selection.addRange(range);
    }
  }

  // Update regex pattern from contenteditable
  function handleRegexInput(e: Event) {
    const target = e.target as HTMLDivElement;
    const text = target.textContent || '';

    // Don't process if it's just the placeholder
    if (text === 'e.g. (\\w+)@(\\w+)\\.com') {
      return;
    }

    filterPattern = text;

    // Clear any pending highlight
    if (highlightTimeout !== null) {
      cancelAnimationFrame(highlightTimeout);
    }

    // Save cursor position before scheduling update
    const cursorPos = getCursorPosition(target);

    // Defer highlighting to next animation frame to avoid interrupting typing
    highlightTimeout = requestAnimationFrame(() => {
      if (!regexInputEl) return;

      // Apply syntax highlighting
      const highlighted = highlightRegexPattern(text);
      if (highlighted) {
        regexInputEl.innerHTML = highlighted;

        // Restore cursor position in the next frame after DOM update
        requestAnimationFrame(() => {
          setCursorPosition(regexInputEl, cursorPos);
        });
      }

      highlightTimeout = null;
    });
  }

  // Initialize with highlighting when panel opens
  $: if (filterPanelVisible && regexInputEl) {
    if (filterPattern) {
      const highlighted = highlightRegexPattern(filterPattern);
      regexInputEl.innerHTML = highlighted;
    } else {
      regexInputEl.textContent = '';
    }
  }

  // Update URL when this file becomes active
  $: if (isActive && file.lines.length > 0) {
    const centerLine = getCurrentViewportCenterLine();
    updateUrlState({
      path: file.path,
      line: centerLine,
      syntaxHighlighting: file.syntaxHighlighting,
    });
  }

  // Reactive scroll to target line
  $: if (file.scrollToLine !== undefined && contentEl && file.lines.length > 0) {
    isScrollingToTarget = true;
    scrollToTargetLine(file.scrollToLine);
  }

  // Keyboard shortcuts
  function handleKeyDown(e: KeyboardEvent) {
    // Check if user is typing in an input field (search, goto, or regex filter)
    // Don't intercept shortcuts if they're focused on an input
    const target = e.target as HTMLElement;
    const isInputFocused = target.tagName === 'INPUT' ||
                          target.tagName === 'TEXTAREA' ||
                          target.contentEditable === 'true' ||
                          target.contentEditable === 'plaintext-only';

    // Cmd/Ctrl + F - open search
    if ((e.metaKey || e.ctrlKey) && e.key === 'f') {
      e.preventDefault();
      openSearch();
      return;
    }

    // / - open search (vim style) - only if not in an input field
    if (e.key === '/' && !searchVisible && !dashGotoVisible && !isInputFocused) {
      e.preventDefault();
      openSearch();
      return;
    }

    // : - open goto line (vim style) - only if not in an input field
    if (e.key === ':' && !searchVisible && !dashGotoVisible && !isInputFocused) {
      e.preventDefault();
      openDashGoto();
      return;
    }

    // Escape - close search/goto
    if (e.key === 'Escape') {
      if (searchVisible) {
        closeSearch();
      }
      if (dashGotoVisible) {
        closeDashGoto();
      }
      return;
    }

    // Keyboard scrolling - only if not focused on an input field
    if (!isInputFocused && contentEl) {
      const scrollAmount = 40; // pixels per arrow key press
      const pageScrollAmount = contentEl.clientHeight * 0.9; // 90% of viewport height

      if (e.key === 'ArrowDown') {
        e.preventDefault();
        contentEl.scrollTop += scrollAmount;
        return;
      }

      if (e.key === 'ArrowUp') {
        e.preventDefault();
        contentEl.scrollTop -= scrollAmount;
        return;
      }

      if (e.key === 'PageDown') {
        e.preventDefault();
        contentEl.scrollTop += pageScrollAmount;
        return;
      }

      if (e.key === 'PageUp') {
        e.preventDefault();
        contentEl.scrollTop -= pageScrollAmount;
        return;
      }

      if (e.key === 'Home') {
        e.preventDefault();
        contentEl.scrollTop = 0;
        return;
      }

      if (e.key === 'End') {
        e.preventDefault();
        contentEl.scrollTop = contentEl.scrollHeight;
        return;
      }
    }
  }

  function openSearch() {
    searchVisible = true;
    dashGotoVisible = false;
    setTimeout(() => searchInputEl?.focus(), 10);
  }

  function closeSearch() {
    searchVisible = false;
    searchQuery = '';
    searchMatches = [];
    currentMatchIndex = -1;
  }



  function getCurrentViewportCenterLine(): number {
    if (!contentEl || file.lines.length === 0) return file.startLine;

    const rect = contentEl.getBoundingClientRect();
    const centerY = rect.top + rect.height / 2;

    // Find line element closest to center
    const lineElements = contentEl.querySelectorAll('[data-line]');
    let closestLine = file.startLine;
    let closestDistance = Infinity;

    lineElements.forEach((el) => {
      const lineRect = el.getBoundingClientRect();
      const distance = Math.abs(lineRect.top + lineRect.height / 2 - centerY);
      if (distance < closestDistance) {
        closestDistance = distance;
        closestLine = parseInt(el.getAttribute('data-line') || String(file.startLine), 10);
      }
    });

    return closestLine;
  }

  function performSearch() {
    if (!searchQuery.trim()) {
      searchMatches = [];
      currentMatchIndex = -1;
      return;
    }

    // Find all lines containing the search query (case-insensitive)
    const query = searchQuery.toLowerCase();
    searchMatches = file.lines
      .filter(line => line.content.toLowerCase().includes(query))
      .map(line => line.lineNumber);

    if (searchMatches.length > 0) {
      currentMatchIndex = 0;
      scrollToSearchMatch(0);
    } else {
      currentMatchIndex = -1;
    }
  }

  function nextSearchMatch() {
    if (searchMatches.length === 0) return;
    currentMatchIndex = (currentMatchIndex + 1) % searchMatches.length;
    scrollToSearchMatch(currentMatchIndex);
  }

  function prevSearchMatch() {
    if (searchMatches.length === 0) return;
    currentMatchIndex = (currentMatchIndex - 1 + searchMatches.length) % searchMatches.length;
    scrollToSearchMatch(currentMatchIndex);
  }

  function scrollToSearchMatch(index: number) {
    const lineNumber = searchMatches[index];
    scrollToTargetLine(lineNumber);
  }

  function handleSearchKeyDown(e: KeyboardEvent) {
    if (e.key === 'Enter') {
      e.preventDefault();
      if (e.shiftKey) {
        prevSearchMatch();
      } else {
        if (searchMatches.length === 0) {
          performSearch();
        } else {
          nextSearchMatch();
        }
      }
    }
  }



  function jumpToLineNumber(lineNum: number) {
    // Check if the line is actually in the loaded range
    if (lineNum >= file.startLine && lineNum <= file.endLine) {
      // Line is loaded, just scroll to it
      isScrollingToTarget = true;
      scrollToTargetLine(lineNum);
    } else {
      // Line is not loaded, use the store's jumpToLine which will load it
      files.jumpToLine(file.path, lineNum);
    }
  }

  function openDashGoto() {
    // Calculate center line
    const centerLine = Math.round((file.startLine + file.endLine) / 2);
    dashGotoLineNumber = centerLine.toString();
    dashGotoVisible = true;

    // Focus input after it's rendered
    setTimeout(() => {
      dashGotoInputEl?.focus();
      dashGotoInputEl?.select();
    }, 0);
  }

  function closeDashGoto() {
    dashGotoVisible = false;
    dashGotoLineNumber = '';
  }

  function handleDashGotoKeyDown(e: KeyboardEvent) {
    if (e.key === 'Enter') {
      e.preventDefault();
      const lineNum = parseInt(dashGotoLineNumber, 10);
      if (!isNaN(lineNum) && lineNum > 0) {
        files.jumpToLine(file.path, lineNum);
        closeDashGoto();
      }
    } else if (e.key === 'Escape') {
      e.preventDefault();
      closeDashGoto();
    }
  }

  function scrollToTargetLine(targetLine: number) {
    // Use requestAnimationFrame to ensure DOM is fully rendered
    requestAnimationFrame(() => {
      if (!contentEl) return;

      const lineEl = contentEl.querySelector(`[data-line="${targetLine}"]`);
      if (lineEl) {
        lineEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
        // Clear scroll position after scroll completes
        setTimeout(() => {
          files.clearScrollPosition(file.path);
          isScrollingToTarget = false;
          // Update tracking after target scroll completes
          if (contentEl) {
            previousStartLine = file.startLine;
            previousScrollHeight = contentEl.scrollHeight;
          }
        }, 500);
      } else {
        // If line element not found, retry after a short delay
        setTimeout(() => {
          const retryLineEl = contentEl?.querySelector(`[data-line="${targetLine}"]`);
          if (retryLineEl) {
            retryLineEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
            setTimeout(() => {
              files.clearScrollPosition(file.path);
              isScrollingToTarget = false;
              // Update tracking after target scroll completes
              if (contentEl) {
                previousStartLine = file.startLine;
                previousScrollHeight = contentEl.scrollHeight;
              }
            }, 500);
          } else {
            // Line not found even after retry, reset flag
            isScrollingToTarget = false;
          }
        }, 100);
      }
    });
  }

  // Debounced URL update on scroll
  const updateUrlOnScroll = debounce(() => {
    if (isActive && file.lines.length > 0) {
      const centerLine = getCurrentViewportCenterLine();
      updateUrlState({
        path: file.path,
        line: centerLine,
        syntaxHighlighting: file.syntaxHighlighting,
      });
    }
  }, 500); // Update URL 500ms after user stops scrolling

  function handleScroll() {
    if (!contentEl) return;

    const { scrollTop, scrollHeight, clientHeight } = contentEl;

    // Update URL on scroll (debounced)
    updateUrlOnScroll();

    // Don't trigger loads if we're scrolling to a target line
    if (file.scrollToLine !== undefined || isScrollingToTarget) {
      return;
    }

    // Don't trigger new loads while loading, unless we're at the very edge
    if (file.loading) {
      // Only trigger if we're actually at the edge (scrolled past the threshold)
      const atTop = scrollTop === 0;
      const atBottom = scrollHeight - scrollTop - clientHeight === 0;

      if (!atTop && !atBottom) {
        return;
      }
    }

    // Load more when near top (200px threshold) or at the very top
    if ((scrollTop < 200 || scrollTop === 0) && file.startLine > 1) {
      files.loadMore(file.path, 'before');
    }

    // Load more when near bottom (200px threshold) or at the very bottom
    const distanceFromBottom = scrollHeight - scrollTop - clientHeight;
    if (distanceFromBottom < 200 || distanceFromBottom === 0) {
      files.loadMore(file.path, 'after');
    }
  }

  // Preserve scroll position when loading content at the top
  // But NOT when we're scrolling to a target line
  $: if (contentEl && file.startLine < previousStartLine && !file.loading && !file.scrollToLine) {
    // Content was added at the top - adjust scroll to keep same lines visible
    const newScrollHeight = contentEl.scrollHeight;
    const addedHeight = newScrollHeight - previousScrollHeight;

    if (addedHeight > 0) {
      contentEl.scrollTop += addedHeight;
    }

    previousStartLine = file.startLine;
    previousScrollHeight = newScrollHeight;
  }

  // Update tracking variables when file changes (but not during target scroll)
  $: if (contentEl && !file.scrollToLine) {
    previousStartLine = file.startLine;
    previousScrollHeight = contentEl.scrollHeight;
  }

  // Check if we need to load more after loading completes (only for bottom edge)
  $: if (!file.loading && contentEl && !isScrollingToTarget) {
    // Use setTimeout to ensure DOM is updated
    setTimeout(() => {
      if (!contentEl || isScrollingToTarget) return;

      const { scrollTop, scrollHeight, clientHeight } = contentEl;
      const atBottom = scrollHeight - scrollTop - clientHeight === 0;

      // Only auto-load at bottom edge
      // For top edge, user needs to manually scroll up
      // This prevents the multiple fetches when jumping to a line near the top
      if (atBottom) {
        files.loadMore(file.path, 'after');
      }
    }, 100);
  }

  function handleClose() {
    files.closeFile(file.path);
  }

  function isMatchLine(lineNumber: number): boolean {
    return fileMatches.some((m) => m.lineNumber === lineNumber);
  }

  function isSearchMatch(lineNumber: number): boolean {
    return searchMatches.includes(lineNumber);
  }

  function isCurrentSearchMatch(lineNumber: number): boolean {
    return currentMatchIndex >= 0 && searchMatches[currentMatchIndex] === lineNumber;
  }

  // Calculate the maximum line number width based on loaded lines
  // Dynamically grows to accommodate any line number length
  $: maxLineNumberWidth = file.lines.length > 0
    ? Math.max(...file.lines.map(l => l.lineNumber)).toString().length
    : 7;

  function formatLineNumber(n: number): string {
    // Pad line numbers for alignment using the maximum line number in loaded scope
    return n.toString().padStart(maxLineNumberWidth, ' ');
  }

  function getMatchPatterns(lineNumber: number): string[] {
    return fileMatches
      .filter((m) => m.lineNumber === lineNumber)
      .map((m) => m.pattern);
  }

  function getLineMarkerColor(lineNumber: number): string {
    // No marks by default - mechanism ready for future custom logic
    return 'transparent';
  }

  function replaceInvisibleChars(html: string): string {
    // Replace invisible characters with visible symbols in HTML
    // This needs to work with syntax-highlighted HTML, so we replace carefully
    // to avoid breaking HTML tags

    // Use a regex that avoids replacing inside HTML tags
    // This regex matches text content outside of < and >
    const textOutsideTags = /(?:^|>)([^<]*)(?=<|$)/g;

    return html.replace(textOutsideTags, (match, textContent, offset) => {
      // Keep the delimiters (> or start/end), only transform the text content
      const prefix = match[0] === '>' ? '>' : '';
      const transformed = textContent
        // Space → middle dot (·)
        .replace(/ /g, '<span class="invisible-char">·</span>')
        // Tab → arrow (→)
        .replace(/\t/g, '<span class="invisible-char">→</span>')
        // Carriage return (CR) → ⏎
        .replace(/\r/g, '<span class="invisible-char">⏎</span>')
        // Non-breaking space → ° (degree symbol)
        .replace(/\u00A0/g, '<span class="invisible-char">°</span>')
        // Zero-width space → |
        .replace(/\u200B/g, '<span class="invisible-char">|</span>')
        // Zero-width non-joiner → |
        .replace(/\u200C/g, '<span class="invisible-char">|</span>')
        // Zero-width joiner → |
        .replace(/\u200D/g, '<span class="invisible-char">|</span>')
        // Word joiner → |
        .replace(/\u2060/g, '<span class="invisible-char">|</span>')
        // Zero-width no-break space (BOM) → |
        .replace(/\uFEFF/g, '<span class="invisible-char">|</span>');

      return prefix + transformed;
    });
  }

  function highlightContent(content: string, lineNumber: number, patterns: string[]): string {
    function escapeHtml(text: string): string {
      return text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
    }

    // STEP 1: Apply regex filter first (if enabled)
    // This returns already-escaped HTML with filter transformations
    if (file.regexFilter && file.regexFilter.enabled && file.regexFilter.compiledRegex) {
      let result = applyRegexFilter(content, file.regexFilter);

      // For highlight mode, we can still apply syntax highlighting before the filter
      // For hide/show modes, we skip syntax highlighting as the content is already transformed
      if (file.regexFilter.mode === 'highlight' && syntaxHighlighting) {
        // Re-apply syntax highlighting on the original content first
        result = highlightLine(content, detectedLanguage);
        // Then apply regex highlighting on top
        if (file.regexFilter.compiledRegex) {
          const regex = new RegExp(file.regexFilter.pattern, 'g');
          result = result.replace(regex, (match, ...groups) => {
            // Highlight the entire match
            return `<span class="regex-highlight">${match}</span>`;
          });
        }
      }

      // Apply trace search pattern highlighting
      if (patterns.length > 0) {
        for (const pattern of patterns) {
          try {
            const regex = new RegExp(pattern, 'g');
            result = result.replace(regex, (match) => {
              return `<strong style="background-color: rgba(139, 69, 19, 0.2); border-bottom: 2px solid #8B4513; font-weight: 700;">${match}</strong>`;
            });
          } catch (e) {
            console.warn('Invalid regex pattern:', pattern, e);
          }
        }
      }

      return result;
    }

    // STEP 2: Normal flow (no regex filter) - apply syntax highlighting
    let result = syntaxHighlighting ? highlightLine(content, detectedLanguage) : escapeHtml(content);

    // STEP 3: Apply invisible characters replacement if enabled
    if (file.showInvisibleChars) {
      result = replaceInvisibleChars(result);
    }

    // Highlight regex matches (from trace search) - wrap with strong tag
    if (patterns.length > 0) {
      for (const pattern of patterns) {
        try {
          const regex = new RegExp(pattern, 'g');
          result = result.replace(regex, (match) => {
            return `<strong style="background-color: rgba(139, 69, 19, 0.2); border-bottom: 2px solid #8B4513; font-weight: 700;">${match}</strong>`;
          });
        } catch (e) {
          console.warn('Invalid regex pattern:', pattern, e);
        }
      }
    }

    // Highlight in-file search matches
    if (searchQuery && content.toLowerCase().includes(searchQuery.toLowerCase())) {
      const isCurrentMatch = isCurrentSearchMatch(lineNumber);
      const bgColor = isCurrentMatch ? '#ff9632' : '#ffeb3b';
      const query = searchQuery.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); // Escape special chars
      const regex = new RegExp(`(${query})`, 'gi');
      result = result.replace(regex, (match) => {
        return `<mark style="background-color: ${bgColor}; color: #000; padding: 0 2px; font-weight: 600;">${match}</mark>`;
      });
    }

    return result;
  }

  onMount(() => {
    paneEl?.addEventListener('keydown', handleKeyDown);
  });

  onDestroy(() => {
    paneEl?.removeEventListener('keydown', handleKeyDown);
  });
</script>

<div
  bind:this={paneEl}
  tabindex="-1"
  class="h-full w-full flex flex-col outline-none
         border-r border-gh-border-default dark:border-gh-border-dark-default
         last:border-r-0"
>
  <!-- File header -->
  <div
    class="flex items-center justify-between px-3 py-1.5 gap-2
           bg-gh-canvas-subtle dark:bg-gh-canvas-dark-subtle
           border-b border-gh-border-default dark:border-gh-border-dark-default"
  >
    <div class="flex items-center gap-2 min-w-0">
      <span class="text-sm font-medium truncate" title={file.path}>
        {file.name}
      </span>
      <FileBadges
        isCompressed={file.isCompressed}
        compressionFormat={file.compressionFormat}
        isIndexed={null}
      />
      {#if file.totalLines !== null}
        <span class="text-xs text-gh-fg-muted dark:text-gh-fg-dark-muted">
          {file.totalLines.toLocaleString()} lines
        </span>
      {:else if file.lines.length > 0}
        <span class="text-xs text-gh-fg-muted dark:text-gh-fg-dark-muted flex items-center gap-1">
          <span>lines</span>
          <button
            class="font-bold hover:text-gh-accent-fg dark:hover:text-gh-accent-dark-fg hover:underline"
            on:click={() => jumpToLineNumber(file.startLine)}
            title="Jump to line {file.startLine}"
          >
            {file.startLine.toLocaleString()}
          </button>
          {#if dashGotoVisible}
            <input
              bind:this={dashGotoInputEl}
              bind:value={dashGotoLineNumber}
              on:keydown={handleDashGotoKeyDown}
              on:blur={closeDashGoto}
              type="number"
              class="text-xs bg-gh-canvas-default dark:bg-gh-canvas-dark-default border border-gh-border-default dark:border-gh-border-dark-default rounded px-1 outline-none w-20 text-center font-bold"
            />
          {:else}
            <button
              class="font-bold hover:text-gh-accent-fg dark:hover:text-gh-accent-dark-fg hover:underline"
              on:click={openDashGoto}
              title="Jump to line..."
            >
              —
            </button>
          {/if}
          <button
            class="font-bold hover:text-gh-accent-fg dark:hover:text-gh-accent-dark-fg hover:underline"
            on:click={() => jumpToLineNumber(file.endLine)}
            title="Jump to line {file.endLine}"
          >
            {file.endLine.toLocaleString()}
          </button>
        </span>
      {/if}
    </div>

    <div class="flex items-center gap-2">
      <!-- Search box -->
      {#if searchVisible}
        <div class="flex items-center gap-1 bg-gh-canvas-default dark:bg-gh-canvas-dark-default border border-gh-border-default dark:border-gh-border-dark-default rounded px-2 py-0.5">
          <input
            bind:this={searchInputEl}
            bind:value={searchQuery}
            on:keydown={handleSearchKeyDown}
            on:input={performSearch}
            placeholder="Search in file..."
            class="text-xs bg-transparent outline-none w-40"
          />
          {#if searchMatches.length > 0}
            <span class="text-xs text-gh-fg-muted dark:text-gh-fg-dark-muted">
              {currentMatchIndex + 1}/{searchMatches.length}
            </span>
          {/if}
          <button
            on:click={closeSearch}
            class="text-gh-fg-muted dark:text-gh-fg-dark-muted hover:text-gh-fg-default dark:hover:text-gh-fg-dark-default"
            title="Close search (Esc)"
          >
            <svg class="w-3 h-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M18 6L6 18M6 6l12 12" />
            </svg>
          </button>
        </div>
      {/if}

      <!-- Syntax highlighting toggle -->
      <button
        class="p-1 rounded flex-shrink-0 transition-colors
               {syntaxHighlighting
                 ? 'bg-gh-accent-emphasis dark:bg-gh-accent-dark-emphasis text-white hover:bg-gh-accent-fg dark:hover:bg-gh-accent-dark-fg'
                 : 'bg-gh-canvas-inset dark:bg-gh-canvas-dark-inset text-gh-fg-muted dark:text-gh-fg-dark-muted hover:bg-gh-canvas-subtle dark:hover:bg-gh-canvas-dark-subtle'}"
        title="{syntaxHighlighting ? 'Disable' : 'Enable'} syntax highlighting"
        on:click={toggleSyntaxHighlighting}
      >
        <svg class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M16 18L22 12L16 6M8 6L2 12L8 18" />
        </svg>
      </button>

      <!-- Invisible characters toggle -->
      <button
        class="p-1 rounded flex-shrink-0 transition-colors
               {file.showInvisibleChars
                 ? 'bg-gh-accent-emphasis dark:bg-gh-accent-dark-emphasis text-white hover:bg-gh-accent-fg dark:hover:bg-gh-accent-dark-fg'
                 : 'bg-gh-canvas-inset dark:bg-gh-canvas-dark-inset text-gh-fg-muted dark:text-gh-fg-dark-muted hover:bg-gh-canvas-subtle dark:hover:bg-gh-canvas-dark-subtle'}"
        title="{file.showInvisibleChars ? 'Hide' : 'Show'} invisible characters"
        on:click={toggleInvisibleChars}
      >
        <svg class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M6 12h.01M12 12h.01M18 12h.01M6 18h.01M12 18h.01M18 18h.01M6 6h.01M12 6h.01M18 6h.01" stroke-linecap="round" />
        </svg>
      </button>

      <!-- Regex filter toggle -->
      <button
        class="p-1 rounded flex-shrink-0 transition-colors
               {file.regexFilter?.enabled
                 ? 'bg-gh-accent-emphasis dark:bg-gh-accent-dark-emphasis text-white hover:bg-gh-accent-fg dark:hover:bg-gh-accent-dark-fg'
                 : 'bg-gh-canvas-inset dark:bg-gh-canvas-dark-inset text-gh-fg-muted dark:text-gh-fg-dark-muted hover:bg-gh-canvas-subtle dark:hover:bg-gh-canvas-dark-subtle'}"
        title="Regex filter"
        on:click={toggleFilterPanel}
      >
        <svg class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M3 4a1 1 0 011-1h12a1 1 0 011 1v2.586a1 1 0 01-.293.707l-4.414 4.414a1 1 0 00-.293.707V17l-4 2v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z" />
        </svg>
      </button>

      <button
        class="p-1 rounded hover:bg-gh-canvas-inset dark:hover:bg-gh-canvas-dark-inset
               text-gh-fg-muted dark:text-gh-fg-dark-muted flex-shrink-0"
        title="Close"
        on:click={handleClose}
      >
        <svg class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M18 6L6 18M6 6l12 12" />
        </svg>
      </button>
    </div>
  </div>

  <!-- Regex filter panel (expandable) -->
  {#if filterPanelVisible}
    <div class="px-3 py-2 bg-gh-canvas-subtle dark:bg-gh-canvas-dark-subtle border-b border-gh-border-default dark:border-gh-border-dark-default">
      <div class="flex items-center gap-2 mb-2">
        <label for="regex-filter-input" class="text-xs font-medium text-gh-fg-muted dark:text-gh-fg-dark-muted">
          Regex:
        </label>
        <div
          id="regex-filter-input"
          bind:this={regexInputEl}
          contenteditable="plaintext-only"
          on:input={handleRegexInput}
          on:keydown={handleFilterKeyDown}
          role="textbox"
          tabindex="0"
          aria-label="Regex pattern"
          data-placeholder="e.g. (\w+)@(\w+)\.com"
          class="flex-1 text-sm bg-gh-canvas-default dark:bg-gh-canvas-dark-default
                 border border-gh-border-default dark:border-gh-border-dark-default
                 rounded px-3 py-1.5 outline-none focus:border-gh-accent-fg dark:focus:border-gh-accent-dark-fg
                 font-mono regex-input min-h-[28px]"
        ></div>
        <button
          on:click={applyFilter}
          class="px-3 py-1 text-xs font-medium rounded
                 bg-gh-accent-emphasis dark:bg-gh-accent-dark-emphasis text-white
                 hover:bg-gh-accent-fg dark:hover:bg-gh-accent-dark-fg"
        >
          Apply
        </button>
        <button
          on:click={clearFilter}
          class="px-3 py-1 text-xs font-medium rounded
                 bg-gh-canvas-inset dark:bg-gh-canvas-dark-inset
                 text-gh-fg-muted dark:text-gh-fg-dark-muted
                 hover:bg-gh-canvas-subtle dark:hover:bg-gh-canvas-dark-subtle"
        >
          Cancel
        </button>
      </div>

      <div class="flex items-center gap-4">
        <span class="text-xs font-medium text-gh-fg-muted dark:text-gh-fg-dark-muted">
          Mode:
        </span>
        <label class="flex items-center gap-1 text-xs cursor-pointer">
          <input
            type="radio"
            bind:group={filterMode}
            value="hide"
            on:change={applyFilter}
            class="cursor-pointer"
          />
          <span>Hide groups</span>
        </label>
        <label class="flex items-center gap-1 text-xs cursor-pointer">
          <input
            type="radio"
            bind:group={filterMode}
            value="show"
            on:change={applyFilter}
            class="cursor-pointer"
          />
          <span>Show only</span>
        </label>
        <label class="flex items-center gap-1 text-xs cursor-pointer">
          <input
            type="radio"
            bind:group={filterMode}
            value="highlight"
            on:change={applyFilter}
            class="cursor-pointer"
          />
          <span>Highlight</span>
        </label>
      </div>

      {#if file.regexFilter?.error}
        <div class="mt-2 text-xs text-gh-danger-fg dark:text-gh-danger-dark-fg">
          {file.regexFilter.error}
        </div>
      {/if}
    </div>
  {/if}

  <!-- Content container -->
  <div
    bind:this={contentEl}
    class="flex-1 min-h-0 overflow-y-scroll font-mono"
    style="font-size: {fontSize}px;"
    on:scroll={handleScroll}
  >
    {#if file.loading && file.lines.length === 0}
      <div class="flex items-center justify-center h-full">
        <Spinner size="lg" />
      </div>
    {:else if file.error}
      <div class="flex items-center justify-center h-full text-gh-danger-fg dark:text-gh-danger-dark-fg">
        <div class="text-center p-4">
          <p class="font-medium">Failed to load file</p>
          <p class="text-sm mt-1 opacity-75">{file.error}</p>
        </div>
      </div>
    {:else if file.lines.length === 0}
      <div class="flex items-center justify-center h-full text-gh-fg-muted dark:text-gh-fg-dark-muted">
        <p>Empty file</p>
      </div>
    {:else}
      <!-- Load more indicator (top) -->
      {#if file.loading && file.startLine > 1}
        <div class="flex items-center justify-center py-2">
          <Spinner size="sm" />
        </div>
      {:else if file.startLine > 1}
        <button
          class="w-full py-1 text-xs text-gh-accent-fg dark:text-gh-accent-dark-fg
                 hover:bg-gh-canvas-subtle dark:hover:bg-gh-canvas-dark-subtle"
          on:click={() => files.loadMore(file.path, 'before')}
        >
          Load earlier lines...
        </button>
      {/if}

      <table class="w-full border-collapse">
        <tbody>
          {#key syntaxHighlighting}
            {#each file.lines as line (line.lineNumber)}
              {@const isMatch = isMatchLine(line.lineNumber)}
              {@const isSearch = isSearchMatch(line.lineNumber)}
              {@const patterns = getMatchPatterns(line.lineNumber)}
              {@const highlightedContent = highlightContent(line.content, line.lineNumber, patterns)}
              {@const markerColor = getLineMarkerColor(line.lineNumber)}
              <tr
                data-line={line.lineNumber}
                class="{isMatch
                  ? 'bg-gh-attention-emphasis/10 dark:bg-gh-attention-dark-emphasis/20 border-l-2 border-gh-attention-fg dark:border-gh-attention-dark-fg'
                  : isSearch
                  ? 'bg-yellow-50 dark:bg-yellow-900/20'
                  : 'hover:bg-gh-canvas-subtle dark:hover:bg-gh-canvas-dark-subtle'}"
              >
                {#if showLineNumbers}
                  <td
                    class="select-none text-right pr-2 pl-2 text-gh-fg-subtle dark:text-gh-fg-dark-subtle
                           align-top whitespace-nowrap line-number-col"
                    style="line-height: {fontSize * 1.5}px; width: {maxLineNumberWidth * 0.65 + 2}em; min-width: {maxLineNumberWidth * 0.65 + 2}em; max-width: {maxLineNumberWidth * 0.65 + 2}em;"
                  >
                    {formatLineNumber(line.lineNumber)}
                  </td>
                {/if}
                <!-- Line marker column -->
                <td
                  class="line-marker-col select-none p-0 align-top border-r border-gh-border-default dark:border-gh-border-dark-default"
                  style="line-height: {fontSize * 1.5}px; background-color: {markerColor}; width: 5px; min-width: 5px; max-width: 5px;"
                >
                  &nbsp;
                </td>
                <td
                  class="pl-3 pr-2 whitespace-pre"
                  style="line-height: {fontSize * 1.5}px"
                >
                  {@html highlightedContent}
                </td>
              </tr>
            {/each}
          {/key}
        </tbody>
      </table>

      <!-- Load more indicator (bottom) -->
      {#if file.loading}
        <div class="flex items-center justify-center py-2">
          <Spinner size="sm" />
        </div>
      {:else}
        <button
          class="w-full py-1 text-xs text-gh-accent-fg dark:text-gh-accent-dark-fg
                 hover:bg-gh-canvas-subtle dark:hover:bg-gh-canvas-dark-subtle"
          on:click={() => files.loadMore(file.path, 'after')}
        >
          Load more lines...
        </button>
      {/if}
    {/if}
  </div>
</div>

<style>
  /* Regex filter styles */
  :global(.hidden-bar) {
    display: inline-block;
    width: 5px;
    font-weight: bold;
  }

  :global(.hidden-bar.red) {
    background-color: #ef4444; /* Bright red */
  }

  :global(.hidden-bar.blue) {
    background-color: #3b82f6; /* Bright blue */
  }

  :global(.regex-highlight) {
    background-color: rgba(255, 235, 59, 0.4);
    padding: 0 2px;
    border-radius: 2px;
  }

  /* Regex input contenteditable styles */
  .regex-input {
    white-space: pre;
    overflow-x: auto;
  }

  .regex-input:empty:before {
    content: attr(data-placeholder);
    color: #9ca3af;
  }

  :global(.regex-placeholder) {
    color: #9ca3af;
    pointer-events: none;
  }

  /* Prism.js regex syntax highlighting */
  :global(.token.char-class) {
    color: #0ea5e9; /* Blue for character classes like \w \d */
  }

  :global(.token.quantifier) {
    color: #f59e0b; /* Orange for quantifiers like + * ? */
  }

  :global(.token.anchor) {
    color: #8b5cf6; /* Purple for anchors like ^ $ */
  }

  :global(.token.group) {
    color: #10b981; /* Green for groups () */
    font-weight: 600;
  }

  :global(.token.alternation) {
    color: #ef4444; /* Red for | */
  }

  :global(.token.escape) {
    color: #06b6d4; /* Cyan for escape sequences */
  }

  :global(.token.char-set) {
    color: #8b5cf6; /* Purple for character sets [] */
  }

  /* Fixed width line numbers column */
  .line-number-col {
    box-sizing: border-box;
    overflow: hidden;
  }

  /* Line marker column - thin vertical bar for line status indicators */
  .line-marker-col {
    box-sizing: border-box;
    overflow: hidden;
    padding: 0 !important;
  }

  /* Invisible characters display */
  :global(.invisible-char) {
    color: #9ca3af;
    user-select: none;
  }
</style>
