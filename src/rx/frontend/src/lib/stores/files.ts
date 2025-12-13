import { writable, get } from 'svelte/store';
import { api } from '../api';
import type { OpenFile, FileLine, FileMatch } from '../types';
import { notifications } from './notifications';
import { settings } from './settings';

interface FilesState {
  openFiles: OpenFile[];
  matches: Map<string, FileMatch[]>; // path -> matches
  activeFilePath: string | null; // Currently active/focused file
}

function createFilesStore() {
  const { subscribe, set, update } = writable<FilesState>({
    openFiles: [],
    matches: new Map(),
    activeFilePath: null,
  });

  /**
   * Open a file and load initial content
   */
  async function openFile(
    path: string,
    scrollToLine?: number,
    fileSize?: number | null,
    syntaxHighlightingOverride?: boolean
  ) {
    const state = get({ subscribe });

    // Check if already open
    const existingIndex = state.openFiles.findIndex((f) => f.path === path);
    if (existingIndex >= 0) {
      // File is already open - activate it and update scroll position if provided
      update((s) => ({
        ...s,
        activeFilePath: path,
        openFiles: scrollToLine !== undefined
          ? s.openFiles.map((f, i) =>
              i === existingIndex ? { ...f, scrollToLine } : f
            )
          : s.openFiles,
      }));
      return;
    }

    // Create placeholder file entry
    const name = path.split('/').pop() || path;

    // Default syntax highlighting: enabled for files < 1MB, disabled for larger files
    // Can be overridden by syntaxHighlightingOverride parameter (from URL state)
    const ONE_MB = 1024 * 1024;
    const defaultSyntaxHighlighting = fileSize === null || fileSize === undefined || fileSize < ONE_MB;
    const syntaxHighlighting = syntaxHighlightingOverride !== undefined
      ? syntaxHighlightingOverride
      : defaultSyntaxHighlighting;

    const newFile: OpenFile = {
      path,
      name,
      lines: [],
      totalLines: null,
      startLine: 1,
      endLine: 0,
      loading: true,
      error: null,
      isCompressed: false,
      compressionFormat: null,
      scrollToLine,
      reachedStart: false,
      reachedEnd: false,
      syntaxHighlighting,
      fileSize: fileSize ?? null,
      regexFilter: null, // Disabled by default
      showInvisibleChars: false, // Disabled by default
    };

    // Add file to the end and make it active
    update((s) => ({
      ...s,
      openFiles: [...s.openFiles, newFile],
      activeFilePath: path,
    }));

    // Load initial content
    const linesPerPage = get(settings).linesPerPage;
    if (scrollToLine) {
      // If scrolling to a specific line, load around that line
      await loadLinesAroundCenter(path, scrollToLine, linesPerPage);
    } else {
      // When opening a file, always start from line 1
      await loadLinesFromStart(path, linesPerPage);
    }
  }

  /**
   * Load lines from the start of the file
   * This loads line 1 with specified context after it
   */
  async function loadLinesFromStart(path: string, totalLines: number) {
    update((s) => ({
      ...s,
      openFiles: s.openFiles.map((f) =>
        f.path === path ? { ...f, loading: true, error: null } : f
      ),
    }));

    try {
      const context = totalLines - 1; // Request line 1 with context to get totalLines total

      // Request line 1 with large after context
      const response = await api.getSamples(path, [1], context);

      // Parse response - samples key is the line number as string
      const lines: FileLine[] = [];

      for (const [lineNumStr, contentArr] of Object.entries(response.samples)) {
        const lineNum = parseInt(lineNumStr, 10);

        // The API returns context around the requested line.
        // The requested line appears at position min(lineNum - 1, response.before_context) in the array.
        // For example: requesting line 1 with before=500, after=500
        //   - The array starts at line max(1, 1-500) = 1
        //   - Line 1 is at index 0 (since there are no lines before it)
        // Another example: requesting line 100 with before=50, after=50
        //   - The array starts at line max(1, 100-50) = 50
        //   - Line 100 is at index 50 (100 - 50 = 50)

        const actualBeforeContext = Math.min(lineNum - 1, response.before_context);
        const startLineNum = lineNum - actualBeforeContext;

        for (let i = 0; i < contentArr.length; i++) {
          const actualLineNum = startLineNum + i;
          lines.push({
            lineNumber: actualLineNum,
            content: contentArr[i],
          });
        }
      }

      // Sort by line number and dedupe
      const lineMap = new Map<number, FileLine>();
      for (const line of lines) {
        lineMap.set(line.lineNumber, line);
      }
      const sortedLines = Array.from(lineMap.values()).sort(
        (a, b) => a.lineNumber - b.lineNumber
      );

      update((s) => ({
        ...s,
        openFiles: s.openFiles.map((f) =>
          f.path === path
            ? {
                ...f, // Preserve all existing fields including scrollToLine
                lines: sortedLines,
                startLine: sortedLines.length > 0 ? sortedLines[0].lineNumber : 1,
                endLine: sortedLines.length > 0 ? sortedLines[sortedLines.length - 1].lineNumber : 0,
                loading: false,
                isCompressed: response.is_compressed,
                compressionFormat: response.compression_format,
              }
            : f
        ),
      }));
    } catch (e) {
      const errorMessage = e instanceof Error ? e.message : 'Failed to load file';

      // Check if it's a binary file error
      if (errorMessage.includes('binary') || errorMessage.includes('Binary')) {
        // Show notification instead of creating an error tab
        notifications.error(`Cannot open binary file: ${path.split('/').pop()}`, 5000);

        // Remove the file from open files
        update((s) => ({
          ...s,
          openFiles: s.openFiles.filter((f) => f.path !== path),
        }));
      } else if (errorMessage.includes('EOF reached') || errorMessage.includes('out of bounds')) {
        // Handle EOF errors specifically
        update((s) => ({
          ...s,
          openFiles: s.openFiles.map((f) =>
            f.path === path
              ? {
                  ...f,
                  loading: false,
                  error: errorMessage,
                }
              : f
          ),
        }));
      } else {
        // For other errors, show in the tab
        update((s) => ({
          ...s,
          openFiles: s.openFiles.map((f) =>
            f.path === path
              ? {
                  ...f,
                  loading: false,
                  error: errorMessage,
                }
              : f
          ),
        }));
      }
      console.error('Failed to load file:', e);
    }
  }

  /**
   * Load lines around a center line using context
   * This uses the backend's context feature efficiently
   */
  async function loadLinesAroundCenter(path: string, centerLine: number, totalLines: number) {
    update((s) => ({
      ...s,
      openFiles: s.openFiles.map((f) =>
        f.path === path ? { ...f, loading: true, error: null } : f
      ),
    }));

    try {
      // Calculate context needed
      // We want totalLines around centerLine
      const context = Math.floor(totalLines / 2);

      // For lines near the start, request a line further down to get more content
      const requestLine = centerLine < context ? context : centerLine;

      // Request single line with large context
      const response = await api.getSamples(path, [requestLine], context);

      // Parse response - samples key is the line number as string
      const lines: FileLine[] = [];

      for (const [lineNumStr, contentArr] of Object.entries(response.samples)) {
        const lineNum = parseInt(lineNumStr, 10);

        // The response includes context_before + the line + context_after
        // Each entry in contentArr is a line
        for (let i = 0; i < contentArr.length; i++) {
          const actualLineNum = lineNum - response.before_context + i;
          if (actualLineNum > 0) { // Skip negative line numbers
            lines.push({
              lineNumber: actualLineNum,
              content: contentArr[i],
            });
          }
        }
      }

      // Sort by line number and dedupe
      const lineMap = new Map<number, FileLine>();
      for (const line of lines) {
        lineMap.set(line.lineNumber, line);
      }
      const sortedLines = Array.from(lineMap.values()).sort(
        (a, b) => a.lineNumber - b.lineNumber
      );

      update((s) => ({
        ...s,
        openFiles: s.openFiles.map((f) =>
          f.path === path
            ? {
                ...f, // Preserve all existing fields including scrollToLine
                lines: sortedLines,
                startLine: sortedLines.length > 0 ? sortedLines[0].lineNumber : 1,
                endLine: sortedLines.length > 0 ? sortedLines[sortedLines.length - 1].lineNumber : 0,
                loading: false,
                isCompressed: response.is_compressed,
                compressionFormat: response.compression_format,
              }
            : f
        ),
      }));
    } catch (e) {
      const errorMessage = e instanceof Error ? e.message : 'Failed to load file';

      // Check if it's a binary file error
      if (errorMessage.includes('binary') || errorMessage.includes('Binary')) {
        // Show notification instead of creating an error tab
        notifications.error(`Cannot open binary file: ${path.split('/').pop()}`, 5000);

        // Remove the file from open files
        update((s) => ({
          ...s,
          openFiles: s.openFiles.filter((f) => f.path !== path),
        }));
      } else if (errorMessage.includes('EOF reached') || errorMessage.includes('out of bounds')) {
        // If we tried to load around a line that's out of bounds, fall back to loading from start
        console.warn(`Line ${centerLine} is out of bounds, loading from start instead`);
        await loadLinesFromStart(path, totalLines);
      } else {
        // For other errors, show in the tab
        update((s) => ({
          ...s,
          openFiles: s.openFiles.map((f) =>
            f.path === path
              ? {
                  ...f,
                  loading: false,
                  error: errorMessage,
                }
              : f
          ),
        }));
      }
      console.error('Failed to load file:', e);
    }
  }

  /**
   * Load more lines (before or after current content)
   */
  async function loadMore(path: string, direction: 'before' | 'after') {
    const state = get({ subscribe });
    const file = state.openFiles.find((f) => f.path === path);
    if (!file || file.loading || file.lines.length === 0) return;

    // Check if we've already reached the boundary
    if (direction === 'before' && file.reachedStart) return;
    if (direction === 'after' && file.reachedEnd) return;

    const linesPerPage = get(settings).linesPerPage;
    const centerLine = direction === 'before'
      ? Math.max(1, file.startLine - linesPerPage / 2)
      : file.endLine + linesPerPage / 2;

    update((s) => ({
      ...s,
      openFiles: s.openFiles.map((f) =>
        f.path === path ? { ...f, loading: true } : f
      ),
    }));

    try {
      const context = Math.floor(linesPerPage / 2);
      const response = await api.getSamples(path, [Math.floor(centerLine)], context);

      // Parse response
      const newLines: FileLine[] = [];

      for (const [lineNumStr, contentArr] of Object.entries(response.samples)) {
        const lineNum = parseInt(lineNumStr, 10);

        for (let i = 0; i < contentArr.length; i++) {
          const actualLineNum = lineNum - response.before_context + i;
          if (actualLineNum > 0) {
            newLines.push({
              lineNumber: actualLineNum,
              content: contentArr[i],
            });
          }
        }
      }

      update((s) => ({
        ...s,
        openFiles: s.openFiles.map((f) => {
          if (f.path !== path) return f;

          // Merge with existing lines
          const lineMap = new Map<number, FileLine>();

          // Add existing lines
          for (const line of f.lines) {
            lineMap.set(line.lineNumber, line);
          }

          // Add new lines
          for (const line of newLines) {
            lineMap.set(line.lineNumber, line);
          }

          const mergedLines = Array.from(lineMap.values()).sort(
            (a, b) => a.lineNumber - b.lineNumber
          );

          const newStartLine = mergedLines.length > 0 ? mergedLines[0].lineNumber : 1;
          const newEndLine = mergedLines.length > 0 ? mergedLines[mergedLines.length - 1].lineNumber : 0;

          return {
            ...f,
            lines: mergedLines,
            startLine: newStartLine,
            endLine: newEndLine,
            loading: false,
            reachedStart: newStartLine === 1,
          };
        }),
      }));
    } catch (e) {
      const errorMessage = e instanceof Error ? e.message : String(e);
      const isEOF = errorMessage.includes('EOF reached') || errorMessage.includes('out of bounds');

      update((s) => ({
        ...s,
        openFiles: s.openFiles.map((f) => {
          if (f.path !== path) return f;

          return {
            ...f,
            loading: false,
            // Mark boundary reached if we got EOF error
            reachedEnd: isEOF && direction === 'after',
            reachedStart: isEOF && direction === 'before',
          };
        }),
      }));

      if (!isEOF) {
        console.error('Failed to load more lines:', e);
      }
    }
  }

  /**
   * Jump to a specific line in a file
   */
  async function jumpToLine(path: string, lineNumber: number) {
    const state = get({ subscribe });
    const file = state.openFiles.find((f) => f.path === path);

    if (!file) {
      // Open file and scroll to line
      await openFile(path, lineNumber);
      return;
    }

    // Check if line is already loaded
    if (lineNumber >= file.startLine && lineNumber <= file.endLine) {
      // Line is already loaded, just scroll to it
      update((s) => ({
        ...s,
        openFiles: s.openFiles.map((f) =>
          f.path === path ? { ...f, scrollToLine: lineNumber } : f
        ),
      }));
    } else {
      // Set scroll position BEFORE loading to prevent scroll handlers from triggering
      update((s) => ({
        ...s,
        openFiles: s.openFiles.map((f) =>
          f.path === path ? { ...f, scrollToLine: lineNumber } : f
        ),
      }));

      // Load lines around the target
      const linesPerPage = get(settings).linesPerPage;
      await loadLinesAroundCenter(path, lineNumber, linesPerPage);
    }
  }

  /**
   * Close a file
   */
  function closeFile(path: string) {
    update((s) => ({
      ...s,
      openFiles: s.openFiles.filter((f) => f.path !== path),
      matches: (() => {
        const newMatches = new Map(s.matches);
        newMatches.delete(path);
        return newMatches;
      })(),
    }));
  }

  /**
   * Set matches for a file (for highlighting)
   */
  function setMatches(path: string, matches: FileMatch[]) {
    update((s) => {
      const newMatches = new Map(s.matches);
      newMatches.set(path, matches);
      return { ...s, matches: newMatches };
    });
  }

  /**
   * Clear scroll position after scrolling is done
   */
  function clearScrollPosition(path: string) {
    update((s) => ({
      ...s,
      openFiles: s.openFiles.map((f) =>
        f.path === path ? { ...f, scrollToLine: undefined } : f
      ),
    }));
  }

  /**
   * Reorder files (for drag and drop)
   */
  function reorderFiles(fromIndex: number, toIndex: number) {
    update((s) => {
      const newOpenFiles = [...s.openFiles];
      const [movedFile] = newOpenFiles.splice(fromIndex, 1);
      newOpenFiles.splice(toIndex, 0, movedFile);
      return { ...s, openFiles: newOpenFiles };
    });
  }

  /**
   * Toggle syntax highlighting for a specific file
   */
  function toggleSyntaxHighlighting(path: string) {
    update((s) => ({
      ...s,
      openFiles: s.openFiles.map((f) =>
        f.path === path ? { ...f, syntaxHighlighting: !f.syntaxHighlighting } : f
      ),
    }));
  }

  /**
   * Toggle regex filter for a specific file
   */
  function toggleRegexFilter(path: string) {
    update((s) => ({
      ...s,
      openFiles: s.openFiles.map((f) => {
        if (f.path !== path) return f;

        if (f.regexFilter === null) {
          // Enable with default settings
          return {
            ...f,
            regexFilter: {
              enabled: true,
              pattern: '',
              mode: 'highlight',
              compiledRegex: null,
              error: null,
              applying: false,
            },
          };
        } else {
          // Toggle enabled state
          return {
            ...f,
            regexFilter: {
              ...f.regexFilter,
              enabled: !f.regexFilter.enabled,
            },
          };
        }
      }),
    }));
  }

  /**
   * Update regex filter configuration for a specific file
   */
  function updateRegexFilter(path: string, pattern: string, mode: 'hide' | 'show' | 'highlight') {
    update((s) => ({
      ...s,
      openFiles: s.openFiles.map((f) => {
        if (f.path !== path || !f.regexFilter) return f;

        let compiledRegex: RegExp | null = null;
        let error: string | null = null;

        if (pattern.trim()) {
          try {
            compiledRegex = new RegExp(pattern, 'g');
            // Test if regex has capturing groups
            if (compiledRegex.source.indexOf('(') === -1 || compiledRegex.source.indexOf('(?:') !== -1) {
              error = 'Regex must contain at least one capturing group ()';
              compiledRegex = null;
            }
          } catch (e) {
            error = e instanceof Error ? e.message : 'Invalid regex pattern';
            compiledRegex = null;
          }
        }

        return {
          ...f,
          regexFilter: {
            ...f.regexFilter,
            pattern,
            mode,
            compiledRegex,
            error,
          },
        };
      }),
    }));
  }

  /**
   * Clear regex filter for a specific file
   */
  function clearRegexFilter(path: string) {
    update((s) => ({
      ...s,
      openFiles: s.openFiles.map((f) =>
        f.path === path ? { ...f, regexFilter: null } : f
      ),
    }));
  }

  /**
   * Toggle invisible characters display for a specific file
   */
  function toggleInvisibleChars(path: string) {
    update((s) => ({
      ...s,
      openFiles: s.openFiles.map((f) =>
        f.path === path ? { ...f, showInvisibleChars: !f.showInvisibleChars } : f
      ),
    }));
  }

  return {
    subscribe,
    openFile,
    closeFile,
    loadMore,
    jumpToLine,
    setMatches,
    clearScrollPosition,
    reorderFiles,
    toggleSyntaxHighlighting,
    toggleRegexFilter,
    updateRegexFilter,
    clearRegexFilter,
    toggleInvisibleChars,
  };
}

export const files = createFilesStore();
