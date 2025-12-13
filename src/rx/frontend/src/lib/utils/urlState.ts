/**
 * Utility for managing application state in URL parameters
 */

export interface FileState {
  path: string;
  line: number;
  syntaxHighlighting: boolean;
}

/**
 * Update URL with current file state without reloading the page
 */
export function updateUrlState(state: FileState | null): void {
  if (typeof window === 'undefined') return;

  const url = new URL(window.location.href);

  if (state) {
    url.searchParams.set('file', state.path);
    url.searchParams.set('line', state.line.toString());
    url.searchParams.set('highlight', state.syntaxHighlighting ? '1' : '0');
  } else {
    // Clear file-related params
    url.searchParams.delete('file');
    url.searchParams.delete('line');
    url.searchParams.delete('highlight');
  }

  window.history.replaceState({}, '', url.toString());
}

/**
 * Read file state from URL parameters
 */
export function readUrlState(): FileState | null {
  if (typeof window === 'undefined') return null;

  const params = new URLSearchParams(window.location.search);
  const path = params.get('file');
  const lineStr = params.get('line');
  const highlightStr = params.get('highlight');

  if (!path) return null;

  const line = lineStr ? parseInt(lineStr, 10) : 1;
  const syntaxHighlighting = highlightStr === '1';

  return {
    path,
    line: isNaN(line) ? 1 : line,
    syntaxHighlighting,
  };
}

/**
 * Debounce function for scroll events
 */
export function debounce<T extends (...args: any[]) => any>(
  fn: T,
  delay: number
): (...args: Parameters<T>) => void {
  let timeoutId: ReturnType<typeof setTimeout> | null = null;

  return (...args: Parameters<T>) => {
    if (timeoutId) {
      clearTimeout(timeoutId);
    }
    timeoutId = setTimeout(() => {
      fn(...args);
    }, delay);
  };
}
