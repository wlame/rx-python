import { writable } from 'svelte/store';
import { api } from '../api';
import type { TraceMatch, TraceResponse } from '../types';

interface TraceState {
  query: string;
  isRegex: boolean;
  caseSensitive: boolean;
  searching: boolean;
  response: TraceResponse | null;
  error: string | null;
}

function createTraceStore() {
  const { subscribe, set, update } = writable<TraceState>({
    query: '',
    isRegex: true, // ripgrep uses regex by default
    caseSensitive: true,
    searching: false,
    response: null,
    error: null,
  });

  async function search(
    paths: string[],
    patterns: string[],
    maxResults?: number
  ) {
    if (patterns.length === 0) return;

    update((s) => ({
      ...s,
      query: patterns.join(' | '),
      searching: true,
      error: null,
      response: null,
    }));

    try {
      const response = await api.trace(
        paths,
        patterns,
        maxResults
      );

      update((s) => ({
        ...s,
        searching: false,
        response,
      }));

      return response;
    } catch (e) {
      update((s) => ({
        ...s,
        searching: false,
        error: e instanceof Error ? e.message : 'Search failed',
      }));
      return null;
    }
  }

  function setQuery(query: string) {
    update((s) => ({ ...s, query }));
  }

  function setIsRegex(isRegex: boolean) {
    update((s) => ({ ...s, isRegex }));
  }

  function setCaseSensitive(caseSensitive: boolean) {
    update((s) => ({ ...s, caseSensitive }));
  }

  function clear() {
    set({
      query: '',
      isRegex: true,
      caseSensitive: true,
      searching: false,
      response: null,
      error: null,
    });
  }

  /**
   * Get matches for a specific file path from the last search
   */
  function getMatchesForFile(filePath: string): TraceMatch[] {
    let matches: TraceMatch[] = [];

    const state = getState();
    if (!state.response) return matches;

    // Find the file ID for this path
    const fileId = Object.entries(state.response.files).find(
      ([, path]) => path === filePath
    )?.[0];

    if (!fileId) return matches;

    // Filter matches for this file
    return state.response.matches.filter((m) => m.file === fileId);
  }

  // Helper to get current state synchronously
  function getState(): TraceState {
    let state: TraceState;
    subscribe((s) => (state = s))();
    return state!;
  }

  return {
    subscribe,
    search,
    setQuery,
    setIsRegex,
    setCaseSensitive,
    clear,
    getMatchesForFile,
  };
}

export const trace = createTraceStore();
