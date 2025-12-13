import type {
  HealthResponse,
  TreeResponse,
  SamplesResponse,
  TraceResponse,
  TaskStatus,
} from './types';

const API_BASE = '/v1';

class ApiError extends Error {
  constructor(
    public status: number,
    public statusText: string,
    message: string
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  });

  if (!response.ok) {
    const text = await response.text();
    throw new ApiError(response.status, response.statusText, text);
  }

  return response.json();
}

export const api = {
  /**
   * Check API health status
   */
  async getHealth(): Promise<HealthResponse> {
    return fetchJson<HealthResponse>('/health');
  },

  /**
   * Get directory tree listing
   * @param path - Directory path to list, or undefined for search roots
   */
  async getTree(path?: string): Promise<TreeResponse> {
    const url = path
      ? `${API_BASE}/tree?path=${encodeURIComponent(path)}`
      : `${API_BASE}/tree`;
    return fetchJson<TreeResponse>(url);
  },

  /**
   * Get file content samples around specific line numbers
   * @param path - File path
   * @param lines - Array of line numbers to fetch
   * @param context - Number of context lines before and after (default 50)
   */
  async getSamples(
    path: string,
    lines: number[],
    context: number = 50
  ): Promise<SamplesResponse> {
    const params = new URLSearchParams({
      path,
      lines: lines.join(','),
      context: context.toString(),
    });
    return fetchJson<SamplesResponse>(`${API_BASE}/samples?${params}`);
  },

  /**
   * Search for patterns in files using ripgrep
   * @param paths - File or directory paths to search
   * @param patterns - Regex patterns to search for
   * @param maxResults - Maximum number of results (optional)
   * @param caseSensitive - Case sensitive search (optional)
   * @param contextBefore - Lines of context before match (optional)
   * @param contextAfter - Lines of context after match (optional)
   */
  async trace(
    paths: string[],
    patterns: string[],
    maxResults?: number,
    caseSensitive?: boolean,
    contextBefore?: number,
    contextAfter?: number
  ): Promise<TraceResponse> {
    const params = new URLSearchParams();
    paths.forEach((p) => params.append('path', p));
    patterns.forEach((r) => params.append('regexp', r));
    if (maxResults !== undefined) {
      params.set('max_results', maxResults.toString());
    }
    if (caseSensitive !== undefined) {
      params.set('case_sensitive', caseSensitive.toString());
    }
    if (contextBefore !== undefined) {
      params.set('context_before', contextBefore.toString());
    }
    if (contextAfter !== undefined) {
      params.set('context_after', contextAfter.toString());
    }
    return fetchJson<TraceResponse>(`${API_BASE}/trace?${params}`);
  },

  /**
   * Analyse files to get metadata and statistics
   * @param paths - File or directory paths to analyse
   */
  async analyse(
    paths: string | string[]
  ): Promise<Record<string, unknown>> {
    const pathList = Array.isArray(paths) ? paths : [paths];
    const params = new URLSearchParams();
    pathList.forEach((p) => params.append('path', p));
    return fetchJson<Record<string, unknown>>(`${API_BASE}/analyse?${params}`);
  },

  /**
   * Start file indexing task
   * @param path - File path to index
   * @param force - Force rebuild even if valid index exists
   */
  async index(
    path: string,
    force: boolean = false
  ): Promise<{ task_id: string; status: string; message: string }> {
    return fetchJson(`${API_BASE}/index`, {
      method: 'POST',
      body: JSON.stringify({ path, force }),
    });
  },

  /**
   * Get task status
   * @param taskId - Task ID to check
   */
  async getTaskStatus(taskId: string): Promise<TaskStatus> {
    return fetchJson<TaskStatus>(`${API_BASE}/tasks/${taskId}`);
  },
};

export { ApiError };
