// API Types matching backend models

export interface HealthResponse {
  status: string;
  app_version: string;
  ripgrep_available: boolean;
  search_roots: string[] | null;
}

export interface TreeEntry {
  name: string;
  path: string;
  type: 'file' | 'directory';
  size: number | null;
  size_human: string | null;
  modified_at: string | null;
  is_text: boolean | null;
  is_compressed: boolean | null;
  compression_format: string | null;
  is_indexed: boolean | null;
  line_count: number | null;
  children_count: number | null;
}

export interface TreeResponse {
  path: string;
  parent: string | null;
  is_search_root: boolean;
  entries: TreeEntry[];
  total_entries: number;
  total_size: number | null;
  total_size_human: string | null;
}

// Backend samples endpoint response
// GET /v1/samples?path=file&lines=1,2,3&context=3
export interface SamplesResponse {
  path: string;
  offsets: Record<string, number>; // offset -> line_number
  lines: Record<string, number>; // line_number -> offset
  before_context: number;
  after_context: number;
  samples: Record<string, string[]>; // key (offset or line) -> array of context lines
  is_compressed: boolean;
  compression_format: string | null;
}

// Trace endpoint (GET /v1/trace)
export interface TraceMatch {
  pattern: string; // pattern ID like 'p1'
  file: string; // file ID like 'f1'
  offset: number;
  relative_line_number: number | null;
  absolute_line_number: number;
  line_text: string | null;
}

export interface TraceResponse {
  request_id: string;
  path: string[];
  time: number;
  patterns: Record<string, string>; // pattern_id -> pattern string
  files: Record<string, string>; // file_id -> file path
  matches: TraceMatch[];
  scanned_files: string[];
  skipped_files: string[];
  max_results: number | null;
}

export interface TaskStatus {
  task_id: string;
  status: 'queued' | 'running' | 'completed' | 'failed';
  path: string;
  operation: string;
  started_at: string | null;
  completed_at: string | null;
  error: string | null;
  result: Record<string, unknown> | null;
}

// Frontend-specific types

export interface TreeNode extends TreeEntry {
  expanded: boolean;
  loading: boolean;
  children: TreeNode[];
  level: number;
}

/** A line of content with its line number */
export interface FileLine {
  lineNumber: number;
  content: string;
}

/** Regex filter configuration for content transformation */
export interface RegexFilter {
  enabled: boolean;
  pattern: string;
  mode: 'hide' | 'show' | 'highlight';
  compiledRegex: RegExp | null;
  error: string | null;
  applying: boolean;
}

/** Represents an open file in the editor */
export interface OpenFile {
  path: string;
  name: string;
  lines: FileLine[];
  totalLines: number | null; // null if unknown (file not indexed)
  startLine: number; // First loaded line number
  endLine: number; // Last loaded line number
  loading: boolean;
  error: string | null;
  isCompressed: boolean;
  compressionFormat: string | null;
  scrollToLine?: number;
  reachedStart: boolean; // True if we've reached line 1
  reachedEnd: boolean; // True if we've hit EOF (can't load more after)
  syntaxHighlighting: boolean; // Per-file syntax highlighting state
  fileSize: number | null; // File size in bytes (for determining default highlighting state)
  regexFilter: RegexFilter | null; // Regex-based content filter
  showInvisibleChars: boolean; // Show invisible characters (spaces, tabs, CR, etc.)
}

/** Match info for highlighting in file viewer */
export interface FileMatch {
  lineNumber: number;
  patternId: string;
  pattern: string;
}

export interface AppSettings {
  theme: 'light' | 'dark' | 'system';
  sidebarWidth: number;
  editorFontSize: number;
  showLineNumbers: boolean;
  wrapLines: boolean;
  linesPerPage: number;
}

export type Theme = 'light' | 'dark';
