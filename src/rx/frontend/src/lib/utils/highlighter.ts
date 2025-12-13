/**
 * Syntax highlighter using Highlight.js
 */

declare global {
  interface Window {
    hljs: any;
  }
}

/**
 * Detect language from file extension
 */
export function detectLanguage(filename: string): string | undefined {
  const ext = filename.split('.').pop()?.toLowerCase();
  
  const languageMap: Record<string, string> = {
    'js': 'javascript',
    'jsx': 'javascript',
    'ts': 'typescript',
    'tsx': 'typescript',
    'py': 'python',
    'rb': 'ruby',
    'java': 'java',
    'c': 'c',
    'cpp': 'cpp',
    'cc': 'cpp',
    'cxx': 'cpp',
    'h': 'c',
    'hpp': 'cpp',
    'cs': 'csharp',
    'go': 'go',
    'rs': 'rust',
    'php': 'php',
    'sh': 'bash',
    'bash': 'bash',
    'zsh': 'bash',
    'json': 'json',
    'xml': 'xml',
    'html': 'html',
    'css': 'css',
    'scss': 'scss',
    'sass': 'sass',
    'less': 'less',
    'yml': 'yaml',
    'yaml': 'yaml',
    'toml': 'toml',
    'ini': 'ini',
    'conf': 'nginx',
    'md': 'markdown',
    'sql': 'sql',
    'r': 'r',
    'swift': 'swift',
    'kt': 'kotlin',
    'scala': 'scala',
    'clj': 'clojure',
    'ex': 'elixir',
    'exs': 'elixir',
    'erl': 'erlang',
    'hrl': 'erlang',
    'lua': 'lua',
    'pl': 'perl',
    'pm': 'perl',
    'vim': 'vim',
    'diff': 'diff',
    'patch': 'diff',
  };
  
  return ext ? languageMap[ext] : undefined;
}

/**
 * Highlight a single line of code
 */
export function highlightLine(
  content: string,
  language?: string
): string {
  if (!window.hljs) {
    // Highlight.js not loaded yet, return escaped content
    return escapeHtml(content);
  }

  try {
    if (language) {
      // Use specified language
      const result = window.hljs.highlight(content, { language, ignoreIllegals: true });
      return result.value;
    } else {
      // Auto-detect language
      const result = window.hljs.highlightAuto(content);
      return result.value;
    }
  } catch (e) {
    // If highlighting fails, return escaped content
    return escapeHtml(content);
  }
}

/**
 * Escape HTML special characters
 */
function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}
