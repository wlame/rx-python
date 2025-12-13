/**
 * Regex filter utilities for line content transformation
 */

import type { RegexFilter } from '../types';

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function escapeAttribute(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

/**
 * Hide mode: Remove matched groups, show rest of line with red bars
 */
function hideGroups(content: string, matches: RegExpMatchArray[]): string {
  if (matches.length === 0) {
    return escapeHtml(content);
  }

  const segments: string[] = [];
  let lastIndex = 0;

  for (const match of matches) {
    const matchStart = match.index!;

    // Add content before this match
    if (matchStart > lastIndex) {
      segments.push(escapeHtml(content.slice(lastIndex, matchStart)));
    }

    // Process each capturing group in this match
    let groupStart = matchStart;
    for (let i = 1; i < match.length; i++) {
      const group = match[i];
      if (group !== undefined) {
        // Find where this group starts within the match
        const groupIndex = match[0].indexOf(group, groupStart - matchStart);
        const absoluteGroupStart = matchStart + groupIndex;

        // Add content between last group and this group
        if (absoluteGroupStart > groupStart) {
          segments.push(escapeHtml(content.slice(groupStart, absoluteGroupStart)));
        }

        // Add red bar for hidden group (space with bright red background)
        segments.push(
          `<span class="hidden-bar red" title="${escapeAttribute(group)}">&nbsp;</span>`
        );

        groupStart = absoluteGroupStart + group.length;
      }
    }

    // Add any remaining content within the match after last group
    const matchEnd = matchStart + match[0].length;
    if (matchEnd > groupStart) {
      segments.push(escapeHtml(content.slice(groupStart, matchEnd)));
    }

    lastIndex = matchEnd;
  }

  // Add remaining content after last match
  if (lastIndex < content.length) {
    segments.push(escapeHtml(content.slice(lastIndex)));
  }

  return segments.join('');
}

/**
 * Show only mode: Show only matched groups, hide rest with blue bars
 */
function showOnlyGroups(content: string, matches: RegExpMatchArray[]): string {
  if (matches.length === 0) {
    // No matches - hide entire line (space with bright blue background)
    const hiddenText = escapeAttribute(content);
    return `<span class="hidden-bar blue" title="${hiddenText}">&nbsp;</span>`;
  }

  const segments: string[] = [];
  let lastShownEnd = 0;

  for (const match of matches) {
    const matchStart = match.index!;

    // Hide content before this match (space with bright blue background)
    if (matchStart > lastShownEnd) {
      const hiddenText = content.slice(lastShownEnd, matchStart);
      segments.push(
        `<span class="hidden-bar blue" title="${escapeAttribute(hiddenText)}">&nbsp;</span>`
      );
    }

    // Show each capturing group
    for (let i = 1; i < match.length; i++) {
      const group = match[i];
      if (group !== undefined) {
        segments.push(escapeHtml(group));
      }
    }

    lastShownEnd = matchStart + match[0].length;
  }

  // Hide any remaining content after last match (space with bright blue background)
  if (lastShownEnd < content.length) {
    const hiddenText = content.slice(lastShownEnd);
    segments.push(
      `<span class="hidden-bar blue" title="${escapeAttribute(hiddenText)}">&nbsp;</span>`
    );
  }

  return segments.join('');
}

/**
 * Highlight mode: Highlight matched groups with yellow background
 */
function highlightGroups(content: string, matches: RegExpMatchArray[]): string {
  if (matches.length === 0) {
    return escapeHtml(content);
  }

  const segments: string[] = [];
  let lastIndex = 0;

  for (const match of matches) {
    const matchStart = match.index!;

    // Add content before this match
    if (matchStart > lastIndex) {
      segments.push(escapeHtml(content.slice(lastIndex, matchStart)));
    }

    // Process each capturing group in this match
    let groupStart = matchStart;
    for (let i = 1; i < match.length; i++) {
      const group = match[i];
      if (group !== undefined) {
        // Find where this group starts within the match
        const groupIndex = match[0].indexOf(group, groupStart - matchStart);
        const absoluteGroupStart = matchStart + groupIndex;

        // Add content between last group and this group (non-captured parts of match)
        if (absoluteGroupStart > groupStart) {
          segments.push(escapeHtml(content.slice(groupStart, absoluteGroupStart)));
        }

        // Add highlighted group
        segments.push(
          `<span class="regex-highlight">${escapeHtml(group)}</span>`
        );

        groupStart = absoluteGroupStart + group.length;
      }
    }

    // Add any remaining content within the match after last group
    const matchEnd = matchStart + match[0].length;
    if (matchEnd > groupStart) {
      segments.push(escapeHtml(content.slice(groupStart, matchEnd)));
    }

    lastIndex = matchEnd;
  }

  // Add remaining content after last match
  if (lastIndex < content.length) {
    segments.push(escapeHtml(content.slice(lastIndex)));
  }

  return segments.join('');
}

/**
 * Apply regex filter to line content
 * Returns transformed HTML string
 */
export function applyRegexFilter(content: string, filter: RegexFilter | null): string {
  if (!filter || !filter.enabled || !filter.compiledRegex) {
    return escapeHtml(content);
  }

  // Reset regex lastIndex to ensure consistent matching
  filter.compiledRegex.lastIndex = 0;

  // Find all matches
  const matches = [...content.matchAll(filter.compiledRegex)];

  switch (filter.mode) {
    case 'hide':
      return hideGroups(content, matches);
    case 'show':
      return showOnlyGroups(content, matches);
    case 'highlight':
      return highlightGroups(content, matches);
    default:
      return escapeHtml(content);
  }
}
