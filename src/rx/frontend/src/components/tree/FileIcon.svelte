<script lang="ts">
  import type { TreeNode } from '$lib/types';

  export let node: TreeNode;

  // File extension to color mapping
  const extensionColors: Record<string, string> = {
    // Logs
    log: 'text-yellow-600 dark:text-yellow-400',
    // Code
    py: 'text-blue-500 dark:text-blue-400',
    js: 'text-yellow-500 dark:text-yellow-400',
    ts: 'text-blue-600 dark:text-blue-400',
    json: 'text-green-500 dark:text-green-400',
    yaml: 'text-purple-500 dark:text-purple-400',
    yml: 'text-purple-500 dark:text-purple-400',
    xml: 'text-orange-500 dark:text-orange-400',
    html: 'text-orange-600 dark:text-orange-400',
    css: 'text-blue-400 dark:text-blue-300',
    md: 'text-gray-600 dark:text-gray-400',
    txt: 'text-gray-500 dark:text-gray-400',
    // Compressed
    gz: 'text-red-500 dark:text-red-400',
    zst: 'text-red-600 dark:text-red-400',
    xz: 'text-red-700 dark:text-red-400',
    bz2: 'text-red-500 dark:text-red-400',
    zip: 'text-red-500 dark:text-red-400',
  };

  $: extension = node.name.split('.').pop()?.toLowerCase() || '';
  $: colorClass =
    extensionColors[extension] || 'text-gh-fg-muted dark:text-gh-fg-dark-muted';
  $: isDirectory = node.type === 'directory';
  $: isCompressed = node.is_compressed;
</script>

{#if isDirectory}
  <svg
    class="w-4 h-4 flex-shrink-0 text-gh-accent-fg dark:text-gh-accent-dark-fg"
    viewBox="0 0 24 24"
    fill="currentColor"
  >
    {#if node.expanded}
      <path
        d="M20 18a2 2 0 01-2 2H4a2 2 0 01-2-2V6a2 2 0 012-2h5l2 2h7a2 2 0 012 2v1H4v9h16v-3"
      />
    {:else}
      <path
        d="M22 17a2 2 0 01-2 2H4a2 2 0 01-2-2V7a2 2 0 012-2h5l2 2h9a2 2 0 012 2z"
      />
    {/if}
  </svg>
{:else}
  <svg class="w-4 h-4 flex-shrink-0 {colorClass}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
    <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" />
    <path d="M14 2v6h6" />
    {#if isCompressed}
      <path d="M10 12h4M10 16h4" />
    {/if}
  </svg>
{/if}
