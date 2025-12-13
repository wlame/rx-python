<script lang="ts">
  import { files, trace, health } from '$lib/stores';

  $: openFileCount = $files.openFiles.length;
  $: searchRoots = $health.data?.search_roots || [];
  $: lastSearchTime = $trace.response?.time;
  $: matchCount = $trace.response?.matches.length || 0;
</script>

<footer
  class="h-6 flex-shrink-0 flex items-center justify-between px-3 text-xs
         bg-gh-canvas-subtle dark:bg-gh-canvas-dark-subtle
         border-t border-gh-border-default dark:border-gh-border-dark-default
         text-gh-fg-muted dark:text-gh-fg-dark-muted relative z-10"
>
  <div class="flex items-center gap-4">
    {#if searchRoots.length > 0}
      <span>
        {searchRoots.length} {searchRoots.length === 1 ? 'root' : 'roots'}
      </span>
    {/if}

    {#if openFileCount > 0}
      <span>
        {openFileCount} {openFileCount === 1 ? 'file' : 'files'} open
      </span>
    {/if}

    {#if $trace.response}
      <span class="text-gh-accent-fg dark:text-gh-accent-dark-fg">
        {matchCount} {matchCount === 1 ? 'match' : 'matches'}
        {#if lastSearchTime}
          ({lastSearchTime.toFixed(2)}s)
        {/if}
      </span>
    {/if}
  </div>

  <div class="flex items-center gap-4">
    {#if $health.data?.ripgrep_available}
      <span>ripgrep ready</span>
    {/if}
  </div>
</footer>
