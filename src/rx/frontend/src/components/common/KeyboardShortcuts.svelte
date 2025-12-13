<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { settings } from '$lib/stores';

  let showHelp = false;

  function handleKeydown(event: KeyboardEvent) {
    // Cmd/Ctrl + K: Focus search
    if ((event.metaKey || event.ctrlKey) && event.key === 'k') {
      event.preventDefault();
      const searchInput = document.querySelector('input[placeholder*="Search"]') as HTMLInputElement;
      if (searchInput) {
        searchInput.focus();
      }
    }

    // Cmd/Ctrl + B: Toggle sidebar (future)
    if ((event.metaKey || event.ctrlKey) && event.key === 'b') {
      event.preventDefault();
      // Could toggle sidebar visibility
    }

    // Cmd/Ctrl + /: Show keyboard shortcuts
    if ((event.metaKey || event.ctrlKey) && event.key === '/') {
      event.preventDefault();
      showHelp = !showHelp;
    }

    // Escape: Close help dialog
    if (event.key === 'Escape' && showHelp) {
      showHelp = false;
    }
  }

  onMount(() => {
    window.addEventListener('keydown', handleKeydown);
  });

  onDestroy(() => {
    window.removeEventListener('keydown', handleKeydown);
  });
</script>

<!-- Help dialog -->
{#if showHelp}
  <div
    class="fixed inset-0 bg-black/50 flex items-center justify-center z-50"
    role="button"
    tabindex="-1"
    aria-label="Close dialog"
    on:click={() => (showHelp = false)}
    on:keydown={(e) => e.key === 'Escape' && (showHelp = false)}
  >
    <!-- svelte-ignore a11y-no-noninteractive-element-interactions -->
    <div
      class="bg-gh-canvas-default dark:bg-gh-canvas-dark-default
             border border-gh-border-default dark:border-gh-border-dark-default
             rounded-lg shadow-lg p-6 max-w-md"
      role="dialog"
      aria-modal="true"
      aria-labelledby="shortcuts-title"
      on:click|stopPropagation
      on:keydown|stopPropagation
    >
      <h2 id="shortcuts-title" class="text-lg font-semibold mb-4">Keyboard Shortcuts</h2>

      <div class="space-y-2 text-sm">
        <div class="flex justify-between">
          <span class="text-gh-fg-muted dark:text-gh-fg-dark-muted">Focus Search</span>
          <kbd
            class="px-2 py-1 bg-gh-canvas-subtle dark:bg-gh-canvas-dark-subtle
                   border border-gh-border-default dark:border-gh-border-dark-default
                   rounded text-xs"
          >
            ⌘K / Ctrl+K
          </kbd>
        </div>

        <div class="flex justify-between">
          <span class="text-gh-fg-muted dark:text-gh-fg-dark-muted">Show Shortcuts</span>
          <kbd
            class="px-2 py-1 bg-gh-canvas-subtle dark:bg-gh-canvas-dark-subtle
                   border border-gh-border-default dark:border-gh-border-dark-default
                   rounded text-xs"
          >
            ⌘/ / Ctrl+/
          </kbd>
        </div>

        <div class="flex justify-between">
          <span class="text-gh-fg-muted dark:text-gh-fg-dark-muted">Close Dialog</span>
          <kbd
            class="px-2 py-1 bg-gh-canvas-subtle dark:bg-gh-canvas-dark-subtle
                   border border-gh-border-default dark:border-gh-border-dark-default
                   rounded text-xs"
          >
            Esc
          </kbd>
        </div>
      </div>

      <button class="btn btn-primary w-full mt-4" on:click={() => (showHelp = false)}>
        Close
      </button>
    </div>
  </div>
{/if}
