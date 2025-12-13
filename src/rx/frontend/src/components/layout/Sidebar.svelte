<script lang="ts">
  import { settings } from '$lib/stores';
  import FileTree from '../tree/FileTree.svelte';
  import SearchPanel from '../search/SearchPanel.svelte';

  export let width: number;

  let activeTab: 'tree' | 'search' = 'tree';
  let isResizing = false;
  let startX = 0;
  let startWidth = 0;

  function handleMouseDown(event: MouseEvent) {
    isResizing = true;
    startX = event.clientX;
    startWidth = width;
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
  }

  function handleMouseMove(event: MouseEvent) {
    if (!isResizing) return;
    const delta = event.clientX - startX;
    const newWidth = Math.max(200, Math.min(600, startWidth + delta));
    settings.update((s) => ({ ...s, sidebarWidth: newWidth }));
  }

  function handleMouseUp() {
    isResizing = false;
    document.removeEventListener('mousemove', handleMouseMove);
    document.removeEventListener('mouseup', handleMouseUp);
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
  }
</script>

<aside
  class="flex-shrink-0 flex flex-col
         bg-gh-canvas-default dark:bg-gh-canvas-dark-default
         border-r border-gh-border-default dark:border-gh-border-dark-default
         relative"
  style="width: {width}px"
>
  <!-- Tabs -->
  <div class="flex border-b border-gh-border-default dark:border-gh-border-dark-default">
    <button
      class="flex-1 px-4 py-2 text-sm font-medium
             {activeTab === 'tree'
        ? 'text-gh-fg-default dark:text-gh-fg-dark-default border-b-2 border-gh-accent-emphasis dark:border-gh-accent-dark-emphasis'
        : 'text-gh-fg-muted dark:text-gh-fg-dark-muted hover:text-gh-fg-default dark:hover:text-gh-fg-dark-default'}"
      on:click={() => (activeTab = 'tree')}
    >
      Files
    </button>
    <button
      class="flex-1 px-4 py-2 text-sm font-medium
             {activeTab === 'search'
        ? 'text-gh-fg-default dark:text-gh-fg-dark-default border-b-2 border-gh-accent-emphasis dark:border-gh-accent-dark-emphasis'
        : 'text-gh-fg-muted dark:text-gh-fg-dark-muted hover:text-gh-fg-default dark:hover:text-gh-fg-dark-default'}"
      on:click={() => (activeTab = 'search')}
    >
      Search
    </button>
  </div>

  <!-- Content -->
  <div class="flex-1 overflow-hidden">
    {#if activeTab === 'tree'}
      <FileTree />
    {:else}
      <SearchPanel />
    {/if}
  </div>

  <!-- Resize handle -->
  <button
    type="button"
    aria-label="Resize sidebar"
    class="absolute top-0 right-0 w-1 h-full cursor-col-resize border-0 p-0 bg-transparent
           hover:bg-gh-accent-emphasis dark:hover:bg-gh-accent-dark-emphasis
           {isResizing
      ? 'bg-gh-accent-emphasis dark:bg-gh-accent-dark-emphasis'
      : ''}"
    on:mousedown={handleMouseDown}
  />
</aside>
