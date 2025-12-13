<script lang="ts">
  import { files } from '$lib/stores';
  import EditorPane from '../editor/EditorPane.svelte';
  import FileBadges from '../common/FileBadges.svelte';

  let activeFileIndex = 0;
  let draggedIndex: number | null = null;
  let dragOverIndex: number | null = null;

  // Track if we're manually switching tabs to avoid reactivity conflicts
  let manualTabSwitch = false;

  // React to active file path changes from the store (e.g., from search results)
  $: if ($files.activeFilePath && !manualTabSwitch) {
    const index = $files.openFiles.findIndex(f => f.path === $files.activeFilePath);
    if (index >= 0) {
      activeFileIndex = index;
    }
  }

  // When files change, ensure active index is valid
  // Only adjust if current index is truly out of bounds
  $: if ($files.openFiles.length > 0 && activeFileIndex >= $files.openFiles.length) {
    activeFileIndex = $files.openFiles.length - 1;
  }

  $: activeFile = $files.openFiles[activeFileIndex];

  function selectTab(index: number) {
    manualTabSwitch = true;
    activeFileIndex = index;
    // Reset the flag after reactivity has settled
    setTimeout(() => {
      manualTabSwitch = false;
    }, 0);
  }

  function handleClose(index: number, event: Event) {
    event.stopPropagation();
    const file = $files.openFiles[index];
    files.closeFile(file.path);

    // Adjust active index after close
    if (activeFileIndex >= $files.openFiles.length - 1) {
      activeFileIndex = Math.max(0, $files.openFiles.length - 2);
    }
  }

  // Drag and drop handlers
  function handleDragStart(event: DragEvent, index: number) {
    draggedIndex = index;
    if (event.dataTransfer) {
      event.dataTransfer.effectAllowed = 'move';
    }
  }

  function handleDragOver(event: DragEvent, index: number) {
    event.preventDefault();
    if (event.dataTransfer) {
      event.dataTransfer.dropEffect = 'move';
    }
    dragOverIndex = index;
  }

  function handleDragLeave() {
    dragOverIndex = null;
  }

  function handleDrop(event: DragEvent, dropIndex: number) {
    event.preventDefault();

    if (draggedIndex === null || draggedIndex === dropIndex) {
      draggedIndex = null;
      dragOverIndex = null;
      return;
    }

    // Reorder files
    files.reorderFiles(draggedIndex, dropIndex);

    // Update active index if needed
    if (activeFileIndex === draggedIndex) {
      activeFileIndex = dropIndex;
    } else if (draggedIndex < activeFileIndex && dropIndex >= activeFileIndex) {
      activeFileIndex--;
    } else if (draggedIndex > activeFileIndex && dropIndex <= activeFileIndex) {
      activeFileIndex++;
    }

    draggedIndex = null;
    dragOverIndex = null;
  }

  function handleDragEnd() {
    draggedIndex = null;
    dragOverIndex = null;
  }
</script>

<main class="flex-1 flex flex-col min-w-0 bg-gh-canvas-default dark:bg-gh-canvas-dark-default">
  {#if $files.openFiles.length === 0}
    <!-- Empty state -->
    <div class="flex-1 flex items-center justify-center text-gh-fg-muted dark:text-gh-fg-dark-muted">
      <div class="text-center">
        <svg
          class="w-16 h-16 mx-auto mb-4 opacity-50"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="1"
        >
          <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" />
          <path d="M14 2v6h6" />
          <path d="M16 13H8M16 17H8M10 9H8" />
        </svg>
        <p class="text-lg font-medium">No files open</p>
        <p class="text-sm mt-1">Select a file from the tree to view its contents</p>
      </div>
    </div>
  {:else}
    <!-- Tabs for file switching with drag-to-reorder -->
    <div class="flex items-center gap-0.5 px-2 py-1 bg-gh-canvas-subtle dark:bg-gh-canvas-dark-subtle border-b border-gh-border-default dark:border-gh-border-dark-default overflow-x-auto scrollbar-hide">
      {#each $files.openFiles as file, index (file.path)}
        <button
          draggable="true"
          class="flex items-center gap-2 px-3 py-1.5 rounded-t
                 transition-colors text-sm whitespace-nowrap cursor-pointer
                 {index === activeFileIndex
                   ? 'bg-gh-canvas-default dark:bg-gh-canvas-dark-default border border-b-0 border-gh-border-default dark:border-gh-border-dark-default'
                   : 'bg-transparent hover:bg-gh-canvas-inset dark:hover:bg-gh-canvas-dark-inset text-gh-fg-muted dark:text-gh-fg-dark-muted'}
                 {dragOverIndex === index ? 'border-l-2 border-gh-accent-fg dark:border-gh-accent-dark-fg' : ''}"
          on:click={() => selectTab(index)}
          on:dragstart={(e) => handleDragStart(e, index)}
          on:dragover={(e) => handleDragOver(e, index)}
          on:dragleave={handleDragLeave}
          on:drop={(e) => handleDrop(e, index)}
          on:dragend={handleDragEnd}
        >
          <span class="font-medium truncate max-w-[200px]" title={file.path}>
            {file.name}
          </span>
          <FileBadges
            isCompressed={file.isCompressed}
            compressionFormat={file.compressionFormat}
            isIndexed={null}
          />
          <button
            class="p-0.5 rounded hover:bg-gh-danger-subtle dark:hover:bg-gh-danger-dark-subtle
                   hover:text-gh-danger-fg dark:hover:text-gh-danger-dark-fg
                   transition-colors"
            title="Close"
            on:click={(e) => handleClose(index, e)}
          >
            <svg class="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M18 6L6 18M6 6l12 12" />
            </svg>
          </button>
        </button>
      {/each}
    </div>

    <!-- Single active file editor -->
    <div class="flex-1 min-h-0 overflow-hidden">
      {#if activeFile}
        <EditorPane file={activeFile} hideHeader={true} isActive={true} />
      {/if}
    </div>
  {/if}
</main>
