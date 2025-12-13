<script lang="ts">
  import type { TreeNode as TreeNodeType } from '$lib/types';
  import { tree, files, notifications } from '$lib/stores';
  import { api } from '$lib/api';
  import FileIcon from './FileIcon.svelte';
  import Spinner from '../common/Spinner.svelte';
  import FileBadges from '../common/FileBadges.svelte';

  export let node: TreeNodeType;

  $: isSelected = $tree.selectedPath === node.path;
  $: indentPx = node.level * 16;

  let showContextMenu = false;
  let contextMenuX = 0;
  let contextMenuY = 0;

  // Analyze popup state
  let showAnalyzePopup = false;
  let analyzeLoading = false;
  let analyzeResult: any | null = null;
  let selectedFileId: string | null = null; // For directory analysis
  let expandedDirs: Set<string> = new Set(); // Track expanded directories in analyze tree

  function handleClick() {
    if (node.type === 'directory') {
      tree.toggleExpanded(node.path);
    } else {
      tree.selectPath(node.path);
      files.openFile(node.path, undefined, node.size);
    }
  }

  function handleKeydown(event: KeyboardEvent) {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      handleClick();
    }
  }

  function formatSize(size: number | null): string {
    if (size === null) return '';
    if (size < 1024) return `${size} B`;
    if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
    if (size < 1024 * 1024 * 1024) return `${(size / 1024 / 1024).toFixed(1)} MB`;
    return `${(size / 1024 / 1024 / 1024).toFixed(1)} GB`;
  }

  function handleContextMenu(event: MouseEvent) {
    event.preventDefault();
    contextMenuX = event.clientX;
    contextMenuY = event.clientY;
    showContextMenu = true;
  }

  function closeContextMenu() {
    showContextMenu = false;
  }

  async function handleAnalyse() {
    closeContextMenu();
    analyzeLoading = true;
    showAnalyzePopup = true;
    analyzeResult = null;

    try {
      const result = await api.analyse(node.path);
      console.log('Analyse result:', result);
      analyzeResult = result;
      notifications.success(`Analysis complete for ${node.name}`, 3000);
    } catch (e) {
      const error = e instanceof Error ? e.message : 'Analysis failed';
      notifications.error(error, 5000);
      showAnalyzePopup = false;
    } finally {
      analyzeLoading = false;
    }
  }

  function closeAnalyzePopup() {
    showAnalyzePopup = false;
    analyzeResult = null;
    selectedFileId = null;
    expandedDirs = new Set();
  }

  // Build tree structure from file paths
  function buildFileTree(files: Record<string, string>, results: any[], analyzedPath: string) {
    const tree: any = { children: {} };

    // Normalize the analyzed path (remove trailing slash)
    const basePath = analyzedPath.endsWith('/') ? analyzedPath.slice(0, -1) : analyzedPath;

    Object.entries(files).forEach(([fileId, filePath]) => {
      // Strip the base path to get relative path
      let relativePath = filePath;
      if (filePath.startsWith(basePath + '/')) {
        relativePath = filePath.slice(basePath.length + 1);
      } else if (filePath === basePath) {
        // Single file analysis
        relativePath = filePath.split('/').pop() || filePath;
      }

      const parts = relativePath.split('/');
      let current = tree;

      // Build path through directories
      for (let i = 0; i < parts.length - 1; i++) {
        const dirName = parts[i];
        if (!dirName) continue; // Skip empty parts

        const dirPath = parts.slice(0, i + 1).join('/');
        if (!current.children[dirName]) {
          current.children[dirName] = {
            type: 'dir',
            name: dirName,
            path: dirPath,
            children: {}
          };
          // Auto-expand all directories
          expandedDirs.add(dirPath);
        }
        current = current.children[dirName];
      }

      // Add file
      const fileName = parts[parts.length - 1];
      const fileResult = results.find((r: any) => r.file === fileId);
      current.children[fileName] = {
        type: 'file',
        name: fileName,
        path: relativePath,
        fileId: fileId,
        result: fileResult
      };
    });

    return tree.children;
  }

  // Reactive tree structure
  $: fileTree = analyzeResult && analyzeResult.files
    ? buildFileTree(analyzeResult.files, analyzeResult.results, node.path)
    : {};

  // Toggle directory expansion
  function toggleDir(dirPath: string) {
    if (expandedDirs.has(dirPath)) {
      expandedDirs.delete(dirPath);
    } else {
      expandedDirs.add(dirPath);
    }
    expandedDirs = expandedDirs; // Trigger reactivity
  }

  // Helper to check if result is a directory analysis
  $: isDirectoryAnalysis = analyzeResult && analyzeResult.results && analyzeResult.results.length > 1;

  // Helper to get currently selected file result
  $: currentFileResult = analyzeResult && analyzeResult.results ?
    (selectedFileId
      ? analyzeResult.results.find((r: any) => r.file === selectedFileId)
      : analyzeResult.results[0])
    : null;

  // Helper to format file size
  function formatBytes(bytes: number | null): string {
    if (bytes === null) return 'N/A';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(2)} KB`;
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
    return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`;
  }

  async function handleIndex() {
    closeContextMenu();
    try {
      const result = await api.index(node.path, false);
      console.log('Index result:', result);
      notifications.success(`Indexing started for ${node.name}`, 3000);
    } catch (e) {
      const error = e instanceof Error ? e.message : 'Indexing failed';
      notifications.error(error, 5000);
    }
  }
</script>

<div class="select-none">
  <div
    role="treeitem"
    tabindex="0"
    aria-expanded={node.type === 'directory' ? node.expanded : undefined}
    aria-selected={isSelected}
    class="flex items-center gap-1 px-2 py-0.5 cursor-pointer text-sm
           hover:bg-gh-canvas-subtle dark:hover:bg-gh-canvas-dark-subtle
           {isSelected
      ? 'bg-gh-accent-muted dark:bg-gh-accent-dark-muted'
      : ''}"
    style="padding-left: {indentPx + 8}px"
    on:click={handleClick}
    on:keydown={handleKeydown}
    on:contextmenu={handleContextMenu}
  >
    <!-- Expand/collapse chevron for directories -->
    {#if node.type === 'directory'}
      <span class="w-4 h-4 flex items-center justify-center flex-shrink-0">
        {#if node.loading}
          <Spinner size="sm" />
        {:else}
          <svg
            class="w-3 h-3 text-gh-fg-muted dark:text-gh-fg-dark-muted transition-transform
                   {node.expanded ? 'rotate-90' : ''}"
            viewBox="0 0 24 24"
            fill="currentColor"
          >
            <path d="M9 18l6-6-6-6" />
          </svg>
        {/if}
      </span>
    {:else}
      <span class="w-4 h-4 flex-shrink-0" />
    {/if}

    <!-- File/folder icon -->
    <FileIcon {node} />

    <!-- Name -->
    <span
      class="truncate flex-1 {node.type === 'directory'
        ? 'font-medium'
        : node.is_text === false
        ? 'opacity-50'
        : ''}"
    >
      {node.name}
    </span>

    <!-- Badges for files -->
    {#if node.type === 'file'}
      <span class="flex items-center gap-1 flex-shrink-0">
        <FileBadges
          isCompressed={node.is_compressed}
          compressionFormat={node.compression_format}
          isIndexed={node.is_indexed}
        />
        {#if node.size !== null}
          <span
            class="text-xs text-gh-fg-subtle dark:text-gh-fg-dark-subtle ml-1"
          >
            {formatSize(node.size)}
          </span>
        {/if}
      </span>
    {:else if node.children_count !== null}
      <span class="text-xs text-gh-fg-subtle dark:text-gh-fg-dark-subtle">
        {node.children_count}
      </span>
    {/if}
  </div>

  <!-- Children -->
  {#if node.type === 'directory' && node.expanded && node.children.length > 0}
    <div role="group">
      {#each node.children as child (child.path)}
        <svelte:self node={child} />
      {/each}
    </div>
  {/if}
</div>

<!-- Context Menu -->
{#if showContextMenu}
  <!-- svelte-ignore a11y-click-events-have-key-events -->
  <!-- svelte-ignore a11y-no-static-element-interactions -->
  <div
    class="fixed inset-0 z-40"
    on:click={closeContextMenu}
    on:contextmenu|preventDefault={closeContextMenu}
  />
  <div
    class="fixed z-50 bg-gh-canvas-default dark:bg-gh-canvas-dark-subtle border border-gh-border-default dark:border-gh-border-dark-default rounded-lg shadow-xl py-1 min-w-40"
    style="left: {contextMenuX}px; top: {contextMenuY}px;"
  >
    {#if node.type === 'file'}
      <button
        class="w-full text-left px-3 py-2 text-sm text-gh-fg-default dark:text-gh-fg-dark-default hover:bg-gh-canvas-subtle dark:hover:bg-gh-canvas-dark-inset"
        on:click={handleAnalyse}
      >
        Analyse
      </button>
      {#if node.is_compressed}
        <button
          class="w-full text-left px-3 py-2 text-sm text-gh-fg-default dark:text-gh-fg-dark-default hover:bg-gh-canvas-subtle dark:hover:bg-gh-canvas-dark-inset"
          on:click={handleIndex}
        >
          {node.is_indexed ? 'Re-index' : 'Index'}
        </button>
      {/if}
    {:else}
      <button
        class="w-full text-left px-3 py-2 text-sm text-gh-fg-default dark:text-gh-fg-dark-default hover:bg-gh-canvas-subtle dark:hover:bg-gh-canvas-dark-inset"
        on:click={handleAnalyse}
      >
        Analyse Directory
      </button>
    {/if}
  </div>
{/if}

<!-- Analyze Results Popup -->
{#if showAnalyzePopup}
  <div
    class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
    on:click={closeAnalyzePopup}
  >
    <div
      class="bg-gh-canvas-default dark:bg-gh-canvas-dark-default rounded-lg shadow-xl max-w-3xl w-full h-[70vh] flex flex-col overflow-hidden"
      on:click={(e) => e.stopPropagation()}
    >
      <!-- Header -->
      <div class="flex items-center justify-between px-6 py-4 border-b border-gh-border-default dark:border-gh-border-dark-default flex-shrink-0">
        <h2 class="text-lg font-semibold text-gh-fg-default dark:text-gh-fg-dark-default">
          Analysis Results: {node.name}
        </h2>
        <button
          class="p-1 rounded hover:bg-gh-canvas-subtle dark:hover:bg-gh-canvas-dark-subtle"
          on:click={closeAnalyzePopup}
          aria-label="Close"
        >
          <svg class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M18 6L6 18M6 6l12 12" />
          </svg>
        </button>
      </div>

      <!-- Content -->
      <div class="flex flex-1 overflow-hidden">
        {#if analyzeLoading}
          <div class="flex flex-col items-center justify-center py-12 w-full">
            <Spinner size="lg" />
            <p class="mt-4 text-gh-fg-muted dark:text-gh-fg-dark-muted">Analyzing file...</p>
          </div>
        {:else if analyzeResult}
          <!-- File list sidebar for directory analysis -->
          {#if isDirectoryAnalysis}
            <div class="w-[300px] border-r border-gh-border-default dark:border-gh-border-dark-default overflow-y-auto">
              <div class="p-3 bg-gh-canvas-subtle dark:bg-gh-canvas-dark-subtle border-b border-gh-border-default dark:border-gh-border-dark-default">
                <div class="text-xs font-semibold text-gh-fg-muted dark:text-gh-fg-dark-muted">
                  Files ({analyzeResult.results.length})
                </div>
              </div>
              <!-- Render tree (inline to avoid recursion issues in Svelte 4) -->
              <div class="py-2">
                {#each Object.entries(fileTree) as [rootName, rootItem]}
                  <!-- Level 0 -->
                  {#if rootItem.type === 'dir'}
                    {@const isExpanded0 = expandedDirs.has(rootItem.path)}
                    <div>
                      <button
                        class="w-full flex items-center gap-1 px-2 py-1 text-sm hover:bg-gh-canvas-subtle dark:hover:bg-gh-canvas-dark-subtle"
                        style="padding-left: 8px"
                        on:click={() => toggleDir(rootItem.path)}
                      >
                        <span class="w-4 h-4 flex-shrink-0">
                          <svg class="w-3 h-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            {#if isExpanded0}
                              <path d="M19 9l-7 7-7-7" />
                            {:else}
                              <path d="M9 5l7 7-7 7" />
                            {/if}
                          </svg>
                        </span>
                        <svg class="w-4 h-4 flex-shrink-0 text-gh-fg-muted dark:text-gh-fg-dark-muted" viewBox="0 0 24 24" fill="currentColor">
                          <path d="M10 4H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2h-8l-2-2z"/>
                        </svg>
                        <span class="flex-1 truncate text-left">{rootName}</span>
                      </button>
                      {#if isExpanded0}
                        {#each Object.entries(rootItem.children) as [name1, item1]}
                          <!-- Level 1 -->
                          {#if item1.type === 'dir'}
                            {@const isExpanded1 = expandedDirs.has(item1.path)}
                            <div>
                              <button
                                class="w-full flex items-center gap-1 px-2 py-1 text-sm hover:bg-gh-canvas-subtle dark:hover:bg-gh-canvas-dark-subtle"
                                style="padding-left: 24px"
                                on:click={() => toggleDir(item1.path)}
                              >
                                <span class="w-4 h-4 flex-shrink-0">
                                  <svg class="w-3 h-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    {#if isExpanded1}
                                      <path d="M19 9l-7 7-7-7" />
                                    {:else}
                                      <path d="M9 5l7 7-7 7" />
                                    {/if}
                                  </svg>
                                </span>
                                <svg class="w-4 h-4 flex-shrink-0 text-gh-fg-muted dark:text-gh-fg-dark-muted" viewBox="0 0 24 24" fill="currentColor">
                                  <path d="M10 4H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2h-8l-2-2z"/>
                                </svg>
                                <span class="flex-1 truncate text-left">{name1}</span>
                              </button>
                              {#if isExpanded1}
                                {#each Object.entries(item1.children) as [name2, item2]}
                                  <!-- Level 2+ - Files only (simplified) -->
                                  {#if item2.type === 'file'}
                                    <button
                                      class="w-full flex items-center gap-1 px-2 py-1 text-sm hover:bg-gh-canvas-subtle dark:hover:bg-gh-canvas-dark-subtle
                                             {(selectedFileId || analyzeResult.results[0].file) === item2.fileId ? 'bg-gh-accent-muted dark:bg-gh-accent-dark-muted' : ''}"
                                      style="padding-left: 44px"
                                      on:click={() => selectedFileId = item2.fileId}
                                    >
                                      <svg class="w-4 h-4 flex-shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                        <path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"></path>
                                        <polyline points="13 2 13 9 20 9"></polyline>
                                      </svg>
                                      <span class="flex-1 truncate text-left text-xs" title={item2.path}>{name2}</span>
                                      <span class="text-xs text-gh-fg-muted dark:text-gh-fg-dark-muted flex-shrink-0">
                                        {item2.result?.size_human || formatBytes(item2.result?.size_bytes)}
                                      </span>
                                    </button>
                                  {/if}
                                {/each}
                              {/if}
                            </div>
                          {:else}
                            <!-- Level 1 file -->
                            <button
                              class="w-full flex items-center gap-1 px-2 py-1 text-sm hover:bg-gh-canvas-subtle dark:hover:bg-gh-canvas-dark-subtle
                                     {(selectedFileId || analyzeResult.results[0].file) === item1.fileId ? 'bg-gh-accent-muted dark:bg-gh-accent-dark-muted' : ''}"
                              style="padding-left: 28px"
                              on:click={() => selectedFileId = item1.fileId}
                            >
                              <svg class="w-4 h-4 flex-shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"></path>
                                <polyline points="13 2 13 9 20 9"></polyline>
                              </svg>
                              <span class="flex-1 truncate text-left text-xs" title={item1.path}>{name1}</span>
                              <span class="text-xs text-gh-fg-muted dark:text-gh-fg-dark-muted flex-shrink-0">
                                {item1.result?.size_human || formatBytes(item1.result?.size_bytes)}
                              </span>
                            </button>
                          {/if}
                        {/each}
                      {/if}
                    </div>
                  {:else}
                    <!-- Level 0 file -->
                    <button
                      class="w-full flex items-center gap-1 px-2 py-1 text-sm hover:bg-gh-canvas-subtle dark:hover:bg-gh-canvas-dark-subtle
                             {(selectedFileId || analyzeResult.results[0].file) === rootItem.fileId ? 'bg-gh-accent-muted dark:bg-gh-accent-dark-muted' : ''}"
                      style="padding-left: 8px"
                      on:click={() => selectedFileId = rootItem.fileId}
                    >
                      <svg class="w-4 h-4 flex-shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"></path>
                        <polyline points="13 2 13 9 20 9"></polyline>
                      </svg>
                      <span class="flex-1 truncate text-left text-xs" title={rootItem.path}>{rootName}</span>
                      <span class="text-xs text-gh-fg-muted dark:text-gh-fg-dark-muted flex-shrink-0">
                        {rootItem.result?.size_human || formatBytes(rootItem.result?.size_bytes)}
                      </span>
                    </button>
                  {/if}
                {/each}
              </div>
            </div>
          {/if}

          <!-- File analysis details -->
          <div class="flex-1 p-6 overflow-y-auto">
            {#if currentFileResult}
              {@const filePath = analyzeResult.files[currentFileResult.file]}

              <!-- File header -->
              {#if isDirectoryAnalysis}
                <div class="mb-4 pb-4 border-b border-gh-border-default dark:border-gh-border-dark-default">
                  <div class="text-sm text-gh-fg-muted dark:text-gh-fg-dark-muted font-mono truncate" title={filePath}>
                    {filePath}
                  </div>
                </div>
              {/if}

              <div class="space-y-6">
                <!-- General Information -->
                <div>
                  <h3 class="text-sm font-semibold text-gh-fg-default dark:text-gh-fg-dark-default mb-3 uppercase tracking-wide">
                    General Information
                  </h3>
                  <div class="grid grid-cols-2 gap-4">
                    <div>
                      <div class="text-xs text-gh-fg-muted dark:text-gh-fg-dark-muted mb-1">File Size</div>
                      <div class="text-sm font-medium">{currentFileResult.size_human || formatBytes(currentFileResult.size_bytes)}</div>
                    </div>
                    <div>
                      <div class="text-xs text-gh-fg-muted dark:text-gh-fg-dark-muted mb-1">Type</div>
                      <div class="text-sm font-medium">{currentFileResult.is_text ? 'Text' : 'Binary'}</div>
                    </div>
                    {#if currentFileResult.modified_at}
                      <div>
                        <div class="text-xs text-gh-fg-muted dark:text-gh-fg-dark-muted mb-1">Modified</div>
                        <div class="text-sm font-medium">{new Date(currentFileResult.modified_at).toLocaleString()}</div>
                      </div>
                    {/if}
                    {#if currentFileResult.permissions}
                      <div>
                        <div class="text-xs text-gh-fg-muted dark:text-gh-fg-dark-muted mb-1">Permissions</div>
                        <div class="text-sm font-medium font-mono">{currentFileResult.permissions}</div>
                      </div>
                    {/if}
                  </div>
                </div>

                <!-- Line Statistics (for text files) -->
                {#if currentFileResult.is_text && currentFileResult.line_count !== null}
                  <div>
                    <h3 class="text-sm font-semibold text-gh-fg-default dark:text-gh-fg-dark-default mb-3 uppercase tracking-wide">
                      Line Statistics
                    </h3>
                    <div class="grid grid-cols-2 gap-4">
                      <div>
                        <div class="text-xs text-gh-fg-muted dark:text-gh-fg-dark-muted mb-1">Total Lines</div>
                        <div class="text-lg font-bold text-gh-accent-fg dark:text-gh-accent-dark-fg">
                          {currentFileResult.line_count?.toLocaleString()}
                        </div>
                      </div>
                      {#if currentFileResult.empty_line_count !== null}
                        <div>
                          <div class="text-xs text-gh-fg-muted dark:text-gh-fg-dark-muted mb-1">Empty Lines</div>
                          <div class="text-lg font-bold">
                            {currentFileResult.empty_line_count?.toLocaleString()}
                          </div>
                        </div>
                      {/if}
                      {#if currentFileResult.line_ending}
                        <div>
                          <div class="text-xs text-gh-fg-muted dark:text-gh-fg-dark-muted mb-1">Line Ending</div>
                          <div class="text-sm font-medium font-mono">{currentFileResult.line_ending}</div>
                        </div>
                      {/if}
                    </div>

                    <!-- Line Length Statistics -->
                    {#if currentFileResult.line_length_max !== null}
                      <div class="mt-4">
                        <div class="text-xs font-semibold text-gh-fg-muted dark:text-gh-fg-dark-muted mb-2">Line Length</div>
                        <div class="grid grid-cols-3 gap-3 text-sm">
                          <div>
                            <div class="text-xs text-gh-fg-muted dark:text-gh-fg-dark-muted">Max</div>
                            <div class="font-semibold">{currentFileResult.line_length_max}</div>
                            {#if currentFileResult.line_length_max_line_number}
                              <div class="text-xs text-gh-fg-muted dark:text-gh-fg-dark-muted">Line {currentFileResult.line_length_max_line_number}</div>
                            {/if}
                          </div>
                          {#if currentFileResult.line_length_avg !== null}
                            <div>
                              <div class="text-xs text-gh-fg-muted dark:text-gh-fg-dark-muted">Average</div>
                              <div class="font-semibold">{currentFileResult.line_length_avg.toFixed(1)}</div>
                            </div>
                          {/if}
                          {#if currentFileResult.line_length_median !== null}
                            <div>
                              <div class="text-xs text-gh-fg-muted dark:text-gh-fg-dark-muted">Median</div>
                              <div class="font-semibold">{currentFileResult.line_length_median.toFixed(1)}</div>
                            </div>
                          {/if}
                          {#if currentFileResult.line_length_p95 !== null}
                            <div>
                              <div class="text-xs text-gh-fg-muted dark:text-gh-fg-dark-muted">95th %ile</div>
                              <div class="font-semibold">{currentFileResult.line_length_p95.toFixed(1)}</div>
                            </div>
                          {/if}
                          {#if currentFileResult.line_length_p99 !== null}
                            <div>
                              <div class="text-xs text-gh-fg-muted dark:text-gh-fg-dark-muted">99th %ile</div>
                              <div class="font-semibold">{currentFileResult.line_length_p99.toFixed(1)}</div>
                            </div>
                          {/if}
                          {#if currentFileResult.line_length_stddev !== null}
                            <div>
                              <div class="text-xs text-gh-fg-muted dark:text-gh-fg-dark-muted">Std Dev</div>
                              <div class="font-semibold">{currentFileResult.line_length_stddev.toFixed(1)}</div>
                            </div>
                          {/if}
                        </div>
                      </div>
                    {/if}
                  </div>
                {/if}

                <!-- Compression Information -->
                {#if currentFileResult.is_compressed}
                  <div>
                    <h3 class="text-sm font-semibold text-gh-fg-default dark:text-gh-fg-dark-default mb-3 uppercase tracking-wide">
                      Compression
                    </h3>
                    <div class="grid grid-cols-2 gap-4">
                      <div>
                        <div class="text-xs text-gh-fg-muted dark:text-gh-fg-dark-muted mb-1">Format</div>
                        <div class="text-sm font-medium uppercase">{currentFileResult.compression_format}</div>
                      </div>
                      {#if currentFileResult.compression_ratio !== null}
                        <div>
                          <div class="text-xs text-gh-fg-muted dark:text-gh-fg-dark-muted mb-1">Ratio</div>
                          <div class="text-sm font-medium">{currentFileResult.compression_ratio.toFixed(2)}x</div>
                        </div>
                      {/if}
                      {#if currentFileResult.compressed_size !== null}
                        <div>
                          <div class="text-xs text-gh-fg-muted dark:text-gh-fg-dark-muted mb-1">Compressed Size</div>
                          <div class="text-sm font-medium">{formatBytes(currentFileResult.compressed_size)}</div>
                        </div>
                      {/if}
                      {#if currentFileResult.decompressed_size !== null}
                        <div>
                          <div class="text-xs text-gh-fg-muted dark:text-gh-fg-dark-muted mb-1">Decompressed Size</div>
                          <div class="text-sm font-medium">{formatBytes(currentFileResult.decompressed_size)}</div>
                        </div>
                      {/if}
                    </div>
                  </div>
                {/if}

                <!-- Index Information -->
                {#if currentFileResult.has_index}
                  <div>
                    <h3 class="text-sm font-semibold text-gh-fg-default dark:text-gh-fg-dark-default mb-3 uppercase tracking-wide">
                      Index
                    </h3>
                    <div class="grid grid-cols-2 gap-4">
                      <div>
                        <div class="text-xs text-gh-fg-muted dark:text-gh-fg-dark-muted mb-1">Status</div>
                        <div class="text-sm font-medium">
                          {currentFileResult.index_valid ? '✓ Valid' : '✗ Invalid'}
                        </div>
                      </div>
                      {#if currentFileResult.index_checkpoint_count !== null}
                        <div>
                          <div class="text-xs text-gh-fg-muted dark:text-gh-fg-dark-muted mb-1">Checkpoints</div>
                          <div class="text-sm font-medium">{currentFileResult.index_checkpoint_count}</div>
                        </div>
                      {/if}
                    </div>
                  </div>
                {/if}

                <!-- Custom Metrics -->
                {#if currentFileResult.custom_metrics && Object.keys(currentFileResult.custom_metrics).length > 0}
                  <div>
                    <h3 class="text-sm font-semibold text-gh-fg-default dark:text-gh-fg-dark-default mb-3 uppercase tracking-wide">
                      Custom Metrics
                    </h3>
                    <div class="border border-gh-border-default dark:border-gh-border-dark-default rounded p-4 bg-gh-canvas-subtle dark:bg-gh-canvas-dark-subtle">
                      <pre class="text-xs font-mono overflow-x-auto">{JSON.stringify(currentFileResult.custom_metrics, null, 2)}</pre>
                    </div>
                  </div>
                {/if}
              </div>
            {/if}
          </div>
        {/if}
      </div>

      <!-- Footer -->
      <div class="flex items-center justify-end px-6 py-4 border-t border-gh-border-default dark:border-gh-border-dark-default flex-shrink-0">
        <button
          class="px-4 py-2 text-sm font-medium rounded bg-gh-accent-emphasis dark:bg-gh-accent-dark-emphasis text-white hover:bg-gh-accent-fg dark:hover:bg-gh-accent-dark-fg"
          on:click={closeAnalyzePopup}
        >
          Close
        </button>
      </div>
    </div>
  </div>
{/if}
