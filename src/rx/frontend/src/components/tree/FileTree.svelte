<script lang="ts">
  import { onMount } from 'svelte';
  import { tree } from '$lib/stores';
  import TreeNode from './TreeNode.svelte';
  import Spinner from '../common/Spinner.svelte';

  onMount(() => {
    tree.loadRoots();
  });
</script>

<div class="h-full flex flex-col">
  <div class="px-3 py-2 border-b border-gh-border-default dark:border-gh-border-dark-default">
    <h2
      class="text-xs font-semibold uppercase tracking-wide text-gh-fg-muted dark:text-gh-fg-dark-muted"
    >
      Search Roots
    </h2>
  </div>

  <div class="flex-1 overflow-auto scrollbar-thin py-1" role="tree">
    {#if $tree.loading}
      <div class="flex items-center justify-center py-8">
        <Spinner size="md" />
      </div>
    {:else if $tree.error}
      <div class="px-3 py-4 text-sm text-gh-danger-fg dark:text-gh-danger-dark-fg">
        <p class="font-medium">Failed to load</p>
        <p class="text-xs mt-1 opacity-75">{$tree.error}</p>
        <button
          class="mt-2 text-xs text-gh-accent-fg dark:text-gh-accent-dark-fg hover:underline"
          on:click={() => tree.loadRoots()}
        >
          Retry
        </button>
      </div>
    {:else if $tree.roots.length === 0}
      <div
        class="px-3 py-4 text-sm text-gh-fg-muted dark:text-gh-fg-dark-muted"
      >
        No search roots configured
      </div>
    {:else}
      {#each $tree.roots as node (node.path)}
        <TreeNode {node} />
      {/each}
    {/if}
  </div>
</div>
