<script lang="ts">
  import { health, settings, resolvedTheme } from '$lib/stores';

  function toggleTheme() {
    settings.update((s) => ({
      ...s,
      theme: $resolvedTheme === 'dark' ? 'light' : 'dark',
    }));
  }
</script>

<header
  class="h-12 flex items-center justify-between px-4
         bg-gh-canvas-subtle dark:bg-gh-canvas-dark-subtle
         border-b border-gh-border-default dark:border-gh-border-dark-default"
>
  <div class="flex items-center gap-3">
    <h1 class="text-lg font-semibold">rx-trace</h1>
    {#if $health.data}
      <span class="badge badge-info">v{$health.data.app_version}</span>
      {#if !$health.data.ripgrep_available}
        <span class="badge bg-gh-danger-emphasis/10 text-gh-danger-fg dark:text-gh-danger-dark-fg">
          ripgrep missing
        </span>
      {/if}
    {/if}
  </div>

  <div class="flex items-center gap-4">
    <!-- Connection status -->
    <div class="flex items-center gap-2 text-sm">
      {#if $health.loading}
        <span
          class="w-2 h-2 rounded-full bg-gh-attention-fg dark:bg-gh-attention-dark-fg animate-pulse"
        />
        <span class="text-gh-fg-muted dark:text-gh-fg-dark-muted"
          >Connecting...</span
        >
      {:else if $health.connected}
        <span
          class="w-2 h-2 rounded-full bg-gh-success-fg dark:bg-gh-success-dark-fg"
        />
        <span class="text-gh-fg-muted dark:text-gh-fg-dark-muted">Connected</span
        >
      {:else}
        <span
          class="w-2 h-2 rounded-full bg-gh-danger-fg dark:bg-gh-danger-dark-fg"
        />
        <span class="text-gh-danger-fg dark:text-gh-danger-dark-fg"
          >Disconnected</span
        >
      {/if}
    </div>

    <!-- Theme toggle -->
    <button
      class="p-1.5 rounded-md hover:bg-gh-canvas-inset dark:hover:bg-gh-canvas-dark-inset
             text-gh-fg-muted dark:text-gh-fg-dark-muted"
      title="Toggle theme"
      on:click={toggleTheme}
    >
      {#if $resolvedTheme === 'dark'}
        <svg class="w-5 h-5" viewBox="0 0 24 24" fill="currentColor">
          <path
            d="M12 3a9 9 0 109 9c0-.46-.04-.92-.1-1.36a5.389 5.389 0 01-4.4 2.26 5.403 5.403 0 01-3.14-9.8c-.44-.06-.9-.1-1.36-.1z"
          />
        </svg>
      {:else}
        <svg
          class="w-5 h-5"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
        >
          <circle cx="12" cy="12" r="5" />
          <path
            d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"
          />
        </svg>
      {/if}
    </button>
  </div>
</header>
