<script lang="ts">
  import { onMount } from 'svelte';

  export let fallback: string = 'Something went wrong';

  let hasError = false;
  let errorMessage = '';

  onMount(() => {
    window.addEventListener('error', handleError);
    window.addEventListener('unhandledrejection', handleRejection);

    return () => {
      window.removeEventListener('error', handleError);
      window.removeEventListener('unhandledrejection', handleRejection);
    };
  });

  function handleError(event: ErrorEvent) {
    hasError = true;
    errorMessage = event.message;
    console.error('Caught error:', event.error);
  }

  function handleRejection(event: PromiseRejectionEvent) {
    hasError = true;
    errorMessage = event.reason?.message || 'Promise rejection';
    console.error('Caught rejection:', event.reason);
  }

  function reset() {
    hasError = false;
    errorMessage = '';
  }
</script>

{#if hasError}
  <div
    class="fixed inset-0 bg-gh-canvas-default dark:bg-gh-canvas-dark-default
           flex items-center justify-center z-50"
  >
    <div class="text-center p-8 max-w-md">
      <svg
        class="w-16 h-16 mx-auto mb-4 text-gh-danger-fg dark:text-gh-danger-dark-fg"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
      >
        <circle cx="12" cy="12" r="10" />
        <path d="M12 8v4M12 16h.01" />
      </svg>

      <h1 class="text-xl font-semibold mb-2">{fallback}</h1>
      <p class="text-sm text-gh-fg-muted dark:text-gh-fg-dark-muted mb-4">
        {errorMessage}
      </p>

      <button class="btn btn-primary" on:click={reset}>
        Try Again
      </button>
    </div>
  </div>
{:else}
  <slot />
{/if}
