<script lang="ts">
  import { notifications } from '$lib/stores/notifications';

  function getIcon(type: string) {
    switch (type) {
      case 'error':
        return '✕';
      case 'warning':
        return '⚠';
      case 'success':
        return '✓';
      case 'info':
      default:
        return 'ℹ';
    }
  }

  function getColorClasses(type: string) {
    switch (type) {
      case 'error':
        return 'bg-gh-danger-emphasis dark:bg-gh-danger-dark-emphasis text-white';
      case 'warning':
        return 'bg-gh-attention-emphasis dark:bg-gh-attention-dark-emphasis text-white';
      case 'success':
        return 'bg-gh-success-emphasis dark:bg-gh-success-dark-emphasis text-white';
      case 'info':
      default:
        return 'bg-gh-accent-emphasis dark:bg-gh-accent-dark-emphasis text-white';
    }
  }
</script>

<div class="fixed top-4 right-4 z-50 flex flex-col gap-2 pointer-events-none">
  {#each $notifications as notification (notification.id)}
    <div
      class="pointer-events-auto px-4 py-3 rounded-lg shadow-lg flex items-center gap-3 min-w-64 max-w-md
             {getColorClasses(notification.type)}"
      role="alert"
    >
      <span class="text-lg font-bold">{getIcon(notification.type)}</span>
      <span class="flex-1 text-sm">{notification.message}</span>
      <button
        class="text-lg font-bold opacity-70 hover:opacity-100"
        on:click={() => notifications.dismiss(notification.id)}
        aria-label="Dismiss"
      >
        ✕
      </button>
    </div>
  {/each}
</div>
