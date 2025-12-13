<script lang="ts">
  import { onMount } from 'svelte';
  import { health, settings, files, tree } from '$lib/stores';
  import { readUrlState } from '$lib/utils/urlState';
  import Header from './components/layout/Header.svelte';
  import Sidebar from './components/layout/Sidebar.svelte';
  import MainContent from './components/layout/MainContent.svelte';
  import StatusBar from './components/layout/StatusBar.svelte';
  import KeyboardShortcuts from './components/common/KeyboardShortcuts.svelte';
  import Notifications from './components/common/Notifications.svelte';

  onMount(() => {
    health.startPolling();

    // Read URL params and open file if specified
    const urlState = readUrlState();
    if (urlState) {
      // Open the file
      files.openFile(
        urlState.path,
        urlState.line,
        undefined, // fileSize will be determined by backend
        urlState.syntaxHighlighting
      );

      // Expand tree to show the file
      tree.expandToPath(urlState.path);
    }

    return () => health.stopPolling();
  });
</script>

<div class="h-screen flex flex-col">
  <Header />

  <div class="flex-1 flex min-h-0">
    <Sidebar width={$settings.sidebarWidth} />
    <MainContent />
  </div>

  <StatusBar />
  <KeyboardShortcuts />
  <Notifications />
</div>
