<script lang="ts">
  export let name: string;
  export let size: number = 16;

  // Simple inline SVG icons
  const icons: Record<string, string> = {
    // Chevrons
    'chevron-right': 'M9 18l6-6-6-6',
    'chevron-down': 'M6 9l6 6 6-6',

    // Files
    file: 'M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z M14 2v6h6',
    'file-text':
      'M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z M14 2v6h6 M16 13H8 M16 17H8 M10 9H8',
    folder: 'M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z',
    'folder-open':
      'M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2v1 M2 10h20 M22 10l-2 9H4l-2-9',

    // Actions
    x: 'M18 6L6 18 M6 6l12 12',
    search: 'M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z',
    sun: 'M12 1v2 M12 21v2 M4.22 4.22l1.42 1.42 M18.36 18.36l1.42 1.42 M1 12h2 M21 12h2 M4.22 19.78l1.42-1.42 M18.36 5.64l1.42-1.42 M12 6a6 6 0 100 12 6 6 0 000-12z',
    moon: 'M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z',
    settings: 'M12 8a4 4 0 100 8 4 4 0 000-8z M12 1v2 M12 21v2 M4.22 4.22l1.42 1.42 M18.36 18.36l1.42 1.42 M1 12h2 M21 12h2 M4.22 19.78l1.42-1.42 M18.36 5.64l1.42-1.42',

    // Status
    check: 'M20 6L9 17l-5-5',
    alert: 'M12 9v2 M12 15h.01 M12 2L2 22h20L12 2z',
    info: 'M12 2a10 10 0 100 20 10 10 0 000-20z M12 16v-4 M12 8h.01',

    // Misc
    archive: 'M21 8v13H3V8 M23 3H1v5h22V3z M10 12h4',
    loader: 'M12 2v4 M12 18v4 M4.93 4.93l2.83 2.83 M16.24 16.24l2.83 2.83 M2 12h4 M18 12h4 M4.93 19.07l2.83-2.83 M16.24 7.76l2.83-2.83',
  };

  $: path = icons[name] || '';
</script>

{#if path}
  <svg
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    stroke-width="2"
    stroke-linecap="round"
    stroke-linejoin="round"
    class="inline-block"
  >
    <path d={path} />
  </svg>
{/if}
