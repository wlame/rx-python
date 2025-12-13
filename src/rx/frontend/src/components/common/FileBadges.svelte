<script lang="ts">
  export let isCompressed: boolean | null = null;
  export let compressionFormat: string | null = null;
  export let isIndexed: boolean | null = null;
</script>

<span class="flex items-center gap-1 flex-shrink-0">
  {#if isCompressed && compressionFormat}
    <span
      class="badge badge-info text-[10px] py-0"
      title="Compressed file: {compressionFormat}"
    >
      {compressionFormat}
    </span>
  {/if}
  {#if isIndexed}
    <span class="badge badge-success text-[10px] py-0" title="Indexed">
      idx
    </span>
  {/if}
</span>
