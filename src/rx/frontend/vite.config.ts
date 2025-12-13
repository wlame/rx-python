import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import { resolve } from 'path';

// Plugin to copy Monaco editor workers
function monacoEditorPlugin() {
  return {
    name: 'monaco-editor-plugin',
    configureServer(server: any) {
      server.middlewares.use((req: any, res: any, next: any) => {
        if (req.url?.includes('/monaco-editor/')) {
          // Let Vite handle Monaco editor files
          next();
        } else {
          next();
        }
      });
    },
  };
}

export default defineConfig({
  plugins: [
    svelte({
      onwarn: (warning, handler) => {
        // Suppress accessibility warnings during build
        if (warning.code.startsWith('a11y-')) return;
        // Suppress unused export warnings
        if (warning.code === 'unused-export-let') return;
        handler(warning);
      },
    }),
    monacoEditorPlugin(),
  ],
  resolve: {
    alias: {
      $lib: resolve(__dirname, 'src/lib'),
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    sourcemap: true,
  },
  server: {
    proxy: {
      '/v1': {
        target: 'http://localhost:8080',
        changeOrigin: true,
      },
    },
  },
});
