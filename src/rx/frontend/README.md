# RX-Trace Frontend

A modern web interface for the RX-Trace file search and analysis tool.

## Features

- **File Tree Navigation**: Browse search roots and navigate directory structures
- **File Viewer**: View file contents with syntax highlighting and line numbers
- **Search**: Full-text search with regex support using ripgrep
- **Dynamic Loading**: Efficiently load large files in chunks
- **Dark Mode**: GitHub-inspired light and dark themes
- **Keyboard Shortcuts**: Quick navigation and actions

## Technology Stack

- **Svelte 4**: Reactive UI framework
- **TypeScript**: Type-safe development
- **Tailwind CSS**: Utility-first styling
- **Bun**: Fast package manager and bundler
- **Vite**: Development server

## Development

### Prerequisites

- Docker (for Bun via Docker)
- OR Bun installed locally

### Local Development

```bash
# Install dependencies (using Docker)
make frontend-install

# Start dev server (using Docker)
make frontend-dev

# Or with local Bun
cd src/rx/frontend
bun install
bun run dev
```

The dev server will start at `http://localhost:5173` and proxy API requests to `http://localhost:8080`.

### Building for Production

```bash
# Build (using Docker)
make frontend-build

# Or with local Bun
cd src/rx/frontend
bun run build
```

The build output will be in `src/rx/frontend/dist/`.

## Keyboard Shortcuts

- `Cmd/Ctrl + K`: Focus search input
- `Cmd/Ctrl + /`: Show keyboard shortcuts help
- `Escape`: Close dialogs

## Architecture

### Directory Structure

```
src/rx/frontend/
├── src/
│   ├── lib/
│   │   ├── api.ts              # Backend API client
│   │   ├── types.ts            # TypeScript type definitions
│   │   ├── stores/             # Svelte stores (state management)
│   │   │   ├── health.ts       # API health status
│   │   │   ├── tree.ts         # File tree state
│   │   │   ├── files.ts        # Open files state
│   │   │   ├── trace.ts        # Search state
│   │   │   └── settings.ts     # User settings
│   │   └── utils/              # Utility functions
│   ├── components/
│   │   ├── layout/             # Layout components
│   │   │   ├── Header.svelte
│   │   │   ├── Sidebar.svelte
│   │   │   ├── MainContent.svelte
│   │   │   └── StatusBar.svelte
│   │   ├── tree/               # File tree components
│   │   │   ├── FileTree.svelte
│   │   │   ├── TreeNode.svelte
│   │   │   └── FileIcon.svelte
│   │   ├── editor/             # File viewer components
│   │   │   └── EditorPane.svelte
│   │   ├── search/             # Search components
│   │   │   └── SearchPanel.svelte
│   │   └── common/             # Shared components
│   │       ├── Spinner.svelte
│   │       ├── Icon.svelte
│   │       └── KeyboardShortcuts.svelte
│   ├── App.svelte              # Root component
│   ├── main.ts                 # Entry point
│   └── app.css                 # Global styles
├── public/                     # Static assets
├── index.html                  # HTML template
├── package.json                # Dependencies
├── vite.config.ts              # Vite configuration
├── tailwind.config.js          # Tailwind configuration
└── tsconfig.json               # TypeScript configuration
```

### State Management

The application uses Svelte stores for state management:

- **health**: API connection status and health checks
- **tree**: File tree navigation state
- **files**: Open files and their content
- **trace**: Search queries and results
- **settings**: User preferences (theme, sidebar width, etc.)

### API Integration

The frontend communicates with the RX-Trace backend via REST API:

- `GET /`: Health check
- `GET /v1/tree`: List directory contents
- `GET /v1/samples`: Get file content samples
- `GET /v1/trace`: Search for patterns

## Styling

The application uses a GitHub-inspired color scheme with support for light and dark modes. Colors are defined in `tailwind.config.js` and follow the pattern:

- `gh-canvas-*`: Background colors
- `gh-fg-*`: Foreground/text colors
- `gh-border-*`: Border colors
- `gh-accent-*`: Accent colors
- `gh-success-*`, `gh-danger-*`, `gh-attention-*`: Semantic colors

## Configuration

### Environment Variables

None required - the frontend proxies API requests to the backend.

### Vite Proxy

In development, API requests to `/v1/*` are proxied to `http://localhost:8080`. This is configured in `vite.config.ts`.

## Browser Support

- Modern browsers (Chrome, Firefox, Safari, Edge)
- ES2020+ features required
- No IE11 support

## License

MIT
