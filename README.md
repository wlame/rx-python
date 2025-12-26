# RX (Regex Tracer)

A high-performance tool for searching and analyzing large files, powered by ripgrep.

## Designed for large files.
RX is optimized for processing multi-GB files efficiently through parallel chunking and streaming.
If you need to process many files repeatedly, use the API server (`rx serve`) instead of running CLI commands in a loop. The server mode avoids Python startup overhead on each invocation.

## Key Features

- **Byte-Offset Based**: Returns precise byte offsets for efficient large file processing (line-based indexing available)
- **Parallel Processing**: Automatic chunking and parallel execution for large files
- **Samples output**: Can show arbitrary parts of text files with context when you found interested offsets
- **REST API Server**: All CLI features available via async HTTP API
- **File Analysis**: Extract metadata, statistics, and metrics from files (including compressed files)
- **Regex Complexity Analysis**: Detect ReDoS vulnerabilities before production use
- **Compressed File Support**: Analyze and search gzip, zstd, xz, bzip2 files transparently
- **Seekable Zstd**: Fast random access to seekable zstd compressed files with automatic indexing
- **Analysis Caching**: Cache file analysis results for instant repeated access
- **Background Tasks**: Background compression and indexing with progress tracking
- **Security Sandbox**: Restrict file access to specific directories in server mode

## Prerequisites

**ripgrep must be installed:**

- **macOS**: `brew install ripgrep`
- **Ubuntu/Debian**: `apt install ripgrep`
- **Windows**: `choco install ripgrep`

## Installation

### Option 1: Install from PyPI (Recommended)

```bash
# Requires Python 3.13+
pip install rx-tool

# Now use the rx command
rx "error.*" /var/log/app.log
rx --version
```

**Note:** Requires `ripgrep` to be installed separately (see Prerequisites).

### Option 2: Development with uv

```bash
uv sync
uv run rx "error.*" /var/log/app.log
```

### Option 3: Standalone Binary

```bash
./build.sh
./dist/rx "error.*" /var/log/app.log
```

### Shell Completion

Enable tab completion for `rx` commands and options:

**Zsh** (add to `~/.zshrc`):
```bash
_RX_COMPLETE=zsh_source rx > ~/.rx-complete.zsh
echo 'source ~/.rx-complete.zsh' >> ~/.zshrc
source ~/.zshrc
```

**Bash** (add to `~/.bashrc`):
```bash
_RX_COMPLETE=bash_source rx > ~/.rx-complete.bash
echo 'source ~/.rx-complete.bash' >> ~/.bashrc
source ~/.bashrc
```

**Fish**:
```bash
_RX_COMPLETE=fish_source rx > ~/.config/fish/completions/rx.fish
```

After setup, `rx <Tab>` will suggest subcommands (`index`, `trace`, etc.) and options.

## Quick Start

### Basic Examples

```bash
# Search a file (returns byte offsets)
rx "error.*" /var/log/app.log

# Search a directory
rx "error.*" /var/log/

# Show context lines
rx "error" /var/log/app.log --samples --context=3

# Search piped input (like grep)
cat /var/log/app.log | rx "error"
docker logs mycontainer | rx "error"

# Index and analyze file metadata
rx index /var/log/app.log --analyze

# Check regex complexity
rx check "(a+)+"

# Start API server
rx serve --port=8000
```

## Why Byte Offsets?

RX returns **byte offsets** instead of line numbers for efficiency. Seeking to byte position is O(1), while counting lines is O(n). For large files, this matters significantly.

**Need line numbers?** Use the indexing feature:

```bash
# Create index for a large file
rx index /var/log/huge.log

# Now you can use line-based operations
rx samples /var/log/huge.log -l 1000,2000,3000 --context=5
```

The index enables fast line-to-offset conversion for files >50MB.

## Server Mode (Recommended for Repeated Operations)

The CLI spawns a Python interpreter on each invocation. For processing multiple files or repeated operations, use the API server:

```bash
# Start server
rx serve --port=8000

# Use HTTP API (same endpoints as CLI)
curl "http://localhost:8000/v1/trace?path=/var/log/app.log&regexp=error"
curl "http://localhost:8000/v1/analyze?path=/var/log/"
```

**Benefits:**
- No Python startup overhead per request
- Async processing with configurable workers
- Webhook support for event notifications
- Security sandbox with `--search-root`

### Security Sandbox

Restrict file access in server mode:

```bash
# Only allow access to /var/log
rx serve --search-root=/var/log

# Attempts to access other paths return 403 Forbidden
curl "http://localhost:8000/v1/trace?path=/etc/passwd&regexp=root"
# => 403 Forbidden
```

Prevents directory traversal (`../`) and symlink escape attacks.

## CLI Commands

### `rx` (search)
Search files for regex patterns. Follows ripgrep argument order: `PATTERN [PATH ...]`

```bash
rx "error.*" /var/log/app.log              # Basic search
rx "error.*" /var/log/                     # Search directory
rx "error" /var/log/app.log --samples      # Show context lines
rx "error" /var/log/app.log -i             # Case-insensitive (ripgrep flags work)
rx "error" /var/log/app.log --json         # JSON output

# Pipe input from other commands
cat /var/log/app.log | rx "error"          # Search piped input
docker logs mycontainer | rx "error"       # Search container logs
tail -f /var/log/app.log | rx "error"      # Search live log stream
echo "test error" | rx "error"             # Search from echo
rx "error" -                               # Read from stdin (- means stdin)
```

### `rx index`
Create line-offset index for files with optional full analysis and anomaly detection.

```bash
rx index /var/log/huge.log                # Index large files only (>=50MB)
rx index /var/log/app.log --analyze       # Index all files with full analysis
rx index /var/log/ --analyze -r           # Recursive directory analysis
rx index /var/log/app.log --info          # Show index info
rx index /var/log/app.log --delete        # Delete index
rx index /var/log/app.log --analyze --json # JSON output
```

**Index behavior:**
- Without `--analyze`: Only indexes large files (>=50MB) for line-based access
- With `--analyze`: Indexes ALL files with full analysis + anomaly detection

**Analysis output includes:**
- File size (bytes and human-readable)
- Compression info (format, ratio, decompressed size)
- Line statistics (count, length metrics, endings)
- Anomaly detection (errors, tracebacks, warnings)
- All results cached at `~/.cache/rx/indexes/` for instant repeated access

### `rx check`
Analyze regex complexity and detect ReDoS vulnerabilities.

```bash
rx check "(a+)+"                          # Returns risk level and fixes
```

### `rx samples`
Extract context lines around byte offsets or line numbers.

```bash
rx samples /var/log/app.log -b 12345,67890 --context=3   # Byte offsets with context
rx samples /var/log/app.log -l 100,200 --context=5       # Line numbers with context

# Ranges (get exact lines, ignores context)
rx samples /var/log/app.log -l 100-200                   # Lines 100-200 inclusive
rx samples /var/log/app.log -b 1000-5000                 # Lines covering byte range

# Negative offsets (count from end of file)
rx samples /var/log/app.log -l -1 --context=3            # Last line with context
rx samples /var/log/app.log -l -5,-1                     # 5th from end and last line

# Mix single values and ranges
rx samples /var/log/app.log -l 50,100-200 --context=2    # Line 50 with context + lines 100-200
```

### `rx serve`
Start REST API server.

```bash
rx serve                                  # Start on localhost:8000
rx serve --host=0.0.0.0 --port=8080       # Custom host/port
rx serve --search-root=/var/log           # Restrict to directory
```

## API Endpoints

Once the server is running, visit http://localhost:8000/docs for interactive API documentation.

**Main Endpoints:**
- `GET /v1/trace` - Search files for patterns
- `GET /v1/index` - File indexing with optional analysis and anomaly detection
- `GET /v1/complexity` - Regex complexity analysis
- `GET /v1/samples` - Extract context lines
- `GET /health` - Server health and configuration

**Background Task Endpoints:**
- `POST /v1/compress` - Start background compression to seekable zstd (returns task_id)
- `POST /v1/index` - Start background indexing for large files (returns task_id)
- `GET /v1/tasks/{task_id}` - Check task status (queued, running, completed, failed)

**Examples:**

```bash
# Search
curl "http://localhost:8000/v1/trace?path=/var/log/app.log&regexp=error&max_results=10"

# Index with analysis (works with compressed files too)
curl "http://localhost:8000/v1/index?path=/var/log/app.log&analyze=true"

# Start background compression (returns immediately with task_id)
curl -X POST "http://localhost:8000/v1/compress" \
  -H "Content-Type: application/json" \
  -d '{"input_path": "/var/log/huge.log", "frame_size": "1MB"}'
# Response: {"task_id": "550e8400-e29b-41d4-a716-446655440000", "status": "queued"}

# Check compression progress
curl "http://localhost:8000/v1/tasks/550e8400-e29b-41d4-a716-446655440000"

# With webhooks
curl "http://localhost:8000/v1/trace?path=/var/log/app.log&regexp=error&hook_on_complete=https://example.com/webhook"
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `RX_WORKERS` | Worker processes for server | `1` |
| `RX_LOG_LEVEL` | Log level (DEBUG, INFO, WARNING, ERROR) | `INFO` |
| `RX_MAX_SUBPROCESSES` | Max parallel workers for file processing | `20` |
| `RX_MIN_CHUNK_SIZE_MB` | Min chunk size for splitting files | `20` |

### Server Configuration

```bash
# Production example (8-core, 16GB RAM)
RX_WORKERS=17 \
RX_LIMIT_CONCURRENCY=500 \
RX_LIMIT_MAX_REQUESTS=10000 \
rx serve --host=0.0.0.0 --port=8000 --search-root=/data

# Container/Kubernetes (1 worker per pod, scale with replicas)
RX_WORKERS=1 rx serve --host=0.0.0.0 --port=8000
```

## Compressed File Support

RX can search, analyze, and extract samples from compressed files without manual decompression. Supported formats:
- **gzip** (`.gz`)
- **zstd** (`.zst`) - including seekable zstd
- **xz** (`.xz`)
- **bzip2** (`.bz2`)

### Searching Compressed Files

```bash
# Search a gzip file - works exactly like regular files
rx "error.*" /var/log/syslog.1.gz

# Search with context
rx "error" /var/log/syslog.1.gz --samples --context=3

# All regular options work with compressed files
rx "error" /var/log/app.log.gz -i --json
```

### Analyzing Compressed Files

```bash
# Full analysis with automatic decompression
rx analyze /var/log/app.log.gz

# Output includes compression info:
# Compressed: gzip, ratio: 5.2x, decompressed: 2.5 GB
# Lines: 50000000 total, 0 empty
# Line length: max=256, avg=128.5, median=130.0
```

### Extracting Samples from Compressed Files

For compressed files, use **line numbers** (byte offsets are not meaningful in compressed streams):

```bash
# Get lines 100, 200, 300 with 5 lines of context
rx samples /var/log/syslog.1.gz -l 100,200,300 --context=5
```

### Seekable Zstd Performance

For very large compressed files, convert to **seekable zstd format** for parallel processing:

```bash
# Background compression to seekable zstd (returns immediately)
rx compress /var/log/huge.log --frame-size=1MB

# Or via API:
curl -X POST "http://localhost:8000/v1/compress" \
  -H "Content-Type: application/json" \
  -d '{"input_path": "/var/log/huge.log", "frame_size": "1MB"}'
```

**Seekable zstd benefits:**
- Fast random access without decompressing entire file
- Automatic frame-based indexing
- Better than regular zstd for large files
- Compressed index cached at `~/.cache/rx/indexes/`

### Performance Considerations

- Regular compressed files (gzip, xz, bzip2) processed sequentially
- Seekable zstd supports parallel frame access (Tier 2 - coming soon)
- All indexes cached at `~/.cache/rx/indexes/` for instant repeated access

### API Usage

```bash
# Search compressed file
curl "http://localhost:8000/v1/trace?path=/var/log/syslog.1.gz&regexp=error"

# Index and analyze file (includes compression info, anomaly detection)
curl "http://localhost:8000/v1/index?path=/var/log/app.log.gz&analyze=true"

# Get samples (use lines parameter, not offsets)
curl "http://localhost:8000/v1/samples?path=/var/log/syslog.1.gz&lines=100,200&context=3"

# Get samples with line ranges (ignores context, returns exact lines)
curl "http://localhost:8000/v1/samples?path=/var/log/app.log&lines=100-200"

# Get samples with negative offsets (count from end of file)
curl "http://localhost:8000/v1/samples?path=/var/log/app.log&lines=-1,-5&context=2"

# Start background compression task
curl -X POST "http://localhost:8000/v1/compress" \
  -H "Content-Type: application/json" \
  -d '{"input_path": "/var/log/huge.log", "build_index": true}'
```

## Roadmap

### Completed Features ✅
- **Compressed File Support** - Analyze and search gzip, zstd, xz, bzip2 files
- **File Analysis Caching** - Cache analysis results at `~/.cache/rx/indexes/`
- **Background Tasks** - Compression and indexing endpoints with progress tracking
- **Seekable Zstd** - Frame-based indexing for random access without full decompression
- **Enhanced Analysis** - Compression info, index info, detailed statistics

### In Progress / Coming Soon
- **Parallel Seekable Zstd** - Parallel frame processing for seekable zstd files (Tier 2)
- **Streaming API** - WebSocket endpoint for real-time results
- **Compression Streaming** - Stream compressed output without decompression

## Web Frontend

RX includes a web-based file viewer with advanced navigation and analysis capabilities:

```bash
# Start server with web UI
rx serve --port=8000

# Open browser to http://localhost:8000
```

**Frontend Features:**

• **Multi-file viewer with tabs** - View multiple files simultaneously with tab management and drag-and-drop reordering

• **Advanced search capabilities** - In-file text search with match highlighting and navigation; global trace search across configured roots with regex support

• **Syntax highlighting and filtering** - Toggle-able syntax highlighting; regex-based content filtering with hide/show/highlight modes

• **Invisible characters display mode** - Toggle to visualize spaces (·), tabs (→), carriage returns (⏎), non-breaking spaces (°), and zero-width characters (|)

• **Virtual scrolling for large files** - Efficient on-demand content loading with adjustable lines per page; support for compressed files (gz, zst, xz, bz2)

• **File tree browser with analysis** - Hierarchical directory navigation with context actions to analyze file statistics and create line-offset indexes

• **Theme support and customization** - Light/dark/system themes; configurable font size, line numbers, and keyboard shortcuts

## Development

```bash
# Run tests
uv run pytest -v

# Run with coverage
uv run pytest --cov=rx --cov-report=html

# Build binary
uv sync --group build
./build.sh

# Build frontend
make frontend-build
```

## License

MIT
