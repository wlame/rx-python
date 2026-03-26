# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for RX Tracer

Note: Frontend is now managed separately and downloaded from GitHub releases.
      It is NOT bundled in the binary to keep the executable small.
"""

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

# Frontend is NOT bundled - it will be downloaded from GitHub releases
# This keeps the binary small and allows independent frontend updates
datas = []

print("[INFO] Frontend is not bundled in the binary")
print("[INFO] The server will download frontend from GitHub releases on startup")

# Hidden imports that PyInstaller might miss
hiddenimports = [
    'rx',
    'rx.cli',
    'rx.cli.main',
    'rx.cli.trace',
    'rx.cli.check',
    'rx.cli.index',
    'rx.cli.samples',
    'rx.cli.serve',
    'rx.cli.compress',
    'rx.cli.prometheus',
    'rx.models',
    'rx.trace',
    'rx.trace_cache',
    'rx.analyzer',
    'rx.regex',
    'rx.web',
    'rx.indexer',
    'rx.unified_index',
    'rx.compression',
    'rx.compressed_index',
    'rx.seekable_zstd',
    'rx.seekable_index',
    'rx.file_utils',
    'rx.rg_json',
    'rx.cli_command_builder',
    'rx.hooks',
    'rx.path_security',
    'rx.frontend_manager',
    'rx.prometheus',
    'rx.request_store',
    'rx.task_manager',
    'rx.utils',
    'uvicorn',
    'fastapi',
    'click',
    'sh',
    'httpx',
]

a = Analysis(
    ['src/rx/cli/main.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='rx',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
