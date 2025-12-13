# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for RX Tracer
"""

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

# Collect frontend dist files
frontend_dist = Path('src/rx/frontend/dist')
datas = []

if frontend_dist.exists():
    # Add all files and subdirectories from frontend/dist to the bundle
    # Need to add the directory and its contents recursively
    for item in frontend_dist.rglob('*'):
        if item.is_file():
            # Get relative path from frontend/dist
            rel_path = item.relative_to(frontend_dist.parent)
            dest_path = f'rx/frontend/{rel_path.parent}'
            datas.append((str(item), dest_path))

    print(f"[OK] Including {len(datas)} frontend files from {frontend_dist}")
else:
    print(f"[WARN] Frontend dist not found at {frontend_dist}")
    print("  Run: make frontend-build")

# Hidden imports that PyInstaller might miss
hiddenimports = [
    'rx',
    'rx.cli',
    'rx.cli.main',
    'rx.cli.analyse',
    'rx.cli.check',
    'rx.cli.index',
    'rx.cli.samples',
    'rx.cli.search',
    'rx.cli.serve',
    'rx.parse',
    'rx.models',
    'rx.index',
    'rx.analyse',
    'rx.regex',
    'rx.web',
    'uvicorn',
    'fastapi',
    'click',
    'sh',
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
