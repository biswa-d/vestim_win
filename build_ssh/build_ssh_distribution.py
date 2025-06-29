"""
Build script for Vestim SSH/Remote Client Distribution
Creates a distributable package with SSH remote capabilities
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
import PyInstaller.__main__

def create_ssh_build():
    """Create the SSH-enabled Vestim distribution"""
    
    print("=== Building Vestim SSH/Remote Client ===")
    
    # Get paths
    project_root = Path(__file__).parent.parent
    build_ssh_dir = project_root / "build_ssh"
    dist_dir = build_ssh_dir / "ssh_dist"  # Use separate directory
    
    # Clean previous builds
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    
    # Create build directories
    build_ssh_dir.mkdir(exist_ok=True)
    
    print(f"Project root: {project_root}")
    print(f"Build directory: {build_ssh_dir}")
    
    # Create PyInstaller spec for Universal Client
    # Use raw strings for Windows paths
    project_root_str = str(project_root)
    vestim_client_path = project_root / "vestim_universal_client.py"
    
    # Check if the main script exists
    if not vestim_client_path.exists():
        print(f"ERROR: Script not found: {vestim_client_path}")
        print("Available Python files in project root:")
        for py_file in project_root.glob("*.py"):
            print(f"  - {py_file.name}")
        return
        
    universal_spec_content = f'''# -*- mode: python ; coding: utf-8 -*-
import os

block_cipher = None

a = Analysis(
    [r"{project_root_str}\\vestim_universal_client.py"],
    pathex=[r"{project_root_str}"],
    binaries=[],
    datas=[
        (r"{project_root_str}\\vestim", "vestim"),
        (r"{project_root_str}\\requirements.txt", "."),
        (r"{project_root_str}\\VESTIM_SERVER_CLIENT.md", "."),
        (r"{project_root_str}\\DEPLOYMENT_OPTIONS.md", "."),
        (r"{project_root_str}\\WORKFLOW_EXAMPLE.md", "."),
        (r"{project_root_str}\\setup_server.sh", "."),
    ],
    hiddenimports=[
        "paramiko",
        "scp",
        "cryptography",
        "bcrypt",
        "nacl",
        "keyring",
        "PyQt5.QtCore",
        "PyQt5.QtWidgets",
        "PyQt5.QtGui",
        "vestim.remote.auto_deploying_ssh_manager",
        "vestim.remote.auto_x11_installer",
    ],
    hookspath=[],
    hooksconfig={{}},
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
    name="VestimUniversalClient",
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
    icon=r"{project_root_str}\\vestim\\gui\\resources\\icon.ico" if os.path.exists(r"{project_root_str}\\vestim\\gui\\resources\\icon.ico") else None,
)
'''
    
    # Write spec file
    spec_file = build_ssh_dir / "vestim_universal.spec"
    with open(spec_file, 'w') as f:
        f.write(universal_spec_content)
    
    print("Created PyInstaller spec file")
    
    # Install build dependencies
    print("Installing build dependencies...")
    subprocess.run([
        sys.executable, "-m", "pip", "install", 
        "pyinstaller", "paramiko", "scp", "cryptography", "bcrypt", "pynacl", "keyring"
    ], check=True)
    
    # Run PyInstaller
    print("Running PyInstaller...")
    os.chdir(build_ssh_dir)
    
    PyInstaller.__main__.run([
        str(spec_file),
        "--clean",
        "--noconfirm",
        f"--distpath={dist_dir}",
        f"--workpath={build_ssh_dir}/ssh_build"  # Separate work directory
    ])
    
    # Create additional executables for other clients
    print("Building additional clients...")
    
    # Server Client (pre-configured)
    server_client_spec = f'''# -*- mode: python ; coding: utf-8 -*-
import os

block_cipher = None

a = Analysis(
    [r"{project_root_str}\\vestim_server_client.py"],
    pathex=[r"{project_root_str}"],
    binaries=[],
    datas=[
        (r"{project_root_str}\\vestim\\remote", "vestim/remote"),
    ],
    hiddenimports=[
        "PyQt5.QtCore",
        "PyQt5.QtWidgets", 
        "PyQt5.QtGui",
        "vestim.remote.ssh_manager",
        "vestim.remote.auto_x11_installer",
        "vestim.remote.config_dialog",
    ],
    hookspath=[],
    hooksconfig={{}},
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
    name="VestimServerClient",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=r"{project_root_str}\\vestim\\gui\\resources\\icon.ico" if os.path.exists(r"{project_root_str}\\vestim\\gui\\resources\\icon.ico") else None,
)
'''
    
    server_spec_file = build_ssh_dir / "vestim_server.spec"
    with open(server_spec_file, 'w') as f:
        f.write(server_client_spec)
    
    PyInstaller.__main__.run([
        str(server_spec_file),
        "--clean",
        "--noconfirm", 
        f"--distpath={dist_dir}",
        f"--workpath={build_ssh_dir}/ssh_build"  # Same work directory
    ])
    
    # Create setup wizard executable
    setup_spec = f'''# -*- mode: python ; coding: utf-8 -*-
import os

block_cipher = None

a = Analysis(
    [r"{project_root_str}\\vestim_server_setup.py"],
    pathex=[r"{project_root_str}"],
    binaries=[],
    datas=[
        (r"{project_root_str}\\vestim\\remote", "vestim/remote"),
    ],
    hiddenimports=[
        "PyQt5.QtCore",
        "PyQt5.QtWidgets",
        "PyQt5.QtGui",
    ],
    hookspath=[],
    hooksconfig={{}},
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
    name="VestimServerSetup",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=r"{project_root_str}\\vestim\\gui\\resources\\icon.ico" if os.path.exists(r"{project_root_str}\\vestim\\gui\\resources\\icon.ico") else None,
)
'''
    
    setup_spec_file = build_ssh_dir / "vestim_setup.spec"
    with open(setup_spec_file, 'w') as f:
        f.write(setup_spec)
    
    PyInstaller.__main__.run([
        str(setup_spec_file),
        "--clean",
        "--noconfirm",
        f"--distpath={dist_dir}",
        f"--workpath={build_ssh_dir}/ssh_build"  # Same work directory
    ])
    
    # Copy additional files to distribution
    print("Copying additional files...")
    
    additional_files = [
        "VESTIM_SERVER_CLIENT.md",
        "DEPLOYMENT_OPTIONS.md", 
        "WORKFLOW_EXAMPLE.md",
        "REMOTE_SETUP_GUIDE.md",
        "setup_server.sh",
        "requirements.txt"
    ]
    
    for file_name in additional_files:
        src_file = project_root / file_name
        if src_file.exists():
            shutil.copy2(src_file, dist_dir / file_name)
    
    # Create distribution package info
    create_package_info(dist_dir)
    
    print(f"\\n=== Build Complete ===")
    print(f"SSH Distribution created in: {dist_dir}")
    print(f"\\nExecutables created:")
    print(f"  - VestimUniversalClient.exe  (Auto-deploying client)")
    print(f"  - VestimServerClient.exe     (Pre-configured client)")
    print(f"  - VestimServerSetup.exe      (Setup wizard)")
    print(f"\\nSeparate from main Windows build in: {project_root}\\dist")
    print(f"Ready for SSH distribution!")

def create_package_info(dist_dir):
    """Create package information and usage instructions"""
    
    readme_content = '''# Vestim SSH/Remote Client Distribution

## What's Included

This package contains three executables for different remote access scenarios:

### 1. VestimUniversalClient.exe * RECOMMENDED
- **Universal access** to any Linux server
- **Auto-deployment** - no server setup required
- **Just need SSH access** to any server
- **First-time use**: 5-10 minutes (deploys Vestim)
- **Subsequent use**: Instant launches

### 2. VestimServerClient.exe  
- For **pre-configured servers** where Vestim is already installed
- **Instant launches** every time
- Requires server-side Vestim installation

### 3. VestimServerSetup.exe
- **Setup wizard** for configuring server connections
- **Run once** to save server details
- Used by VestimServerClient.exe

## Quick Start

### For Most Users (Universal Client):
1. **Double-click** `VestimUniversalClient.exe`
2. **Enter** your server details (any Linux server with SSH)
3. **Wait** for auto-deployment (first time only)
4. **Use** Vestim GUI running on remote server!

### For Pre-Configured Servers:
1. **Run** `VestimServerSetup.exe` to configure
2. **Use** `VestimServerClient.exe` for instant access

## Requirements

### Your Computer:
- Windows 10/11
- Internet connection
- The executables handle everything else automatically!

### Remote Server:
- Any Linux server with SSH access
- Internet connection (for downloading dependencies)
- Optional: sudo access (for installing system packages)

## Features

- ✅ **Zero server setup** required (Universal Client)
- ✅ **Auto-installs X11 server** (VcXsrv) if needed  
- ✅ **Automatic deployment** of Vestim to any server
- ✅ **Secure credential storage**
- ✅ **One-click launching** after initial setup
- ✅ **Works with cloud servers** (AWS, Azure, GCP, etc.)

## Support

For detailed documentation, see:
- `VESTIM_SERVER_CLIENT.md` - Complete user guide
- `DEPLOYMENT_OPTIONS.md` - Comparison of different approaches
- `WORKFLOW_EXAMPLE.md` - Step-by-step examples
- `REMOTE_SETUP_GUIDE.md` - Technical details

## Server Setup (Optional)

If you want to manually set up a server, use:
- `setup_server.sh` - Automated server installation script

Most users don't need this - the Universal Client handles server setup automatically!
'''
    
    with open(dist_dir / "README.md", 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    # Create a simple batch launcher
    batch_content = '''@echo off
title Vestim Universal Client Launcher
echo.
echo ===================================
echo   Vestim Universal Client
echo ===================================
echo.
echo Choose your launcher:
echo.
echo 1. Universal Client (Recommended)
echo    - Works with any Linux server
echo    - Auto-deploys Vestim
echo.
echo 2. Server Client (Pre-configured)
echo    - For pre-setup servers
echo    - Instant launches
echo.
echo 3. Server Setup Wizard
echo    - Configure server connections
echo.
echo 4. Exit
echo.
set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" goto universal
if "%choice%"=="2" goto server  
if "%choice%"=="3" goto setup
if "%choice%"=="4" goto exit

:universal
echo Starting Universal Client...
VestimUniversalClient.exe
goto end

:server
echo Starting Server Client...
VestimServerClient.exe
goto end

:setup
echo Starting Setup Wizard...
VestimServerSetup.exe
goto end

:exit
echo Goodbye!
goto end

:end
pause
'''
    
    with open(dist_dir / "Launch_Vestim.bat", 'w', encoding='utf-8') as f:
        f.write(batch_content)

if __name__ == "__main__":
    create_ssh_build()
