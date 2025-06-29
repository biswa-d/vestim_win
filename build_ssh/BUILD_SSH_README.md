# Vestim SSH/Remote Build System

This directory contains the build system for creating Vestim SSH/Remote client distributions, completely separate from the standalone Windows build.

## Build Structure

```
build_ssh/
├── build_ssh_distribution.py     # Main build script
├── build_ssh.bat                # Windows batch runner  
├── build_complete.bat           # Complete build process
├── build_requirements.txt       # SSH-specific dependencies
├── vestim_ssh_installer.iss     # Inno Setup installer script
├── BUILD_SSH_README.md         # This file
├── dist/                       # Generated executables (after build)
└── build/                      # Temporary build files
```

## What Gets Built

### Executables:
1. **VestimUniversalClient.exe** - Auto-deploying client for any server
2. **VestimServerClient.exe** - Pre-configured server client  
3. **VestimServerSetup.exe** - Server configuration wizard

### Distribution Packages:
1. **VestimSSHInstaller.exe** - Full Windows installer
2. **VestimSSH_Portable.zip** - Portable package
3. **Launch_Vestim.bat** - Menu launcher

## Build Process

### Quick Build (Executables Only):
```batch
build_ssh.bat
```

### Complete Build (All Packages):
```batch
build_complete.bat
```

### Manual Build:
```batch
# Install dependencies
pip install -r build_requirements.txt

# Run build script
python build_ssh_distribution.py
```

## Prerequisites

### Required:
- Python 3.8+
- PyInstaller
- SSH build dependencies (in build_requirements.txt)

### Optional:
- **Inno Setup** (for .exe installer creation)
- **7-Zip** (for portable ZIP creation)

## Build Features

### Separation from Main Build:
- ✅ **Independent** from existing Windows standalone build
- ✅ **No conflicts** with main build system
- ✅ **Clean merging** between branches
- ✅ **Separate dependencies** and configurations

### Smart Dependency Handling:
- ✅ **Auto-installs** required packages for SSH functionality
- ✅ **Minimal footprint** - only includes what's needed
- ✅ **Hidden imports** properly configured for PyInstaller

### Multiple Distribution Formats:
- ✅ **Professional installer** (Inno Setup)
- ✅ **Portable package** (ZIP file)  
- ✅ **Individual executables** (direct use)

## Usage After Build

### For End Users:
1. **Installer**: Run `VestimSSHInstaller.exe` for full installation
2. **Portable**: Extract `VestimSSH_Portable.zip` and run
3. **Individual**: Use executables directly from `dist/` folder

### For Developers:
1. Test executables in `dist/` folder
2. Distribute via installer or portable package
3. Update build scripts as needed

## Key Differences from Main Build

| Feature | Main Build | SSH Build |
|---------|------------|-----------|
| **Purpose** | Standalone Windows app | SSH remote clients |
| **Size** | ~1GB (full Vestim) | ~50MB (SSH client) |
| **Dependencies** | All Vestim libs | SSH + minimal core |
| **Target** | Local execution | Remote server connection |
| **Build Dir** | `build/` | `build_ssh/` |
| **Conflicts** | None | None |

## Troubleshooting

### Build Fails:
1. Check Python version (3.8+ required)
2. Install build dependencies: `pip install -r build_requirements.txt`
3. Ensure PyInstaller is working: `pyinstaller --version`

### Missing Modules in Executable:
1. Add to `hiddenimports` in build script
2. Check import paths in source code
3. Test with `python -c "import module_name"`

### Installer Creation Fails:
1. Install Inno Setup from https://jrsoftware.org/isinfo.php
2. Ensure `iscc.exe` is in PATH
3. Check file paths in `.iss` script

## Deployment

### Internal Testing:
```batch
# Build and test locally
build_ssh.bat
cd dist
VestimUniversalClient.exe
```

### Production Release:
```batch
# Create all distribution formats
build_complete.bat

# Upload to distribution platform
# Files: VestimSSHInstaller.exe, VestimSSH_Portable.zip
```

## Branch Management

This build system is designed to coexist with the main build:

```bash
# Safe to merge - no conflicts
git merge main

# Independent builds
git checkout ssh-client-branch
build_ssh/build_complete.bat

git checkout main  
build.bat  # Main Windows build
```

Both build systems can run independently without interfering with each other.
