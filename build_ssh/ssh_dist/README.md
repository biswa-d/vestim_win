# Vestim SSH/Remote Client Distribution

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
