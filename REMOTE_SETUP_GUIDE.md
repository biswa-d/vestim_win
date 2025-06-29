# Vestim Remote GUI Setup Guide

This guide explains how to set up and use Vestim's remote GUI functionality, which allows you to run Vestim on a remote server while displaying the GUI on your local machine.

## How It Works

The Vestim Remote Client provides **zero-setup remote access**:

1. **User Input**: Enter any server address + credentials
2. **Auto-Detection**: Client detects server OS and capabilities  
3. **Auto-Deployment**: Client uploads and installs Vestim automatically
4. **Launch**: GUI appears locally, runs on remote server

### Automatic Deployment Process

```
[User enters server details] 
         ↓
[SSH connection established]
         ↓  
[Detect server: Ubuntu/CentOS/etc.]
         ↓
[Install system dependencies automatically]
         ↓
[Upload Vestim package to server]
         ↓
[Create Python environment remotely] 
         ↓
[Install Python dependencies]
         ↓
[Launch GUI with X11 forwarding]
         ↓
[Vestim appears on local screen]
```

**No server-side setup required!** Works with any Linux server you have SSH access to.

## Prerequisites

### Local Machine Requirements

1. **Python 3.8+** with Vestim installed
2. **X11 Server** (for GUI display):
   - **Windows**: VcXsrv or Xming
   - **macOS**: XQuartz  
   - **Linux**: X11 (usually pre-installed)
3. **SSH Client** (usually pre-installed on modern systems)

### Remote Server Requirements

1. **Any Linux server** with SSH access
2. **Basic system capabilities**:
   - Python 3.8+ (or ability to install it)
   - Internet connection (for downloading dependencies)
   - Sudo access (for installing system packages) OR pre-installed dependencies
3. **No Vestim installation required** - the client handles this automatically

> **Key Feature**: The Vestim client automatically deploys itself to any compatible remote server. Users only need SSH access - no pre-installation required!

## Installation

### 1. Install Remote Dependencies

Run the setup script to install required packages:

```bash
python setup_remote.py
```

This installs:
- `paramiko` for SSH connections
- `keyring` for secure password storage
- Platform-specific packages (pywin32, winshell on Windows)

### 2. X11 Server Setup

#### Windows
Install VcXsrv (recommended):
1. Download from [https://sourceforge.net/projects/vcxsrv/](https://sourceforge.net/projects/vcxsrv/)
2. Install with default settings
3. Vestim will auto-detect and start it when needed

Alternative: Xming from [https://sourceforge.net/projects/xming/](https://sourceforge.net/projects/xming/)

#### macOS
Install XQuartz:
1. Download from [https://www.xquartz.org/](https://www.xquartz.org/)
2. Install and log out/in
3. Or use Homebrew: `brew install --cask xquartz`

#### Linux
X11 is usually pre-installed. If not:
```bash
# Ubuntu/Debian
sudo apt install xorg

# CentOS/RHEL  
sudo yum install xorg-x11-server-Xorg
```

## Configuration

### 1. Launch Configuration GUI

Run the configuration dialog:
```bash
python vestim/remote/launcher.py --gui
```

Or use the desktop shortcut "Vestim Remote" (created during installation).

### 2. Configure Connection Settings

#### SSH Connection Tab
- **Hostname/IP**: Your server address (e.g., `server.university.edu`)
- **Port**: SSH port (usually 22)
- **Username**: Your username on the remote server
- **Password**: Your SSH password (stored securely with keyring)
- **Connection Timeout**: How long to wait for connection (default: 30s)
- **Enable Compression**: Recommended for better performance

#### Display Setup Tab
- **X11 Server Status**: Shows if local X11 server is running
- **Remote DISPLAY**: Usually `:0` (auto-detected)
- **Auto-detect**: Automatically finds the correct DISPLAY variable

#### Project Sync Tab
- **Enable Synchronization**: Sync projects between local and remote
- **Remote Vestim Path**: Where Vestim is installed on server (e.g., `/opt/vestim`)
- **Local Project Path**: Your local project directory
- **Sync Options**: When to sync (before launch, after completion)

### 3. Test Connection

Use the "Connection Test" tab to verify:
- SSH connection works
- Credentials are correct
- Remote server is accessible
- DISPLAY forwarding functions

## Usage

### Method 1: Desktop Shortcut
1. Double-click "Vestim Remote" desktop icon
2. Choose to launch with existing config or modify settings
3. GUI appears on your local screen, running on remote server

### Method 2: Command Line
```bash
# Interactive GUI configuration
python vestim/remote/launcher.py --gui

# Direct launch (if configured)
python vestim/remote/launcher.py --host server.edu --username myuser
```

### Method 3: From Vestim Main GUI
Look for "Remote Launch" option in the main Vestim interface.

## Project Synchronization

When enabled, the system automatically:

1. **Before Launch**: Uploads your local project to the remote server
2. **During Execution**: All processing happens on the remote server
3. **After Completion**: Downloads results back to your local project folder

Sync paths:
- Local: `<your_project>/` → Remote: `<remote_vestim_path>/projects/<project_name>/`
- Results: Remote: `<remote_vestim_path>/projects/<project_name>/output/` → Local: `<your_project>/output/`

## Troubleshooting

### Connection Issues
```bash
# Test SSH manually
ssh -X username@hostname

# Check if X11 forwarding works  
ssh -X username@hostname 'echo $DISPLAY'
```

### Display Issues
- **Linux**: Ensure `DISPLAY` is set: `echo $DISPLAY`
- **Windows**: Check if VcXsrv/Xming is running in system tray
- **macOS**: Ensure XQuartz is running and X11 forwarding enabled

### Permission Issues
```bash
# On remote server, ensure Vestim is executable
chmod +x /opt/vestim/Vestim

# Check X11 permissions
xauth list
```

### Performance Optimization
- Enable SSH compression (default: enabled)
- Use wired connection for better stability
- Close unnecessary applications on both machines
- Consider using SSH key authentication instead of passwords

## Security Considerations

- **Password Storage**: Uses system keyring (Windows Credential Store, macOS Keychain, Linux Secret Service)
- **SSH Keys**: Recommended for production use instead of passwords
- **Firewall**: Ensure SSH port (22) is open on remote server
- **VPN**: Consider using VPN for connections over public networks

## Advanced Configuration

### SSH Key Authentication
1. Generate SSH key pair:
   ```bash
   ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
   ```

2. Copy public key to server:
   ```bash
   ssh-copy-id username@hostname
   ```

3. Leave password field empty in Vestim Remote config

### Custom Remote Installation
If Vestim is installed in a non-standard location:
1. Update "Remote Vestim Path" in configuration
2. Ensure the path contains the executable `Vestim` file
3. Verify permissions: `ls -la /path/to/vestim/Vestim`

### Batch Operations
For multiple projects or automated workflows:
```python
from vestim.remote.ssh_manager import RemoteSSHManager, SSHConfig

# Programmatic launch
config = SSHConfig(host="server.edu", username="user")
manager = RemoteSSHManager()
process = manager.launch_remote_gui(config)
```

## Server Environment Setup

### Automated Setup (Recommended)

Use the provided setup script to automatically configure the server:

```bash
# Copy setup_server.sh to your remote server
scp setup_server.sh username@server:/tmp/
ssh username@server

# Run the setup script
chmod +x /tmp/setup_server.sh
/tmp/setup_server.sh
```

The script will:
- Install system dependencies (Python, X11 libraries, SSH server)
- Configure SSH for X11 forwarding
- Set up Python virtual environment
- Install Vestim and all dependencies
- Create launcher scripts
- Test the installation

### Manual Setup

### Option 1: Install from Source (Recommended)

On the remote server, set up the complete Vestim environment:

```bash
# 1. Clone Vestim repository
git clone <vestim-repository-url>
cd vestim_micros

# 2. Create Python virtual environment
python3 -m venv vestim_env
source vestim_env/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
pip install -e .

# 4. Test installation
python -m vestim.gui.src.main_window --help
```

### Option 2: Use Existing Installation

If Vestim is already installed system-wide:

```bash
# Verify installation
which python
python -c "import vestim; print('Vestim available')"

# Test GUI dependencies
python -c "from PyQt5.QtWidgets import QApplication; print('PyQt5 OK')"

# Check X11 forwarding
echo $DISPLAY
```

### Environment Requirements

The remote server must have these Python packages installed:

```txt
PyQt5>=5.15.0
torch>=1.9.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
joblib>=1.0.0
paramiko>=2.7.0
# ... (all packages from requirements.txt)
```

### Deployment Options

#### Single-User Setup
```bash
# Install in user home directory
cd /home/username
git clone <vestim-repo>
cd vestim_micros
python3 -m venv vestim_env
# ... continue with installation
```

#### Multi-User Setup (System-wide)
```bash
# Install in shared location (requires admin)
sudo mkdir -p /opt/vestim
sudo chown $USER:$USER /opt/vestim
cd /opt/vestim
git clone <vestim-repo>
# ... continue with installation

# Make accessible to all users
sudo chmod -R 755 /opt/vestim
```

#### Docker Deployment
```dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libx11-dev libxext-dev libxrender-dev libxtst-dev \
    git openssh-server

# Create vestim user
RUN useradd -m -s /bin/bash vestim

# Install Vestim
WORKDIR /opt/vestim
COPY . .
RUN pip install -r requirements.txt
RUN pip install -e .

# Configure SSH
RUN mkdir /var/run/sshd
RUN echo 'X11Forwarding yes' >> /etc/ssh/sshd_config
RUN echo 'X11UseLocalhost no' >> /etc/ssh/sshd_config

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]
```

## Getting Help

1. **Configuration Issues**: Use the "Connection Test" tab
2. **SSH Problems**: Test with standard SSH client first
3. **Display Issues**: Check X11 server status in configuration GUI
4. **Performance**: Enable compression, check network connection
5. **Logs**: Check `~/.vestim/remote/` for configuration and log files

For additional support, consult the main Vestim documentation or contact the development team.
