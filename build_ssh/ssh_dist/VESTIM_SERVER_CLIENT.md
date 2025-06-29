# Vestim Server Client

## Overview

Vestim Server Client provides one-click access to remote Vestim installations. After a simple one-time setup, users can launch the Vestim GUI running on a powerful remote server with a single click.

## How It Works

1. **Setup**: Configure your remote server details once during installation
2. **Launch**: Click the "Vestim Server" icon to instantly connect and launch the remote GUI
3. **Use**: The Vestim interface appears on your local machine, but all processing happens on the remote server

## Installation Workflow

### For End Users

1. **Run Installer**: During Vestim installation, check "Run server setup wizard after installation"
2. **Configure Server**: Enter your remote server details in the setup wizard
3. **Launch**: Use the "Vestim Server" desktop shortcut for one-click access

### Manual Setup

If you need to reconfigure your server settings:

```bash
# Run setup wizard
python vestim_server_setup.py

# Or use the client's built-in setup
python vestim_server_client.py --setup
```

## Requirements

### Local Machine
- Windows 10/11 or macOS 10.12+
- Python 3.7+
- PyQt5
- X11 server (automatically installed if needed)

### Remote Server
- Linux server with Vestim installed
- SSH access enabled
- X11 forwarding support

## Configuration

The setup wizard collects:
- **Server address** (hostname or IP)
- **SSH credentials** (username/password or key)
- **Vestim installation path** on the server
- **Connection preferences**

Configuration is saved securely in `~/.vestim/server/server_config.json`

## Usage

### First Time
1. Run the installer or setup wizard
2. Enter your server details
3. Test the connection
4. Launch via desktop shortcut

### Daily Use
Simply double-click the "Vestim Server" icon for instant access to your remote Vestim installation.

### Command Line Options

```bash
# Launch remote GUI (default)
python vestim_server_client.py

# Reconfigure server settings
python vestim_server_client.py --setup

# Show help
python vestim_server_client.py --help
```

## Troubleshooting

### Connection Issues
- Verify server address and SSH credentials
- Check firewall settings on both machines
- Ensure X11 forwarding is enabled on the server

### X11 Display Issues
- The client automatically installs and configures X11 servers
- On Windows: VcXsrv is used
- On macOS: XQuartz is used

### Reconfiguration
Run the setup wizard again if server details change:
```bash
python vestim_server_client.py --setup
```

## Security

- Passwords are stored securely using OS credential storage
- SSH connections use standard encryption
- X11 forwarding is secured through SSH tunneling

## Integration

The server client integrates seamlessly with:
- Vestim installer (automatic setup option)
- Desktop shortcuts and start menu
- Existing Vestim workflows and projects

This provides the smoothest possible experience for users who need to access Vestim on remote servers.
