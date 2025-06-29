# Vestim Server Client - What It Actually Does

## Step-by-Step Execution Flow

### When User Clicks "Vestim Server" Icon:

```
[User clicks desktop shortcut]
         ↓
[vestim_server_client.py starts]
         ↓
[Loads ~/.vestim/server/server_config.json]
         ↓
[Checks if X11 server is running]
         ↓
[If no X11] → [Shows dialog: "Install X11?"] → [Auto-installs VcXsrv]
         ↓
[Connects via SSH to remote server]
         ↓
[Sets up X11 forwarding: ssh -X user@server]
         ↓
[Executes on remote: cd /opt/vestim && python -m vestim.gui.src.main_window]
         ↓
[Vestim GUI appears on local screen]
```

## Real Example Scenario

**User**: Dr. Smith at University
**Local Machine**: Windows laptop 
**Remote Server**: Powerful Linux cluster `cluster.university.edu`

### Installation (Once):
```
1. Dr. Smith installs Vestim on her laptop
2. Setup wizard asks: "Enter server details"
3. She enters:
   - Server: cluster.university.edu
   - Username: dsmith  
   - Password: ********
   - Vestim path: /opt/vestim
4. Config saved to C:\Users\dsmith\.vestim\server\server_config.json
```

### Daily Use:
```
1. Dr. Smith double-clicks "Vestim Server" on desktop
2. Script runs:
   - Loads her server config
   - Checks X11 → VcXsrv not running → Starts it automatically
   - SSH connects: ssh -X dsmith@cluster.university.edu
   - X11 forwarding established
   - Runs: cd /opt/vestim && python -m vestim.gui.src.main_window
3. Vestim GUI appears on her Windows laptop
4. All processing happens on the university cluster
5. She works normally - GUI is local, compute is remote
```

## Technical Details

The script performs these operations:

### 1. Configuration Management
```python
# Loads saved config
with open("~/.vestim/server/server_config.json") as f:
    config = json.load(f)
# Contains: hostname, username, password, remote_path
```

### 2. X11 Server Management  
```python
# Windows: Check if VcXsrv is running
if not x11_installer.is_x11_running():
    # Auto-download and install VcXsrv
    x11_installer.install_x11_server()
    x11_installer.start_x11_server()
```

### 3. SSH Connection with X11 Forwarding
```python
# Equivalent to: ssh -X username@hostname
ssh_manager.connect(
    hostname=config['hostname'],
    username=config['username'], 
    password=config['password']
)
ssh_manager.setup_x11_forwarding()
```

### 4. Remote GUI Launch
```python
# Execute on remote server
remote_command = f"cd {config['remote_vestim_path']} && python -m vestim.gui.src.main_window"
ssh_manager.execute_gui_command(remote_command)
```

## Why This Approach?

### **Problem**: 
- Users have powerful remote servers with Vestim installed
- Want to use Vestim GUI but don't want to install locally
- Need seamless "one-click" access

### **Solution**:
- **One-time setup**: Enter server details during installation
- **Daily use**: Single click → automatic connection → remote GUI appears locally
- **No manual SSH commands** or technical knowledge required

### **Benefits**:
- **For End Users**: Dead simple - just click and go
- **For IT Admins**: Centralized Vestim installation on servers
- **For Performance**: Heavy computation on powerful remote hardware
- **For Licensing**: Single Vestim license can serve multiple users

## File Structure After Installation

```
C:\Program Files\Vestim\
├── Vestim.exe                    # Local Vestim (if needed)
├── vestim_server_client.py       # Main server client script  
├── vestim_server_client.bat      # Windows launcher
├── vestim_server_setup.py        # Setup wizard
└── vestim\remote\                # SSH and X11 management modules

C:\Users\{user}\.vestim\server\
└── server_config.json            # Saved server configuration

Desktop:
├── Vestim.lnk                    # Local Vestim shortcut
└── Vestim Server.lnk             # Remote server shortcut ← This is the magic
```

The "Vestim Server" shortcut runs the batch file which loads the saved config and establishes the remote connection automatically.
