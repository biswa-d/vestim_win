"""
Remote SSH Configuration Manager for Vestim
Handles SSH connection setup, DISPLAY forwarding, and credential management
"""

import os
import json
import subprocess
import platform
import socket
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import paramiko
from dataclasses import dataclass, asdict
import keyring
from .auto_x11_installer import AutoX11ServerManager, get_x11_installation_guide


@dataclass
class SSHConfig:
    """SSH connection configuration"""
    host: str
    port: int = 22
    username: str = ""
    display: str = ":0"
    remote_vestim_path: str = "/opt/vestim"
    project_sync_enabled: bool = True
    compression: bool = True
    timeout: int = 30


class RemoteSSHManager:
    """Manages SSH connections and remote GUI launching"""
    
    def __init__(self):
        self.config_dir = Path.home() / ".vestim" / "remote"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config_file = self.config_dir / "ssh_config.json"
        self.known_hosts_file = self.config_dir / "known_hosts"
        self.x11_manager = AutoX11ServerManager()
        
    def save_config(self, config: SSHConfig) -> None:
        """Save SSH configuration to file"""
        with open(self.config_file, 'w') as f:
            json.dump(asdict(config), f, indent=2)
    
    def load_config(self) -> Optional[SSHConfig]:
        """Load SSH configuration from file"""
        if not self.config_file.exists():
            return None
        try:
            with open(self.config_file, 'r') as f:
                data = json.load(f)
            return SSHConfig(**data)
        except Exception:
            return None
    
    def save_password(self, host: str, username: str, password: str) -> None:
        """Securely save SSH password using keyring"""
        service_name = f"vestim_ssh_{host}"
        keyring.set_password(service_name, username, password)
    
    def get_password(self, host: str, username: str) -> Optional[str]:
        """Retrieve SSH password from keyring"""
        service_name = f"vestim_ssh_{host}"
        return keyring.get_password(service_name, username)
    
    def detect_local_display(self) -> str:
        """Auto-detect local DISPLAY variable or use default"""
        if platform.system() == "Windows":
            # For Windows with VcXsrv or similar X11 server
            return "localhost:0.0"
        elif platform.system() == "Darwin":
            # macOS with XQuartz
            return ":0"
        else:
            # Linux
            return os.environ.get("DISPLAY", ":0")
    
    def test_ssh_connection(self, config: SSHConfig, password: str = None) -> Tuple[bool, str]:
        """Test SSH connection and return status"""
        try:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            if not password:
                password = self.get_password(config.host, config.username)
            
            client.connect(
                hostname=config.host,
                port=config.port,
                username=config.username,
                password=password,
                timeout=config.timeout,
                compress=config.compression
            )
            
            # Test if we can run basic commands
            stdin, stdout, stderr = client.exec_command("echo 'Connection test successful'")
            result = stdout.read().decode().strip()
            
            client.close()
            return True, "Connection successful"
            
        except Exception as e:
            return False, f"Connection failed: {str(e)}"
    
    def detect_remote_display(self, config: SSHConfig, password: str = None) -> str:
        """Auto-detect remote DISPLAY environment"""
        try:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            if not password:
                password = self.get_password(config.host, config.username)
            
            client.connect(
                hostname=config.host,
                port=config.port,
                username=config.username,
                password=password,
                timeout=config.timeout
            )
            
            # Try to detect DISPLAY
            commands = [
                "echo $DISPLAY",
                "who | grep '(' | awk '{print $2}' | head -1",  # Get display from logged users
                "ps aux | grep -i 'xorg\\|x11' | grep -v grep | head -1 | awk '{print \":0\"}'",
                "echo ':0'"  # fallback
            ]
            
            for cmd in commands:
                stdin, stdout, stderr = client.exec_command(cmd)
                result = stdout.read().decode().strip()
                if result and result != "":
                    client.close()
                    return result if result.startswith(":") else f":{result}"
            
            client.close()
            return ":0"  # default fallback
            
        except Exception:
            return ":0"
    
    def setup_x11_forwarding(self, config: SSHConfig) -> List[str]:
        """Generate SSH command with X11 forwarding"""
        local_display = self.detect_local_display()
        
        ssh_cmd = [
            "ssh",
            "-X",  # Enable X11 forwarding
            "-C",  # Enable compression
            "-o", "ForwardX11=yes",
            "-o", "ForwardX11Trusted=yes",
            "-o", "ExitOnForwardFailure=yes",
            "-p", str(config.port),
            f"{config.username}@{config.host}"
        ]
        
        return ssh_cmd
    
    def sync_project_to_remote(self, config: SSHConfig, local_project_path: str, 
                              password: str = None) -> Tuple[bool, str]:
        """Sync local project to remote server using SCP"""
        if not config.project_sync_enabled:
            return True, "Project sync disabled"
        
        try:
            # Use scp to copy project files
            remote_path = f"{config.username}@{config.host}:{config.remote_vestim_path}/projects/"
            
            scp_cmd = [
                "scp", "-r", "-P", str(config.port),
                local_project_path,
                remote_path
            ]
            
            result = subprocess.run(scp_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                return True, "Project synced successfully"
            else:
                return False, f"Sync failed: {result.stderr}"
                
        except Exception as e:
            return False, f"Sync error: {str(e)}"
    
    def launch_remote_gui(self, config: SSHConfig, password: str = None) -> subprocess.Popen:
        """Launch Vestim GUI on remote server with X11 forwarding"""
        try:
            # Ensure X11 server is available locally first
            print("Ensuring X11 server is available...")
            x11_available, x11_message = self.x11_manager.ensure_x11_available()
            
            if not x11_available:
                error_msg = f"X11 server setup failed: {x11_message}\n\n"
                error_msg += "Manual installation guide:\n"
                error_msg += get_x11_installation_guide()
                raise Exception(error_msg)
            
            print(f"✓ X11 server ready: {x11_message}")
            
            # Update remote DISPLAY
            remote_display = self.detect_remote_display(config, password)
            
            # Build SSH command with GUI launch
            ssh_cmd = self.setup_x11_forwarding(config)
            
            # Add the remote command to launch Vestim
            remote_command = f"cd {config.remote_vestim_path} && " \
                           f"DISPLAY={remote_display} ./Vestim"
            
            ssh_cmd.append(remote_command)
            
            # Launch SSH process
            process = subprocess.Popen(
                ssh_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            return process
            
        except Exception as e:
            raise Exception(f"Failed to launch remote GUI: {str(e)}")
    
    def sync_results_back(self, config: SSHConfig, local_project_path: str,
                         password: str = None) -> Tuple[bool, str]:
        """Sync results back from remote server to local project"""
        if not config.project_sync_enabled:
            return True, "Project sync disabled"
        
        try:
            # Download results from remote
            remote_results_path = f"{config.username}@{config.host}:{config.remote_vestim_path}/projects/*/output/"
            local_results_path = os.path.join(local_project_path, "output")
            
            scp_cmd = [
                "scp", "-r", "-P", str(config.port),
                remote_results_path,
                local_results_path
            ]
            
            result = subprocess.run(scp_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                return True, "Results synced back successfully"
            else:
                return False, f"Results sync failed: {result.stderr}"
                
        except Exception as e:
            return False, f"Results sync error: {str(e)}"


class X11ServerManager:
    """Manages local X11 server for Windows/macOS"""
    
    def __init__(self):
        self.platform = platform.system()
    
    def is_x11_available(self) -> bool:
        """Check if X11 server is running locally"""
        if self.platform == "Windows":
            return self._check_vcxsrv() or self._check_xming()
        elif self.platform == "Darwin":
            return self._check_xquartz()
        else:
            return True  # Linux typically has X11
    
    def _check_vcxsrv(self) -> bool:
        """Check if VcXsrv is running on Windows"""
        try:
            result = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq vcxsrv.exe"],
                capture_output=True, text=True
            )
            return "vcxsrv.exe" in result.stdout
        except:
            return False
    
    def _check_xming(self) -> bool:
        """Check if Xming is running on Windows"""
        try:
            result = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq Xming.exe"],
                capture_output=True, text=True
            )
            return "Xming.exe" in result.stdout
        except:
            return False
    
    def _check_xquartz(self) -> bool:
        """Check if XQuartz is running on macOS"""
        try:
            result = subprocess.run(
                ["pgrep", "-f", "XQuartz"],
                capture_output=True, text=True
            )
            return result.returncode == 0
        except:
            return False
    
    def start_x11_server(self) -> Tuple[bool, str]:
        """Attempt to start X11 server if not running"""
        if self.is_x11_available():
            return True, "X11 server already running"
        
        if self.platform == "Windows":
            return self._start_windows_x11()
        elif self.platform == "Darwin":
            return self._start_xquartz()
        else:
            return True, "X11 should be available on Linux"
    
    def _start_windows_x11(self) -> Tuple[bool, str]:
        """Start X11 server on Windows"""
        # Try VcXsrv first
        vcxsrv_paths = [
            r"C:\Program Files\VcXsrv\vcxsrv.exe",
            r"C:\Program Files (x86)\VcXsrv\vcxsrv.exe"
        ]
        
        for path in vcxsrv_paths:
            if os.path.exists(path):
                try:
                    subprocess.Popen([path, ":0", "-ac", "-terminate"])
                    time.sleep(2)  # Wait for startup
                    return True, "VcXsrv started successfully"
                except:
                    continue
        
        # Try Xming as fallback
        xming_paths = [
            r"C:\Program Files\Xming\Xming.exe",
            r"C:\Program Files (x86)\Xming\Xming.exe"
        ]
        
        for path in xming_paths:
            if os.path.exists(path):
                try:
                    subprocess.Popen([path, ":0", "-ac"])
                    time.sleep(2)
                    return True, "Xming started successfully"
                except:
                    continue
        
        return False, "No X11 server found. Please install VcXsrv or Xming."
    
    def _start_xquartz(self) -> Tuple[bool, str]:
        """Start XQuartz on macOS"""
        try:
            subprocess.Popen(["open", "-a", "XQuartz"])
            time.sleep(3)  # Wait for XQuartz to start
            return True, "XQuartz started successfully"
        except:
            return False, "Failed to start XQuartz. Please install XQuartz."
