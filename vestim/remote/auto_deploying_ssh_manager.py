"""
Auto-Deploying SSH Manager for Vestim
Automatically deploys Vestim to any remote Linux server
"""

import os
import sys
import time
import tempfile
import tarfile
import subprocess
from pathlib import Path
import paramiko
from scp import SCPClient
import json

class AutoDeployingSSHManager:
    """SSH Manager that automatically deploys Vestim to remote servers"""
    
    def __init__(self):
        self.ssh_client = None
        self.scp_client = None
        self.remote_vestim_path = None
        
    def connect(self, hostname, username, password=None, port=22, ssh_key_path=None):
        """Connect to remote server"""
        try:
            self.ssh_client = paramiko.SSHClient()
            self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            # Use key or password authentication
            if ssh_key_path and os.path.exists(ssh_key_path):
                self.ssh_client.connect(hostname, port=port, username=username, key_filename=ssh_key_path)
            else:
                self.ssh_client.connect(hostname, port=port, username=username, password=password)
            
            self.scp_client = SCPClient(self.ssh_client.get_transport())
            print(f"Connected to {hostname}")
            return True
            
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
    
    def detect_server_environment(self):
        """Detect server OS and Python capabilities"""
        try:
            # Detect OS
            stdin, stdout, stderr = self.ssh_client.exec_command("cat /etc/os-release")
            os_info = stdout.read().decode()
            
            # Parse OS info
            os_details = {}
            for line in os_info.split('\n'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    os_details[key] = value.strip('"')
            
            # Detect Python
            stdin, stdout, stderr = self.ssh_client.exec_command("python3 --version")
            python_version = stdout.read().decode().strip()
            
            # Check sudo access
            stdin, stdout, stderr = self.ssh_client.exec_command("sudo -n true")
            has_sudo = stderr.read().decode() == ""
            
            return {
                'os_name': os_details.get('NAME', 'Unknown'),
                'os_id': os_details.get('ID', 'unknown'),
                'python_version': python_version,
                'has_sudo': has_sudo,
                'is_ubuntu': 'ubuntu' in os_details.get('ID', '').lower(),
                'is_centos': 'centos' in os_details.get('ID', '').lower() or 'rhel' in os_details.get('ID', '').lower()
            }
            
        except Exception as e:
            print(f"Failed to detect server environment: {e}")
            return None
    
    def check_vestim_installation(self, remote_path="~/vestim"):
        """Check if Vestim is already installed on the server"""
        try:
            # Expand path
            stdin, stdout, stderr = self.ssh_client.exec_command(f"echo {remote_path}")
            expanded_path = stdout.read().decode().strip()
            
            # Check if Vestim directory exists and has the main module
            command = f"test -d {expanded_path} && test -f {expanded_path}/vestim/__init__.py"
            stdin, stdout, stderr = self.ssh_client.exec_command(command)
            return stdout.channel.recv_exit_status() == 0
            
        except Exception as e:
            print(f"Error checking Vestim installation: {e}")
            return False
    
    def install_system_dependencies(self, server_env, progress_callback=None):
        """Install required system dependencies"""
        try:
            if progress_callback:
                progress_callback(10, "Installing system dependencies...")
            
            if server_env['is_ubuntu']:
                # Ubuntu/Debian commands
                commands = [
                    "sudo apt update",
                    "sudo apt install -y python3 python3-pip python3-venv",
                    "sudo apt install -y libx11-dev libxext-dev libxrender-dev libxtst-dev",
                    "sudo apt install -y build-essential git"
                ]
            elif server_env['is_centos']:
                # CentOS/RHEL commands
                commands = [
                    "sudo yum update -y",
                    "sudo yum install -y python3 python3-pip",
                    "sudo yum install -y libX11-devel libXext-devel libXrender-devel libXtst-devel",
                    "sudo yum install -y gcc gcc-c++ make git"
                ]
            else:
                print("Unsupported OS for auto-installation")
                return False
            
            # Execute commands
            for i, command in enumerate(commands):
                if progress_callback:
                    progress_callback(10 + i * 10, f"Running: {command}")
                
                stdin, stdout, stderr = self.ssh_client.exec_command(command)
                exit_status = stdout.channel.recv_exit_status()
                
                if exit_status != 0:
                    error = stderr.read().decode()
                    print(f"Command failed: {command}\nError: {error}")
                    # Continue with other commands - some might be already installed
            
            return True
            
        except Exception as e:
            print(f"Failed to install system dependencies: {e}")
            return False
    
    def create_vestim_package(self):
        """Create a deployable Vestim package"""
        try:
            # Get current Vestim directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            vestim_root = os.path.dirname(current_dir)  # Go up from vestim/remote/
            
            # Create temporary tar file
            temp_dir = tempfile.mkdtemp()
            package_path = os.path.join(temp_dir, "vestim_package.tar.gz")
            
            with tarfile.open(package_path, "w:gz") as tar:
                # Add essential Vestim files
                essential_dirs = [
                    "vestim",
                    "requirements.txt",
                    "setup.py",
                    "pyproject.toml"
                ]
                
                for item in essential_dirs:
                    item_path = os.path.join(vestim_root, item)
                    if os.path.exists(item_path):
                        if os.path.isdir(item_path):
                            tar.add(item_path, arcname=item)
                        else:
                            tar.add(item_path, arcname=item)
            
            return package_path
            
        except Exception as e:
            print(f"Failed to create Vestim package: {e}")
            return None
    
    def upload_and_extract_vestim(self, package_path, remote_path="~/vestim", progress_callback=None):
        """Upload and extract Vestim package to server"""
        try:
            if progress_callback:
                progress_callback(50, "Uploading Vestim package...")
            
            # Expand remote path
            stdin, stdout, stderr = self.ssh_client.exec_command(f"echo {remote_path}")
            expanded_path = stdout.read().decode().strip()
            
            # Create remote directory
            self.ssh_client.exec_command(f"mkdir -p {expanded_path}")
            
            # Upload package
            remote_package = f"{expanded_path}/vestim_package.tar.gz"
            self.scp_client.put(package_path, remote_package)
            
            if progress_callback:
                progress_callback(60, "Extracting Vestim package...")
            
            # Extract package
            extract_command = f"cd {expanded_path} && tar -xzf vestim_package.tar.gz --strip-components=0"
            stdin, stdout, stderr = self.ssh_client.exec_command(extract_command)
            exit_status = stdout.channel.recv_exit_status()
            
            if exit_status != 0:
                error = stderr.read().decode()
                print(f"Extraction failed: {error}")
                return False
            
            # Clean up remote package
            self.ssh_client.exec_command(f"rm {remote_package}")
            
            self.remote_vestim_path = expanded_path
            return True
            
        except Exception as e:
            print(f"Failed to upload Vestim: {e}")
            return False
    
    def setup_python_environment(self, progress_callback=None):
        """Set up Python virtual environment and install dependencies"""
        try:
            if progress_callback:
                progress_callback(70, "Setting up Python environment...")
            
            # Create virtual environment
            venv_command = f"cd {self.remote_vestim_path} && python3 -m venv vestim_env"
            stdin, stdout, stderr = self.ssh_client.exec_command(venv_command)
            exit_status = stdout.channel.recv_exit_status()
            
            if exit_status != 0:
                error = stderr.read().decode()
                print(f"Failed to create virtual environment: {error}")
                return False
            
            if progress_callback:
                progress_callback(80, "Installing Python dependencies...")
            
            # Install dependencies
            pip_commands = [
                f"cd {self.remote_vestim_path} && source vestim_env/bin/activate && pip install --upgrade pip",
                f"cd {self.remote_vestim_path} && source vestim_env/bin/activate && pip install -r requirements.txt",
                f"cd {self.remote_vestim_path} && source vestim_env/bin/activate && pip install -e ."
            ]
            
            for command in pip_commands:
                stdin, stdout, stderr = self.ssh_client.exec_command(command)
                exit_status = stdout.channel.recv_exit_status()
                
                if exit_status != 0:
                    error = stderr.read().decode()
                    print(f"Pip command failed: {command}\nError: {error}")
                    # Continue - some packages might already be installed
            
            if progress_callback:
                progress_callback(90, "Testing installation...")
            
            # Test installation
            test_command = f"cd {self.remote_vestim_path} && source vestim_env/bin/activate && python -c 'import vestim; print(\"Vestim imported successfully\")'"
            stdin, stdout, stderr = self.ssh_client.exec_command(test_command)
            exit_status = stdout.channel.recv_exit_status()
            
            if exit_status != 0:
                error = stderr.read().decode()
                print(f"Vestim import test failed: {error}")
                return False
            
            return True
            
        except Exception as e:
            print(f"Failed to setup Python environment: {e}")
            return False
    
    def auto_deploy_vestim(self, remote_path="~/vestim", progress_callback=None):
        """Complete auto-deployment process"""
        try:
            if progress_callback:
                progress_callback(0, "Starting deployment...")
            
            # 1. Detect server environment
            if progress_callback:
                progress_callback(5, "Detecting server environment...")
            
            server_env = self.detect_server_environment()
            if not server_env:
                return False
            
            print(f"Server: {server_env['os_name']} with {server_env['python_version']}")
            
            # 2. Install system dependencies (if sudo available)
            if server_env['has_sudo']:
                if not self.install_system_dependencies(server_env, progress_callback):
                    print("Warning: System dependency installation failed")
            else:
                print("No sudo access - assuming dependencies are pre-installed")
            
            # 3. Create Vestim package
            if progress_callback:
                progress_callback(40, "Creating Vestim package...")
            
            package_path = self.create_vestim_package()
            if not package_path:
                return False
            
            try:
                # 4. Upload and extract
                if not self.upload_and_extract_vestim(package_path, remote_path, progress_callback):
                    return False
                
                # 5. Setup Python environment
                if not self.setup_python_environment(progress_callback):
                    return False
                
                if progress_callback:
                    progress_callback(100, "Deployment complete!")
                
                print("Vestim deployed successfully!")
                return True
                
            finally:
                # Clean up local package
                if os.path.exists(package_path):
                    os.remove(package_path)
            
        except Exception as e:
            print(f"Auto-deployment failed: {e}")
            return False
    
    def setup_x11_forwarding(self):
        """Set up X11 forwarding"""
        try:
            # Test X11 forwarding
            stdin, stdout, stderr = self.ssh_client.exec_command("echo $DISPLAY")
            display = stdout.read().decode().strip()
            
            if display:
                print(f"X11 forwarding active: DISPLAY={display}")
                return True
            else:
                print("X11 forwarding not available")
                return False
                
        except Exception as e:
            print(f"X11 setup failed: {e}")
            return False
    
    def launch_remote_gui(self, remote_path="~/vestim"):
        """Launch Vestim GUI on remote server"""
        try:
            # Use the deployed path if available
            if self.remote_vestim_path:
                remote_path = self.remote_vestim_path
            
            # Command to launch Vestim
            launch_command = f"cd {remote_path} && source vestim_env/bin/activate && python -m vestim.gui.src.main_window"
            
            print(f"Launching Vestim: {launch_command}")
            
            # Execute in background
            stdin, stdout, stderr = self.ssh_client.exec_command(launch_command)
            
            return True
            
        except Exception as e:
            print(f"Failed to launch GUI: {e}")
            return False
    
    def disconnect(self):
        """Clean up connections"""
        if self.scp_client:
            self.scp_client.close()
        if self.ssh_client:
            self.ssh_client.close()
