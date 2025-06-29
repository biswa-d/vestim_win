#!/usr/bin/env python3
"""
Universal Vestim Server Client
Connects to any Linux server and automatically deploys Vestim
"""

import os
import sys
import json
import tkinter as tk
from tkinter import messagebox, simpledialog
import threading
import time

# Add vestim package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from vestim.remote.auto_deploying_ssh_manager import AutoDeployingSSHManager
    from vestim.remote.auto_x11_installer import AutoX11ServerManager
except ImportError as e:
    print(f"Failed to import required modules: {e}")
    print("Make sure you have installed the required dependencies:")
    print("pip install paramiko scp PyQt5")
    
    # Try alternative import for standalone executable
    try:
        import sys
        import os
        # Add current directory to path for standalone execution
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if hasattr(sys, '_MEIPASS'):
            # Running as PyInstaller bundle
            bundle_dir = sys._MEIPASS
            sys.path.insert(0, bundle_dir)
        sys.path.insert(0, current_dir)
        
        from vestim.remote.auto_deploying_ssh_manager import AutoDeployingSSHManager
        from vestim.remote.auto_x11_installer import AutoX11ServerManager
    except ImportError:
        print("Critical: Cannot import SSH modules")
        input("Press Enter to exit...")
        sys.exit(1)

class UniversalVestimClient:
    """Universal client that can connect to any server"""
    
    def __init__(self):
        self.config_file = os.path.join(os.path.expanduser("~"), ".vestim_universal_config.json")
        self.ssh_manager = AutoDeployingSSHManager()
        self.x11_installer = AutoX11ServerManager()
        
    def get_server_details(self):
        """Get server connection details from user"""
        root = tk.Tk()
        root.withdraw()  # Hide main window
        
        messagebox.showinfo(
            "Universal Vestim Client", 
            "Connect to any Linux server with SSH access!\n\n"
            "Vestim will be automatically deployed and launched.\n"
            "No server-side setup required."
        )
        
        # Get server details
        hostname = simpledialog.askstring("Server Connection", "Enter server hostname or IP address:")
        if not hostname:
            return None
            
        username = simpledialog.askstring("Server Connection", f"Enter username for {hostname}:")
        if not username:
            return None
            
        port = simpledialog.askinteger("Server Connection", "Enter SSH port:", initialvalue=22)
        if not port:
            port = 22
            
        # Authentication method
        use_password = messagebox.askyesno(
            "Authentication", 
            "Use password authentication?\n\n"
            "Click 'No' for SSH key authentication."
        )
        
        password = None
        ssh_key_path = None
        
        if use_password:
            password = simpledialog.askstring(
                "Authentication", 
                f"Enter password for {username}@{hostname}:",
                show='*'
            )
            if not password:
                return None
        else:
            ssh_key_path = simpledialog.askstring(
                "SSH Key", 
                "Enter path to SSH private key:",
                initialvalue=os.path.expanduser("~/.ssh/id_rsa")
            )
            if not ssh_key_path:
                ssh_key_path = os.path.expanduser("~/.ssh/id_rsa")
        
        # Remote installation path
        remote_path = simpledialog.askstring(
            "Installation Path", 
            "Enter remote installation path for Vestim:",
            initialvalue="~/vestim"
        )
        if not remote_path:
            remote_path = "~/vestim"
        
        root.destroy()
        
        return {
            'hostname': hostname,
            'username': username,
            'port': port,
            'password': password,
            'ssh_key_path': ssh_key_path,
            'remote_path': remote_path,
            'use_password': use_password
        }
    
    def save_config(self, config):
        """Save configuration for future use"""
        try:
            # Don't save password for security
            save_config = config.copy()
            if 'password' in save_config:
                del save_config['password']
                
            with open(self.config_file, 'w') as f:
                json.dump(save_config, f, indent=2)
                
        except Exception as e:
            print(f"Could not save config: {e}")
    
    def load_config(self):
        """Load saved configuration"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Could not load config: {e}")
        return None
    
    def ensure_x11_available(self):
        """Ensure X11 server is available"""
        print("Checking X11 server...")
        
        # Use the correct method that returns (bool, str)
        success, message = self.x11_installer.ensure_x11_available()
        
        if not success:
            root = tk.Tk()
            root.withdraw()
            
            install = messagebox.askyesno(
                "X11 Server Required",
                f"X11 server issue: {message}\n\n"
                "Try automatic installation?"
            )
            
            root.destroy()
            
            if not install:
                return False
                
            print("Installing X11 server...")
            success, install_message = self.x11_installer.auto_install_x11()
            if not success:
                messagebox.showerror("Installation Failed", f"Failed to install X11 server: {install_message}")
                return False
            
            # Try again after installation
            success, message = self.x11_installer.ensure_x11_available()
            if not success:
                messagebox.showerror("X11 Error", f"X11 still not available: {message}")
                return False
        
        print(f"X11 server ready: {message}")
        return True
    
    def show_progress_window(self, title="Connecting..."):
        """Show a progress window"""
        progress_window = tk.Toplevel()
        progress_window.title(title)
        progress_window.geometry("400x150")
        progress_window.resizable(False, False)
        
        # Center the window
        progress_window.transient()
        progress_window.grab_set()
        
        status_label = tk.Label(progress_window, text="Initializing...", wraplength=380)
        status_label.pack(pady=20)
        
        progress_bar = tk.Canvas(progress_window, width=360, height=20, bg='white', relief='sunken', bd=1)
        progress_bar.pack(pady=10)
        
        def update_progress(percentage, message):
            status_label.config(text=message)
            progress_bar.delete("all")
            if percentage > 0:
                bar_width = int(360 * percentage / 100)
                progress_bar.create_rectangle(0, 0, bar_width, 20, fill='blue', outline='blue')
            progress_window.update()
        
        return progress_window, update_progress
    
    def connect_and_deploy(self, config):
        """Connect to server and deploy Vestim"""
        try:
            # Show progress
            progress_window, update_progress = self.show_progress_window("Deploying Vestim")
            
            try:
                # Connect to server
                update_progress(5, f"Connecting to {config['hostname']}...")
                
                if not self.ssh_manager.connect(
                    hostname=config['hostname'],
                    username=config['username'],
                    port=config['port'],
                    password=config['password'],
                    ssh_key_path=config['ssh_key_path']
                ):
                    return False, "Failed to connect to server"
                
                update_progress(10, "Connected! Checking existing installation...")
                
                # Check if Vestim already exists
                if self.ssh_manager.check_vestim_installation(config['remote_path']):
                    update_progress(100, "Vestim already installed!")
                    time.sleep(1)
                else:
                    update_progress(15, "Vestim not found. Starting deployment...")
                    
                    # Auto-deploy Vestim
                    if not self.ssh_manager.auto_deploy_vestim(
                        remote_path=config['remote_path'],
                        progress_callback=update_progress
                    ):
                        return False, "Failed to deploy Vestim"
                
                # Set up X11 forwarding
                update_progress(95, "Setting up display forwarding...")
                if not self.ssh_manager.setup_x11_forwarding():
                    return False, "Failed to set up X11 forwarding"
                
                update_progress(100, "Ready to launch!")
                time.sleep(1)
                
                return True, "Deployment successful"
                
            finally:
                progress_window.destroy()
                
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def launch_gui(self, config):
        """Launch the remote GUI"""
        try:
            print("Launching Vestim GUI...")
            
            if self.ssh_manager.launch_remote_gui(config['remote_path']):
                messagebox.showinfo(
                    "Success!", 
                    "Vestim GUI launched successfully!\n\n"
                    "The GUI should appear shortly.\n"
                    "Close this dialog when done."
                )
                return True
            else:
                messagebox.showerror("Launch Error", "Failed to launch Vestim GUI")
                return False
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch GUI: {str(e)}")
            return False
    
    def run(self):
        """Main execution flow"""
        print("=== Universal Vestim Client ===")
        print("Connect to any Linux server and launch Vestim automatically!")
        
        # Ensure X11 is available
        if not self.ensure_x11_available():
            print("X11 setup failed")
            return False
        
        # Try to use saved config first
        config = self.load_config()
        if config:
            root = tk.Tk()
            root.withdraw()
            
            use_saved = messagebox.askyesno(
                "Saved Configuration",
                f"Use saved server: {config.get('username', 'unknown')}@{config.get('hostname', 'unknown')}?"
            )
            
            root.destroy()
            
            if use_saved:
                # Get password for saved config
                if config.get('use_password', True):
                    root = tk.Tk()
                    root.withdraw()
                    password = simpledialog.askstring(
                        "Password", 
                        f"Enter password for {config['username']}@{config['hostname']}:",
                        show='*'
                    )
                    root.destroy()
                    
                    if not password:
                        print("Password required")
                        return False
                    
                    config['password'] = password
            else:
                config = None
        
        # Get new server details if needed
        if not config:
            config = self.get_server_details()
            if not config:
                print("Server configuration cancelled")
                return False
            
            # Save for future use
            self.save_config(config)
        
        try:
            # Connect and deploy
            success, message = self.connect_and_deploy(config)
            
            if not success:
                messagebox.showerror("Connection Failed", message)
                return False
            
            # Launch GUI
            return self.launch_gui(config)
            
        finally:
            # Clean up
            self.ssh_manager.disconnect()

def main():
    """Main entry point"""
    try:
        client = UniversalVestimClient()
        success = client.run()
        
        if not success:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")
        sys.exit(1)

if __name__ == "__main__":
    main()
