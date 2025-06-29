"""
Vestim Server Client - Pre-configured Remote Client
This is the main executable for connecting to a remote Vestim server
"""

import sys
import os
import json
from pathlib import Path
import subprocess
import platform

# Add vestim package to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from PyQt5.QtWidgets import QApplication, QMessageBox, QProgressDialog
    from PyQt5.QtCore import QThread, pyqtSignal, Qt
    from vestim.remote.ssh_manager import RemoteSSHManager, SSHConfig
    from vestim.remote.auto_x11_installer import AutoX11ServerManager
except ImportError as e:
    print(f"Failed to import required modules: {e}")
    sys.exit(1)


class ServerConnectionThread(QThread):
    """Thread for establishing server connection and launching GUI"""
    status_update = pyqtSignal(str)
    finished = pyqtSignal(bool, str)
    
    def __init__(self, config_path):
        super().__init__()
        self.config_path = config_path
    
    def run(self):
        try:
            # Load server configuration
            self.status_update.emit("Loading server configuration...")
            with open(self.config_path, 'r') as f:
                config_data = json.load(f)
            
            config = SSHConfig(**config_data)
            
            # Initialize managers
            ssh_manager = RemoteSSHManager()
            x11_manager = AutoX11ServerManager()
            
            # Ensure X11 is available
            self.status_update.emit("Checking X11 server...")
            x11_available, x11_message = x11_manager.ensure_x11_available()
            if not x11_available:
                self.finished.emit(False, f"X11 setup failed: {x11_message}")
                return
            
            # Test connection
            self.status_update.emit("Connecting to server...")
            password = ssh_manager.get_password(config.host, config.username)
            if not password:
                self.finished.emit(False, "No saved password found. Please run setup again.")
                return
            
            success, message = ssh_manager.test_ssh_connection(config, password)
            if not success:
                self.finished.emit(False, f"Connection failed: {message}")
                return
            
            # Sync project if enabled
            if config.project_sync_enabled:
                self.status_update.emit("Synchronizing project files...")
                # Add project sync logic here
            
            # Launch remote GUI
            self.status_update.emit("Launching Vestim on remote server...")
            process = ssh_manager.launch_remote_gui(config, password)
            
            self.finished.emit(True, "Remote Vestim launched successfully!")
            
        except Exception as e:
            self.finished.emit(False, f"Connection error: {str(e)}")


class VestimServerClient:
    """Main Vestim Server Client application"""
    
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.app.setApplicationName("Vestim Server Client")
        self.app.setApplicationVersion("1.0")
        
        self.config_dir = Path.home() / ".vestim" / "server"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config_file = self.config_dir / "server_config.json"
    
    def run(self):
        """Main entry point"""
        # Check if configuration exists
        if not self.config_file.exists():
            self.show_first_time_setup()
            return
        
        # Load and validate configuration
        try:
            with open(self.config_file, 'r') as f:
                config_data = json.load(f)
            
            # Validate required fields
            required_fields = ['host', 'username']
            if not all(field in config_data for field in required_fields):
                self.show_reconfigure_dialog()
                return
                
        except Exception:
            self.show_reconfigure_dialog()
            return
        
        # Show connection progress and launch
        self.launch_with_progress()
    
    def show_first_time_setup(self):
        """Show first-time setup message"""
        reply = QMessageBox.question(
            None, "Vestim Server Client - First Run",
            "Welcome to Vestim Server Client!\n\n"
            "This application connects to a remote Vestim server and displays\n"
            "the GUI on your local machine while processing runs on the server.\n\n"
            "You need to configure your server connection first.\n\n"
            "Would you like to set up your server connection now?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        
        if reply == QMessageBox.Yes:
            self.open_configuration()
        else:
            QMessageBox.information(
                None, "Setup Required",
                "Server configuration is required to use Vestim Server Client.\n\n"
                "You can run the setup later by:\n"
                "• Using the 'Vestim Server Setup' program\n"
                "• Running: vestim_server_client.py --setup"
            )
            sys.exit(0)
    
    def show_reconfigure_dialog(self):
        """Show reconfiguration dialog"""
        reply = QMessageBox.question(
            None, "Configuration Issue",
            "Server configuration is missing or invalid.\n\n"
            "Would you like to reconfigure your server connection?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        
        if reply == QMessageBox.Yes:
            self.open_configuration()
        else:
            sys.exit(0)
    
    def open_configuration(self):
        """Open configuration dialog"""
        try:
            from vestim.remote.config_dialog import RemoteSetupDialog
            
            dialog = RemoteSetupDialog()
            if dialog.exec_() == dialog.Accepted:
                # Configuration saved, now launch
                self.launch_with_progress()
            else:
                sys.exit(0)
                
        except Exception as e:
            QMessageBox.critical(
                None, "Configuration Error",
                f"Failed to open configuration: {str(e)}\n\n"
                "Please ensure all dependencies are installed."
            )
            sys.exit(1)
    
    def launch_with_progress(self):
        """Launch connection with progress dialog"""
        # Create progress dialog
        progress = QProgressDialog("Connecting to Vestim Server...", "Cancel", 0, 0)
        progress.setWindowModality(Qt.WindowModal)
        progress.setWindowTitle("Vestim Server Client")
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.show()
        
        # Start connection thread
        self.connection_thread = ServerConnectionThread(self.config_file)
        self.connection_thread.status_update.connect(progress.setLabelText)
        self.connection_thread.finished.connect(
            lambda success, message: self.on_connection_finished(success, message, progress)
        )
        
        # Handle cancel button
        progress.canceled.connect(self.connection_thread.terminate)
        
        self.connection_thread.start()
        
        # Keep application alive
        self.app.exec_()
    
    def on_connection_finished(self, success, message, progress_dialog):
        """Handle connection completion"""
        progress_dialog.close()
        
        if success:
            QMessageBox.information(
                None, "Success", 
                f"{message}\n\nThe Vestim GUI should appear shortly.\n"
                "You can close this dialog - the remote session will continue running."
            )
        else:
            # Show error with option to reconfigure
            reply = QMessageBox.critical(
                None, "Connection Failed", 
                f"{message}\n\nWould you like to reconfigure your server settings?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.open_configuration()
                return
        
        sys.exit(0)


def main():
    """Main entry point for Vestim Server Client"""
    try:
        # Handle command line arguments
        if len(sys.argv) > 1:
            if sys.argv[1] in ['--setup', '--configure', '-c']:
                # Run setup only
                app = QApplication(sys.argv)
                from vestim.remote.config_dialog import RemoteSetupDialog
                dialog = RemoteSetupDialog()
                result = dialog.exec_()
                sys.exit(0 if result == dialog.Accepted else 1)
                
            elif sys.argv[1] in ['--help', '-h']:
                print("Vestim Server Client")
                print("One-click launcher for remote Vestim GUI")
                print()
                print("Usage:")
                print("  vestim_server_client.py           # Launch remote GUI")
                print("  vestim_server_client.py --setup   # Configure server settings")
                print("  vestim_server_client.py --help    # Show this help")
                print()
                print("On first run, you'll be guided through server configuration.")
                print("After setup, just double-click the icon to connect!")
                sys.exit(0)
                
            elif sys.argv[1] == '--version':
                print("Vestim Server Client v1.0")
                sys.exit(0)
        
        # Default action: run the client
        client = VestimServerClient()
        client.run()
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}")
        if '--debug' in sys.argv:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
