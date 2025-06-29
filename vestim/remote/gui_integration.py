"""
Remote Launch Integration for Vestim GUI
Add this to existing Vestim GUI classes to enable remote launching
"""

from PyQt5.QtWidgets import QMenuBar, QMenu, QAction, QMessageBox
from PyQt5.QtCore import QThread, pyqtSignal


class RemoteLaunchThread(QThread):
    """Thread for launching remote GUI without blocking main GUI"""
    finished = pyqtSignal(bool, str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
    
    def run(self):
        try:
            from vestim.remote.ssh_manager import RemoteSSHManager
            from vestim.remote.config_dialog import RemoteSetupDialog
            from PyQt5.QtWidgets import QDialog
            
            # Check if configuration exists
            ssh_manager = RemoteSSHManager()
            config = ssh_manager.load_config()
            
            if config and config.host:
                # Try to launch with existing config
                password = ssh_manager.get_password(config.host, config.username)
                if password:
                    process = ssh_manager.launch_remote_gui(config, password)
                    self.finished.emit(True, "Remote Vestim launched successfully")
                    return
            
            # Need configuration - signal main thread to show dialog
            self.finished.emit(False, "SHOW_CONFIG")
            
        except ImportError:
            self.finished.emit(False, "Remote functionality not available. Please run setup_remote.py first.")
        except Exception as e:
            self.finished.emit(False, f"Failed to launch remote GUI: {str(e)}")


class RemoteGUIIntegration:
    """Mixin class to add remote launch capability to existing GUIs"""
    
    def add_remote_menu(self):
        """Add remote launch option to menu bar"""
        if not hasattr(self, 'menuBar'):
            # Create menu bar if it doesn't exist
            menubar = self.menuBar()
        else:
            menubar = self.menuBar
        
        # Add Remote menu
        remote_menu = menubar.addMenu('Remote')
        
        # Launch Remote action
        launch_action = QAction('Launch on Remote Server', self)
        launch_action.setStatusTip('Launch Vestim GUI on remote server')
        launch_action.triggered.connect(self.launch_remote_gui)
        remote_menu.addAction(launch_action)
        
        # Configure Remote action
        config_action = QAction('Configure Remote Connection', self)
        config_action.setStatusTip('Set up remote server connection')
        config_action.triggered.connect(self.configure_remote)
        remote_menu.addAction(config_action)
        
        # Separator
        remote_menu.addSeparator()
        
        # Help action
        help_action = QAction('Remote Setup Help', self)
        help_action.setStatusTip('View remote setup documentation')
        help_action.triggered.connect(self.show_remote_help)
        remote_menu.addAction(help_action)
    
    def add_remote_button(self, layout):
        """Add remote launch button to existing layout"""
        from PyQt5.QtWidgets import QPushButton
        
        remote_btn = QPushButton('Launch Remote')
        remote_btn.setToolTip('Launch this application on a remote server')
        remote_btn.clicked.connect(self.launch_remote_gui)
        layout.addWidget(remote_btn)
        
        return remote_btn
    
    def launch_remote_gui(self):
        """Launch remote GUI in background thread"""
        self.remote_thread = RemoteLaunchThread()
        self.remote_thread.finished.connect(self.on_remote_launch_finished)
        self.remote_thread.start()
    
    def on_remote_launch_finished(self, success, message):
        """Handle remote launch completion"""
        if success:
            QMessageBox.information(self, "Success", message)
        elif message == "SHOW_CONFIG":
            self.configure_remote()
        else:
            QMessageBox.warning(self, "Remote Launch Failed", message)
    
    def configure_remote(self):
        """Show remote configuration dialog"""
        try:
            from vestim.remote.config_dialog import RemoteSetupDialog
            
            dialog = RemoteSetupDialog(self)
            dialog.exec_()
            
        except ImportError:
            QMessageBox.warning(
                self, "Remote Not Available",
                "Remote functionality not available.\n\n"
                "Please run the following command to set up remote features:\n"
                "python setup_remote.py"
            )
        except Exception as e:
            QMessageBox.critical(self, "Configuration Error", f"Failed to open configuration: {str(e)}")
    
    def show_remote_help(self):
        """Show remote help information"""
        help_text = """
Remote Vestim Setup Help

The remote functionality allows you to run Vestim on a remote server while 
displaying the GUI on your local machine.

Prerequisites:
• Remote server with Vestim installed
• SSH access to the remote server  
• X11 server running locally (VcXsrv/Xming on Windows, XQuartz on macOS)

Quick Start:
1. Click 'Configure Remote Connection' to set up your server details
2. Test the connection using the test tab
3. Click 'Launch Remote' to start Vestim on the remote server

For detailed setup instructions, see REMOTE_SETUP_GUIDE.md

Common Issues:
• No GUI appears: Check X11 server is running locally
• Connection failed: Verify SSH credentials and server access
• Permission denied: Check Vestim is executable on remote server
        """
        
        QMessageBox.information(self, "Remote Setup Help", help_text.strip())


# Example integration for existing GUI classes
def integrate_remote_to_gui(gui_class):
    """
    Decorator to add remote functionality to existing GUI classes
    
    Usage:
    @integrate_remote_to_gui
    class MyExistingGUI(QMainWindow):
        # existing code...
    """
    
    # Add RemoteGUIIntegration as a mixin
    class IntegratedGUI(RemoteGUIIntegration, gui_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Add remote menu after initialization
            self.add_remote_menu()
    
    return IntegratedGUI


# Function to patch existing GUI instances
def add_remote_to_existing_gui(gui_instance):
    """
    Add remote functionality to an existing GUI instance
    
    Usage:
    my_gui = ExistingGUI()
    add_remote_to_existing_gui(my_gui)
    """
    
    # Mix in the remote functionality
    gui_instance.__class__ = type(
        gui_instance.__class__.__name__ + "WithRemote",
        (RemoteGUIIntegration, gui_instance.__class__),
        {}
    )
    
    # Add the remote menu
    gui_instance.add_remote_menu()
    
    return gui_instance
