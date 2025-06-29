"""
Vestim Server Installation Setup Wizard
Collects remote server configuration during installation
"""

import sys
import os
import json
from pathlib import Path

try:
    from PyQt5.QtWidgets import (
        QApplication, QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
        QLabel, QLineEdit, QPushButton, QCheckBox, QSpinBox, QTextEdit,
        QMessageBox, QWizard, QWizardPage, QGroupBox, QProgressBar
    )
    from PyQt5.QtCore import Qt, QThread, pyqtSignal
    from PyQt5.QtGui import QFont, QPixmap
except ImportError:
    print("PyQt5 not available. Running in console mode.")
    PyQt5_available = False
else:
    PyQt5_available = True


class ServerConfigWizard(QWizard):
    """Installation wizard for server configuration"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vestim Server Setup")
        self.setFixedSize(600, 400)
        
        # Add wizard pages
        self.addPage(WelcomePage())
        self.addPage(ServerDetailsPage())
        self.addPage(CredentialsPage())
        self.addPage(TestConnectionPage())
        self.addPage(CompletionPage())
        
        # Set wizard style
        self.setWizardStyle(QWizard.ModernStyle)
        self.setOption(QWizard.HaveHelpButton, False)
    
    def save_configuration(self):
        """Save configuration to file"""
        config_dir = Path.home() / ".vestim" / "server"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / "server_config.json"
        
        # Collect data from wizard pages
        server_page = self.page(1)  # ServerDetailsPage
        creds_page = self.page(2)   # CredentialsPage
        
        config = {
            "host": server_page.host_edit.text(),
            "port": server_page.port_spin.value(),
            "username": creds_page.username_edit.text(),
            "display": ":0",
            "remote_vestim_path": server_page.vestim_path_edit.text(),
            "project_sync_enabled": server_page.sync_checkbox.isChecked(),
            "compression": True,
            "timeout": 30
        }
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save password securely if requested
        if creds_page.save_password_checkbox.isChecked():
            try:
                from vestim.remote.ssh_manager import RemoteSSHManager
                ssh_manager = RemoteSSHManager()
                ssh_manager.save_password(
                    config["host"], 
                    config["username"], 
                    creds_page.password_edit.text()
                )
            except Exception as e:
                print(f"Warning: Could not save password: {e}")
        
        return config_file


class WelcomePage(QWizardPage):
    """Welcome page"""
    
    def __init__(self):
        super().__init__()
        self.setTitle("Welcome to Vestim Server")
        self.setSubTitle("Configure your remote Vestim server connection")
        
        layout = QVBoxLayout()
        
        welcome_text = QLabel("""
<h3>Welcome to Vestim Server Client!</h3>

<p>This application allows you to connect to and use Vestim running on a remote server. 
The GUI will appear on your local machine while all processing happens on the powerful 
remote server.</p>

<h4>What you'll need:</h4>
<ul>
<li><b>Remote server</b> with Vestim installed</li>
<li><b>SSH access</b> to the remote server</li>
<li><b>Username and password</b> for the server</li>
</ul>

<p>This setup wizard will guide you through the configuration process.</p>
        """)
        welcome_text.setWordWrap(True)
        layout.addWidget(welcome_text)
        
        self.setLayout(layout)


class ServerDetailsPage(QWizardPage):
    """Server details configuration page"""
    
    def __init__(self):
        super().__init__()
        self.setTitle("Server Details")
        self.setSubTitle("Enter your remote server information")
        
        layout = QGridLayout()
        
        # Server connection group
        server_group = QGroupBox("Server Connection")
        server_layout = QGridLayout(server_group)
        
        server_layout.addWidget(QLabel("Server Address:"), 0, 0)
        self.host_edit = QLineEdit()
        self.host_edit.setPlaceholderText("e.g., server.university.edu or 192.168.1.100")
        server_layout.addWidget(self.host_edit, 0, 1)
        
        server_layout.addWidget(QLabel("SSH Port:"), 1, 0)
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(22)
        server_layout.addWidget(self.port_spin, 1, 1)
        
        # Vestim installation group
        vestim_group = QGroupBox("Vestim Installation")
        vestim_layout = QGridLayout(vestim_group)
        
        vestim_layout.addWidget(QLabel("Vestim Path on Server:"), 0, 0)
        self.vestim_path_edit = QLineEdit("/opt/vestim")
        self.vestim_path_edit.setPlaceholderText("Path where Vestim is installed on the server")
        vestim_layout.addWidget(self.vestim_path_edit, 0, 1)
        
        # Options group
        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout(options_group)
        
        self.sync_checkbox = QCheckBox("Enable automatic project synchronization")
        self.sync_checkbox.setChecked(True)
        self.sync_checkbox.setToolTip("Automatically sync your projects to/from the server")
        options_layout.addWidget(self.sync_checkbox)
        
        layout.addWidget(server_group, 0, 0)
        layout.addWidget(vestim_group, 1, 0)
        layout.addWidget(options_group, 2, 0)
        
        self.setLayout(layout)
        
        # Register fields for validation
        self.registerField("host*", self.host_edit)
    
    def isComplete(self):
        """Validate that required fields are filled"""
        return bool(self.host_edit.text().strip())


class CredentialsPage(QWizardPage):
    """Credentials configuration page"""
    
    def __init__(self):
        super().__init__()
        self.setTitle("Server Credentials")
        self.setSubTitle("Enter your login credentials for the remote server")
        
        layout = QVBoxLayout()
        
        # Credentials group
        creds_group = QGroupBox("Login Information")
        creds_layout = QGridLayout(creds_group)
        
        creds_layout.addWidget(QLabel("Username:"), 0, 0)
        self.username_edit = QLineEdit()
        self.username_edit.setPlaceholderText("Your username on the remote server")
        creds_layout.addWidget(self.username_edit, 0, 1)
        
        creds_layout.addWidget(QLabel("Password:"), 1, 0)
        self.password_edit = QLineEdit()
        self.password_edit.setEchoMode(QLineEdit.Password)
        self.password_edit.setPlaceholderText("Your password (will be stored securely)")
        creds_layout.addWidget(self.password_edit, 1, 1)
        
        self.save_password_checkbox = QCheckBox("Save password securely")
        self.save_password_checkbox.setChecked(True)
        self.save_password_checkbox.setToolTip("Store password in system keyring for automatic login")
        creds_layout.addWidget(self.save_password_checkbox, 2, 0, 1, 2)
        
        # Security notice
        security_label = QLabel("""
<b>Security Note:</b> Your password will be stored using your system's secure credential store 
(Windows Credential Manager, macOS Keychain, or Linux Secret Service).
        """)
        security_label.setWordWrap(True)
        security_label.setStyleSheet("color: #666; font-size: 10px;")
        
        layout.addWidget(creds_group)
        layout.addWidget(security_label)
        
        self.setLayout(layout)
        
        # Register fields for validation
        self.registerField("username*", self.username_edit)
        self.registerField("password*", self.password_edit)
    
    def isComplete(self):
        """Validate that required fields are filled"""
        return (bool(self.username_edit.text().strip()) and 
                bool(self.password_edit.text().strip()))


class TestConnectionPage(QWizardPage):
    """Connection testing page"""
    
    def __init__(self):
        super().__init__()
        self.setTitle("Test Connection")
        self.setSubTitle("Verify your server connection")
        
        layout = QVBoxLayout()
        
        # Instructions
        instructions = QLabel("""
Click 'Test Connection' to verify that we can connect to your server with the provided credentials.
        """)
        layout.addWidget(instructions)
        
        # Test button and progress
        test_layout = QHBoxLayout()
        self.test_button = QPushButton("Test Connection")
        self.test_button.clicked.connect(self.test_connection)
        test_layout.addWidget(self.test_button)
        test_layout.addStretch()
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        # Results area
        self.results_text = QTextEdit()
        self.results_text.setMaximumHeight(150)
        self.results_text.setReadOnly(True)
        
        layout.addLayout(test_layout)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.results_text)
        
        self.setLayout(layout)
        
        self.connection_tested = False
    
    def test_connection(self):
        """Test the server connection"""
        self.test_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.results_text.clear()
        self.results_text.append("Testing connection...")
        
        # Get configuration from previous pages
        wizard = self.wizard()
        host = wizard.page(1).host_edit.text()
        port = wizard.page(1).port_spin.value()
        username = wizard.page(2).username_edit.text()
        password = wizard.page(2).password_edit.text()
        
        try:
            from vestim.remote.ssh_manager import RemoteSSHManager, SSHConfig
            
            config = SSHConfig(host=host, port=port, username=username)
            ssh_manager = RemoteSSHManager()
            
            success, message = ssh_manager.test_ssh_connection(config, password)
            
            if success:
                self.results_text.append(f"✓ {message}")
                self.results_text.setStyleSheet("color: green;")
                self.connection_tested = True
                self.completeChanged.emit()
            else:
                self.results_text.append(f"✗ {message}")
                self.results_text.setStyleSheet("color: red;")
                self.connection_tested = False
                
        except Exception as e:
            self.results_text.append(f"✗ Test failed: {str(e)}")
            self.results_text.setStyleSheet("color: red;")
            self.connection_tested = False
        
        finally:
            self.test_button.setEnabled(True)
            self.progress_bar.setVisible(False)
            self.completeChanged.emit()
    
    def isComplete(self):
        """Require successful connection test"""
        return self.connection_tested


class CompletionPage(QWizardPage):
    """Final completion page"""
    
    def __init__(self):
        super().__init__()
        self.setTitle("Setup Complete")
        self.setSubTitle("Your Vestim Server client is ready to use")
        
        layout = QVBoxLayout()
        
        completion_text = QLabel("""
<h3>🎉 Setup Complete!</h3>

<p>Your Vestim Server client has been configured successfully.</p>

<h4>What happens next:</h4>
<ul>
<li><b>Desktop shortcut</b> will be created for easy access</li>
<li><b>Click "Vestim Server"</b> to connect and launch the remote GUI</li>
<li><b>X11 server</b> will be automatically set up if needed</li>
<li><b>Projects sync</b> automatically between local and remote</li>
</ul>

<p><b>To use Vestim Server:</b><br>
Simply click the "Vestim Server" icon on your desktop or start menu. 
The system will automatically connect to your server and display the GUI locally.</p>

<p>You can reconfigure these settings anytime by running this setup wizard again.</p>
        """)
        completion_text.setWordWrap(True)
        layout.addWidget(completion_text)
        
        self.setLayout(layout)


def run_console_setup():
    """Fallback console-based setup if PyQt5 is not available"""
    print("=" * 60)
    print("Vestim Server Setup (Console Mode)")
    print("=" * 60)
    
    config_dir = Path.home() / ".vestim" / "server"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "server_config.json"
    
    print("\nEnter your remote server details:")
    
    host = input("Server address: ").strip()
    if not host:
        print("Error: Server address is required")
        return False
    
    try:
        port = int(input("SSH port (default 22): ") or "22")
    except ValueError:
        port = 22
    
    username = input("Username: ").strip()
    if not username:
        print("Error: Username is required")
        return False
    
    vestim_path = input("Vestim path on server (default /opt/vestim): ").strip() or "/opt/vestim"
    
    sync_enabled = input("Enable project sync? (y/N): ").lower().startswith('y')
    
    config = {
        "host": host,
        "port": port,
        "username": username,
        "display": ":0",
        "remote_vestim_path": vestim_path,
        "project_sync_enabled": sync_enabled,
        "compression": True,
        "timeout": 30
    }
    
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✓ Configuration saved to {config_file}")
    print("\nNote: You'll need to enter your password when first connecting.")
    return True


def main():
    """Main setup function"""
    print("Setting up Vestim Server client...")
    
    if PyQt5_available:
        # GUI setup
        app = QApplication(sys.argv)
        app.setApplicationName("Vestim Server Setup")
        
        wizard = ServerConfigWizard()
        
        if wizard.exec_() == QDialog.Accepted:
            config_file = wizard.save_configuration()
            print(f"Configuration saved to {config_file}")
            return True
        else:
            print("Setup cancelled by user")
            return False
    else:
        # Console setup
        return run_console_setup()


if __name__ == "__main__":
    if main():
        print("\n🎉 Vestim Server setup complete!")
        print("You can now use the 'Vestim Server' shortcut to connect.")
    else:
        print("\n❌ Setup incomplete. Please run setup again.")
        sys.exit(1)
