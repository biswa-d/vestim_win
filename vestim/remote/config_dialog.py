"""
GUI Configuration Dialog for Remote SSH Setup
"""

import sys
import os
from pathlib import Path
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QLineEdit,
    QPushButton, QCheckBox, QSpinBox, QTextEdit, QTabWidget, QWidget,
    QGroupBox, QProgressBar, QMessageBox, QFileDialog, QComboBox
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QFont, QIcon

from .ssh_manager import RemoteSSHManager, SSHConfig
from .auto_x11_installer import AutoX11ServerManager, get_x11_installation_guide


class ConnectionTestThread(QThread):
    """Thread for testing SSH connection without blocking UI"""
    result_ready = pyqtSignal(bool, str)
    
    def __init__(self, config, password):
        super().__init__()
        self.config = config
        self.password = password
        self.ssh_manager = RemoteSSHManager()
    
    def run(self):
        success, message = self.ssh_manager.test_ssh_connection(self.config, self.password)
        self.result_ready.emit(success, message)


class RemoteSetupDialog(QDialog):
    """Main dialog for configuring remote SSH connection"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ssh_manager = RemoteSSHManager()
        self.x11_manager = AutoX11ServerManager()
        self.config = self.ssh_manager.load_config() or SSHConfig(host="")
        
        self.setup_ui()
        self.load_current_config()
    
    def setup_ui(self):
        """Setup the user interface"""
        self.setWindowTitle("Remote Vestim Configuration")
        self.setFixedSize(600, 500)
        
        layout = QVBoxLayout(self)
        
        # Create tab widget
        tabs = QTabWidget()
        tabs.addTab(self.create_connection_tab(), "SSH Connection")
        tabs.addTab(self.create_display_tab(), "Display Setup")
        tabs.addTab(self.create_sync_tab(), "Project Sync")
        tabs.addTab(self.create_test_tab(), "Connection Test")
        
        layout.addWidget(tabs)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.save_btn = QPushButton("Save Configuration")
        self.launch_btn = QPushButton("Launch Remote Vestim")
        self.cancel_btn = QPushButton("Cancel")
        
        self.save_btn.clicked.connect(self.save_configuration)
        self.launch_btn.clicked.connect(self.launch_remote_vestim)
        self.cancel_btn.clicked.connect(self.reject)
        
        button_layout.addWidget(self.save_btn)
        button_layout.addWidget(self.launch_btn)
        button_layout.addStretch()
        button_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(button_layout)
    
    def create_connection_tab(self) -> QWidget:
        """Create SSH connection configuration tab"""
        widget = QWidget()
        layout = QGridLayout(widget)
        
        # Server details
        server_group = QGroupBox("Server Details")
        server_layout = QGridLayout(server_group)
        
        server_layout.addWidget(QLabel("Hostname/IP:"), 0, 0)
        self.host_edit = QLineEdit()
        server_layout.addWidget(self.host_edit, 0, 1)
        
        server_layout.addWidget(QLabel("Port:"), 1, 0)
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(22)
        server_layout.addWidget(self.port_spin, 1, 1)
        
        # Credentials
        cred_group = QGroupBox("Credentials")
        cred_layout = QGridLayout(cred_group)
        
        cred_layout.addWidget(QLabel("Username:"), 0, 0)
        self.username_edit = QLineEdit()
        cred_layout.addWidget(self.username_edit, 0, 1)
        
        cred_layout.addWidget(QLabel("Password:"), 1, 0)
        self.password_edit = QLineEdit()
        self.password_edit.setEchoMode(QLineEdit.Password)
        cred_layout.addWidget(self.password_edit, 1, 1)
        
        self.save_password_cb = QCheckBox("Save password securely (recommended)")
        cred_layout.addWidget(self.save_password_cb, 2, 0, 1, 2)
        
        # Advanced options
        advanced_group = QGroupBox("Advanced Options")
        advanced_layout = QGridLayout(advanced_group)
        
        advanced_layout.addWidget(QLabel("Connection Timeout (s):"), 0, 0)
        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(5, 300)
        self.timeout_spin.setValue(30)
        advanced_layout.addWidget(self.timeout_spin, 0, 1)
        
        self.compression_cb = QCheckBox("Enable compression")
        self.compression_cb.setChecked(True)
        advanced_layout.addWidget(self.compression_cb, 1, 0, 1, 2)
        
        layout.addWidget(server_group, 0, 0)
        layout.addWidget(cred_group, 1, 0)
        layout.addWidget(advanced_group, 2, 0)
        layout.addStretch()
        
        return widget
    
    def create_display_tab(self) -> QWidget:
        """Create display configuration tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # X11 Server Status
        x11_group = QGroupBox("Local X11 Server")
        x11_layout = QVBoxLayout(x11_group)
        
        self.x11_status_label = QLabel("Checking X11 server status...")
        x11_layout.addWidget(self.x11_status_label)
        
        x11_button_layout = QHBoxLayout()
        self.check_x11_btn = QPushButton("Check X11 Status")
        self.start_x11_btn = QPushButton("Start X11 Server")
        
        self.check_x11_btn.clicked.connect(self.check_x11_status)
        self.start_x11_btn.clicked.connect(self.start_x11_server)
        
        x11_button_layout.addWidget(self.check_x11_btn)
        x11_button_layout.addWidget(self.start_x11_btn)
        x11_button_layout.addStretch()
        
        x11_layout.addLayout(x11_button_layout)
        
        # Display Configuration
        display_group = QGroupBox("Display Configuration")
        display_layout = QGridLayout(display_group)
        
        display_layout.addWidget(QLabel("Remote DISPLAY:"), 0, 0)
        self.display_edit = QLineEdit(":0")
        display_layout.addWidget(self.display_edit, 0, 1)
        
        self.auto_detect_display_cb = QCheckBox("Auto-detect remote DISPLAY")
        self.auto_detect_display_cb.setChecked(True)
        display_layout.addWidget(self.auto_detect_display_cb, 1, 0, 1, 2)
        
        layout.addWidget(x11_group)
        layout.addWidget(display_group)
        layout.addStretch()
        
        return widget
    
    def create_sync_tab(self) -> QWidget:
        """Create project synchronization tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Sync Configuration
        sync_group = QGroupBox("Project Synchronization")
        sync_layout = QGridLayout(sync_group)
        
        self.sync_enabled_cb = QCheckBox("Enable project synchronization")
        self.sync_enabled_cb.setChecked(True)
        sync_layout.addWidget(self.sync_enabled_cb, 0, 0, 1, 2)
        
        sync_layout.addWidget(QLabel("Remote Vestim Path:"), 1, 0)
        self.remote_path_edit = QLineEdit("/opt/vestim")
        sync_layout.addWidget(self.remote_path_edit, 1, 1)
        
        sync_layout.addWidget(QLabel("Local Project Path:"), 2, 0)
        project_layout = QHBoxLayout()
        self.local_path_edit = QLineEdit()
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self.browse_local_path)
        
        project_layout.addWidget(self.local_path_edit)
        project_layout.addWidget(self.browse_btn)
        sync_layout.addLayout(project_layout, 2, 1)
        
        # Sync Options
        options_group = QGroupBox("Sync Options")
        options_layout = QVBoxLayout(options_group)
        
        self.sync_before_cb = QCheckBox("Sync project to remote before launch")
        self.sync_before_cb.setChecked(True)
        options_layout.addWidget(self.sync_before_cb)
        
        self.sync_after_cb = QCheckBox("Sync results back after completion")
        self.sync_after_cb.setChecked(True)
        options_layout.addWidget(self.sync_after_cb)
        
        layout.addWidget(sync_group)
        layout.addWidget(options_group)
        layout.addStretch()
        
        return widget
    
    def create_test_tab(self) -> QWidget:
        """Create connection testing tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Test Controls
        test_group = QGroupBox("Connection Test")
        test_layout = QVBoxLayout(test_group)
        
        self.test_btn = QPushButton("Test SSH Connection")
        self.test_btn.clicked.connect(self.test_connection)
        test_layout.addWidget(self.test_btn)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        test_layout.addWidget(self.progress_bar)
        
        # Results
        results_group = QGroupBox("Test Results")
        results_layout = QVBoxLayout(results_group)
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(150)
        results_layout.addWidget(self.results_text)
        
        layout.addWidget(test_group)
        layout.addWidget(results_group)
        layout.addStretch()
        
        return widget
    
    def load_current_config(self):
        """Load current configuration into form fields"""
        self.host_edit.setText(self.config.host)
        self.port_spin.setValue(self.config.port)
        self.username_edit.setText(self.config.username)
        self.display_edit.setText(self.config.display)
        self.remote_path_edit.setText(self.config.remote_vestim_path)
        self.timeout_spin.setValue(self.config.timeout)
        self.compression_cb.setChecked(self.config.compression)
        self.sync_enabled_cb.setChecked(self.config.project_sync_enabled)
        
        # Check X11 status on load
        self.check_x11_status()
    
    def check_x11_status(self):
        """Check local X11 server status"""
        if self.x11_manager.is_x11_running():
            self.x11_status_label.setText("✓ X11 server is running")
            self.x11_status_label.setStyleSheet("color: green;")
            self.start_x11_btn.setEnabled(False)
        elif self.x11_manager.has_x11_installed():
            self.x11_status_label.setText("⚠ X11 server installed but not running")
            self.x11_status_label.setStyleSheet("color: orange;")
            self.start_x11_btn.setEnabled(True)
            self.start_x11_btn.setText("Start X11 Server")
        else:
            self.x11_status_label.setText("✗ X11 server not installed")
            self.x11_status_label.setStyleSheet("color: red;")
            self.start_x11_btn.setEnabled(True)
            self.start_x11_btn.setText("Install & Start X11")
    
    def start_x11_server(self):
        """Start or install X11 server"""
        if self.x11_manager.has_x11_installed():
            # Just start existing installation
            success, message = self.x11_manager.start_x11_server()
            if success:
                QMessageBox.information(self, "Success", message)
                self.check_x11_status()
            else:
                QMessageBox.warning(self, "Error", message)
        else:
            # Need to install first
            reply = QMessageBox.question(
                self, "Install X11 Server",
                "X11 server is required for remote GUI functionality.\n\n"
                "Would you like to automatically install VcXsrv (Windows) or XQuartz (macOS)?\n\n"
                "This may take a few minutes and might require administrator permissions.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            
            if reply == QMessageBox.Yes:
                # Show progress dialog
                progress = QMessageBox(self)
                progress.setWindowTitle("Installing X11 Server")
                progress.setText("Downloading and installing X11 server...\nThis may take several minutes.")
                progress.setStandardButtons(QMessageBox.NoButton)
                progress.show()
                progress.repaint()
                
                try:
                    success, message = self.x11_manager.ensure_x11_available()
                    progress.close()
                    
                    if success:
                        QMessageBox.information(self, "Installation Complete", 
                                              f"X11 server installed successfully!\n\n{message}")
                        self.check_x11_status()
                    else:
                        error_msg = f"Automatic installation failed:\n{message}\n\n"
                        error_msg += "Manual installation guide:\n"
                        error_msg += get_x11_installation_guide()
                        QMessageBox.warning(self, "Installation Failed", error_msg)
                        
                except Exception as e:
                    progress.close()
                    error_msg = f"Installation error: {str(e)}\n\n"
                    error_msg += "Manual installation guide:\n"
                    error_msg += get_x11_installation_guide()
                    QMessageBox.critical(self, "Installation Error", error_msg)
            else:
                # Show manual installation guide
                QMessageBox.information(self, "Manual Installation", get_x11_installation_guide())
    
    def browse_local_path(self):
        """Browse for local project path"""
        path = QFileDialog.getExistingDirectory(
            self, "Select Project Directory", self.local_path_edit.text()
        )
        if path:
            self.local_path_edit.setText(path)
    
    def test_connection(self):
        """Test SSH connection in background thread"""
        # Create config from current form values
        config = SSHConfig(
            host=self.host_edit.text(),
            port=self.port_spin.value(),
            username=self.username_edit.text(),
            display=self.display_edit.text(),
            remote_vestim_path=self.remote_path_edit.text(),
            project_sync_enabled=self.sync_enabled_cb.isChecked(),
            compression=self.compression_cb.isChecked(),
            timeout=self.timeout_spin.value()
        )
        
        if not config.host or not config.username:
            QMessageBox.warning(self, "Error", "Please enter hostname and username")
            return
        
        password = self.password_edit.text()
        if not password:
            password = self.ssh_manager.get_password(config.host, config.username)
            if not password:
                QMessageBox.warning(self, "Error", "Please enter password")
                return
        
        self.test_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.results_text.clear()
        self.results_text.append("Testing SSH connection...")
        
        # Start test thread
        self.test_thread = ConnectionTestThread(config, password)
        self.test_thread.result_ready.connect(self.on_test_complete)
        self.test_thread.start()
    
    def on_test_complete(self, success: bool, message: str):
        """Handle test completion"""
        self.test_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        if success:
            self.results_text.append(f"✓ {message}")
            self.results_text.setStyleSheet("color: green;")
        else:
            self.results_text.append(f"✗ {message}")
            self.results_text.setStyleSheet("color: red;")
    
    def save_configuration(self):
        """Save current configuration"""
        config = SSHConfig(
            host=self.host_edit.text(),
            port=self.port_spin.value(),
            username=self.username_edit.text(),
            display=self.display_edit.text(),
            remote_vestim_path=self.remote_path_edit.text(),
            project_sync_enabled=self.sync_enabled_cb.isChecked(),
            compression=self.compression_cb.isChecked(),
            timeout=self.timeout_spin.value()
        )
        
        if not config.host or not config.username:
            QMessageBox.warning(self, "Error", "Please enter hostname and username")
            return
        
        self.ssh_manager.save_config(config)
        
        # Save password if requested
        if self.save_password_cb.isChecked() and self.password_edit.text():
            self.ssh_manager.save_password(
                config.host, config.username, self.password_edit.text()
            )
        
        QMessageBox.information(self, "Success", "Configuration saved successfully")
        self.accept()
    
    def launch_remote_vestim(self):
        """Launch remote Vestim with current configuration"""
        # First save configuration
        self.save_configuration()
        
        # Then launch remote GUI
        try:
            config = SSHConfig(
                host=self.host_edit.text(),
                port=self.port_spin.value(),
                username=self.username_edit.text(),
                display=self.display_edit.text(),
                remote_vestim_path=self.remote_path_edit.text(),
                project_sync_enabled=self.sync_enabled_cb.isChecked(),
                compression=self.compression_cb.isChecked(),
                timeout=self.timeout_spin.value()
            )
            
            password = self.password_edit.text()
            if not password:
                password = self.ssh_manager.get_password(config.host, config.username)
            
            # Sync project if enabled
            if config.project_sync_enabled and self.sync_before_cb.isChecked():
                local_path = self.local_path_edit.text()
                if local_path:
                    success, message = self.ssh_manager.sync_project_to_remote(
                        config, local_path, password
                    )
                    if not success:
                        QMessageBox.warning(self, "Sync Error", message)
                        return
            
            # Launch remote GUI
            process = self.ssh_manager.launch_remote_gui(config, password)
            QMessageBox.information(
                self, "Success", 
                "Remote Vestim launched successfully. The GUI should appear shortly."
            )
            
            self.accept()
            
        except Exception as e:
            QMessageBox.critical(self, "Launch Error", f"Failed to launch remote GUI: {str(e)}")


if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    dialog = RemoteSetupDialog()
    dialog.show()
    sys.exit(app.exec_())
