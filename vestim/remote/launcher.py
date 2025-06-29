#!/usr/bin/env python3
"""
Remote Vestim Launcher
Creates desktop shortcut and handles remote GUI launching
"""

import sys
import os
import argparse
from pathlib import Path

# Add vestim package to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from PyQt5.QtWidgets import QApplication, QMessageBox
    from vestim.remote.config_dialog import RemoteSetupDialog
    from vestim.remote.ssh_manager import RemoteSSHManager
except ImportError as e:
    print(f"Failed to import required modules: {e}")
    print("Please ensure PyQt5 and required dependencies are installed.")
    sys.exit(1)


def create_desktop_shortcut():
    """Create desktop shortcut for remote Vestim launcher"""
    import platform
    
    if platform.system() == "Windows":
        create_windows_shortcut()
    elif platform.system() == "Darwin":
        create_macos_shortcut()
    else:
        create_linux_shortcut()


def create_windows_shortcut():
    """Create Windows desktop shortcut"""
    try:
        import winshell
        from win32com.client import Dispatch
        
        desktop = winshell.desktop()
        shortcut_path = os.path.join(desktop, "Vestim Remote.lnk")
        
        shell = Dispatch('WScript.Shell')
        shortcut = shell.CreateShortCut(shortcut_path)
        shortcut.Targetpath = sys.executable
        shortcut.Arguments = f'"{__file__}" --gui'
        shortcut.WorkingDirectory = os.path.dirname(__file__)
        shortcut.IconLocation = get_icon_path()
        shortcut.Description = "Launch Vestim on Remote Server"
        shortcut.save()
        
        print(f"Desktop shortcut created: {shortcut_path}")
        
    except ImportError:
        print("Warning: winshell not available. Creating batch file instead.")
        create_windows_batch()


def create_windows_batch():
    """Create Windows batch file as alternative to shortcut"""
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    batch_path = os.path.join(desktop, "Vestim Remote.bat")
    
    batch_content = f'''@echo off
cd /d "{os.path.dirname(__file__)}"
"{sys.executable}" "{__file__}" --gui
pause
'''
    
    with open(batch_path, 'w') as f:
        f.write(batch_content)
    
    print(f"Batch file created: {batch_path}")


def create_macos_shortcut():
    """Create macOS application bundle"""
    applications_dir = os.path.join(os.path.expanduser("~"), "Applications")
    app_dir = os.path.join(applications_dir, "Vestim Remote.app")
    contents_dir = os.path.join(app_dir, "Contents")
    macos_dir = os.path.join(contents_dir, "MacOS")
    
    os.makedirs(macos_dir, exist_ok=True)
    
    # Create Info.plist
    plist_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>vestim_remote</string>
    <key>CFBundleIdentifier</key>
    <string>com.vestim.remote</string>
    <key>CFBundleName</key>
    <string>Vestim Remote</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
</dict>
</plist>'''
    
    with open(os.path.join(contents_dir, "Info.plist"), 'w') as f:
        f.write(plist_content)
    
    # Create executable script
    exec_script = f'''#!/bin/bash
cd "{os.path.dirname(__file__)}"
"{sys.executable}" "{__file__}" --gui
'''
    
    exec_path = os.path.join(macos_dir, "vestim_remote")
    with open(exec_path, 'w') as f:
        f.write(exec_script)
    
    os.chmod(exec_path, 0o755)
    print(f"macOS application created: {app_dir}")


def create_linux_shortcut():
    """Create Linux desktop entry"""
    desktop_dir = os.path.join(os.path.expanduser("~"), "Desktop")
    applications_dir = os.path.join(os.path.expanduser("~"), ".local", "share", "applications")
    
    desktop_entry = f'''[Desktop Entry]
Name=Vestim Remote
Comment=Launch Vestim on Remote Server
Exec={sys.executable} "{__file__}" --gui
Icon={get_icon_path()}
Terminal=false
Type=Application
Categories=Science;Education;
'''
    
    # Create in both desktop and applications
    for directory in [desktop_dir, applications_dir]:
        os.makedirs(directory, exist_ok=True)
        entry_path = os.path.join(directory, "vestim-remote.desktop")
        
        with open(entry_path, 'w') as f:
            f.write(desktop_entry)
        
        os.chmod(entry_path, 0o755)
        print(f"Desktop entry created: {entry_path}")


def get_icon_path():
    """Get path to Vestim icon"""
    # Try to find icon in various locations
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "..", "gui", "resources", "icon.ico"),
        os.path.join(os.path.dirname(__file__), "..", "gui", "resources", "icon.png"),
        os.path.join(os.path.dirname(__file__), "..", "gui", "resources", "vestim.ico"),
        os.path.join(os.path.dirname(__file__), "..", "gui", "resources", "vestim.png"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return os.path.abspath(path)
    
    return ""  # No icon found


def launch_gui():
    """Launch the remote configuration GUI"""
    app = QApplication(sys.argv)
    app.setApplicationName("Vestim Remote")
    app.setApplicationVersion("1.0")
    
    # Check if configuration exists
    ssh_manager = RemoteSSHManager()
    config = ssh_manager.load_config()
    
    if config and config.host:
        # Configuration exists, ask user what to do
        reply = QMessageBox.question(
            None, "Vestim Remote",
            f"Found existing configuration for {config.host}.\n\n"
            "What would you like to do?",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            QMessageBox.Yes
        )
        
        if reply == QMessageBox.Yes:
            # Launch with existing config
            try:
                password = ssh_manager.get_password(config.host, config.username)
                if not password:
                    # Need to get password
                    dialog = RemoteSetupDialog()
                    if dialog.exec_() == QDialog.Accepted:
                        sys.exit(0)
                    else:
                        sys.exit(1)
                
                process = ssh_manager.launch_remote_gui(config, password)
                QMessageBox.information(
                    None, "Success",
                    "Remote Vestim launched successfully. The GUI should appear shortly."
                )
                sys.exit(0)
                
            except Exception as e:
                QMessageBox.critical(
                    None, "Launch Error",
                    f"Failed to launch remote GUI: {str(e)}\n\n"
                    "Opening configuration dialog..."
                )
        elif reply == QMessageBox.Cancel:
            sys.exit(0)
    
    # Show configuration dialog
    dialog = RemoteSetupDialog()
    result = dialog.exec_()
    sys.exit(0 if result == QDialog.Accepted else 1)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Vestim Remote Launcher")
    parser.add_argument("--gui", action="store_true", help="Launch GUI configuration")
    parser.add_argument("--create-shortcut", action="store_true", help="Create desktop shortcut")
    parser.add_argument("--host", help="SSH hostname")
    parser.add_argument("--username", help="SSH username")
    parser.add_argument("--password", help="SSH password")
    
    args = parser.parse_args()
    
    if args.create_shortcut:
        create_desktop_shortcut()
        return
    
    if args.gui or not args.host:
        launch_gui()
        return
    
    # Command line launch
    try:
        from vestim.remote.ssh_manager import SSHConfig
        
        config = SSHConfig(
            host=args.host,
            username=args.username or os.getenv("USER", ""),
        )
        
        ssh_manager = RemoteSSHManager()
        password = args.password or ssh_manager.get_password(config.host, config.username)
        
        if not password:
            print("Error: Password required")
            sys.exit(1)
        
        process = ssh_manager.launch_remote_gui(config, password)
        print("Remote Vestim launched successfully")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
