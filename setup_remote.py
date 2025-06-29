#!/usr/bin/env python3
"""
Remote Setup Script for Vestim
Installs required dependencies and sets up remote functionality
"""

import sys
import subprocess
import platform
import os
from pathlib import Path


def install_package(package_name):
    """Install a Python package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return True
    except subprocess.CalledProcessError:
        return False


def check_package(package_name):
    """Check if a package is installed"""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False


def install_remote_dependencies():
    """Install dependencies required for remote functionality"""
    print("Setting up Vestim Remote dependencies...")
    
    required_packages = [
        "paramiko>=3.4.0",
        "keyring>=24.0.0",
    ]
    
    # Platform-specific packages
    if platform.system() == "Windows":
        required_packages.extend([
            "pywin32>=306",
            "winshell>=0.6"
        ])
    
    failed_packages = []
    
    for package in required_packages:
        package_name = package.split(">=")[0].split("==")[0]
        print(f"Checking {package_name}...")
        
        if not check_package(package_name):
            print(f"Installing {package}...")
            if install_package(package):
                print(f"✓ {package_name} installed successfully")
            else:
                print(f"✗ Failed to install {package_name}")
                failed_packages.append(package)
        else:
            print(f"✓ {package_name} already installed")
    
    if failed_packages:
        print("\nFailed to install the following packages:")
        for package in failed_packages:
            print(f"  - {package}")
        print("\nPlease install these manually using:")
        for package in failed_packages:
            print(f"  pip install {package}")
        return False
    
    print("\n✓ All remote dependencies installed successfully!")
    return True


def setup_x11_server_windows():
    """Provide instructions for setting up X11 server on Windows"""
    print("\nX11 Server Setup for Windows:")
    print("=" * 40)
    print("For remote GUI functionality, an X11 server is required.")
    print("\n🚀 AUTOMATIC INSTALLATION AVAILABLE!")
    print("Vestim Remote can automatically install VcXsrv for you.")
    print("Just click 'Install & Start X11' in the configuration dialog.")
    print("\nSupported automatic installation methods:")
    print("• Chocolatey (if installed): choco install vcxsrv")
    print("• winget (Windows 10+): winget install VcXsrv.VcXsrv") 
    print("• Scoop (if installed): scoop install vcxsrv")
    print("• Direct download from SourceForge")
    print("\nManual installation options:")
    print("1. VcXsrv (Recommended):")
    print("   - Download from: https://sourceforge.net/projects/vcxsrv/")
    print("   - Install with default settings")
    print("\n2. Xming (Alternative):")
    print("   - Download from: https://sourceforge.net/projects/xming/")
    print("\n3. WSL2 with WSLg (Windows 11):")
    print("   - Built-in X11 support: wsl --install")


def setup_x11_server_macos():
    """Provide instructions for setting up X11 server on macOS"""
    print("\nX11 Server Setup for macOS:")
    print("=" * 40)
    print("For remote GUI functionality, XQuartz is required.")
    print("\n🚀 AUTOMATIC INSTALLATION AVAILABLE!")
    print("Vestim Remote can automatically install XQuartz for you.")
    print("Just click 'Install & Start X11' in the configuration dialog.")
    print("\nSupported automatic installation methods:")
    print("• Homebrew (if installed): brew install --cask xquartz")
    print("• Direct download from XQuartz.org")
    print("\nManual installation:")
    print("1. Download XQuartz from: https://www.xquartz.org/")
    print("2. Install the .dmg file")
    print("3. Log out and log back in")
    print("\nAfter installation:")
    print("XQuartz will be auto-detected by Vestim Remote")


def setup_ssh_client():
    """Provide instructions for SSH client setup"""
    print("\nSSH Client Setup:")
    print("=" * 20)
    
    if platform.system() == "Windows":
        print("Windows 10/11 includes OpenSSH client by default.")
        print("If SSH is not available, enable it via:")
        print("  Settings > Apps > Optional Features > Add Feature > OpenSSH Client")
    elif platform.system() == "Darwin":
        print("macOS includes SSH client by default.")
    else:
        print("Most Linux distributions include SSH client by default.")
        print("If not installed, use your package manager:")
        print("  Ubuntu/Debian: sudo apt install openssh-client")
        print("  CentOS/RHEL: sudo yum install openssh-clients")
        print("  Fedora: sudo dnf install openssh-clients")


def create_example_config():
    """Create an example remote configuration"""
    config_dir = Path.home() / ".vestim" / "remote"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    example_config = {
        "host": "your-server.example.com",
        "port": 22,
        "username": "your-username",
        "display": ":0",
        "remote_vestim_path": "/opt/vestim",
        "project_sync_enabled": True,
        "compression": True,
        "timeout": 30
    }
    
    example_file = config_dir / "example_config.json"
    
    import json
    with open(example_file, 'w') as f:
        json.dump(example_config, f, indent=2)
    
    print(f"\nExample configuration created at: {example_file}")
    print("Edit this file with your server details, then rename to ssh_config.json")


def main():
    """Main setup function"""
    print("Vestim Remote Setup")
    print("=" * 50)
    
    # Install Python dependencies
    if not install_remote_dependencies():
        print("\nSetup failed due to missing dependencies.")
        sys.exit(1)
    
    # Platform-specific X11 setup instructions
    if platform.system() == "Windows":
        setup_x11_server_windows()
    elif platform.system() == "Darwin":
        setup_x11_server_macos()
    else:
        print("\nLinux detected - X11 should be available by default.")
    
    # SSH client setup
    setup_ssh_client()
    
    # Create example configuration
    create_example_config()
    
    print("\n" + "=" * 50)
    print("Setup Complete!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Set up your X11 server (if on Windows/macOS)")
    print("2. Configure your remote server details using the GUI:")
    print("   - Run 'python vestim/remote/launcher.py --gui'")
    print("   - Or use the 'Vestim Remote' desktop shortcut")
    print("3. Test the connection before first use")
    print("\nFor help, see the documentation or run with --help")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ["--help", "-h"]:
        print(__doc__)
        print("\nUsage: python setup_remote.py")
        print("\nThis script will:")
        print("- Install required Python packages")
        print("- Provide X11 server setup instructions")
        print("- Create example configuration files")
        sys.exit(0)
    
    main()
