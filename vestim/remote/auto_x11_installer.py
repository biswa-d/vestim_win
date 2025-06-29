"""
Enhanced X11 Server Manager with Automatic Installation
Handles detection, download, and installation of X11 servers
"""

import os
import sys
import platform
import subprocess
import tempfile
import urllib.request
import zipfile
import shutil
import winreg
from pathlib import Path
from typing import Tuple, Optional


class AutoX11ServerManager:
    """Enhanced X11 server manager with automatic installation capabilities"""
    
    def __init__(self):
        self.platform = platform.system()
        self.downloads_dir = Path.home() / "Downloads" / "VestimX11"
        self.downloads_dir.mkdir(parents=True, exist_ok=True)
    
    def ensure_x11_available(self) -> Tuple[bool, str]:
        """Ensure X11 server is available, installing if necessary"""
        print("Checking X11 server availability...")
        
        if self.is_x11_running():
            return True, "X11 server is already running"
        
        if self.has_x11_installed():
            print("X11 server found but not running. Starting...")
            return self.start_x11_server()
        
        print("No X11 server found. Beginning automatic installation...")
        return self.auto_install_x11()
    
    def is_x11_running(self) -> bool:
        """Check if X11 server is currently running"""
        if self.platform == "Windows":
            return self._check_vcxsrv_running() or self._check_xming_running()
        elif self.platform == "Darwin":
            return self._check_xquartz_running()
        else:
            return True  # Linux typically has X11
    
    def has_x11_installed(self) -> bool:
        """Check if any X11 server is installed"""
        if self.platform == "Windows":
            return self._find_vcxsrv_path() or self._find_xming_path() or self._check_wsl_x11()
        elif self.platform == "Darwin":
            return self._find_xquartz_path() is not None
        else:
            return True  # Linux typically has X11
    
    def auto_install_x11(self) -> Tuple[bool, str]:
        """Automatically download and install appropriate X11 server"""
        if self.platform == "Windows":
            return self._auto_install_windows_x11()
        elif self.platform == "Darwin":
            return self._auto_install_macos_x11()
        else:
            return True, "Linux X11 should be available by default"
    
    def _auto_install_windows_x11(self) -> Tuple[bool, str]:
        """Auto-install X11 server on Windows"""
        print("Installing VcXsrv for Windows...")
        
        # Option 1: Try Chocolatey if available
        if self._has_chocolatey():
            print("Detected Chocolatey. Installing VcXsrv via Chocolatey...")
            try:
                result = subprocess.run(
                    ["choco", "install", "vcxsrv", "-y"],
                    capture_output=True, text=True, timeout=300
                )
                if result.returncode == 0:
                    return True, "VcXsrv installed successfully via Chocolatey"
            except Exception as e:
                print(f"Chocolatey installation failed: {e}")
        
        # Option 2: Try winget if available (Windows 10+)
        if self._has_winget():
            print("Detected winget. Installing VcXsrv via winget...")
            try:
                result = subprocess.run(
                    ["winget", "install", "VcXsrv.VcXsrv"],
                    capture_output=True, text=True, timeout=300
                )
                if result.returncode == 0:
                    return True, "VcXsrv installed successfully via winget"
            except Exception as e:
                print(f"winget installation failed: {e}")
        
        # Option 3: Try scoop if available
        if self._has_scoop():
            print("Detected Scoop. Installing VcXsrv via Scoop...")
            try:
                result = subprocess.run(
                    ["scoop", "install", "vcxsrv"],
                    capture_output=True, text=True, timeout=300
                )
                if result.returncode == 0:
                    return True, "VcXsrv installed successfully via Scoop"
            except Exception as e:
                print(f"Scoop installation failed: {e}")
        
        # Option 4: Direct download and install
        print("Package managers not available. Downloading VcXsrv directly...")
        return self._download_and_install_vcxsrv()
    
    def _download_and_install_vcxsrv(self) -> Tuple[bool, str]:
        """Download and install VcXsrv directly"""
        try:
            # VcXsrv download URL (latest release)
            vcxsrv_url = "https://sourceforge.net/projects/vcxsrv/files/latest/download"
            
            print("Downloading VcXsrv...")
            installer_path = self.downloads_dir / "vcxsrv-installer.exe"
            
            # Download with progress
            urllib.request.urlretrieve(vcxsrv_url, installer_path)
            print(f"Downloaded VcXsrv to {installer_path}")
            
            # Run installer silently
            print("Running VcXsrv installer...")
            result = subprocess.run([
                str(installer_path), 
                "/S",  # Silent install
                "/D=C:\\Program Files\\VcXsrv"  # Install directory
            ], timeout=300)
            
            if result.returncode == 0:
                # Clean up
                installer_path.unlink(missing_ok=True)
                return True, "VcXsrv installed successfully"
            else:
                return False, "VcXsrv installation failed"
                
        except Exception as e:
            return False, f"Failed to download/install VcXsrv: {str(e)}"
    
    def _auto_install_macos_x11(self) -> Tuple[bool, str]:
        """Auto-install XQuartz on macOS"""
        print("Installing XQuartz for macOS...")
        
        # Option 1: Try Homebrew if available
        if self._has_homebrew():
            print("Detected Homebrew. Installing XQuartz via Homebrew...")
            try:
                result = subprocess.run(
                    ["brew", "install", "--cask", "xquartz"],
                    capture_output=True, text=True, timeout=300
                )
                if result.returncode == 0:
                    return True, "XQuartz installed successfully via Homebrew"
            except Exception as e:
                print(f"Homebrew installation failed: {e}")
        
        # Option 2: Direct download
        print("Downloading XQuartz directly...")
        return self._download_and_install_xquartz()
    
    def _download_and_install_xquartz(self) -> Tuple[bool, str]:
        """Download and install XQuartz directly"""
        try:
            # XQuartz download URL
            xquartz_url = "https://github.com/XQuartz/XQuartz/releases/latest/download/XQuartz-2.8.5.dmg"
            
            print("Downloading XQuartz...")
            dmg_path = self.downloads_dir / "XQuartz.dmg"
            
            urllib.request.urlretrieve(xquartz_url, dmg_path)
            print(f"Downloaded XQuartz to {dmg_path}")
            
            # Mount DMG and install
            print("Installing XQuartz...")
            subprocess.run(["hdiutil", "attach", str(dmg_path)])
            subprocess.run(["sudo", "installer", "-pkg", "/Volumes/XQuartz*/XQuartz.pkg", "-target", "/"])
            subprocess.run(["hdiutil", "detach", "/Volumes/XQuartz*"])
            
            # Clean up
            dmg_path.unlink(missing_ok=True)
            return True, "XQuartz installed successfully"
            
        except Exception as e:
            return False, f"Failed to download/install XQuartz: {str(e)}"
    
    # Package manager detection methods
    def _has_chocolatey(self) -> bool:
        """Check if Chocolatey is installed"""
        try:
            subprocess.run(["choco", "--version"], capture_output=True, timeout=5)
            return True
        except:
            return False
    
    def _has_winget(self) -> bool:
        """Check if winget is available"""
        try:
            subprocess.run(["winget", "--version"], capture_output=True, timeout=5)
            return True
        except:
            return False
    
    def _has_scoop(self) -> bool:
        """Check if Scoop is installed"""
        try:
            subprocess.run(["scoop", "--version"], capture_output=True, timeout=5)
            return True
        except:
            return False
    
    def _has_homebrew(self) -> bool:
        """Check if Homebrew is installed"""
        try:
            subprocess.run(["brew", "--version"], capture_output=True, timeout=5)
            return True
        except:
            return False
    
    # Detection methods
    def _check_vcxsrv_running(self) -> bool:
        """Check if VcXsrv is running"""
        try:
            result = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq vcxsrv.exe"],
                capture_output=True, text=True
            )
            return "vcxsrv.exe" in result.stdout
        except:
            return False
    
    def _check_xming_running(self) -> bool:
        """Check if Xming is running"""
        try:
            result = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq Xming.exe"],
                capture_output=True, text=True
            )
            return "Xming.exe" in result.stdout
        except:
            return False
    
    def _check_xquartz_running(self) -> bool:
        """Check if XQuartz is running"""
        try:
            result = subprocess.run(
                ["pgrep", "-f", "XQuartz"],
                capture_output=True, text=True
            )
            return result.returncode == 0
        except:
            return False
    
    def _find_vcxsrv_path(self) -> Optional[str]:
        """Find VcXsrv installation path"""
        possible_paths = [
            r"C:\Program Files\VcXsrv\vcxsrv.exe",
            r"C:\Program Files (x86)\VcXsrv\vcxsrv.exe",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # Check registry
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\VcXsrv") as key:
                install_path = winreg.QueryValueEx(key, "InstallLocation")[0]
                vcxsrv_exe = os.path.join(install_path, "vcxsrv.exe")
                if os.path.exists(vcxsrv_exe):
                    return vcxsrv_exe
        except:
            pass
        
        return None
    
    def _find_xming_path(self) -> Optional[str]:
        """Find Xming installation path"""
        possible_paths = [
            r"C:\Program Files\Xming\Xming.exe",
            r"C:\Program Files (x86)\Xming\Xming.exe",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def _find_xquartz_path(self) -> Optional[str]:
        """Find XQuartz installation"""
        xquartz_paths = [
            "/Applications/Utilities/XQuartz.app",
            "/opt/X11/bin/X"
        ]
        
        for path in xquartz_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def _check_wsl_x11(self) -> bool:
        """Check if WSL with X11 forwarding is available"""
        try:
            result = subprocess.run(
                ["wsl", "--list", "--quiet"],
                capture_output=True, text=True
            )
            return result.returncode == 0 and result.stdout.strip()
        except:
            return False
    
    def start_x11_server(self) -> Tuple[bool, str]:
        """Start the installed X11 server"""
        if self.platform == "Windows":
            return self._start_windows_x11()
        elif self.platform == "Darwin":
            return self._start_xquartz()
        else:
            return True, "Linux X11 should start automatically"
    
    def _start_windows_x11(self) -> Tuple[bool, str]:
        """Start X11 server on Windows"""
        # Try VcXsrv first
        vcxsrv_path = self._find_vcxsrv_path()
        if vcxsrv_path:
            try:
                subprocess.Popen([vcxsrv_path, ":0", "-ac", "-terminate"])
                return True, "VcXsrv started successfully"
            except Exception as e:
                print(f"Failed to start VcXsrv: {e}")
        
        # Try Xming
        xming_path = self._find_xming_path()
        if xming_path:
            try:
                subprocess.Popen([xming_path, ":0", "-ac"])
                return True, "Xming started successfully"
            except Exception as e:
                print(f"Failed to start Xming: {e}")
        
        # Try WSL
        if self._check_wsl_x11():
            return True, "WSL X11 forwarding available"
        
        return False, "No X11 server could be started"
    
    def _start_xquartz(self) -> Tuple[bool, str]:
        """Start XQuartz on macOS"""
        try:
            subprocess.Popen(["open", "-a", "XQuartz"])
            return True, "XQuartz started successfully"
        except Exception as e:
            return False, f"Failed to start XQuartz: {str(e)}"


def get_x11_installation_guide() -> str:
    """Get platform-specific installation guide if auto-install fails"""
    system = platform.system()
    
    if system == "Windows":
        return """
X11 Server Installation Guide for Windows:

Automatic installation failed. Please install manually:

Option 1 - VcXsrv (Recommended):
1. Download from: https://sourceforge.net/projects/vcxsrv/
2. Run installer with default settings
3. After installation, Vestim will auto-detect and start it

Option 2 - Package Manager:
• Chocolatey: choco install vcxsrv
• winget: winget install VcXsrv.VcXsrv  
• Scoop: scoop install vcxsrv

Option 3 - WSL2 with WSLg (Windows 11):
• Run: wsl --install
• Built-in X11 support
        """
    
    elif system == "Darwin":
        return """
X11 Server Installation Guide for macOS:

Automatic installation failed. Please install manually:

Option 1 - XQuartz:
1. Download from: https://www.xquartz.org/
2. Install the .dmg file
3. Log out and log back in

Option 2 - Homebrew:
• Run: brew install --cask xquartz
        """
    
    else:
        return """
X11 Installation Guide for Linux:

X11 should be pre-installed. If not available:

Ubuntu/Debian: sudo apt install xorg
CentOS/RHEL: sudo yum install xorg-x11-server-Xorg  
Fedora: sudo dnf install xorg-x11-server-Xorg
        """
