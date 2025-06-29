#!/usr/bin/env python3
"""
Build script for Vestim with Remote functionality
Handles packaging with remote SSH capabilities
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def build_with_remote():
    """Build Vestim executable with remote functionality included"""
    print("Building Vestim with Remote functionality...")
    
    # Ensure remote dependencies are available
    print("Installing remote dependencies...")
    remote_deps = [
        "paramiko>=3.4.0",
        "keyring>=24.0.0",
        "pywin32>=306; sys_platform == 'win32'",
        "winshell>=0.6; sys_platform == 'win32'"
    ]
    
    for dep in remote_deps:
        subprocess.run([sys.executable, "-m", "pip", "install", dep], check=False)
    
    # Check if PyInstaller spec exists
    spec_file = "Vestim.spec"
    if not os.path.exists(spec_file):
        print("Creating PyInstaller spec file...")
        create_pyinstaller_spec()
    else:
        print("Updating existing PyInstaller spec...")
        update_pyinstaller_spec(spec_file)
    
    # Run PyInstaller
    print("Running PyInstaller...")
    result = subprocess.run([
        sys.executable, "-m", "PyInstaller", 
        "--clean", 
        spec_file
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ Build successful!")
        print("Executable created in dist/ directory")
        
        # Copy remote files to dist
        copy_remote_files()
        
        # Create installer
        create_installer()
        
    else:
        print("✗ Build failed!")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return False
    
    return True


def create_pyinstaller_spec():
    """Create PyInstaller spec file with remote functionality"""
    spec_content = '''# -*- mode: python ; coding: utf-8 -*-

import sys
import os
from pathlib import Path

# Add vestim package to analysis paths
vestim_path = str(Path.cwd())
sys.path.insert(0, vestim_path)

block_cipher = None

# Collect all vestim modules including remote
vestim_modules = []
for root, dirs, files in os.walk('vestim'):
    for file in files:
        if file.endswith('.py') and file != '__init__.py':
            module_path = os.path.relpath(os.path.join(root, file), '.')
            module_name = module_path.replace(os.sep, '.').replace('.py', '')
            vestim_modules.append(module_name)

a = Analysis(
    ['vestim_main.py'],  # Main entry point
    pathex=[vestim_path],
    binaries=[],
    datas=[
        ('vestim/gui/resources', 'vestim/gui/resources'),
        ('vestim/remote', 'vestim/remote'),
        ('hyperparams.json', '.'),
        ('README.md', '.'),
        ('LICENSE', '.'),
        ('REMOTE_SETUP_GUIDE.md', '.'),
    ],
    hiddenimports=[
        # Core vestim imports
        'vestim.gui.src.training_task_gui_qt',
        'vestim.gui.src.testing_gui_qt',
        'vestim.gui.src.data_import_gui_qt',
        'vestim.gui.src.data_augment_gui_qt',
        'vestim.services.model_training.src.LSTM_model_service_test',
        'vestim.services.model_testing.src.continuous_testing_service',
        
        # Remote functionality imports
        'vestim.remote.ssh_manager',
        'vestim.remote.config_dialog', 
        'vestim.remote.launcher',
        'vestim.remote.gui_integration',
        
        # External dependencies
        'paramiko',
        'keyring',
        'keyring.backends',
        'keyring.backends.Windows',
        'keyring.backends.SecretService',
        'keyring.backends.macOS',
    ] + vestim_modules,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Vestim',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # GUI application
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='vestim/gui/resources/icon.ico'
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Vestim',
)
'''
    
    with open("Vestim.spec", 'w') as f:
        f.write(spec_content)
    
    print("PyInstaller spec file created")


def update_pyinstaller_spec(spec_file):
    """Update existing spec file to include remote functionality"""
    print("Updating PyInstaller spec file for remote functionality...")
    
    # Read existing spec
    with open(spec_file, 'r') as f:
        content = f.read()
    
    # Add remote imports if not present
    remote_imports = [
        "'vestim.remote.ssh_manager'",
        "'vestim.remote.config_dialog'", 
        "'vestim.remote.launcher'",
        "'vestim.remote.gui_integration'",
        "'paramiko'",
        "'keyring'",
    ]
    
    for import_name in remote_imports:
        if import_name not in content:
            # Add to hiddenimports list
            if "hiddenimports=[" in content:
                content = content.replace(
                    "hiddenimports=[",
                    f"hiddenimports=[\n        {import_name},"
                )
    
    # Add remote data files if not present
    if "'vestim/remote', 'vestim/remote'" not in content:
        if "datas=[" in content:
            content = content.replace(
                "datas=[",
                "datas=[\n        ('vestim/remote', 'vestim/remote'),"
            )
    
    # Write updated spec
    with open(spec_file, 'w') as f:
        f.write(content)


def copy_remote_files():
    """Copy remote files to dist directory"""
    print("Copying remote files to distribution...")
    
    dist_dir = Path("dist/Vestim")
    remote_source = Path("vestim/remote")
    remote_dest = dist_dir / "vestim" / "remote"
    
    if remote_source.exists():
        shutil.copytree(remote_source, remote_dest, dirs_exist_ok=True)
        print(f"✓ Remote files copied to {remote_dest}")
    
    # Copy setup script
    if Path("setup_remote.py").exists():
        shutil.copy2("setup_remote.py", dist_dir / "setup_remote.py")
        print("✓ Remote setup script copied")
    
    # Copy documentation
    if Path("REMOTE_SETUP_GUIDE.md").exists():
        shutil.copy2("REMOTE_SETUP_GUIDE.md", dist_dir / "REMOTE_SETUP_GUIDE.md")
        print("✓ Remote documentation copied")


def create_installer():
    """Create installer with remote functionality"""
    print("Creating installer with remote functionality...")
    
    # Check for Inno Setup
    inno_setup_path = None
    possible_paths = [
        r"C:\\Program Files (x86)\\Inno Setup 6\\ISCC.exe",
        r"C:\\Program Files\\Inno Setup 6\\ISCC.exe",
        r"C:\\Program Files (x86)\\Inno Setup 5\\ISCC.exe",
        r"C:\\Program Files\\Inno Setup 5\\ISCC.exe",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            inno_setup_path = path
            break
    
    if inno_setup_path:
        print(f"Found Inno Setup at: {inno_setup_path}")
        result = subprocess.run([inno_setup_path, "vestim_installer.iss"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Installer created successfully!")
            print("Check installer_output/ directory for the installer")
        else:
            print("✗ Installer creation failed")
            print("Error:", result.stderr)
    else:
        print("Inno Setup not found. Creating portable ZIP instead...")
        create_portable_zip()


def create_portable_zip():
    """Create portable ZIP distribution"""
    import zipfile
    
    print("Creating portable ZIP distribution...")
    
    zip_path = "installer_output/vestim-portable-with-remote.zip"
    os.makedirs("installer_output", exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add all files from dist/Vestim
        dist_path = Path("dist/Vestim")
        for file_path in dist_path.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(dist_path)
                zipf.write(file_path, arcname)
    
    print(f"✓ Portable ZIP created: {zip_path}")


def main():
    """Main build function"""
    print("Vestim Build with Remote Functionality")
    print("=" * 50)
    
    if not build_with_remote():
        print("Build failed!")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("Build completed successfully!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Test the executable in dist/Vestim/")
    print("2. Run setup_remote.py to configure remote functionality")
    print("3. Use the installer from installer_output/ for distribution")
    print("\nFor remote setup help, see REMOTE_SETUP_GUIDE.md")


if __name__ == "__main__":
    main()
