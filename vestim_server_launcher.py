"""
Vestim Server Client - Simplified standalone version
This creates an executable that can run independently
"""

import sys
import os
import subprocess
import platform

def find_python():
    """Find Python executable."""
    # Try common Python paths
    python_paths = [
        "python",
        "python3", 
        "py",
        sys.executable
    ]
    
    for python_cmd in python_paths:
        try:
            result = subprocess.run([python_cmd, "--version"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                return python_cmd
        except FileNotFoundError:
            continue
    
    return None

def main():
    """Main entry point for standalone client."""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the main server client script
    client_script = os.path.join(script_dir, "vestim_server_client.py")
    
    if not os.path.exists(client_script):
        print(f"Error: Could not find vestim_server_client.py in {script_dir}")
        input("Press Enter to exit...")
        return 1
    
    # Find Python
    python_cmd = find_python()
    if not python_cmd:
        print("Error: Could not find Python installation.")
        print("Please ensure Python is installed and available in PATH.")
        input("Press Enter to exit...")
        return 1
    
    try:
        # Run the main client script
        print("Starting Vestim Server Client...")
        os.chdir(script_dir)
        subprocess.run([python_cmd, client_script] + sys.argv[1:])
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        input("Press Enter to exit...")
        return 1

if __name__ == "__main__":
    sys.exit(main())
