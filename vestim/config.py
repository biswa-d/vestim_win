import os

# Define the root of your project
# For PyInstaller compatibility, use current working directory instead of __file__ location
def get_root_dir():
    """Get the root directory, using current working directory for PyInstaller compatibility"""
    return os.getcwd()

def get_output_dir():
    """Get the output directory, dynamically based on current working directory"""
    return os.path.join(get_root_dir(), 'output')

# For backward compatibility, provide the variables
ROOT_DIR = get_root_dir()
OUTPUT_DIR = get_output_dir()

