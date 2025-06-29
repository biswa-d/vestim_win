#!/bin/bash
# Vestim Server Environment Setup Script
# Run this on the remote server to set up Vestim for remote GUI access

set -e  # Exit on any error

echo "=== Vestim Server Environment Setup ==="
echo "This script will install Vestim and its dependencies on the remote server."
echo

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo "Warning: Running as root. Consider running as a regular user instead."
   read -p "Continue anyway? (y/N): " -n 1 -r
   echo
   if [[ ! $REPLY =~ ^[Yy]$ ]]; then
       exit 1
   fi
fi

# Detect OS
if [[ -f /etc/os-release ]]; then
    . /etc/os-release
    OS=$NAME
    OS_VERSION=$VERSION_ID
else
    echo "Cannot detect OS. This script supports Ubuntu/Debian and CentOS/RHEL."
    exit 1
fi

echo "Detected OS: $OS $OS_VERSION"

# Install system dependencies
echo "Installing system dependencies..."
if [[ $OS == *"Ubuntu"* ]] || [[ $OS == *"Debian"* ]]; then
    sudo apt update
    sudo apt install -y \
        python3 python3-pip python3-venv \
        libx11-dev libxext-dev libxrender-dev libxtst-dev \
        git openssh-server \
        build-essential
        
elif [[ $OS == *"CentOS"* ]] || [[ $OS == *"Red Hat"* ]]; then
    sudo yum update -y
    sudo yum install -y \
        python3 python3-pip \
        libX11-devel libXext-devel libXrender-devel libXtst-devel \
        git openssh-server \
        gcc gcc-c++ make
        
    # Install python3-venv on CentOS
    sudo yum install -y python3-devel
else
    echo "Unsupported OS: $OS"
    echo "Please install dependencies manually:"
    echo "- Python 3.8+"
    echo "- X11 development libraries"
    echo "- Git and SSH server"
    exit 1
fi

# Configure SSH for X11 forwarding
echo "Configuring SSH for X11 forwarding..."
if ! grep -q "X11Forwarding yes" /etc/ssh/sshd_config; then
    echo "X11Forwarding yes" | sudo tee -a /etc/ssh/sshd_config
fi
if ! grep -q "X11UseLocalhost no" /etc/ssh/sshd_config; then
    echo "X11UseLocalhost no" | sudo tee -a /etc/ssh/sshd_config
fi

# Restart SSH service
sudo systemctl restart sshd
sudo systemctl enable sshd

# Choose installation directory
echo
echo "Choose Vestim installation directory:"
echo "1) User home directory: $HOME/vestim"
echo "2) System-wide: /opt/vestim (requires sudo)"
echo "3) Custom path"
read -p "Enter choice (1-3): " -n 1 -r
echo

case $REPLY in
    1)
        VESTIM_DIR="$HOME/vestim"
        ;;
    2)
        VESTIM_DIR="/opt/vestim"
        if [[ ! -w /opt ]]; then
            sudo mkdir -p $VESTIM_DIR
            sudo chown $USER:$USER $VESTIM_DIR
        fi
        ;;
    3)
        read -p "Enter custom path: " VESTIM_DIR
        mkdir -p $VESTIM_DIR
        ;;
    *)
        echo "Invalid choice. Using default: $HOME/vestim"
        VESTIM_DIR="$HOME/vestim"
        ;;
esac

echo "Installing Vestim to: $VESTIM_DIR"

# Clone or copy Vestim
if [[ -d "$VESTIM_DIR" ]]; then
    echo "Directory $VESTIM_DIR already exists."
    read -p "Remove and reinstall? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf $VESTIM_DIR
    else
        echo "Installation cancelled."
        exit 1
    fi
fi

mkdir -p $VESTIM_DIR
cd $VESTIM_DIR

# Option to clone from Git or copy from local
echo
echo "Vestim source code options:"
echo "1) Clone from Git repository"
echo "2) Copy from local directory"
read -p "Enter choice (1-2): " -n 1 -r
echo

case $REPLY in
    1)
        read -p "Enter Git repository URL: " GIT_URL
        git clone $GIT_URL .
        ;;
    2)
        read -p "Enter local Vestim directory path: " LOCAL_DIR
        if [[ -d "$LOCAL_DIR" ]]; then
            cp -r $LOCAL_DIR/* .
        else
            echo "Local directory not found: $LOCAL_DIR"
            exit 1
        fi
        ;;
    *)
        echo "Invalid choice."
        exit 1
        ;;
esac

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv vestim_env
source vestim_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
echo "Installing Python dependencies..."
if [[ -f requirements.txt ]]; then
    pip install -r requirements.txt
else
    echo "requirements.txt not found. Installing common dependencies..."
    pip install PyQt5 torch pandas numpy scikit-learn matplotlib joblib paramiko keyring
fi

# Install Vestim package
if [[ -f setup.py ]] || [[ -f pyproject.toml ]]; then
    pip install -e .
fi

# Test installation
echo "Testing Vestim installation..."
python -c "import vestim; print('✓ Vestim package imported successfully')" || {
    echo "✗ Failed to import Vestim package"
    exit 1
}

python -c "from PyQt5.QtWidgets import QApplication; print('✓ PyQt5 available')" || {
    echo "✗ PyQt5 not available"
    exit 1
}

# Create launcher script
echo "Creating launcher script..."
cat > vestim_launcher.sh << EOF
#!/bin/bash
# Vestim Remote Launcher Script
cd $VESTIM_DIR
source vestim_env/bin/activate
export DISPLAY=\${DISPLAY:-:0}
python -m vestim.gui.src.main_window "\$@"
EOF

chmod +x vestim_launcher.sh

# Create environment activation script
cat > activate_vestim.sh << EOF
#!/bin/bash
# Activate Vestim environment
cd $VESTIM_DIR
source vestim_env/bin/activate
echo "Vestim environment activated."
echo "Run: python -m vestim.gui.src.main_window"
EOF

chmod +x activate_vestim.sh

# Add to PATH if desired
echo
read -p "Add Vestim to system PATH? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "export PATH=\"$VESTIM_DIR:\$PATH\"" >> ~/.bashrc
    echo "Added to ~/.bashrc"
fi

# Final instructions
echo
echo "=== Installation Complete ==="
echo "Vestim installed to: $VESTIM_DIR"
echo
echo "To use Vestim:"
echo "1. Activate environment: source $VESTIM_DIR/activate_vestim.sh"
echo "2. Launch GUI: $VESTIM_DIR/vestim_launcher.sh"
echo
echo "For remote access:"
echo "- SSH server is configured for X11 forwarding"
echo "- Connect with: ssh -X username@$(hostname -I | awk '{print $1}')"
echo "- Run: $VESTIM_DIR/vestim_launcher.sh"
echo
echo "Test connection:"
echo "ssh -X $USER@localhost '$VESTIM_DIR/vestim_launcher.sh'"
echo

echo "Setup complete! The server is ready for remote Vestim connections."
