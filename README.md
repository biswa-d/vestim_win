# Vestim - Voltage Estimation Tool for Lithium-ion Batteries

Vestim is a comprehensive tool for voltage estimation in lithium-ion batteries, featuring a user-friendly GUI for data import, processing, training, and testing of machine learning models.

## Features

- **Multi-format Data Import**: Support for Arbin, STLA, and Digatron data formats
- **Data Processing & Augmentation**: Advanced data preprocessing and augmentation capabilities
- **Machine Learning Models**: Training and testing of voltage estimation models
- **User-friendly GUI**: PyQt5-based graphical interface
- **Cross-platform**: Works on Windows, Linux, and macOS

## Installation

### Option 1: Pre-built Installer (Recommended for End Users)

1. Download the appropriate installer for your platform:
   - **Windows**: `vestim_installer.zip`
   - **Linux/macOS**: `vestim_installer.tar.gz`

2. Extract the downloaded archive

3. Run the installer:
   - **Windows**: Double-click `install_windows.bat` or run `python install.py`
   - **Linux/macOS**: Run `bash install_unix.sh` or `python install.py`

4. After installation, use the desktop shortcut or run the launcher script:
   - **Windows**: `scripts\vestim.bat`
   - **Linux/macOS**: `scripts/vestim.sh`

### Option 2: Vestim Server Client (Remote Access)

For users who need to access Vestim running on a remote server:

1. Install Vestim normally (Option 1)
2. During installation, check "Run server setup wizard after installation"
3. Configure your remote server details (hostname, credentials, Vestim path)
4. Use the "Vestim Server" shortcut for one-click remote access

This enables seamless access to Vestim running on powerful remote servers with GUI display on your local machine. See [VESTIM_SERVER_CLIENT.md](VESTIM_SERVER_CLIENT.md) for detailed setup instructions.

### Option 3: Manual Installation (For Developers)

```bash
# Clone the repository
git clone <repository-url>
cd vestim_micros

# Create virtual environment
python -m venv vestim_env

# Activate virtual environment
# On Windows:
vestim_env\Scripts\activate
# On Linux/macOS:
source vestim_env/bin/activate

# Install dependencies
pip install -e .

# Run the application
python -m vestim.gui.src.data_import_gui_qt
```

### Option 3: Using pip (If published to PyPI)

```bash
pip install vestim
vestim  # Launch the GUI
```

## Usage

1. **Data Import**: Start with the data import GUI to select and process your battery data files
2. **Data Augmentation**: Use the data augmentation tools to enhance your dataset
3. **Model Training**: Configure and train machine learning models for voltage estimation
4. **Model Testing**: Evaluate model performance on test datasets

## Supported Data Formats

- **Arbin**: `.mat` files from Arbin battery testing systems
- **STLA**: Stellantis battery data format
- **Digatron**: `.csv` files from Digatron battery testers

## Requirements

- Python 3.8 or later
- PyQt5
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- PyTorch
- SciPy

## Project Structure

```
vestim/
├── gui/                    # GUI components
│   ├── src/               # GUI source files
│   └── resources/         # Icons and resources
├── services/              # Core services
│   ├── data_import/       # Data import functionality
│   ├── data_processor/    # Data processing services
│   ├── model_training/    # ML model training
│   └── model_testing/     # Model evaluation
└── gateway/               # API gateway components
```

## Building Distribution

To create a distributable package:

```bash
# Install build dependencies
pip install build

# Run the build script
python build_dist.py

# Or on Windows, simply run:
build.bat
```

This creates:
- `vestim_installer.zip` for Windows users
- `vestim_installer.tar.gz` for Linux/macOS users
- Python packages in the `dist/` directory

## Development

### Setting up Development Environment

1. Clone the repository
2. Create and activate a virtual environment
3. Install in development mode: `pip install -e .`
4. Install development dependencies: `pip install -e .[dev]`

### Running Tests

```bash
pytest
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run the test suite
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Biswanath Dehury**  
Email: dehuryb@mcmaster.ca

## Support

For questions, issues, or support, please contact the author or create an issue in the repository.

## Changelog

### Version 1.0.0
- Initial release
- Multi-format data import support
- GUI-based data processing
- Machine learning model training and testing
- Cross-platform installation support
