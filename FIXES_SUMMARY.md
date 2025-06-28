# Vestim Standalone Executable Fixes

## Issues Fixed

### 1. **Multiprocessing Issue - Multiple GUI Instances**
- **Problem**: When training started, multiple GUI instances launched due to PyTorch DataLoader multiprocessing
- **Root Cause**: PyInstaller + DataLoader `num_workers > 0` causes worker processes to re-execute main script
- **Fix**: 
  - Added `multiprocessing.freeze_support()` to main entry point
  - Auto-disable DataLoader multiprocessing (`num_workers=0`) when running from PyInstaller
  - Added multiprocessing guards to all GUI entry points

### 2. **Output Directory Issue - Files in Temp Location**
- **Problem**: Job folders created in PyInstaller temp directory (`_MEI...`) instead of user-accessible location
- **Root Cause**: Static `OUTPUT_DIR` calculated at import time using temp directory
- **Fix**:
  - Made `OUTPUT_DIR` dynamic with `get_output_dir()` function
  - Working directory now set to `[exe_location]/Vestim_Projects/`
  - All outputs now go to: `Vestim_Projects/output/job_YYYYMMDD-HHMMSS/`

### 3. **Logging Chaos - Scattered Log Files**
- **Problem**: Multiple log files created in launch directory (`data_augment_service.log`, etc.)
- **Root Cause**: Modules creating individual loggers before working directory set
- **Fix**:
  - Centralized logging to use standard logger hierarchy
  - Job-specific logs in each job folder: `job_YYYYMMDD-HHMMSS/logs/`
  - Main app log in: `Vestim_Projects/logs/vestim.log`

### 4. **Testing/Denormalization Error - Scale Mismatch**
- **Problem**: 
  ```
  Pred: [0.884836...] (normalized 0-1)
  True: [-1.3594128...] (already denormalized voltage)
  IndexError: index 2 is out of bounds for axis 0 with size 1
  ```
- **Root Cause**: 
  - Processed test files have denormalized true values but normalized predictions expected
  - Scaler dimension mismatch (trying to access feature index that doesn't exist)
- **Fix**:
  - Smart range detection: Check if values are actually normalized (0-1 range)
  - Selective denormalization: Only denormalize what's actually normalized
  - Robust scaler index handling with bounds checking and fallbacks

## Current Directory Structure

When running `Vestim.exe`:
```
[Directory where Vestim.exe is located]/
├── Vestim.exe
└── Vestim_Projects/                    # Auto-created project directory
    ├── logs/
    │   └── vestim.log                 # Main application log
    └── output/                        # All job outputs
        └── job_20250627-184037/       # Individual job folder
            ├── train_data/
            ├── test_data/
            ├── models/
            ├── logs/                  # Job-specific logs
            └── results/
```

## Key Files Modified

1. **vestim/gui/src/data_import_gui_qt.py** - Main entry point with multiprocessing guard and working directory setup
2. **vestim/services/model_training/src/data_loader_service.py** - Disabled multiprocessing for PyInstaller
3. **vestim/config.py** - Made OUTPUT_DIR dynamic
4. **vestim/gateway/src/job_manager_qt.py** - Updated to use dynamic output directory
5. **vestim/services/model_testing/src/continuous_testing_service.py** - Fixed denormalization logic
6. **Multiple logger files** - Removed individual log file creation

## Testing Checklist

- [ ] Build executable: `python build_exe.py`
- [ ] Copy `Vestim.exe` to clean directory 
- [ ] Run executable - should create `Vestim_Projects/` folder
- [ ] Complete full workflow: Data Import → Training → Testing
- [ ] Verify no multiple GUI instances during training
- [ ] Check job folder created in `Vestim_Projects/output/`
- [ ] Verify logs in job-specific folders
- [ ] Test denormalization works correctly (voltage values in proper range)
