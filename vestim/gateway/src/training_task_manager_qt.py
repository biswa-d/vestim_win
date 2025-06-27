import time, os, sys, math, json # Added json
import csv
import sqlite3
import torch
from PyQt5.QtCore import QThread, pyqtSignal
from vestim.gateway.src.job_manager_qt import JobManager
from vestim.gateway.src.training_setup_manager_qt import VEstimTrainingSetupManager
from vestim.services.model_training.src.data_loader_service import DataLoaderService
from vestim.services.model_training.src.training_task_service import TrainingTaskService
import logging, wandb

def format_time(seconds):
    """Convert seconds to mm:ss format."""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

class TrainingTaskManager:
    def __init__(self, global_params=None):
        self.logger = logging.getLogger(__name__)
        self.job_manager = JobManager()
        self.data_loader_service = DataLoaderService()
        self.training_service = TrainingTaskService()
        self.training_setup_manager = VEstimTrainingSetupManager()
        self.current_task = None
        self.stop_requested = False
        self.global_params = global_params if global_params else {}
        self.loaded_scaler = None # For storing the loaded scaler
        self.scaler_metadata = {} # For storing normalization metadata (path, columns, target)
        
        # Determine device based on global_params or fallback
        selected_device_str = self.global_params.get('DEVICE_SELECTION', 'cuda:0')
        try:
            if selected_device_str.startswith("cuda") and not torch.cuda.is_available():
                self.logger.warning(f"CUDA device {selected_device_str} selected, but CUDA is not available. Falling back to CPU.")
                self.device = torch.device("cpu")
            elif selected_device_str.startswith("cuda"):
                # Attempt to use the specific CUDA device. torch.device will raise an error if invalid.
                self.device = torch.device(selected_device_str)
                if not torch.cuda.is_available() or torch.cuda.current_device() != self.device.index:
                    # This check is a bit redundant if torch.device(selected_device_str) worked,
                    # but good for an explicit log if a specific CUDA device isn't the one torch ends up using.
                    # More robust check would be to try a small tensor operation on that device.
                    # For now, we assume torch.device handles the validation.
                    pass # self.logger.info(f"Successfully set device to {selected_device_str}")
            elif selected_device_str == "CPU":
                self.device = torch.device("cpu")
            else: # Default fallback if string is unrecognized
                self.logger.warning(f"Unrecognized device selection '{selected_device_str}'. Falling back to cuda:0 if available, else CPU.")
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        except Exception as e:
            self.logger.error(f"Error setting device to '{selected_device_str}': {e}. Falling back to CPU.")
            self.device = torch.device("cpu")
        
        self.logger.info(f"TrainingTaskManager initialized with device: {self.device}")

        self.training_thread = None  # Initialize the training thread here for PyQt
       
        # WandB setup (optional)
        self.use_wandb = False  # Set to False to disable WandB
        self.wandb_enabled = False
        if self.use_wandb:
            try:
                import wandb
                wandb.init(project="VEstim", config={"task_name": "LSTM Model Training"})
                self.wandb_enabled = True
                self.logger.info("WandB initialized successfully.")
            except Exception as e:
                self.wandb_enabled = False
                self.logger.error(f"Failed to initialize WandB: {e}")

    def log_to_csv(self, task, epoch, train_loss, val_loss, elapsed_time, current_lr, best_val_loss, delta_t_epoch):
        """Log richer data to CSV file."""
        csv_log_file = task['csv_log_file']  # Fetch the csv log file path from the task
        with open(csv_log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['Epoch', 'Train Loss', 'Val Loss', 'Elapsed Time', 'Learning Rate', 'Best Val Loss', 'Train Time Per Epoch'])
            writer.writerow({
                'Epoch': epoch,
                'Train Loss': train_loss,
                'Val Loss': val_loss,
                'Elapsed Time': elapsed_time,
                'Learning Rate': current_lr,
                'Best Val Loss': best_val_loss,
                'Train Time Per Epoch': delta_t_epoch
            })

    def log_to_sqlite(self, task, epoch, train_loss, val_loss, best_val_loss, elapsed_time, avg_batch_time, early_stopping, model_memory_usage, current_lr):
        """Log epoch-level data to a SQLite database with the updated learning rate."""
        sqlite_db_file = task['db_log_file']
        conn = sqlite3.connect(sqlite_db_file)
        cursor = conn.cursor()

        # Insert data with updated learning rate
        cursor.execute('''INSERT INTO task_logs (task_id, epoch, train_loss, val_loss, elapsed_time, avg_batch_time, learning_rate, 
                        best_val_loss, num_learnable_params, batch_size, lookback, max_epochs, early_stopping, model_memory_usage, device)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                    (task['task_id'], epoch, train_loss, val_loss, elapsed_time, avg_batch_time, current_lr, best_val_loss,
                    task['hyperparams']['NUM_LEARNABLE_PARAMS'], task['hyperparams']['BATCH_SIZE'], task['hyperparams']['LOOKBACK'], 
                    task['hyperparams']['MAX_EPOCHS'], early_stopping, model_memory_usage, self.device.type))

        conn.commit()
        conn.close()

    def process_task(self, task, update_progress_callback):
        """Process a single training task and set up logging."""
        try:
            # Concise log for starting task
            h_params_summary = {
                k: task['hyperparams'].get(k) for k in ['MODEL_TYPE', 'LAYERS', 'HIDDEN_UNITS', 'MAX_EPOCHS', 'INITIAL_LR', 'BATCH_SIZE', 'LOOKBACK'] if k in task['hyperparams']
            }
            self.logger.info(f"Starting task_id: {task.get('task_id', 'N/A')} with key hyperparams: {h_params_summary}")

            # Load normalization metadata and scaler if applicable
            self.load_normalization_info_and_scaler()

            # Setup logging (SQL and CSV) for the job
            self.setup_job_logging(task)

            # Task initialization and logging
            self.current_task = task
            self.stop_requested = False

            # Ensure the task contains a valid model
            if 'model' not in task or task['model'] is None:
                raise ValueError("Task does not contain a valid model instance.")

            # Configuring DataLoader (send progress update via signal)
            self.logger.info("Configuring DataLoader")
            update_progress_callback.emit({'status': 'Configuring DataLoader...'})

            # Create data loaders for the task
            train_loader, val_loader = self.create_data_loaders(task)
            self.logger.info(f"DataLoaders configured for task_id: {task.get('task_id', 'N/A')}")
            print(f" dataloader size, Train: {len(train_loader)} | Validation: {len(val_loader)}")

            # Update progress for starting training
            update_progress_callback.emit({'status': f'Training LSTM model for {task["hyperparams"]["MAX_EPOCHS"]} epochs...'})

            # Run training with all necessary parameters
            self.run_training(task, update_progress_callback, train_loader, val_loader, self.device)

        except Exception as e:
            self.logger.error(f"Error during task processing: {str(e)}")
            update_progress_callback.emit({'task_error': str(e)})

    def setup_job_logging(self, task):
        """
        Set up the database and logging environment for the job.
        This ensures that the database tables are created if they do not exist.
        """
        job_id = self.job_manager.get_job_id()  # Get the job ID
        model_dir = task.get('model_dir')  # Path where task-related logs are stored

        # Retrieve log file paths from the task
        csv_log_file = task.get('csv_log_file')
        db_log_file = task.get('db_log_file')

        # Log information about the task and log file paths
        output_dir_root = os.path.dirname(self.job_manager.get_job_folder()) # e.g., 'output'
        
        log_model_dir = os.path.relpath(model_dir, output_dir_root) if model_dir and output_dir_root in model_dir else model_dir
        log_csv_file = os.path.relpath(csv_log_file, output_dir_root) if csv_log_file and output_dir_root in csv_log_file else csv_log_file
        log_db_file = os.path.relpath(db_log_file, output_dir_root) if db_log_file and output_dir_root in db_log_file else db_log_file

        print(f"Setting up logging for job: {job_id}")
        print(f"Model directory (relative to output): {log_model_dir}")
        print(f"Log files for task {task['task_id']} (relative to output): CSV: {log_csv_file}, DB: {log_db_file}")

        # Ensure the model_dir exists
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)  # Create the directory if it does not exist

        # Create SQLite tables if they do not exist
        self.create_sql_tables(db_log_file)

    def create_sql_tables(self, db_log_file):
        """Create the necessary SQL tables for task-level and batch-level logging."""
        try:
            # Ensure the database file path is valid
            if not os.path.isfile(db_log_file):
                output_dir_root_db = os.path.dirname(self.job_manager.get_job_folder())
                log_db_path_create = os.path.relpath(db_log_file, output_dir_root_db) if db_log_file and output_dir_root_db in db_log_file else db_log_file
                self.logger.info(f"Creating new database file at (relative to output): {log_db_path_create}")

            # Connect to the database and create tables
            conn = sqlite3.connect(db_log_file)
            cursor = conn.cursor()

            # Create table for high-level task logs (epoch-level)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS task_logs (
                    task_id TEXT,
                    epoch INTEGER,
                    train_loss REAL,
                    val_loss REAL,
                    elapsed_time REAL,
                    avg_batch_time REAL,  -- New column for average batch time
                    learning_rate REAL,
                    best_val_loss REAL,
                    num_learnable_params INTEGER,
                    batch_size INTEGER,
                    lookback INTEGER,
                    max_epochs INTEGER,
                    early_stopping INTEGER,  -- New column for early stopping flag (1 if stopped early, 0 otherwise)
                    model_memory_usage REAL,  -- New column for memory usage (optional)
                    device TEXT,  -- Add this new column for the device
                    PRIMARY KEY(task_id, epoch)
                )
            ''')

            # Create table for fine-grained batch logs
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS batch_logs (
                    task_id TEXT,
                    epoch INTEGER,
                    batch_idx INTEGER,
                    batch_time REAL,
                    phase TEXT,
                    learning_rate REAL,
                    num_learnable_params INTEGER,
                    batch_size INTEGER,
                    lookback INTEGER,
                    device TEXT,  -- Add the device column here
                    FOREIGN KEY(task_id, epoch) REFERENCES task_logs(task_id, epoch)
                )
            ''')

            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            self.logger.error(f"SQLite error: {e}")
            raise e

    def load_normalization_info_and_scaler(self):
        """Loads normalization metadata and the scaler if normalization was applied."""
        self.loaded_scaler = None
        self.scaler_metadata = {}
        job_folder = self.job_manager.get_job_folder()
        if not job_folder:
            self.logger.warning("Job folder not set in JobManager. Cannot load normalization info.")
            return

        metadata_file_path = os.path.join(job_folder, "job_metadata.json")
        if os.path.exists(metadata_file_path):
            try:
                with open(metadata_file_path, 'r') as f_meta:
                    job_meta = json.load(f_meta)
                
                if job_meta.get('normalization_applied', False):
                    scaler_path_relative = job_meta.get('scaler_path')
                    normalized_columns = job_meta.get('normalized_columns')
                    # Assuming target_column is available in task later, or we can add it to job_meta if it's globally unique
                    
                    if scaler_path_relative and normalized_columns:
                        scaler_path_absolute = os.path.join(job_folder, scaler_path_relative)
                        from vestim.services import normalization_service # Local import
                        self.loaded_scaler = normalization_service.load_scaler(scaler_path_absolute)
                        if self.loaded_scaler:
                            self.scaler_metadata = {
                                'scaler_path': scaler_path_absolute,
                                'normalized_columns': normalized_columns,
                                'normalization_applied': True
                                # 'target_column' will be derived from task params later
                            }
                            self.logger.info(f"Successfully loaded scaler from {scaler_path_absolute} and normalization metadata.")
                        else:
                            self.logger.error(f"Failed to load scaler from {scaler_path_absolute}. Reporting will be on normalized scale.")
                    else:
                        self.logger.warning("Normalization metadata incomplete (scaler_path or normalized_columns missing). Reporting on normalized scale.")
                else:
                    self.logger.info("Normalization was not applied according to job_metadata.json.")
            except Exception as e:
                self.logger.error(f"Error loading normalization metadata from {metadata_file_path}: {e}")
        else:
            self.logger.info(f"job_metadata.json not found at {metadata_file_path}. Assuming no normalization or scaler to load.")


    def create_data_loaders(self, task):
        """Create data loaders for the current task."""
        feature_cols = task['data_loader_params']['feature_columns']
        target_col = task['data_loader_params']['target_column']
        train_val_split = float(task['data_loader_params'].get('train_val_split', 0.7))
        num_workers = int(task['hyperparams'].get('NUM_WORKERS', 4))
        seed = int(task['hyperparams'].get('SEED', 2000))
        
        training_method = task['hyperparams'].get('TRAINING_METHOD', 'Sequence-to-Sequence') # Default if not present
        model_type = task['hyperparams'].get('MODEL_TYPE', 'LSTM') # Default if not present

        self.logger.info(f"Selected Training Method: {training_method}, Model Type: {model_type}")

        if training_method == 'Whole Sequence' and model_type in ['LSTM', 'GRU']: # Check for RNN types
            self.logger.info("Using concatenated whole sequence loader for RNN.")
            train_loader, val_loader = self.data_loader_service.create_concatenated_whole_sequence_loaders(
                folder_path=self.job_manager.get_train_folder(),
                feature_cols=feature_cols,
                target_col=target_col,
                num_workers=num_workers,
                train_split=train_val_split,
                seed=seed
            )
        else: # Default to lookback-based sequence loading
            if training_method == 'Whole Sequence' and model_type not in ['LSTM', 'GRU']:
                 self.logger.info(f"Training method is 'Whole Sequence' but model type is {model_type}. Using standard sequence loader (this path is typically for FNNs with whole_sequence_fnn_data_handler or similar).")
            
            lookback = int(task['data_loader_params'].get('lookback', 50)) # Default lookback
            user_batch_size = int(task['data_loader_params'].get('batch_size', 32))
            
            batch_training_enabled = task['hyperparams'].get('BATCH_TRAINING', True)
            use_full_train_batch_flag = not batch_training_enabled

            self.logger.info(f"Using standard sequence loader. Batch training enabled: {batch_training_enabled}, User batch size: {user_batch_size}, Use full train batch flag: {use_full_train_batch_flag}, Lookback: {lookback}")
            
            train_loader, val_loader = self.data_loader_service.create_data_loaders(
                folder_path=self.job_manager.get_train_folder(),
                training_method=training_method,
                lookback=lookback,
                feature_cols=feature_cols,
                target_col=target_col,
                batch_size=user_batch_size,
                num_workers=num_workers,
                # use_full_train_batch=use_full_train_batch_flag, # Removed as it's not an accepted arg by production DataLoaderService
                train_split=train_val_split,
                seed=seed
            )

        return train_loader, val_loader

    def run_training(self, task, update_progress_callback, train_loader, val_loader, device):
        """Run the training process for a single task."""
        try:
            self.logger.info(f"--- Starting run_training for task: {task['task_id']} ---") # Added detailed log
            self.logger.info("Starting training loop")
            # Initialize/reset task-specific best original scale validation RMSE tracker
            # Using a unique attribute name per task to avoid conflicts if manager instance is reused for different tasks sequentially
            # though typically a new manager or thread might be used. This is safer.
            setattr(self, f'_task_{task["task_id"]}_best_val_rmse_orig', float('inf'))

            hyperparams = self.convert_hyperparams(task['hyperparams']) # This ensures BATCH_SIZE is int if it exists
            model = task['model'].to(device)
            
            # Ensure BATCH_SIZE from hyperparams (which might be the string from QLineEdit) is correctly converted and available
            # The actual batch size used by train_loader is now determined by DataLoaderService based on use_full_train_batch_flag
            # However, other parts of the code might still refer to hyperparams['BATCH_SIZE']
            # For logging or other purposes, ensure it's an int.
            # The convert_hyperparams method already handles BATCH_SIZE if it's a direct hyperparam.
            # If BATCH_SIZE is under data_loader_params, it's handled in create_data_loaders above.
            
            max_epochs = hyperparams['MAX_EPOCHS']
            valid_freq = hyperparams['ValidFrequency']
            valid_patience = hyperparams['VALID_PATIENCE']
            #patience_threshold = int(valid_patience * 0.5) 
            current_lr = hyperparams['INITIAL_LR']
            lr_drop_period = hyperparams['LR_DROP_PERIOD']
            lr_drop_factor = hyperparams['LR_DROP_FACTOR']
            # Define a buffer period after which LR drops can happen again, e.g., 100 epochs.
            lr_drop_buffer = 50
            last_lr_drop_epoch = 0  # Initialize the epoch of the last LR drop
            # weight_decay = hyperparams.get('WEIGHT_DECAY', 1e-5)

            best_validation_loss = float('inf')
            patience_counter = 0
            loop_start_time = time.time() # Renamed from start_time to avoid confusion with overall training start
            last_validation_time = loop_start_time
            early_stopping = False  # Initialize early stopping flag

            # Max training time logic
            max_training_time_seconds = int(task.get('training_params', {}).get('max_training_time_seconds', 0))
            overall_training_start_time = time.time() # For max training time check
            self.logger.info(f"Max training time set to: {max_training_time_seconds} seconds.")

            self.optimizer = torch.optim.Adam(model.parameters(), lr=current_lr)
            # self.scheduler = self.training_service.get_scheduler(self.optimizer, gamma=lr_drop_factor)
            #self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=lr_drop_factor)
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=lr_drop_period,  # Number of epochs between drops
                gamma=lr_drop_factor       # Multiplicative factor for the drop
            )
            optimizer = self.optimizer
            scheduler = self.scheduler

            # Initialize CSV logging for epoch-wise data
            csv_log_file = task['csv_log_file']
            # Ensure the directory for csv_log_file exists (it's task_dir/logs/)
            os.makedirs(os.path.dirname(csv_log_file), exist_ok=True)
            with open(csv_log_file, 'w', newline='') as f: # Added newline=''
                csv_writer = csv.writer(f)
                csv_writer.writerow(["epoch", "train_loss_norm", "val_loss_norm", "best_val_loss_norm", "learning_rate", "elapsed_time_sec", "avg_batch_time_sec", "patience_counter", "model_memory_mb"]) # Header

            # Training loop
            for epoch in range(1, max_epochs + 1):
                if self.stop_requested:  # Ensure thread safety here
                    self.logger.info("Training stopped by user")
                    print("Stopping training...")
                    break
                
                # Check for max training time exceeded
                if max_training_time_seconds > 0:
                    current_training_duration = time.time() - overall_training_start_time
                    if current_training_duration > max_training_time_seconds:
                        self.logger.info(f"Max training time ({max_training_time_seconds}s) exceeded. Stopping training.")
                        print(f"Max training time ({max_training_time_seconds}s) exceeded. Stopping training.")
                        self.stop_requested = True # Use existing flag to gracefully stop
                        early_stopping = True # Indicate it was a form of early stop
                        # Also update task results if possible here or after loop
                        task['results']['early_stopped_reason'] = 'Max training time exceeded'
                        break # Exit epoch loop

                # Initialize hidden states for training phase
                # Use the actual batch size from the train_loader
                actual_train_batch_size = train_loader.batch_size
                if actual_train_batch_size is None:
                    self.logger.warning(f"train_loader.batch_size is None. Falling back to hyperparams BATCH_SIZE ({hyperparams.get('BATCH_SIZE', 'N/A')}) for training hidden state init.")
                    actual_train_batch_size = int(hyperparams.get('BATCH_SIZE', 32)) # Default if all else fails

                self.logger.info(f"Initializing training hidden state with batch size: {actual_train_batch_size}")
                h_s = torch.zeros(model.num_layers, actual_train_batch_size, model.hidden_units).to(device)
                h_c = torch.zeros(model.num_layers, actual_train_batch_size, model.hidden_units).to(device)

                # Measure time for the training loop
                epoch_start_time = time.time()

                # Train the model for one epoch
                model_type = task.get('model_metadata', {}).get('model_type', task.get('hyperparams', {}).get('MODEL_TYPE', 'LSTM')) # Get model_type
                # train_epoch now returns: avg_batch_time, avg_loss (normalized), all_train_y_pred_normalized, all_train_y_true_normalized
                avg_batch_time, train_loss_norm, epoch_train_preds_norm, epoch_train_trues_norm = self.training_service.train_epoch(
                    model, model_type, train_loader, optimizer, h_s, h_c, epoch, device, self.stop_requested, task
                )

                epoch_end_time = time.time()
                epoch_duration = epoch_end_time - epoch_start_time
                formatted_epoch_time = format_time(epoch_duration)  # Convert epoch time to mm:ss format

                if self.stop_requested:
                    self.logger.info("Training stopped by user")
                    print("Training stopped after training phase.")
                    self.logger.info("Training stopped after training phase.")
                    break

                # Only validate at specified frequency
                if epoch == 1 or epoch % valid_freq == 0 or epoch == max_epochs:
                    # Initialize hidden states for validation phase
                    # Use the actual batch size from the val_loader
                    actual_val_batch_size = val_loader.batch_size
                    if actual_val_batch_size is None:
                        self.logger.warning(f"val_loader.batch_size is None. Falling back to hyperparams BATCH_SIZE ({hyperparams.get('BATCH_SIZE', 'N/A')}) for validation hidden state init.")
                        actual_val_batch_size = int(hyperparams.get('BATCH_SIZE', 32)) # Default if all else fails
                    
                    self.logger.info(f"Initializing validation hidden state with batch size: {actual_val_batch_size}")
                    h_s_val = torch.zeros(model.num_layers, actual_val_batch_size, model.hidden_units).to(device)
                    h_c_val = torch.zeros(model.num_layers, actual_val_batch_size, model.hidden_units).to(device)

                    model_type = task.get('model_metadata', {}).get('model_type', task.get('hyperparams', {}).get('MODEL_TYPE', 'LSTM')) # Get model_type
                    # validate_epoch now returns: avg_loss (normalized), all_val_y_pred_normalized, all_val_y_true_normalized
                    val_loss_norm, epoch_val_preds_norm, epoch_val_trues_norm = self.training_service.validate_epoch(
                        model, model_type, val_loader, h_s_val, h_c_val, epoch, device, self.stop_requested, task
                    )

                    current_time = time.time()
                    elapsed_time = current_time - loop_start_time # Use loop_start_time for per-epoch/validation cycle timing
                    delta_t_epoch = (current_time - last_validation_time) / valid_freq
                    last_validation_time = current_time

                    current_lr = optimizer.param_groups[0]['lr']
                    
                    if val_loss_norm < best_validation_loss: # best_validation_loss is also on normalized scale
                        print(f"Epoch: {epoch}, Validation loss improved from {best_validation_loss:.6f} to {val_loss_norm:.6f}. Saving model...")
                        best_validation_loss = val_loss_norm
                        # Save to best_model_path
                        best_model_save_path = task.get('training_params', {}).get('best_model_path')
                        if best_model_save_path:
                            self.save_model(task, save_path=best_model_save_path)
                            self.logger.info(f"Best model saved to: {best_model_save_path}")
                        else:
                            self.logger.warning(f"best_model_path not found in task for epoch {epoch}. Best model not saved.")
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    # Determine target variable specifics for error reporting
                    target_column = task['data_loader_params']['target_column']
                    error_unit_label = "RMS Error"  # Default
                    multiplier = 1.0

                    if "voltage" in target_column.lower():
                        error_unit_label = "RMS Error [mV]"
                        multiplier = 1000.0
                    elif "soc" in target_column.lower():
                        error_unit_label = "RMS Error [% SOC]"
                        multiplier = 100.0
                    elif "soe" in target_column.lower(): # Added SOE
                        error_unit_label = "RMS Error [% SOE]"
                        multiplier = 100.0
                    elif "sop" in target_column.lower(): # Added SOP
                        error_unit_label = "RMS Error [% SOP]"
                        multiplier = 100.0
                    elif "temperature" in target_column.lower() or "temp" in target_column.lower():
                        error_unit_label = "RMS Error [Deg C]"
                        multiplier = 1.0
                    
                    # Calculate scaled RMSE values.
                    train_rmse_for_gui = float('nan')
                    val_rmse_for_gui = float('nan')
                    # Retrieve the running best original scale validation RMSE for this task
                    best_val_rmse_orig_scale_for_gui = getattr(self, f'_task_{task["task_id"]}_best_val_rmse_orig', float('inf'))

                    target_col_for_scaler = task['data_loader_params']['target_column']

                    if self.loaded_scaler and target_col_for_scaler in self.scaler_metadata.get('normalized_columns', []):
                        from vestim.services import normalization_service # Local import
                        import pandas as pd # Local import for DataFrame
                        import numpy as np # Ensure numpy is imported

                        # --- Train RMSE on original scale (if epoch_train_preds_norm available) ---
                        if epoch_train_preds_norm is not None and epoch_train_trues_norm is not None and len(epoch_train_preds_norm) > 0:
                            try:
                                # Ensure tensors are on CPU and converted to numpy
                                e_t_p_n_cpu = epoch_train_preds_norm.cpu().numpy() if epoch_train_preds_norm.is_cuda else epoch_train_preds_norm.numpy()
                                e_t_t_n_cpu = epoch_train_trues_norm.cpu().numpy() if epoch_train_trues_norm.is_cuda else epoch_train_trues_norm.numpy()

                                temp_df_train_pred = pd.DataFrame(0, index=np.arange(len(e_t_p_n_cpu)), columns=self.scaler_metadata['normalized_columns'])
                                temp_df_train_pred[target_col_for_scaler] = e_t_p_n_cpu.flatten()
                                df_train_pred_inv = normalization_service.inverse_transform_data(temp_df_train_pred, self.loaded_scaler, self.scaler_metadata['normalized_columns'])
                                train_pred_orig = df_train_pred_inv[target_col_for_scaler].values

                                temp_df_train_true = pd.DataFrame(0, index=np.arange(len(e_t_t_n_cpu)), columns=self.scaler_metadata['normalized_columns'])
                                temp_df_train_true[target_col_for_scaler] = e_t_t_n_cpu.flatten()
                                df_train_true_inv = normalization_service.inverse_transform_data(temp_df_train_true, self.loaded_scaler, self.scaler_metadata['normalized_columns'])
                                train_true_orig = df_train_true_inv[target_col_for_scaler].values
                                
                                train_mse_orig = np.mean((train_pred_orig - train_true_orig)**2)
                                train_rmse_for_gui = np.sqrt(train_mse_orig) * multiplier
                            except Exception as e_inv_train:
                                self.logger.error(f"Error during inverse transform for training data (epoch {epoch}): {e_inv_train}. Falling back for train_rmse_for_gui.")
                                if train_loss_norm is not None and not math.isnan(train_loss_norm):
                                    train_rmse_for_gui = math.sqrt(max(0, train_loss_norm)) * multiplier
                        else:
                             if train_loss_norm is not None and not math.isnan(train_loss_norm):
                                train_rmse_for_gui = math.sqrt(max(0, train_loss_norm)) * multiplier
                        
                        # --- Validation RMSE on original scale ---
                        if epoch_val_preds_norm is not None and epoch_val_trues_norm is not None and len(epoch_val_preds_norm) > 0:
                            try:
                                e_v_p_n_cpu = epoch_val_preds_norm.cpu().numpy() if epoch_val_preds_norm.is_cuda else epoch_val_preds_norm.numpy()
                                e_v_t_n_cpu = epoch_val_trues_norm.cpu().numpy() if epoch_val_trues_norm.is_cuda else epoch_val_trues_norm.numpy()

                                temp_df_val_pred = pd.DataFrame(0, index=np.arange(len(e_v_p_n_cpu)), columns=self.scaler_metadata['normalized_columns'])
                                temp_df_val_pred[target_col_for_scaler] = e_v_p_n_cpu.flatten()
                                df_val_pred_inv = normalization_service.inverse_transform_data(temp_df_val_pred, self.loaded_scaler, self.scaler_metadata['normalized_columns'])
                                val_pred_orig = df_val_pred_inv[target_col_for_scaler].values

                                temp_df_val_true = pd.DataFrame(0, index=np.arange(len(e_v_t_n_cpu)), columns=self.scaler_metadata['normalized_columns'])
                                temp_df_val_true[target_col_for_scaler] = e_v_t_n_cpu.flatten()
                                df_val_true_inv = normalization_service.inverse_transform_data(temp_df_val_true, self.loaded_scaler, self.scaler_metadata['normalized_columns'])
                                val_true_orig = df_val_true_inv[target_col_for_scaler].values

                                val_mse_orig = np.mean((val_pred_orig - val_true_orig)**2)
                                current_val_rmse_orig_scale = np.sqrt(val_mse_orig) * multiplier
                                val_rmse_for_gui = current_val_rmse_orig_scale
                                
                                if current_val_rmse_orig_scale < best_val_rmse_orig_scale_for_gui:
                                    best_val_rmse_orig_scale_for_gui = current_val_rmse_orig_scale
                                    setattr(self, f'_task_{task["task_id"]}_best_val_rmse_orig', best_val_rmse_orig_scale_for_gui) # Update the stored best
                                
                            except Exception as e_inv_val:
                                self.logger.error(f"Error during inverse transform for validation data (epoch {epoch}): {e_inv_val}. Falling back for val_rmse_for_gui.")
                                if val_loss_norm is not None and not math.isnan(val_loss_norm):
                                    val_rmse_for_gui = math.sqrt(max(0, val_loss_norm)) * multiplier
                        else:
                            if val_loss_norm is not None and not math.isnan(val_loss_norm):
                                val_rmse_for_gui = math.sqrt(max(0, val_loss_norm)) * multiplier
                        
                        # Use the overall best original-scale validation RMSE for this task for display
                        best_val_rmse_for_gui = best_val_rmse_orig_scale_for_gui

                    else: # No scaler loaded or target not in normalized columns - use normalized loss for GUI RMSE
                        if train_loss_norm is not None and not math.isnan(train_loss_norm):
                            train_rmse_for_gui = math.sqrt(max(0, train_loss_norm)) * multiplier
                        if val_loss_norm is not None and not math.isnan(val_loss_norm):
                            val_rmse_for_gui = math.sqrt(max(0, val_loss_norm)) * multiplier
                        # If no scaler, best_val_rmse_for_gui is based on best_validation_loss (normalized)
                        if best_validation_loss != float('inf') and not math.isnan(best_validation_loss):
                             best_val_rmse_for_gui = math.sqrt(max(0, best_validation_loss)) * multiplier
                        else:
                             best_val_rmse_for_gui = float('inf') # Ensure it's inf if best_validation_loss is inf
                    
                    # Log to CSV (after validation)
                    model_memory_usage_val = torch.cuda.memory_allocated(device=self.device) if self.device.type == 'cuda' else 0
                    model_memory_usage_mb_val = model_memory_usage_val / (1024 * 1024) if model_memory_usage_val > 0 else 0
                    with open(csv_log_file, 'a', newline='') as f:
                        csv_writer_val = csv.writer(f)
                        csv_writer_val.writerow([
                            epoch,
                            f"{train_loss_norm:.6f}" if train_loss_norm is not None else 'nan',
                            f"{val_loss_norm:.6f}" if val_loss_norm is not None else 'nan',
                            f"{best_validation_loss:.6f}" if best_validation_loss is not None else 'nan', # best_validation_loss is normalized
                            f"{current_lr:.1e}" if current_lr is not None else 'nan',
                            f"{elapsed_time:.2f}" if elapsed_time is not None else 'nan', # elapsed_time for validation epoch
                            f"{avg_batch_time:.4f}" if avg_batch_time is not None else 'nan', # avg_batch_time for train part of this epoch
                            patience_counter if patience_counter is not None else 'nan',
                            f"{model_memory_usage_mb_val:.3f}" if model_memory_usage_mb_val is not None else 'nan'
                        ])
                    
                    self.logger.info(f"Epoch {epoch} | Train Loss (Norm): {train_loss_norm:.6f} | Val Loss (Norm): {val_loss_norm:.6f} | GUI Train RMSE: {train_rmse_for_gui:.4f} {error_unit_label} | GUI Val RMSE: {val_rmse_for_gui:.4f} {error_unit_label} | LR: {current_lr} | Epoch Time: {formatted_epoch_time} | Best Val Loss (Norm): {best_validation_loss:.6f} | GUI Best Val RMSE: {best_val_rmse_for_gui:.4f} {error_unit_label} | Patience: {patience_counter}")
                    progress_data = {
                        'epoch': epoch,
                        'train_loss': train_loss_norm,
                        'val_loss': val_loss_norm,
                        'train_rmse_scaled': train_rmse_for_gui,
                        'val_rmse_scaled': val_rmse_for_gui,
                        'error_unit_label': error_unit_label,
                        'elapsed_time': elapsed_time,
                        'delta_t_epoch': formatted_epoch_time,
                        'learning_rate': current_lr,
                        'best_val_loss': best_validation_loss, # This is normalized best MSE
                        'best_val_rmse_scaled': best_val_rmse_for_gui,
                        'patience_counter': patience_counter,
                    }
                    update_progress_callback.emit(progress_data)

                    if patience_counter > valid_patience:
                        early_stopping = True
                        print(f"Early stopping at epoch {epoch} due to no improvement.")
                        self.logger.info(f"Early stopping at epoch {epoch} due to no improvement.")
                        
                        model_memory_usage = torch.cuda.memory_allocated() if torch.cuda.is_available() else sys.getsizeof(model)
                        model_memory_usage_mb = model_memory_usage / (1024 * 1024)
                        
                        # self.log_to_sqlite(
                        #     task=task, epoch=epoch, train_loss=train_loss_norm,
                        #     val_loss=val_loss_norm, best_val_loss=best_validation_loss,
                        #     elapsed_time=elapsed_time, avg_batch_time=avg_batch_time,
                        #     early_stopping=early_stopping, model_memory_usage=round(model_memory_usage_mb, 3),
                        #     current_lr=current_lr
                        # )
                        break
                
                # Log to SQLite for non-validation epochs or if not early stopped on a validation epoch
                if not early_stopping:
                    if not (epoch == 1 or epoch % valid_freq == 0 or epoch == max_epochs): # If not a validation epoch
                        model_memory_usage = torch.cuda.memory_allocated() if torch.cuda.is_available() else sys.getsizeof(model)
                        model_memory_usage_mb = model_memory_usage / (1024 * 1024)
# This block is for epochs where validation did not run.
                        current_time_train_only = time.time()
                        elapsed_time_train_only = current_time_train_only - loop_start_time # Changed start_time to loop_start_time
                        # model_memory_usage_mb is already calculated at the start of this 'if not (epoch ...)' block
                        
                        # Calculate scaled RMSE for training for GUI if possible
                        train_rmse_for_gui_no_val = float('nan')
                        target_column_no_val = task['data_loader_params']['target_column']
                        # Determine error_unit_label and multiplier for this context
                        # These would have been set during a validation epoch if one occurred,
                        # otherwise, we need defaults or to fetch them.
                        # 'error_unit_label' is defined in the outer scope of the validation block
                        # We need to ensure it's available or use a default if this non-validation epoch occurs before any validation.
                        current_error_unit_label_no_val = "RMS Error" # Default
                        multiplier_no_val = 1.0 # Default
                        if 'error_unit_label' in locals() and 'multiplier' in locals(): # Check if set by validation block
                            current_error_unit_label_no_val = error_unit_label
                            multiplier_no_val = multiplier
                        else: # Recalculate if not set from validation context (e.g. first few epochs before validation)
                            if "voltage" in target_column_no_val.lower():
                                current_error_unit_label_no_val = "RMS Error [mV]"
                                multiplier_no_val = 1000.0
                            elif "soc" in target_column_no_val.lower():
                                current_error_unit_label_no_val = "RMS Error [% SOC]"
                                multiplier_no_val = 100.0
                            elif "soe" in target_column_no_val.lower():
                                current_error_unit_label_no_val = "RMS Error [% SOE]"
                                multiplier_no_val = 100.0
                            elif "sop" in target_column_no_val.lower():
                                current_error_unit_label_no_val = "RMS Error [% SOP]"
                                multiplier_no_val = 100.0
                            elif "temperature" in target_column_no_val.lower() or "temp" in target_column_no_val.lower():
                                current_error_unit_label_no_val = "RMS Error [Deg C]"
                                multiplier_no_val = 1.0

                        if self.loaded_scaler and target_column_no_val in self.scaler_metadata.get('normalized_columns', []):
                            if epoch_train_preds_norm is not None and epoch_train_trues_norm is not None and len(epoch_train_preds_norm) > 0:
                                try:
                                    import pandas as pd 
                                    import numpy as np
                                    from vestim.services import normalization_service 
                                    e_t_p_n_cpu_no_val = epoch_train_preds_norm.cpu().numpy() if epoch_train_preds_norm.is_cuda else epoch_train_preds_norm.numpy()
                                    e_t_t_n_cpu_no_val = epoch_train_trues_norm.cpu().numpy() if epoch_train_trues_norm.is_cuda else epoch_train_trues_norm.numpy()
                                    temp_df_train_pred_no_val = pd.DataFrame(0, index=np.arange(len(e_t_p_n_cpu_no_val)), columns=self.scaler_metadata['normalized_columns'])
                                    temp_df_train_pred_no_val[target_column_no_val] = e_t_p_n_cpu_no_val.flatten()
                                    df_train_pred_inv_no_val = normalization_service.inverse_transform_data(temp_df_train_pred_no_val, self.loaded_scaler, self.scaler_metadata['normalized_columns'])
                                    train_pred_orig_no_val = df_train_pred_inv_no_val[target_column_no_val].values
                                    temp_df_train_true_no_val = pd.DataFrame(0, index=np.arange(len(e_t_t_n_cpu_no_val)), columns=self.scaler_metadata['normalized_columns'])
                                    temp_df_train_true_no_val[target_column_no_val] = e_t_t_n_cpu_no_val.flatten()
                                    df_train_true_inv_no_val = normalization_service.inverse_transform_data(temp_df_train_true_no_val, self.loaded_scaler, self.scaler_metadata['normalized_columns'])
                                    train_true_orig_no_val = df_train_true_inv_no_val[target_column_no_val].values
                                    train_mse_orig_no_val = np.mean((train_pred_orig_no_val - train_true_orig_no_val)**2)
                                    train_rmse_for_gui_no_val = np.sqrt(train_mse_orig_no_val) * multiplier_no_val
                                except Exception as e_inv_train_no_val:
                                    self.logger.error(f"Error during inverse transform for training data (non-val epoch {epoch}): {e_inv_train_no_val}.")
                                    if train_loss_norm is not None and not math.isnan(train_loss_norm): train_rmse_for_gui_no_val = math.sqrt(max(0, train_loss_norm)) * multiplier_no_val
                            else: 
                                if train_loss_norm is not None and not math.isnan(train_loss_norm): train_rmse_for_gui_no_val = math.sqrt(max(0, train_loss_norm)) * multiplier_no_val
                        else: 
                            if train_loss_norm is not None and not math.isnan(train_loss_norm): train_rmse_for_gui_no_val = math.sqrt(max(0, train_loss_norm)) * multiplier_no_val

                        # Log to CSV for non-validation epochs
                        with open(csv_log_file, 'a', newline='') as f:
                            csv_writer_train_only = csv.writer(f)
                            csv_writer_train_only.writerow([
                                epoch,
                                f"{train_loss_norm:.6f}" if train_loss_norm is not None else 'nan',
                                'nan', 
                                f"{best_validation_loss:.6f}" if best_validation_loss is not None else 'nan', 
                                f"{current_lr:.1e}" if current_lr is not None else 'nan',
                                f"{elapsed_time_train_only:.2f}" if elapsed_time_train_only is not None else 'nan',
                                f"{avg_batch_time:.4f}" if avg_batch_time is not None else 'nan',
                                patience_counter if patience_counter is not None else 'nan',
                                f"{model_memory_usage_mb:.3f}" if model_memory_usage_mb is not None else 'nan'
                            ])
                        
                        # Update GUI via signal (for non-validation epochs)
                        progress_data_train_only = {
                            'epoch': epoch,
                            'train_loss_norm': train_loss_norm, 
                            'val_loss_norm': float('nan'),      
                            'train_rmse_scaled': train_rmse_for_gui_no_val, 
                            'val_rmse_scaled': float('nan'),         
                            'best_val_rmse_scaled': getattr(self, f'_task_{task["task_id"]}_best_val_rmse_orig', float('inf')), 
                            'error_unit_label': current_error_unit_label_no_val, 
                            'delta_t_epoch': formatted_epoch_time, 
                            'elapsed_time': format_time(elapsed_time_train_only), 
                            'patience_counter': patience_counter, 
                            'learning_rate': current_lr, 
                            'status': f"Epoch {epoch}/{max_epochs} - Training..."
                        }
                        update_progress_callback.emit(progress_data_train_only)
# This block is for epochs where validation did not run.
                        current_time_train_only = time.time()
                        elapsed_time_train_only = current_time_train_only - loop_start_time # Changed start_time to loop_start_time
                        model_memory_usage_train_only = torch.cuda.memory_allocated(device=self.device) if self.device.type == 'cuda' else 0
                        model_memory_usage_mb_train_only = model_memory_usage_train_only / (1024 * 1024) if model_memory_usage_train_only > 0 else 0
                        
                        # Calculate scaled RMSE for training for GUI if possible
                        train_rmse_for_gui_no_val = float('nan')
                        target_column_no_val = task['data_loader_params']['target_column']
                        # Determine error_unit_label and multiplier for this context
                        current_error_unit_label_no_val = "RMS Error" # Default
                        multiplier_no_val = 1.0 # Default
                        if "voltage" in target_column_no_val.lower():
                            current_error_unit_label_no_val = "RMS Error [mV]"
                            multiplier_no_val = 1000.0
                        elif "soc" in target_column_no_val.lower():
                            current_error_unit_label_no_val = "RMS Error [% SOC]"
                            multiplier_no_val = 100.0
                        elif "soe" in target_column_no_val.lower():
                            current_error_unit_label_no_val = "RMS Error [% SOE]"
                            multiplier_no_val = 100.0
                        elif "sop" in target_column_no_val.lower():
                            current_error_unit_label_no_val = "RMS Error [% SOP]"
                            multiplier_no_val = 100.0
                        elif "temperature" in target_column_no_val.lower() or "temp" in target_column_no_val.lower():
                            current_error_unit_label_no_val = "RMS Error [Deg C]"
                            multiplier_no_val = 1.0

                        if self.loaded_scaler and target_column_no_val in self.scaler_metadata.get('normalized_columns', []):
                            if epoch_train_preds_norm is not None and epoch_train_trues_norm is not None and len(epoch_train_preds_norm) > 0:
                                try:
                                    # Ensure pandas and numpy are available
                                    import pandas as pd 
                                    import numpy as np
                                    from vestim.services import normalization_service 
                                    e_t_p_n_cpu_no_val = epoch_train_preds_norm.cpu().numpy() if epoch_train_preds_norm.is_cuda else epoch_train_preds_norm.numpy()
                                    e_t_t_n_cpu_no_val = epoch_train_trues_norm.cpu().numpy() if epoch_train_trues_norm.is_cuda else epoch_train_trues_norm.numpy()
                                    temp_df_train_pred_no_val = pd.DataFrame(0, index=np.arange(len(e_t_p_n_cpu_no_val)), columns=self.scaler_metadata['normalized_columns'])
                                    temp_df_train_pred_no_val[target_column_no_val] = e_t_p_n_cpu_no_val.flatten()
                                    df_train_pred_inv_no_val = normalization_service.inverse_transform_data(temp_df_train_pred_no_val, self.loaded_scaler, self.scaler_metadata['normalized_columns'])
                                    train_pred_orig_no_val = df_train_pred_inv_no_val[target_column_no_val].values
                                    temp_df_train_true_no_val = pd.DataFrame(0, index=np.arange(len(e_t_t_n_cpu_no_val)), columns=self.scaler_metadata['normalized_columns'])
                                    temp_df_train_true_no_val[target_column_no_val] = e_t_t_n_cpu_no_val.flatten()
                                    df_train_true_inv_no_val = normalization_service.inverse_transform_data(temp_df_train_true_no_val, self.loaded_scaler, self.scaler_metadata['normalized_columns'])
                                    train_true_orig_no_val = df_train_true_inv_no_val[target_column_no_val].values
                                    train_mse_orig_no_val = np.mean((train_pred_orig_no_val - train_true_orig_no_val)**2)
                                    train_rmse_for_gui_no_val = np.sqrt(train_mse_orig_no_val) * multiplier_no_val
                                except Exception as e_inv_train_no_val:
                                    self.logger.error(f"Error during inverse transform for training data (non-val epoch {epoch}): {e_inv_train_no_val}.")
                                    if train_loss_norm is not None and not math.isnan(train_loss_norm): train_rmse_for_gui_no_val = math.sqrt(max(0, train_loss_norm)) * multiplier_no_val
                            else: 
                                if train_loss_norm is not None and not math.isnan(train_loss_norm): train_rmse_for_gui_no_val = math.sqrt(max(0, train_loss_norm)) * multiplier_no_val
                        else: 
                            if train_loss_norm is not None and not math.isnan(train_loss_norm): train_rmse_for_gui_no_val = math.sqrt(max(0, train_loss_norm)) * multiplier_no_val

                        # Log to CSV for non-validation epochs
                        with open(csv_log_file, 'a', newline='') as f:
                            csv_writer_train_only = csv.writer(f)
                            csv_writer_train_only.writerow([
                                epoch,
                                f"{train_loss_norm:.6f}" if train_loss_norm is not None else 'nan',
                                'nan', 
                                f"{best_validation_loss:.6f}" if best_validation_loss is not None else 'nan', 
                                f"{current_lr:.1e}" if current_lr is not None else 'nan',
                                f"{elapsed_time_train_only:.2f}" if elapsed_time_train_only is not None else 'nan',
                                f"{avg_batch_time:.4f}" if avg_batch_time is not None else 'nan',
                                patience_counter if patience_counter is not None else 'nan',
                                f"{model_memory_usage_mb_train_only:.3f}" if model_memory_usage_mb_train_only is not None else 'nan'
                            ])
                        
                        # Update GUI via signal (for non-validation epochs)
                        progress_data_train_only = {
                            'epoch': epoch,
                            'train_loss_norm': train_loss_norm, 
                            'val_loss_norm': float('nan'),      
                            'train_rmse_scaled': train_rmse_for_gui_no_val, 
                            'val_rmse_scaled': float('nan'),         
                            'best_val_rmse_scaled': getattr(self, f'_task_{task["task_id"]}_best_val_rmse_orig', float('inf')), 
                            'error_unit_label': current_error_unit_label_no_val, 
                            'delta_t_epoch': formatted_epoch_time, 
                            'elapsed_time': format_time(elapsed_time_train_only), 
                            'patience_counter': patience_counter, 
                            'learning_rate': current_lr, 
                            'status': f"Epoch {epoch}/{max_epochs} - Training..."
                        }
                        update_progress_callback.emit(progress_data_train_only)
                        # self.log_to_sqlite(
                        #     task=task, epoch=epoch, train_loss=train_loss_norm,
                        #     val_loss=float('nan'),
                        #     best_val_loss=best_validation_loss,
                        #     elapsed_time=time.time() - start_time,
                        #     avg_batch_time=avg_batch_time, early_stopping=False,
                        #     model_memory_usage=round(model_memory_usage_mb, 3), current_lr=current_lr
                        # )
                    elif (epoch == 1 or epoch % valid_freq == 0 or epoch == max_epochs): # If it IS a validation epoch but did NOT early stop
                        model_memory_usage = torch.cuda.memory_allocated() if torch.cuda.is_available() else sys.getsizeof(model)
                        model_memory_usage_mb = model_memory_usage / (1024 * 1024)
                        # self.log_to_sqlite(
                        #     task=task, epoch=epoch, train_loss=train_loss_norm,
                        #     val_loss=val_loss_norm,
                        #     best_val_loss=best_validation_loss,
                        #     elapsed_time=elapsed_time,
                        #     avg_batch_time=avg_batch_time, early_stopping=False,
                        #     model_memory_usage=round(model_memory_usage_mb, 3), current_lr=current_lr
                        # )
                
                scheduler.step()

                # Log data to CSV and SQLite after each epoch (whether validated or not)
                #print(f"Checking log files for the task: {task['task_id']}: task['csv_log_file'], task['db_log_file']")

                # Save log data to CSV and SQLite
                # self.log_to_csv(task, epoch, train_loss, val_loss, elapsed_time, current_lr, best_validation_loss, delta_t_epoch)
                model_memory_usage = torch.cuda.memory_allocated() if torch.cuda.is_available() else sys.getsizeof(model)
                model_memory_usage_mb = model_memory_usage / (1024 * 1024)  # Convert to MB

                new_lr = optimizer.param_groups[0]['lr']
                if new_lr != current_lr:
                    print(f"Learning rate changed from {current_lr:.8f} to {new_lr:.8f} at epoch {epoch}")
                    logging.info(f"Learning rate changed from {current_lr:.8f} to {new_lr:.8f} at epoch {epoch}")
                    current_lr = new_lr

                # Scheduler step condition: Either when lr_drop_period is reached or patience_counter exceeds the threshold
                # Scheduler step condition: Check for drop period or patience_counter with buffer consideration
                # if (epoch % lr_drop_period == 0 or patience_counter > patience_threshold) and (epoch - last_lr_drop_epoch > lr_drop_buffer):
                #     print(f"Learning rate before scheduler step: {optimizer.param_groups[0]['lr']: .8f}\n")
                #     scheduler.step()
                #     current_lr = optimizer.param_groups[0]['lr']
                #     print(f"Current learning rate updated at epoch {epoch}: {current_lr: .8f}\n")
                #     logging.info(f"Current learning rate updated at epoch {epoch}: {current_lr: .8f}\n")
                #     last_lr_drop_epoch = epoch
                # else:
                #     print(f"Epoch {epoch}: No LR drop. patience_counter={patience_counter}, patience_threshold={patience_threshold}\n")
    
                # Log data to SQLite
                #commented out for testing db error
                # self.log_to_sqlite(
                #     task=task,
                #     epoch=epoch,
                #     train_loss=train_loss,
                #     val_loss=val_loss,
                #     best_val_loss=best_validation_loss,
                #     elapsed_time=elapsed_time,
                #     avg_batch_time=avg_batch_time,
                #     early_stopping=early_stopping,
                #     model_memory_usage=round(model_memory_usage_mb, 3),  # Memory in MB
                #     current_lr=current_lr  # Pass updated learning rate here
                # )

            if self.stop_requested:
                print("Training was stopped early. Exiting...")
                self.logger.info("Training was stopped early. Exiting...")

            # Final save and cleanup
            # self.save_model(task) # REMOVED: Best model is saved during validation improvement.
            # self.logger.info(f"Training loop finished for task {task['task_id']}. Best model is at: {task.get('training_params', {}).get('best_model_path')}") # Redundant
            
            # Log final summary
            with open(os.path.join(task['model_dir'], 'training_summary.txt'), 'w') as f:
                f.write(f"Training completed\n")
                f.write(f"Best validation loss: {best_validation_loss:.6f}\n")
                f.write(f"Final learning rate: {optimizer.param_groups[0]['lr']:.8f}\n")
                f.write(f"Stopped at epoch: {epoch}/{max_epochs}\n")
            
            # REMOVED unconditional final save: self.save_model(task)
            self.logger.info(f"Training loop finished for task {task['task_id']}. Best model is at: {task.get('training_params', {}).get('best_model_path')}")

            update_progress_callback.emit({'task_completed': True})
            self.logger.info("Training task completed")
        # Correctly indented except for the try block starting at line 318
        except Exception as e:
            self.logger.error(f"Error during training for task {task.get('task_id', 'N/A')}: {str(e)}", exc_info=True)
            update_progress_callback.emit({'task_error': str(e)})
        # Correctly indented finally for the try block starting at line 318
        finally:
            best_model_path_final = task.get('training_params', {}).get('best_model_path', 'N/A')
            job_folder_final = self.job_manager.get_job_folder()
            if best_model_path_final != 'N/A' and job_folder_final in best_model_path_final:
                relative_best_model_path = os.path.relpath(best_model_path_final, job_folder_final)
            else:
                relative_best_model_path = best_model_path_final
            self.logger.info(f"Finished run_training attempt for task {task.get('task_id', 'N/A')}. Best model (if saved): {relative_best_model_path}")
    # End of run_training method, convert_hyperparams should be at class level indentation

    def convert_hyperparams(self, hyperparams):
        """Converts all relevant hyperparameters to the correct types."""
        hyperparams['LAYERS'] = int(hyperparams['LAYERS'])
        hyperparams['HIDDEN_UNITS'] = int(hyperparams['HIDDEN_UNITS'])
        hyperparams['BATCH_SIZE'] = int(hyperparams['BATCH_SIZE'])
        hyperparams['MAX_EPOCHS'] = int(hyperparams['MAX_EPOCHS'])
        hyperparams['INITIAL_LR'] = float(hyperparams['INITIAL_LR'])
        
        # Update scheduler parameter names to match task info
        if hyperparams['SCHEDULER_TYPE'] == 'StepLR':
            hyperparams['LR_DROP_PERIOD'] = int(hyperparams['LR_PERIOD'])  # Map LR_PERIOD to LR_DROP_PERIOD
            hyperparams['LR_DROP_FACTOR'] = float(hyperparams['LR_PARAM'])  # Map LR_PARAM to LR_DROP_FACTOR
        else:
            hyperparams['PLATEAU_PATIENCE'] = int(hyperparams['PLATEAU_PATIENCE'])
            hyperparams['PLATEAU_FACTOR'] = float(hyperparams['PLATEAU_FACTOR'])
        hyperparams['VALID_PATIENCE'] = int(hyperparams['VALID_PATIENCE'])
        hyperparams['ValidFrequency'] = int(hyperparams['ValidFrequency'])
        hyperparams['LOOKBACK'] = int(hyperparams['LOOKBACK'])
        hyperparams['REPETITIONS'] = int(hyperparams['REPETITIONS'])
        return hyperparams

    def save_model(self, task, save_path=None):
        """Save the trained model to disk. Uses save_path if provided, else defaults to task['model_path']."""
        try:
            path_to_save = save_path if save_path else task.get('model_path')
            
            if path_to_save is None:
                self.logger.error("No valid save path provided or found in task for saving model.")
                raise ValueError("No valid save path for model.")

            model = task['model']
            if model is None:
                self.logger.error("No model instance found in task.")
                raise ValueError("No model instance found in task.")

            # Save full model for internal use (current workflow)
            torch.save(model, path_to_save) # Use path_to_save
            self.logger.info(f"Model saved to {path_to_save}") # Log actual save path

            # Save portable version (if this is the best model)
            is_best_model_save = save_path and os.path.basename(save_path) == 'best_model.pth'
            if is_best_model_save:
                task_dir = os.path.dirname(path_to_save) # Use path_to_save
                export_path = os.path.join(task_dir, 'best_model_export.pt') # Differentiate export name
                
                # Get model definition code
                model_def = """
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_units, num_layers, output_size=1, device='cpu'):
        super(LSTMModel, self).__init__()
        self.input_size = input_size
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.output_size = output_size
        self.device = device

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_units,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_units, output_size)
    
    def forward(self, x, h_s=None, h_c=None):
        # Initialize hidden state and cell state if not provided
        if h_s is None or h_c is None:
            h_s = torch.zeros(self.num_layers, x.size(0), self.hidden_units).to(self.device)
            h_c = torch.zeros(self.num_layers, x.size(0), self.hidden_units).to(self.device)
        
        # Forward pass through LSTM
        out, (h_s, h_c) = self.lstm(x, (h_s, h_c))
        
        # Get output from last time step
        out = self.fc(out[:, -1, :])
        return out, (h_s, h_c)
"""

                # Create export dictionary with all necessary information
                export_dict = {
                    'state_dict': model.state_dict(),
                    'model_definition': model_def,
                    'model_metadata': task['model_metadata'],
                    'hyperparams': {
                        'input_size': task['hyperparams']['INPUT_SIZE'],
                        'hidden_size': task['hyperparams']['HIDDEN_UNITS'],
                        'num_layers': task['hyperparams']['LAYERS'],
                        'output_size': task['hyperparams']['OUTPUT_SIZE']
                    },
                    'data_config': {
                        'feature_columns': task['data_loader_params']['feature_columns'],
                        'target_column': task['data_loader_params']['target_column'],
                        'lookback': task['data_loader_params']['lookback']
                    },
                    'model_type': task['model_metadata']['model_type'],
                    'export_timestamp': time.strftime("%Y%m%d-%H%M%S")
                }

                # Save the export dictionary
                torch.save(export_dict, export_path)
                self.logger.info(f"Best model exported to {export_path}")

                # Create a README file with multiple loading options for the best model
                readme_path = os.path.join(task_dir, 'BEST_MODEL_LOADING_INSTRUCTIONS.md')
                readme_content = f"""# Model Loading Instructions

## Model Details
- Model Type: {task['model_metadata']['model_type']}
- Input Size: {task['hyperparams']['INPUT_SIZE']}
- Hidden Units: {task['hyperparams']['HIDDEN_UNITS']}
- Layers: {task['hyperparams']['LAYERS']}
- Output Size: {task['hyperparams']['OUTPUT_SIZE']}
- Lookback: {task['data_loader_params']['lookback']}

## Feature Configuration
- Input Features: {', '.join(task['data_loader_params']['feature_columns'])}
- Target Variable: {task['data_loader_params']['target_column']}

## Loading Options

### Option 1: Using VEstim Environment
```python
import torch
from vestim.services.model_training.src.LSTM_model_service_test import LSTMModelService

# Load the exported model
checkpoint = torch.load('model_export.pt')

# Create model instance
model_service = LSTMModelService()
model = model_service.create_model(
    input_size=checkpoint['hyperparams']['input_size'],
    hidden_size=checkpoint['hyperparams']['hidden_size'],
    num_layers=checkpoint['hyperparams']['num_layers'],
    output_size=checkpoint['hyperparams']['output_size']
)

# Load state dict
model.load_state_dict(checkpoint['state_dict'])
model.eval()  # Set to evaluation mode
```

### Option 2: Standalone Usage (No VEstim Required)
```python
import torch
import torch.nn as nn

# Load the checkpoint
checkpoint = torch.load('model_export.pt')

# Execute the model definition code (included in the checkpoint)
exec(checkpoint['model_definition'])

# Create model instance
model = LSTMModel(
    input_size=checkpoint['hyperparams']['input_size'],
    hidden_units=checkpoint['hyperparams']['hidden_size'],
    num_layers=checkpoint['hyperparams']['num_layers'],
    output_size=checkpoint['hyperparams']['output_size']
)

# Load state dict
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Example usage:
def predict(model, input_data):
    with torch.no_grad():
        output, _ = model(input_data)
    return output
```

## Input Data Format
- Input shape should be: (batch_size, lookback, input_size)
- Features should be in order: {', '.join(task['data_loader_params']['feature_columns'])}
- All inputs should be normalized using the same scaling as training data

## Example Preprocessing
```python
import numpy as np

def preprocess_data(data, lookback={task['data_loader_params']['lookback']}):
    # Ensure data is normalized using the same scaling as training
    # Create sequences of length 'lookback'
    sequences = []
    for i in range(len(data) - lookback + 1):
        sequences.append(data[i:(i + lookback)])
    return torch.FloatTensor(np.array(sequences))
```

## Making Predictions
```python
# Example prediction
input_sequence = preprocess_data(your_data)  # Shape: (1, lookback, input_size)
with torch.no_grad():
    prediction, _ = model(input_sequence)
```
"""
                with open(readme_path, 'w') as f:
                    f.write(readme_content)
                self.logger.info(f"Best model loading instructions saved to {readme_path}")

        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise

    def stop_task(self):
        self.stop_requested = True  # Set the flag to request a stop
        if self.training_thread and self.training_thread.isRunning():  # Use isRunning() instead of is_alive()
            print("Attempting to gracefully stop the training thread...")
            self.training_thread.quit()  # Gracefully stop the thread
            if self.training_thread.wait(7000):  # Wait for the thread to finish cleanly
                print("Training thread has finished.")
                self.logger.info("Training thread has finished after stop request.")
            else:
                print("Training thread did not finish cleanly after 7 seconds.")
                self.logger.warning("Training thread did not finish cleanly after stop request and 7s wait.")
