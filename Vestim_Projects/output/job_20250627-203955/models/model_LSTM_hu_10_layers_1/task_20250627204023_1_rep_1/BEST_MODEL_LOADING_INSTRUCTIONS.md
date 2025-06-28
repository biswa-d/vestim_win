# Model Loading Instructions

## Model Details
- Model Type: LSTM
- Input Size: 3
- Hidden Units: 10
- Layers: 1
- Output Size: 1
- Lookback: 400

## Feature Configuration
- Input Features: Battery_Temp_degC, SOC, Power
- Target Variable: Voltage

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
- Features should be in order: Battery_Temp_degC, SOC, Power
- All inputs should be normalized using the same scaling as training data

## Example Preprocessing
```python
import numpy as np

def preprocess_data(data, lookback=400):
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
