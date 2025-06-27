import torch
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_units, num_layers, output_size=1, dropout_prob=0.0, device='cpu'):
        """
        Gated Recurrent Unit (GRU) Model.

        :param input_size: Number of input features.
        :param hidden_units: Number of features in the hidden state h.
        :param num_layers: Number of recurrent layers.
        :param output_size: Number of output features (typically 1 for regression).
        :param dropout_prob: Dropout probability for GRU layers (if num_layers > 1) and an optional final dropout.
        :param device: The device to run the model on ('cpu' or 'cuda').
        """
        super(GRUModel, self).__init__()
        self.input_size = input_size
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout_prob = dropout_prob
        self.device = device

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_units,
            num_layers=num_layers,
            batch_first=True, # Expects input: (batch, seq, feature)
            dropout=dropout_prob if num_layers > 1 else 0.0
        ).to(self.device)

        # Optional: A dropout layer before the final fully connected layer
        if dropout_prob > 0:
            self.dropout = nn.Dropout(p=dropout_prob).to(self.device)
        else:
            self.dropout = None

        self.fc = nn.Linear(hidden_units, output_size).to(self.device)

    def forward(self, x, h_0=None):
        """
        Forward pass for the GRU model.

        :param x: Input tensor of shape [batch_size, sequence_length, input_size].
        :param h_0: Initial hidden state of shape [num_layers, batch_size, hidden_units].
                    If None, it will be initialized to zeros.
        :return: Output tensor of shape [batch_size, sequence_length, output_size] (if fc applied to all outputs)
                 or [batch_size, output_size] (if fc applied to last output only),
                 and the final hidden state.
                 For simplicity and common use in sequence-to-one or sequence-to-sequence where
                 only the last output's prediction or all outputs are processed by fc,
                 this example applies fc to all outputs of GRU.
        """
        x = x.to(self.device)
        if h_0 is None:
            # Initialize hidden state if not provided
            h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_units, device=self.device).requires_grad_()
        else:
            h_0 = h_0.to(self.device)

        # GRU output: output features for each time step, and the final hidden state
        out, h_n = self.gru(x, h_0)
        # out shape: (batch_size, seq_len, hidden_units)
        # h_n shape: (num_layers, batch_size, hidden_units)

        if self.dropout:
            out = self.dropout(out) # Apply dropout to the GRU outputs

        # Apply the fully connected layer to each time step's output
        # This is a common approach for sequence tagging or if every step's output is needed.
        # If you only want to predict based on the last time step:
        # out = self.fc(out[:, -1, :]) # This would change output shape to [batch_size, output_size]
        # For now, let's apply to all time steps, consistent with how LSTMModel was structured.
        out = self.fc(out)
        # out shape: (batch_size, seq_len, output_size)

        return out, h_n