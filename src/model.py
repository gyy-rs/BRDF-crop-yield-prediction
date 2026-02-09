# File: model.py
# Description: LSTM-based model with multi-head attention for crop yield estimation

import torch
import torch.nn as nn


class AttentionLSTMModel(nn.Module):
    """
    Crop yield estimation model combining LSTM and multi-head self-attention mechanism.
    
    This model captures temporal dependencies in vegetation indices and SIF data
    while using attention to identify key temporal periods for yield prediction.
    """
    
    def __init__(self, input_dim, lstm_hidden_dim, n_heads, n_layers, dropout_rate):
        """
        Initialize the AttentionLSTMModel.
        
        Parameters
        ----------
        input_dim : int
            Number of input features per time step.
        lstm_hidden_dim : int
            Number of hidden units in LSTM layers.
        n_heads : int
            Number of attention heads in multi-head attention.
        n_layers : int
            Number of LSTM layers.
        dropout_rate : float
            Dropout rate for regularization (0.0 to 1.0).
        """
        super(AttentionLSTMModel, self).__init__()
        
        # LSTM layers for temporal sequence processing
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout_rate if n_layers > 1 else 0,
            bidirectional=False
        )
        
        # Multi-head attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden_dim,
            num_heads=n_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Regression head: maps LSTM hidden state to yield prediction
        self.regressor = nn.Sequential(
            nn.Linear(lstm_hidden_dim, lstm_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_hidden_dim // 2, 1)
        )

    def forward(self, x):
        """
        Forward pass of the model.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, time_steps, input_dim)
        
        Returns
        -------
        torch.Tensor
            Yield prediction of shape (batch_size, 1)
        """
        # Process temporal sequence through LSTM
        lstm_out, _ = self.lstm(x)
        
        # Apply multi-head self-attention
        attn_output, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use the last time step representation
        final_representation = attn_output[:, -1, :]
        
        # Generate yield prediction
        output = self.regressor(self.dropout(final_representation))
        
        return output
