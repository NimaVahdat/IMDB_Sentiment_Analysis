import torch.nn as nn
import torch

import torch.nn as nn
import torch

class RNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim,
        n_layers,
        bidirectional,
        dropout_rate,
        pad_index,
    ):
        """
        Initialize the RNN model.

        Args:
            vocab_size (int): The size of the vocabulary.
            embedding_dim (int): The dimension of word embeddings.
            hidden_dim (int): The dimension of the hidden state.
            output_dim (int): The dimension of the output.
            n_layers (int): Number of recurrent layers.
            bidirectional (bool): Whether to use bidirectional RNN.
            dropout_rate (float): Dropout rate.
            pad_index (int): Index used for padding.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        self.rnn = nn.RNN(
            embedding_dim,
            hidden_dim,
            n_layers,
            bidirectional=bidirectional,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, ids, length):
        """
        Forward pass of the RNN model.

        Args:
            ids (torch.Tensor): Input sequences.
            length (torch.Tensor): Lengths of input sequences.

        Returns:
            torch.Tensor: Output predictions.
        """
        # Embed input sequences
        embedded = self.dropout(self.embedding(ids))
        
        # Pack padded sequences
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, length, batch_first=True, enforce_sorted=False
        )

        # Pass through RNN layer
        packed_output, hidden = self.rnn(packed_embedded)

        # Unpack packed sequences
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output)

        # Extract the hidden state (last layer) and apply dropout
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat([hidden[-2], hidden[-1]], dim=-1))
        else:
            hidden = self.dropout(hidden[-1])

        # Pass through fully connected layer
        prediction = self.fc(hidden)

        return prediction


class LSTM(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim,
        n_layers,
        bidirectional,
        dropout_rate,
        pad_index,
    ):
        """
        Initialize the LSTM model.

        Args:
            vocab_size (int): The size of the vocabulary.
            embedding_dim (int): The dimension of word embeddings.
            hidden_dim (int): The dimension of the hidden state.
            output_dim (int): The dimension of the output.
            n_layers (int): Number of recurrent layers.
            bidirectional (bool): Whether to use bidirectional LSTM.
            dropout_rate (float): Dropout rate.
            pad_index (int): Index used for padding.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            n_layers,
            bidirectional=bidirectional,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, ids, length):
        """
        Forward pass of the LSTM model.

        Args:
            ids (torch.Tensor): Input sequences.
            length (torch.Tensor): Lengths of input sequences.

        Returns:
            torch.Tensor: Output predictions.
        """
        # Embed input sequences
        embedded = self.dropout(self.embedding(ids))

        # Pack padded sequences
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, length, batch_first=True, enforce_sorted=False
        )

        # Pass through LSTM layer
        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        # Unpack packed sequences
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output)

        # Extract the hidden state (last layer) and apply dropout
        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat([hidden[-2], hidden[-1]], dim=-1))
        else:
            hidden = self.dropout(hidden[-1])

        # Pass through fully connected layer
        prediction = self.fc(hidden)

        return prediction


import torch.nn as nn
import torch

class GRU(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim,
        n_layers,
        bidirectional,
        dropout_rate,
        pad_index,
    ):
        """
        Initialize the GRU model.

        Args:
            vocab_size (int): The size of the vocabulary.
            embedding_dim (int): The dimension of word embeddings.
            hidden_dim (int): The dimension of the hidden state.
            output_dim (int): The dimension of the output.
            n_layers (int): Number of recurrent layers.
            bidirectional (bool): Whether to use bidirectional GRU.
            dropout_rate (float): Dropout rate.
            pad_index (int): Index used for padding.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            n_layers,
            bidirectional=bidirectional,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, ids, length):
        """
        Forward pass of the GRU model.

        Args:
            ids (torch.Tensor): Input sequences.
            length (torch.Tensor): Lengths of input sequences.

        Returns:
            torch.Tensor: Output predictions.
        """
        # Embed input sequences
        embedded = self.dropout(self.embedding(ids))

        # Pack padded sequences
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, length, batch_first=True, enforce_sorted=False
        )

        # Pass through GRU layer
        packed_output, hidden = self.gru(packed_embedded)

        # Unpack packed sequences
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output)

        # Extract the hidden state (last layer) and apply dropout
        if self.gru.bidirectional:
            hidden = self.dropout(torch.cat([hidden[-2], hidden[-1]], dim=-1))
        else:
            hidden = self.dropout(hidden[-1])

        # Pass through fully connected layer
        prediction = self.fc(hidden)

        return prediction
