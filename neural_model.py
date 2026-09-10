"""Red neuronal compacta para predecir el siguiente pictograma por ID."""

import torch
from torch import nn


class GRUPictogramPredictor(nn.Module):
    def __init__(self, vocabulary_size: int, output_size: int, embedding_dim: int = 64,
                 hidden_dim: int = 128, layers: int = 1, dropout: float = 0.15) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, output_size)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)
        return self.output(self.dropout(hidden[-1]))
