import math

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence


class PositionalEncoding(nn.Module):
    def __init__(self, feature_size, dropout, max_length):
        super().__init__()

        den = torch.exp(-torch.arange(0, feature_size, 2)* math.log(10000) / feature_size)
        pos = torch.arange(0, max_length).reshape(max_length, 1)
        pos_embedding = torch.zeros((max_length, feature_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0)
        
        self._dropout = nn.Dropout(dropout)
        self.register_buffer('_pos_embedding', pos_embedding)

    def forward(self, features):
        return self._dropout(features + self._pos_embedding[:, :features.size(1), :features.size(2)])
        

class TransformerModel(nn.Module):
  def __init__(
      self,
      input_size,
      static_input_size,
      sequence_length,  # length of each sequence, since our inputs are fixed-length
      d_model=512,
      nhead=16,
      dim_feedforward=256,
      dropout=0.2,
      num_encoder_layers=6
  ):
      super().__init__()

      encoder_layer = nn.TransformerEncoderLayer(
          d_model=d_model,
          nhead=nhead,
          dim_feedforward=dim_feedforward,
          dropout=dropout,
          batch_first=True
      )
      merged_input_size = input_size + static_input_size

      self.embedding = nn.Embedding(1, d_model)  # contains only 1 embedding for CLS-like token
      self.positional_encoding = PositionalEncoding(d_model, dropout, sequence_length + 2)

      self.input_linear = nn.Linear(merged_input_size, d_model)
      self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
      self.output_linear = nn.Linear(d_model, 1)

  def forward(self, x1, x2):
      # merge x1 and x2
      _, seq_len, _ = x1.shape
      x2 = x2.view(x2.shape[0], -1, x2.shape[-1])
      x2 = x2.repeat(1, seq_len, 1)
      x = torch.cat([x1, x2], dim=-1)
      
      x = self.input_linear(x)  # up-projection
      
      # append cls-like token to the beginning of the sequence
      cls = self.embedding(torch.tensor([[0]], device=x.device).long())
      cls = cls.repeat(x.shape[0], 1, 1)
      x = torch.cat([cls, x], dim=1)

      # forward
      x = self.positional_encoding(x)
      x = self.transformer_encoder(x)
      x = self.output_linear(x)

      x = torch.flatten(x[:,0,:])  # extract output corresponding to cls
      
      return x

