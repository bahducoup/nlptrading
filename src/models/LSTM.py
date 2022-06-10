import torch.nn as nn
import torch

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, static_input_size = 1,  hidden_layer_size=32, num_layers=2, output_size=1, dropout=0.2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.linear_1 = nn.Linear(input_size, hidden_layer_size)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(hidden_layer_size, hidden_size=self.hidden_layer_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(static_input_size, hidden_layer_size)
        self.linear_3 = nn.Linear((1 + num_layers)*hidden_layer_size, output_size)
        
        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                 nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                 nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                 nn.init.orthogonal_(param)

    def forward(self, x1, x2):
        batchsize = x1.shape[0]

        # layer 1
        x1 = self.linear_1(x1)
        x1 = self.relu(x1)


        # layer 1
        x2 = self.linear_2(x2)
        x2 = self.relu(x2)
        
        # LSTM layer
        lstm_out, (h_n, c_n) = self.lstm(x1)

        # reshape output from hidden cell into [batch, features] for `linear_2`
        x1 = h_n.permute(1, 0, 2).reshape(batchsize, -1) 
        # layer 2
        
        x = torch.cat((x1, x2), 1)
        x = self.dropout(x)

        predictions = self.linear_3(x)
        return predictions[:,-1]