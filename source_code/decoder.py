import torch
import torch.nn as nn


#Decoder
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        
    def forward(self, input, hidden):
        
        output = self.embedding(input).view(1, 1, -1)
        output, hidden = self.lstm(output, hidden)
        output = self.out(output[0])
        return output, hidden
