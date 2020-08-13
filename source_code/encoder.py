import torch
import torch.nn as nn


#Encoder
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)

    def forward(self, input, hidden):
        # Necessary modification for LSTM
        hidden = (hidden,hidden)
        
        output = self.embedding(input).view(1, 1, -1)
        output, hidden = self.lstm(output, hidden)
        
        return output, hidden[0]


