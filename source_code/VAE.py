import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
import random

MAX_PRED_LENGTH = 25


#Encoder
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.lstm(output, hidden)
        
        return output, hidden


#Decoder
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        
    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        
        output, hidden = self.lstm(output, hidden)
        return output, hidden


# VAE model
class VAE(nn.Module):
    def __init__(self,input_size, hidden_size, output_size, teacher_forcing_ratio = 1.0):
        super(VAE, self).__init__()

        self.encoder   = EncoderRNN(input_size, hidden_size)
        self.fc_mu     = nn.Linear(hidden_size, hidden_size)
        self.fc_logvar = nn.Linear(hidden_size, hidden_size)
        self.decoder   = DecoderRNN(hidden_size, output_size)
        self.teacher_forcing_ratio = teacher_forcing_ratio
        
        #The number of vocabulary
        self.vocab_size = 28
        self.SOS_token = 0
        self.EOS_token = self.vocab_size-1
        
    def encode(self, input_seq, hidden):
        for c in input_seq:
            input_index = torch.LongTensor([[c]])
            _, hidden = self.encoder(input_index, hidden)
        
        mu     = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        
        # eps is generated from N(0,I)
        eps = torch.randn_like(std)
        
        return mu + eps*std

    def decode(self, hidden, use_teacher_forcing = False, target_tensor = None):
        result = []
               
        # Set decoder_input as SOS
        decoder_input = decoder_input = torch.tensor([[self.SOS_token]])
        
        if use_teacher_forcing:
            # target_tensor should not be None
            assert target_tensor is not None, "target_tensor should be specified"
            
            # Teacher forcing: Feed the target as the next input
            for i in range(len(target_tensor)):
                decoder_output, hidden = self.decoder(decoder_input, hidden)
                topv, topi = decoder_output.topk(1)
                result.append(topi.squeeze().detach())
                
                decoder_input = target_tensor[i]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for i in range(MAX_PRED_LENGTH):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1)                
                result.append(topi.squeeze().detach())
                
                decoder_input = topi.squeeze().detach()  # detach from history as input

                if decoder_input.item() == self.EOS_token: 
                    break
    
        return result

    def forward(self, x, hidden, use_teacher_forcing = False, target_tensor = None):
        # Encode
        mu, logvar = self.encode(x, hidden)
        
        # Reparameterize
        hidden = self.reparameterize(mu, logvar)
        
        result = self.decode(hidden, use_teacher_forcing, target_tensor)
        
        return result, mu, logvar


# Reference: https://github.com/pytorch/examples/blob/master/vae
# Reconstruction + KL divergence losses summed over all elements and batch
def VAE_Loss(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

