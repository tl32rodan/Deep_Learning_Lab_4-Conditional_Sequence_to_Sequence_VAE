import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from encoder import EncoderRNN
from decoder import DecoderRNN

MAX_PRED_LENGTH = 25


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
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def encode(self, input_seq, hidden):
        # Necessary modification for LSTM
        hidden = (hidden,hidden)
        
        for c in input_seq:
            input_index = torch.tensor([[c]], device=self.device)
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
        decoder_input = torch.tensor([[self.SOS_token]], device=self.device)
        
        if use_teacher_forcing:
            # target_tensor should not be None
            assert target_tensor is not None, "target_tensor should be specified"
            
            # Teacher forcing: Feed the target as the next input
            for i in range(len(target_tensor)):
                decoder_output, hidden = self.decoder(decoder_input, hidden)
                result.append(decoder_output[0])
                
                decoder_input = target_tensor[i]  # Teacher forcing
                decoder_input = torch.tensor([[decoder_input]], device=self.device)

        else:
            # Without teacher forcing: use its own predictions as the next input
            for i in range(MAX_PRED_LENGTH):
                decoder_output, hidden = self.decoder(decoder_input, hidden)
                topv, topi = decoder_output.topk(1)                
                result.append(decoder_output[0])
                
                decoder_input = topi.squeeze().detach()  # detach from history as input
                decoder_input = torch.tensor([[decoder_input]], device=self.device)

                if decoder_input.item() == self.EOS_token: 
                    break
    
        return torch.stack()

    def forward(self, x, hidden, use_teacher_forcing = False, target_tensor = None):
        # Encode
        mu, logvar = self.encode(x, hidden)
        
        # Reparameterize
        hidden = self.reparameterize(mu, logvar)
        
        result = self.decode(hidden, use_teacher_forcing, target_tensor)
        
        return result, mu, logvar


# Conditional VAE model
class CondVAE(nn.Module):
    def __init__(self,input_size, hidden_size, output_size, teacher_forcing_ratio = 1.0):
        super(CondVAE, self).__init__()
        self.hidden_size = hidden_size
        # Condition matters
        self.condition_embedding_size = 8
        self.num_conditions = 4
        
        self.latent_size = 32
        self.teacher_forcing_ratio = teacher_forcing_ratio
        
        # Embedding of condition
        self.encoder_condition_embedding = nn.Embedding(self.num_conditions, self.condition_embedding_size)
        self.encoder   = EncoderRNN(input_size, self.hidden_size+self.condition_embedding_size)
        
        self.fc_mu     = nn.Linear(self.hidden_size+self.condition_embedding_size, self.latent_size)
        self.fc_logvar = nn.Linear(self.hidden_size+self.condition_embedding_size, self.latent_size)
        self.fc_extend_latent = nn.Linear(self.latent_size, self.hidden_size)
        
        self.decoder_condition_embedding = nn.Embedding(self.num_conditions, self.condition_embedding_size)
        self.decoder   = DecoderRNN(self.hidden_size+self.condition_embedding_size, output_size)
        
        #The number of vocabulary
        self.vocab_size = 28
        self.SOS_token = 0
        self.EOS_token = self.vocab_size-1
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def encode(self, input_seq, hidden, encode_cond):
        # Necessary modification for LSTM
        hidden = (hidden,hidden)
        
        # Get condition embedding
        encode_cond = torch.tensor([[encode_cond]], device=self.device)
        cond_embeded = self.encoder_condition_embedding(encode_cond).view(1, 1, -1)
        # Concat hidden and condition
        hidden = (torch.cat((hidden[0], cond_embeded),2), torch.cat((hidden[1], cond_embeded),2))
        
        for c in input_seq:
            input_index = torch.tensor([[c]], device=self.device)
            _, hidden = self.encoder(input_index, hidden)
        
        
        ######### RELU hidden ############
        hidden = F.relu(hidden[1])
        
        mu     = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        
        # eps is generated from N(0,I)
        eps = torch.randn_like(std)
        
        latent = mu + eps*std
        
        # Extend latent
        return self.fc_extend_latent(latent)

    def decode(self, hidden, decode_cond, use_teacher_forcing = False, target_tensor = None):
        result = []
        
        # Get condition embedding
        decode_cond = torch.tensor([[decode_cond]], device=self.device)
        cond_embeded = self.decoder_condition_embedding(decode_cond).view(1, 1, -1)
        
        # Concat hidden and condition
        hidden = (torch.cat((torch.zeros(1, 1, self.hidden_size, device=self.device), cond_embeded),2),\
                  torch.cat((hidden, cond_embeded),2))
        
        # Set decoder_input as SOS
        decoder_input = torch.tensor([[self.SOS_token]], device=self.device)
        
        if use_teacher_forcing:
            # target_tensor should not be None
            assert target_tensor is not None, "target_tensor should be specified"
            
            # Teacher forcing: Feed the target as the next input
            for i in range(len(target_tensor)):
                decoder_output, hidden = self.decoder(decoder_input, hidden)
                result.append(decoder_output[0])
                
                decoder_input = target_tensor[i]  # Teacher forcing
                decoder_input = torch.tensor([[decoder_input]], device=self.device)

        else:
            # Without teacher forcing: use its own predictions as the next input
            for i in range(MAX_PRED_LENGTH):
                decoder_output, hidden = self.decoder(decoder_input, hidden)
                topv, topi = decoder_output.topk(1)                
                result.append(decoder_output[0])
                
                decoder_input = topi.squeeze().detach()  # detach from history as input
                decoder_input = torch.tensor([[decoder_input]], device=self.device)

                if decoder_input.item() == self.EOS_token:
                    break
                    
        return torch.stack(result)

    def forward(self, x, hidden, encode_cond, decode_cond, use_teacher_forcing = False, target_tensor = None):
        # Encode
        mu, logvar = self.encode(x, hidden, encode_cond)
        
        # Reparameterize
        hidden = self.reparameterize(mu, logvar)
        
        result = self.decode(hidden, decode_cond, use_teacher_forcing, target_tensor)
        
        return result, mu, logvar


# Reference: https://github.com/pytorch/examples/blob/master/vae
# Reconstruction + KL divergence losses summed over all elements and batch
def VAE_Loss(recon_x, x, mu, logvar):
    CE = nn.CrossEntropyLoss(reduction='sum')
    CE_loss = CE(recon_x, x)
    CE_loss = CE_loss/len(x)
    
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return CE_loss+KLD


def VAE_Loss_CE(recon_x, x):
    CE = nn.CrossEntropyLoss(reduction='sum')
    CE_loss = CE(recon_x, x)
    CE_loss = CE_loss/len(x)
    return CE_loss


def VAE_Loss_KLD(mu, logvar):
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return KLD


## Use KL annealing
def KL_annealing(current_iter, policy = 'mono', mono_reach_max = 300000,
                cycl_reach_max = 50000, cycl_period = 100000,
                heuristic_base = 1e-4, heuristic_rate = 1e-3,
                heuristic_grow_every = 50):
    if policy == 'mono':
        beta = 1 if current_iter >= mono_reach_max else (current_iter+1)/mono_reach_max
    elif policy == 'cyclical':
        beta = 1 if current_iter%cycl_period >= cycl_reach_max else ((current_iter+1)%cycl_period)/cycl_reach_max
    elif policy == 'heuristic':
        beta = heuristic_base*( (1+heuristic_rate)**(current_iter/heuristic_grow_every) )
    else:
        raise ValueError
        
    return beta
