# +
from dataloader import *
from VAE import *
from scores import *
from utils import *

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import random

# +
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab_size = 28 #The number of vocabulary
SOS_token = 0
EOS_token = vocab_size-1


# -

def infer_by_simple(vae_model, data_tuple):
    pred_tuple = []
    
    vae_model.eval()
    
    with torch.no_grad():
        for i in range(len(data_tuple)):
            input_tense = 0  # Input: simple present
            target_tense = i # Target: 4 tense results
            input_seq, target_seq = (seq_from_str(data_tuple[input_tense]),seq_from_str(data_tuple[target_tense])) 
            
            # Initialize hidden feature
            hidden = torch.zeros(1, 1, hidden_size, device=device)

            result, mu, logvar = vae_model(input_seq, hidden, input_tense, target_tense)
            
            pred_seq = str_from_tensor(result)
            pred_tuple.append(pred_seq[:-1])
            
    return pred_tuple


def train_condVAE(vae_model, input_seq, input_cond, target_seq, target_cond, use_teacher_forcing, optimizer,                   criterion_CE, criterion_KLD, kl_annealing_beta = 1):    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize hidden feature
    hidden = torch.zeros(1, 1, hidden_size, device=device)
        
    # Run model
    optimizer.zero_grad()
    if use_teacher_forcing:
        # input_cond is encoder condition; targer_cond is decoder condition
        result, mu, logvar = vae_model(input_seq, hidden, input_cond, target_cond, use_teacher_forcing, target_seq)
    else:
        result, mu, logvar = vae_model(input_seq, hidden, input_cond, target_cond, use_teacher_forcing, None)
            
            
    # Ground truth should have EOS in the end
    target_seq.append(EOS_token)
    
    # Calculate loss
    # First, we should strim the sequences by the length of smaller one
    min_len = min(len(target_seq),len(result))
        
    # hat_y need not to do one-hot encoding
    hat_y = result[:min_len]
    y = torch.tensor(target_seq[:min_len], device=device)
        
    ce_loss = criterion_CE(hat_y, y)
    kld_loss = criterion_KLD(mu, logvar)
    #print('------------------------------')
    #print('before: ',kld_loss )
    kld_loss = kl_annealing_beta * kld_loss # KL annealing
    #print('after: ',kld_loss )
    
    loss = ce_loss + kld_loss
        
    loss.backward()
    optimizer.step()
    
    return ce_loss.item(), kld_loss.item(), hat_y


def trainIter_condVAE(vae_model, data, n_epochs, iter_per_epoch = 300, 
                      print_every=100, save_every=100, 
                      learning_rate=0.01, teacher_forcing_ratio = 1.0, 
                      optimizer = None, scheduler = None,
                      criterion_CE = VAE_Loss_CE, criterion_KLD = VAE_Loss_KLD,
                      ckp_path = '', kl_annealing = 'mono'):
    '''
        data: A list of 4-tuple
              the tense order should be : (simple present, third person, present progressive, past)
    '''
    loss_list = []
    ce_loss_list = []
    kld_loss_list = []
    bleu_list = []
  
    # Check optimizer; default: SGD
    if optimizer is None:
        optimizer = optim.SGD(vae_model.parameters(), lr=learning_rate)
    
    avg_bleu = 0.
    avg_loss = 0.
    avg_ce   = 0.
    avg_kld  = 0.
    avg_counter = 0
    bleu_counter = 0
    
    for epoch in range(n_epochs): 
        # Randomly pick data
        data_tuples = [random.choice(data) for i in range(iter_per_epoch)]
        
        
        # KL annealing
        beta = KL_annealing(epoch, policy=kl_annealing)
        #beta = 1e-4
        #print('beta = ',beta)
        
        for data_tuple in data_tuples:
            
            # Calculate BLEU-4 score
            # Should execute before updating the model
            pred = infer_by_simple(vae_model, data_tuple)
            avg_bleu += compute_bleu(pred, data_tuple)
            bleu_counter += 1
            
            for i in range(4):
                for j in range(4):
                    input_tense = i # input tense
                    target_tense = j# target tense
                    input_seq = seq_from_str(data_tuple[input_tense])
                    target_seq = seq_from_str(data_tuple[target_tense])                  
                    # Determine whether to use teacher forcing
                    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

                    # Training
                    vae_model.train()
                    ce_loss, kld_loss, hat_y = train_condVAE(vae_model, input_seq, input_tense,                                                             target_seq, target_tense,                                                             use_teacher_forcing, optimizer,                                                              criterion_CE, criterion_KLD, beta)
                    # Loss
                    loss = ce_loss + kld_loss
                    #print('loss = ',loss,'; ce_loss = ', ce_loss, '; kld_loss = ',kld_loss)
                    avg_loss += loss
                    avg_ce   += ce_loss
                    avg_kld  += kld_loss                
                    avg_counter += 1 
        
        
        if scheduler is not None:
            scheduler.step()
        
            
        if (epoch+1) % print_every == 0:
            avg_bleu = avg_bleu/bleu_counter
            avg_loss = avg_loss/avg_counter
            avg_ce   = avg_ce/avg_counter
            avg_kld  = avg_kld/avg_counter
            
            loss_list.append(avg_loss)
            ce_loss_list.append(avg_ce)
            kld_loss_list.append(avg_kld)
            bleu_list.append(avg_bleu)

            print('-----------------')
            print('Iter %d: avg_loss = %.4f' % (epoch+1, avg_loss))
            print('Avg CE = ', avg_ce)
            print('Avg KLD = ', avg_kld)
            print('Beta = ',beta)
            print('Avg BLEU-4 score = ', avg_bleu)
            data_tuple = random.choice(data)
            pred_seq = infer_by_simple(vae_model, data_tuple)
            
            print('=========================')
            print('|| pred_seq = ', pred_seq)
            print('|| target_seq = ', data_tuple)
            print('=========================')
            # Reset 
            avg_bleu = 0.
            avg_loss = 0.
            avg_ce   = 0.
            avg_kld  = 0.
            avg_counter = 0
            bleu_counter = 0
            
        if (epoch+1) % save_every == 0:
            torch.save(vae_model, ckp_path+str(epoch+1))
    
    return loss_list, ce_loss_list, kld_loss_list, bleu_list
