#!/usr/bin/env python
# coding: utf-8
# %%
from dataloader import *
from VAE import *
from scores import *
from train import *

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import random
import argparse


# %%
parser = argparse.ArgumentParser(description='Conditional VAE - Seq2Seq model')
parser.add_argument('--ckp_postfix', type=str, default='0', metavar='DATE_TIME',
                    help='Checkpoint storing name\'s postfix')
parser.add_argument('--annealing', type=str, default='mono', metavar='mono|cyclical|heuristic',
                    help='KL cost annealing strategy (default: mono)')
parser.add_argument('--teacher_forcing_ratio', type=float, default=0.7, metavar='0.0~1.0',
                   help='Teacher forcing ratio (default=0.7)')
parser.add_argument('--lr', type=float, default=0.05,
                   help='Learning rate (default=0.05)')
parser.add_argument('--n_epochs', type=int, default=10000, metavar='N',
                    help='Number of epochs (default: 10000)')
parser.add_argument('--print_every', type=int, default=10, metavar='N',
                    help='Print after how many epochs (default: 10)')
parser.add_argument('--save_every', type=int, default=10, metavar='N',
                    help='Save losses after how many epochs (default: 10)')
args = parser.parse_args()

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare data

# %%
train_vocab = load_data('./data/train.txt')
print('Data loaded!')

# %%
# Train VAE

vocab_size = 28 #The number of vocabulary
SOS_token = 0
EOS_token = vocab_size-1

# %%
## Setting hyperparameters
#----------Hyper Parameters----------#
hidden_size = 256
latent_size = 32


# %%
my_vae = CondVAE(vocab_size, hidden_size, vocab_size, args.teacher_forcing_ratio).to(device)


# %%
optimizer = optim.SGD(my_vae.parameters(), lr=args.lr)
lr_sch = optim.lr_scheduler.StepLR(optimizer, 500, gamma=0.8)

# %%
print('Start training...')
loss_list, ce_loss_list, kld_loss_list, bleu_list =  \
    trainIter_condVAE(my_vae, train_vocab, n_epochs=args.n_epochs, iter_per_epoch = 10,\
                      print_every=args.print_every, save_every=args.save_every, \
                      learning_rate=args.lr,teacher_forcing_ratio=args.teacher_forcing_ratio,\
                      optimizer= optimizer, scheduler = lr_sch, criterion_CE = VAE_Loss_CE,\
                      criterion_KLD = VAE_Loss_KLD, ckp_path = './models/model_'+args.ckp_postfix, \
                      kl_annealing = args.annealing)
# %%
plt.plot(kld_loss_list)
