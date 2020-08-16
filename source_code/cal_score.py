from dataloader import *
from scores import *
from utils import *
import torch


def cal_bleu(model, hidden_size = 256,print_result = False):
    test_vocab = load_data('./data/test.txt')
    input_tense_list  = [0, 0, 0, 0, 3, 0, 3, 2, 2, 2]
    target_tense_list = [3, 2, 1, 1, 1, 2, 0, 0, 3, 1]
    
    score = 0.
    
    model.eval()
    for i in range(10):
        input_tense = input_tense_list[i]  
        target_tense = target_tense_list[i]
        input_seq = seq_from_str(test_vocab[i][0])
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize hidden feature
        hidden = torch.zeros(1, 1, hidden_size, device=device)
        # Eval
        result, mu, logvar = model(input_seq, hidden, input_tense, target_tense)
        # Strim EOS
        pred_seq = str_from_tensor(result)[:-1]
        # Calculate BLEU-4 score
        score += compute_bleu(test_vocab[i][1], pred_seq)
        if print_result:
            print('---------------------------------')
            print('input : ', test_vocab[i][0])
            print('target: ', test_vocab[i][1])
            print('pred  : ', pred_seq)
    
    score = score/10.
    if print_result:
        print('=================================')
        print('Average BLEU-4 score = ',score/10.)
        
    return score


def cal_gaussian(model, latent_size=32, print_result = False):
    result = []
    model.eval()
    for i in range(100):
        pred_tuple = []
        latent = torch.randn(1, 1, latent_size, device=model.device)
        
        for cond in range(4):
            pred_seq = model.decode(model.fc_extend_latent(latent), cond)
            pred_seq = str_from_tensor(pred_seq)
            pred_tuple.append(pred_seq[:-1])
            
        result.append(pred_tuple)
        
    score = Gaussian_score(result)
    if print_result:
        for seq_tuple in result:
            print(seq_tuple)
        print('Gaussian score: ', score)
    
    return score

