# +
def seq_from_str(target):
    ord_a = ord('a')
    seq = [ord(c) - ord_a + 1 for c in target]
    
    return seq

def str_from_tensor(target):
    seq = ''
    for output in target:
        _, c = output.topk(1)
        seq += chr(c+ord('a')-1)

    return seq
