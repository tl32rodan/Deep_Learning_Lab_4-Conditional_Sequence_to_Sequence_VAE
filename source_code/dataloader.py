def load_data(filename):
    vocab = []
    with open(filename) as f:
        for line in iter(f):
            line = line[:-1].split(' ')
            vocab.append(line)
        f.close()
    
    return vocab
