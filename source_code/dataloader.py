def load_data():
    train_vocab = []
    test_vocab = []
    with open('data/train.txt') as f:
        for line in iter(f):
            line = line[:-1].split(' ')
            train_vocab.append(line)
        f.close()
    
    with open('data/test.txt') as f:
        for line in iter(f):
            line = line[:-1].split(' ')
            test_vocab.append(line)
        f.close()

    return train_vocab, test_vocab
