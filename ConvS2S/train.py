import torch
from torch import nn
from torch import optim

from model import ConvEncoder, ConvDecoder, ConvS2S

from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator

import spacy

import random
import math
import time


def train(model, iterator, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()

        output, _ = model(src, trg[:,:-1])

        output = output.contiguous().view(-1, output.shape[-1])
        trg = trg[:, 1:].contiguous().view(-1)

        loss = criterion(output, trg)

        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            output, _ = model(src, trg[:, :-1])

            output = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def tokenize_de(text):

    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):

    return [tok.text for tok in spacy_en.tokenizer(text)]

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time/ 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

if __name__ == "__main__":
    SEED = 1234

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    spacy_de = spacy.load('de')
    spacy_en = spacy.load('en')

    SRC = Field(tokenize=tokenize_de,
            init_token = '<sos>',
            eos_token = '<eos>',
            lower = True,
            batch_first = True)

    TRG = Field(tokenize=tokenize_en,
                init_token = '<sos>',
                eos_token = '<eos>',
                lower = True,
                batch_first = True)

    train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'),
                                                    fields=(SRC, TRG))


    SRC.build_vocab(train_data, min_freq = 2)
    TRG.build_vocab(train_data, min_freq = 2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    BATCH_SIZE = 128

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
         batch_size = BATCH_SIZE,
         device = device)

    src_vocab_size = len(SRC.vocab)
    trg_vocab_size = len(TRG.vocab)
    embed_dim = 256
    hidden_dim = 512
    num_layers = 10
    kernel_size = 3
    dropout = 0.25
    pad_idx = TRG.vocab.stoi['<pad>']

    encoder = ConvEncoder(src_vocab_size, embed_dim, hidden_dim, num_layers, kernel_size, dropout, device)
    decoder = ConvDecoder(trg_vocab_size, embed_dim, hidden_dim, num_layers, kernel_size, dropout, pad_idx, device)
    model = ConvS2S(encoder, decoder, device).to(device)

    optimizer = optim.Adam(model.parameters())

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    num_epochs = 10
    clip = 1

    best_valid_loss = float('inf')

    for epoch in range(num_epochs):

        start_time = time.time()

        train_loss = train(model, train_iterator, optimizer, criterion, clip)
        valid_loss = evaluate(model, valid_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'convseq-model.pt')

        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
