import os
import random
import torch
import torch.optim as optim
import torch.nn as nn
import time
import math
import dill

from torchtext.data import Field, BucketIterator, TabularDataset
from torch_utils import init_weights, count_parameters, train, epoch_time, translate_sentence, list2str_response
from models import Encoder, Decoder, Seq2Seq
from tokenizer import tokenize_kr

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

SEED = 1234

random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

SRC = Field(tokenize=tokenize_kr,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True,
            include_lengths=True)

TRG = Field(tokenize=tokenize_kr,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True)

train_data = TabularDataset(path='../datasets/gachon_chatbot.tsv', format='tsv', fields=[('src', SRC), ('trg', TRG)])
SRC.build_vocab(train_data, min_freq = 2)
TRG.build_vocab(train_data, min_freq = 2)
device = torch.device('cuda:0')

BATCH_SIZE = 128

train_iterator = BucketIterator(
    train_data, batch_size=BATCH_SIZE, sort_within_batch=True, sort_key=lambda x: len(x.src), device=device)

INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
PAD_IDX = SRC.vocab.stoi['<pad>']
SOS_IDX = TRG.vocab.stoi['<sos>']
EOS_IDX = TRG.vocab.stoi['<eos>']

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

model = Seq2Seq(enc, dec, PAD_IDX, SOS_IDX, EOS_IDX, device).to(device)
model.apply(init_weights)
print(f'The model has {count_parameters(model):,} trainable parameters')
optimizer = optim.Adam(model.parameters())

PAD_IDX = TRG.vocab.stoi['<pad>']
print(f'PAD_IDX : {PAD_IDX}')
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

with open("../model/SRC.Field","wb")as f:
     dill.dump(SRC, f)
with open("../model/TRG.Field","wb")as f:
     dill.dump(TRG, f)

N_EPOCHS = 10
CLIP = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    start_time = time.time()

    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    torch.save(model.state_dict(), '../model/seq2seq_model.pt')

    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')


