import os
import dill
import torch
from models import Encoder, Decoder, Seq2Seq
from tokenizer import tokenize_kr
from torch_utils import translate_sentence, list2str_response

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
device = torch.device('cpu')

with open('../model/SRC.Field', 'rb') as file:
    SRC = dill.load(file)
with open('../model/TRG.Field', 'rb') as file:
    TRG = dill.load(file)

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

model.load_state_dict(torch.load('../model/seq2seq_model.pt', map_location=torch.device('cpu')))

raw_input = input('질문하세요 : ')
ko_input = tokenize_kr(raw_input)

response = translate_sentence(model, ko_input, SRC, TRG, device)
print('---------------------------질문---------------------------\n')
print(f'{raw_input}')
print('---------------------------답변---------------------------\n')
print(f'{list2str_response(response)}')
