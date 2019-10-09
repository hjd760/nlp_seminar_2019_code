import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ConvEncoder(nn.Module):
    def __init__(self, src_vocab_size, embed_dim, hidden_dim, num_layers,
                    kernel_size, dropout, device):
        super().__init__()

        assert kernel_size % 2 == 1

        self.src_vocab_size = src_vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.device = device

        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)

        self.token_embedding = nn.Embedding(src_vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(100, embed_dim)

        self.embed2hidden = nn.Linear(embed_dim, hidden_dim)
        self.hidden2embed = nn.Linear(hidden_dim, embed_dim)

        conv_layer = nn.Conv1d(in_channels=hidden_dim, out_channels = 2 * hidden_dim,
                            kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.convs = nn.ModuleList([conv_layer for _ in range(num_layers)])

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        position = torch.arange(0, src.shape[1]).unsqueeze(0).repeat(src.shape[0], 1).to(self.device)

        token_embedded = self.token_embedding(src)
        position_embedded = self.position_embedding(position)

        enc_input = self.dropout(token_embedded + position_embedded)

        conv_input = self.embed2hidden(enc_input)
        conv_input = conv_input.permute(0, 2, 1)


        for i, conv in enumerate(self.convs):
            conv_output = conv(self.dropout(conv_input))

            conv_output = F.glu(conv_output, dim=1)
            conv_output = (conv_output + conv_input) * self.scale
            conv_input = conv_output

        enc_output = self.hidden2embed(conv_output.permute(0, 2, 1))
        condition = (enc_output + enc_input) * self.scale

        return enc_output, condition

class ConvDecoder(nn.Module):
    def __init__(self, trg_vocab_size, embed_dim, hidden_dim, num_layers, kernel_size,
                    dropout, pad_idx, device):
        super().__init__()

        self.trg_vocab_size = trg_vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.pad_idx = pad_idx
        self.device = device

        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)

        self.token_embedding = nn.Embedding(trg_vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(100, embed_dim)

        self.embed2hidden = nn.Linear(embed_dim, hidden_dim)
        self.hidden2embed = nn.Linear(hidden_dim, embed_dim)

        self.attn_hidden2embed = nn.Linear(hidden_dim, embed_dim)
        self.attn_embed2hidden = nn.Linear(embed_dim, hidden_dim)

        self.output = nn.Linear(embed_dim, trg_vocab_size)

        conv_layer = nn.Conv1d(hidden_dim, 2 * hidden_dim, kernel_size)

        self.convs = nn.ModuleList([conv_layer for _ in range(num_layers)])

        self.dropout = nn.Dropout(dropout)

    def compute_attention(self, decoder_input, decoder_output, encoder_output, condition):
        dec_output = self.attn_hidden2embed(decoder_output.permute(0, 2, 1))
        dec_output = (dec_output + decoder_input) * self.scale

        attention = torch.matmul(dec_output, encoder_output.permute(0, 2, 1))
        attention = F.softmax(attention, dim=2)

        conditional_input = torch.matmul(attention, condition)
        conditional_input = self.attn_embed2hidden(conditional_input)

        attended_output = (decoder_output + conditional_input.permute(0, 2, 1)) * self.scale

        return attention, attended_output

    def forward(self, target, encoder_output, condition):
        position = torch.arange(0, target.shape[1]).unsqueeze(0).repeat(target.shape[0], 1).to(self.device)

        token_embedded = self.token_embedding(target)
        position_embedded = self.position_embedding(position)

        input_embedded = self.dropout(token_embedded + position_embedded)

        conv_input = self.embed2hidden(input_embedded)
        conv_input = conv_input.permute(0, 2, 1)

        for i, conv in enumerate(self.convs):
            conv_input = self.dropout(conv_input)
            print('input', conv_input.size())
            padding = torch.zeros(conv_input.shape[0], conv_input.shape[1], self.kernel_size-1)
            padding = padding.fill_(self.pad_idx).to(self.device)
            print('check :', padding.size())
            padded_conv_input = torch.cat((padding, conv_input), dim=2)
            print(padded_conv_input.size())
            conv_output = conv(padded_conv_input)
            conv_output = F.glu(conv_output, dim=1)

            attention, conv_output = self.compute_attention(
                                    input_embedded, conv_output, encoder_output, condition)

            conv_output = (conv_output + conv_input) * self.scale

            conv_input = conv_output

        conv_output = self.hidden2embed(conv_output.permute(0, 2, 1))

        dec_output = self.output(self.dropout(conv_output))

        return dec_output, attention

class ConvS2S(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg):
        enc_output, condition = self.encoder(src)

        dec_output, attention = self.decoder(trg, enc_output, condition)

        return dec_output, attention
