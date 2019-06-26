from torch import nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from collections import namedtuple
import torch.nn.functional as F


class Generator(nn.Module):

    def __init__(self, hidden_size, output_size):
        super(Generator, self).__init__()

        self.output = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)  ##마지막 dim에 softmax를 하라...

    def forward(self, x):
        # |x| = (batch_size, length, hidden_size)

        y = self.softmax(self.output(x))
        # |y| = (batch_size, length, output_size)

        # Return log-probability instead of just probability.
        return y


class Seq2Seq(nn.Module):

    def __init__(self,
                 input_size,
                 word_vec_dim,
                 hidden_size,
                 output_size,
                 n_layers=4,
                 dropout_p=.2
                 ):
        self.input_size = input_size
        self.word_vec_dim = word_vec_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        super(Seq2Seq, self).__init__()

        self.embedding = nn.Embedding(input_size, word_vec_dim)
        self.rnn = nn.GRU(input_size=word_vec_dim,
                          hidden_size=hidden_size,
                          num_layers=n_layers,
                          batch_first=True,
                          dropout=dropout_p,
                          bidirectional=False
                          )

        self.generator = Generator(hidden_size, output_size)

    def encode(self, emb):

        if isinstance(emb, tuple):
            x, lengths = emb
            x = pack(x, lengths.tolist(), batch_first=True)
        else:
            x = emb

        y, h = self.rnn(x)

        if isinstance(emb, tuple):
            y, _ = unpack(y, batch_first=True)

        return y, h

    def decode(self, x, h_t_1):
        y, h = self.rnn(x, h_t_1)
        return y, h

    def forward(self, src, tgt):

        x_length = None

        if isinstance(src, tuple):
            x, x_length = src
        else:
            x = src

        if isinstance(tgt, tuple):
            tgt = tgt[0]

        # Get Movie embedding vectors for every time-step of input sentence.
        emb_src = self.embedding(x)
        # |emb_src| = (batch_size, length, word_vec_dim)

        # The last hidden state of the encoder would be a initial hidden state of decoder.
        h_src, h_0_tgt = self.encode((emb_src, x_length))
        # |h_src| = (batch_size, length, hidden_size)
        # |h_0_tgt| = (n_layers, batch_size, hidden_size)

        emb_tgt = self.embedding(tgt)
        # |emb_tgt| = (batch_size, length, word_vec_dim)
        decoder_hidden = h_0_tgt

        output = []

        # Run decoder until the end of the time-step.
        for t in range(tgt.size(1)):  ##|tgt| = (bs, l2)
            emb_t = emb_tgt[:, t, :].unsqueeze(1)  ##unsqueeze -> 첫번쨰 dimenttion에 배치를 하나 더 만들어주라...
            # |emb_t| = (batch_size, 1, word_vec_dim)
            # |h_t_tilde| = (batch_size, 1, hidden_size)

            decoder_output, decoder_hidden = self.decode(emb_t, decoder_hidden)

            output += [decoder_output]

        output = torch.cat(output, dim=1)
        # |output| = (batch_size, length, hidden_size)

        y_hat = self.generator(output)
        # |y_hat| = (batch_size, length, output_size)

        return y_hat

class SeqModel2(nn.Module):

    def __init__(self,
                 input_size,
                 word_vec_dim,
                 hidden_size,
                 output_size,
                 n_layers=4,
                 dropout_p=.2
                 ):
        self.input_size = int(input_size)
        self.word_vec_dim = int(word_vec_dim)
        self.hidden_size = int(hidden_size)
        self.output_size = int(output_size)
        self.n_layers = int(n_layers)
        self.dropout_p = float(dropout_p)

        super(SeqModel2, self).__init__()

        self.embedding = nn.Embedding(input_size, word_vec_dim)
        self.rnn = nn.GRU(input_size=word_vec_dim,
                          hidden_size=hidden_size,
                          num_layers=n_layers,
                          batch_first=True,
                          dropout=dropout_p,
                          bidirectional=False
                          )

        self.output = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)  ##마지막 dim에 softmax를 하라...

    def encode(self, emb):

        if isinstance(emb, tuple):
            x, lengths = emb
            x = pack(x, lengths.tolist(), batch_first=True)  ##torch.text는 매핑 작업을 잘 해줌.....
        else:
            x = emb

        y, h = self.rnn(x)
        # |y| = (batch_size, length, hidden_size)  ###hidden size는 bidirection으로 뽑아낸 마지막 hidden을 concat한거임.
        # |h[0]| = (num_layers * 2, batch_size, hidden_size / 2)   ##batch first해도 이렇게 나옴

        if isinstance(emb, tuple):
            y, _ = unpack(y, batch_first=True)

        return y, h

    def forward(self, src):
        # |src| = (batch, length_1, element)

        x_length = None

        if isinstance(src, tuple):
            x, x_length = src
        else:
            x = src

        # Get Movie embedding vectors for every time-step of input sentence.
        emb_src = self.embedding(x)
        # |emb_src| = (batch_size, length, word_vec_dim)

        # The last hidden state of the encoder would be a initial hidden state of decoder.
        h_src, h_0_tgt = self.encode((emb_src, x_length))
        # |h_src| = (batch_size, length, hidden_size)
        # |h_0_tgt| = (n_layers * 2, batch_size, hidden_size / 2)

        last_hidden = []
        for index, length in enumerate(x_length):
            last_hidden.append(h_src[index, length - 1, :].unsqueeze(0))

        last_hidden = torch.cat(last_hidden, dim=0)
        y_hat = self.output(last_hidden)
        y_hat = self.softmax(y_hat)
        return y_hat

class SeqModel3(nn.Module):

    def __init__(self,
                 input_size,
                 word_vec_dim,
                 hidden_size,
                 output_size,
                 n_layers=4,
                 dropout_p=.2,
                 device = 'cpu'
                 ):
        self.input_size = int(input_size)
        self.word_vec_dim = int(word_vec_dim)
        self.hidden_size = int(hidden_size)
        self.output_size = int(output_size)
        self.n_layers = int(n_layers)
        self.dropout_p = float(dropout_p)
        self.device = device

        super(SeqModel3, self).__init__()

        self.embedding = nn.Embedding(input_size, word_vec_dim)
        self.rnn = nn.GRU(input_size=word_vec_dim,
                          hidden_size=hidden_size,
                          num_layers=n_layers,
                          batch_first=True,
                          dropout=dropout_p,
                          bidirectional=False
                          )

        self.output = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)  ##마지막 dim에 softmax를 하라...

    def forward(self, x):
        length = x.size(1)

        x = self.embedding(x)
        h_src, h_0_tgt = self.rnn(x)

        y_hat = []
        for index in range(length):
            y_temp = self.output(h_src[:,index,:])
            y_temp = self.softmax(y_temp).unsqueeze(1)
            y_hat.append(y_temp)
        y_hat = torch.cat(y_hat, dim=1)
        return y_hat

    def search(self, x):
        x = self.embedding(x)
        h_src, h_0_tgt = self.rnn(x)
        last_hidden = h_src[:, -1, :]
        y_hat = self.output(last_hidden)
        y_hat = self.softmax(y_hat)
        return y_hat




class SeqModel(nn.Module):

    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 n_layers=4,
                 dropout_p=.2,
                 device='cpu'
                 ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.device = device

        super(SeqModel, self).__init__()

        self.rnn = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=n_layers,
                          batch_first=True,
                          dropout=dropout_p,
                          bidirectional=False
                          ).to(self.device)

        self.output = nn.Linear(hidden_size, output_size).to(self.device)
        self.softmax = nn.LogSoftmax(dim=-1).to(self.device)  ##마지막 dim에 softmax를 하라...

    def forward(self, x):
        # |src| = (batch, )
        # [extra] = (batch, length, extra_dim)

        # The last hidden state of the encoder would be a initial hidden state of decoder.
        h_src, h_0_tgt = self.rnn(x)
        # |h_src| = (batch_size, length, hidden_size)

        last_hidden = h_src[:, -1, :]
        y_hat = self.output(last_hidden)
        y_hat = self.softmax(y_hat)
        return y_hat





