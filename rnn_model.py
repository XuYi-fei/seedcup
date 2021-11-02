# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         rnn_model
# Description:
# Author:       梁超
# Date:         2021/11/1
# -------------------------------------------------------------------------------
import torch
import torch.nn as nn


class RnnNet(nn.Module):
    def __init__(self, embedding_dim, num_embeddings, output_size, hidden_size, layer_num, bidirectional, device):
        super(RnnNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.device = device
        self.output_size = output_size
        self.num_embeddings = num_embeddings
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(num_embeddings=self.num_embeddings, embedding_dim=self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_size, self.layer_num,
                            batch_first=True, bidirectional=self.bidirectional)
        self.fc = nn.Linear(self.hidden_size * 2, self.output_size) \
            if self.bidirectional else nn.Linear(self.hidden_size, output_size)
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.embedding(x)
        h0 = torch.zeros(self.layer_num*2, x.size(0), self.hidden_size) if self.bidirectional \
            else torch.zeros(self.layer_num, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.layer_num * 2, x.size(0), self.hidden_size) if self.bidirectional \
            else torch.zeros(self.layer_num, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.sm(out)
        return out


if __name__ == '__main__':
    md = RnnNet(embedding_dim=500, num_embeddings=1000, output_size=2, hidden_size=16,
                layer_num=1, bidirectional=True, device=torch.device('cpu'))
    temp = torch.randint(0, 900, (30, 33))
    temp = torch.LongTensor(temp)
    out = md.forward(temp)
    print(out.size())
