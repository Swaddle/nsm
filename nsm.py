import torch
import torch.nn as nn
import torch.nn.functional as F
from random import randint
from itertools import permutations, repeat


def to_rand(token, emd_dim):
    return torch.rand(emd_dim)


class NSM(nn.Module):
    embd_dim = 7
    out_dim = 4
    batch = 10
    N = 3
    L = 3

    def __init__(self):
        super().__init__()

        self.W = torch.eye(
            self.embd_dim,
            requires_grad=True
        )

        self.property_W = torch.stack(
            repeat(
                torch.eye(self.embd_dim, requires_grad=True), 
                range(self.L + 1)
            ),
            dim=0
        )

        self.W_L_plus_1 = torch.eye(
            self.embd_dim,
            requires_grad=True
        )

        self.W_r = nn.Linear(
            self.embd_dim,
            1,
            bias=False
        )

        self.W_s = nn.Linear(
            self.embd_dim,
            1,
            bias=False
        )

        # encoder 
        self.encoder_lstm = nn.LSTM(
            input_size=self.embd_dim,
            hidden_size=self.embd_dim,
            batch_first=True,
            bidirectional=False
        )

        # recurrent decoder
        self.decoder_lstm = nn.LSTM(
            input_size=self.embd_dim, 
            hidden_size=self.embd_dim, 
            batch_first=True, 
            bidirectional=False
        )

        # final classifier
        self.classifier = nn.Sequential(
            nn.Linear(2*self.embd_dim, 2*self.embd_dim), 
            nn.ELU(),
            nn.Linear(2*self.embd_dim, self.out_dim)
        )

    def forward(self, questions, C, D, E, S, adjacency_mask):     
        embd_questions = torch.stack( 
            map(lambda x: torch.stack(to_rand(x, self.embd_dim)), questions)
        )

        P_i = torch.softmax(torch.bmm(
            torch.bmm(
                embd_questions,
                self.W.expand(self.batch, self.embd_dim, self.embd_dim)
            ),
            C.expand(self.batch, -1, self.embd_dim).transpose(1, 2)
        ), dim=2)

        return 0