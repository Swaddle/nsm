import torch
import torch.nn as nn
import torch.nn.functional as F
from random import randint
from itertools import permutations, repeat, chain
from functools import partial


torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def dummy(token, emd_dim) -> torch.Tensor:
    return torch.rand(emd_dim)


class NSM(nn.Module):
    embd_dim = 7
    out_dim = 4
    batch = 10
    N = 3
    L = 3
    nodes = ['1']

    def __init__(self, ns, e, o, b, n, l):
        super().__init__()

        self.nodes = ns
        self.embd_dim = e
        self.out_dim = o
        self.batch = b
        self.N = n
        self.L = l

        self.W = torch.eye(
            self.embd_dim,
            requires_grad=True
        )  # Type : Tensor

        self.property_W = torch.stack(
            list(repeat(
                torch.eye(self.embd_dim, requires_grad=True),
                self.L + 1
            )),
            dim=0
        )  # Type : Tensor

        self.W_L_plus_1 = torch.eye(
            self.embd_dim,
            requires_grad=True
        )  # Type : Tensor

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

    def embed(self, token) -> torch.Tensor:
        return dummy(token, self.embd_dim)

    def forward(self, questions, C, D, E, S, adjacency_mask):

        embd_questions = torch.stack([
            torch.stack(list(map(self.embed, question)))
            for question in questions
        ])

        P_i = torch.softmax(
            torch.bmm(
                torch.bmm(
                    embd_questions,
                    self.W.expand(self.batch, self.embd_dim, self.embd_dim)
                ),
                C.expand(self.batch, -1, self.embd_dim).transpose(1, 2)
                ),
            dim=2
        )  # Type : Tensor

        V = (P_i[:, :, -1]).unsqueeze(2) * embd_questions + torch.bmm(
            P_i[:, :, :-1], C[:-1, :].expand(self.batch, -1, self.embd_dim))

        # run encoder on normalized sequence
        _, encoder_hidden = self.encoder_lstm(V)
        (q, _) = encoder_hidden
        q = q.view(self.batch, 1, self.embd_dim)

        # run decoder
        h, _ = self.decoder_lstm(
            q.expand(self.batch, (self.N)+1, self.embd_dim), 
            encoder_hidden
        )

        r = torch.bmm(torch.softmax(torch.bmm(h, V.transpose(1, 2)), dim=2), V)
        p_i = torch.ones(self.batch, len(self.nodes)) / len(self.nodes)
        for i in range(self.N):
            # r_i is the appropiate reasoning inst for the ith step
            r_i = r[:, i, :]

            R_i = F.softmax(torch.bmm(
                D.expand(self.batch, -1, self.embd_dim),
                r_i.unsqueeze(2)
            ), dim=1).squeeze(2)

            # degree to which reasoning inst is concerned with sem rel
            r_i_prime = R_i[:, -1].unsqueeze(1)

            property_R_i = R_i[:, :-1]

            # bilinear proj (one for each property) init to identity.
            gamma_i_s = F.elu(torch.sum(
                torch.mul(
                    property_R_i.view(self.batch, -1, 1, 1),
                    torch.mul(
                        torch.matmul(
                            S.transpose(2, 1),
                            self.property_W
                        ), r_i.view(self.batch, 1, 1, self.embd_dim)
                    )
                ), dim=1
            ))

            # bilinear projection
            gamma_i_e = F.elu(
                torch.mul(
                    torch.bmm(
                        E.view(self.batch, -1, self.embd_dim),
                        self.W_L_plus_1.expand(
                            self.batch,
                            self.embd_dim,
                            self.embd_dim
                        )
                    ), r_i.unsqueeze(1))
            ).view(self.batch, len(self.nodes), len(self.nodes), self.embd_dim)

            # update state probabilities (via relevant relation)
            p_i_r = F.softmax(
                self.W_r(
                    torch.sum(
                        torch.mul(
                            gamma_i_e,
                            p_i.view(self.batch, -1, 1, 1)
                        ), dim=1)
                ).squeeze(2), dim=1)

            # update state probabilities (property lookup)
            p_i_s = F.softmax(self.W_s(gamma_i_s).squeeze(2), dim=1)

            p_i = r_i_prime * p_i_r + (1 - r_i_prime) * p_i_s

        # Sumarize final NSM state
        r_N = r[:, self.N, :]

        property_R_N = F.softmax(
            torch.bmm(
                D.expand(self.batch, -1, self.embd_dim),
                r_N.unsqueeze(2)
            ),
            dim=1
        ).squeeze(2)[:, :-1]

        # equivalent to:
        # torch.sum(
        #   p_i.unsqueeze(2) * torch.sum(
        #           property_R_N.view(10, 1, 3, 1) * S,
        #       dim=2),
        # dim=1)
        m = torch.bmm(
            p_i.unsqueeze(1),
            torch.sum(
                property_R_N.view(self.batch, 1, (self.L)+1, 1) * S, 
                dim=2
            )
        )

        pre_logits = self.classifier(torch.cat([m, q], dim=2).squeeze(1))

        return pre_logits
