import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, n_dis, temperature, attn_dropout=0.2):
        super().__init__()

        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.n_dis = n_dis

        self.b = torch.nn.Parameter(torch.DoubleTensor(1), requires_grad=True)
        self.b.data.fill_(1e-5)
        self.c = torch.nn.Parameter(torch.DoubleTensor(1), requires_grad=True)
        self.c.data.fill_(1e-1)

    # def forward(self, q, k, v, geo_, mask=None):
    def forward(self, q, k, v, inner_dis, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))  # [32, 8, 85, 512]  [32, 8, 85, 512]

        # # #
        # # # inner_dis = F.normalize(inner_dis, dim=-1)
        # # # # #

        if inner_dis is not None:
            for i in range(self.n_dis):
                attn[:, i:i+1, :, :] *= torch.unsqueeze(inner_dis,1)  # *tim_sim+self.c
                # n = tim_sim.size()[1]
                # tim_sim += torch.eye(n, n,device='cuda:0')
                # attn[:, i:i+1, :, :] += torch.unsqueeze(tim_sim,1)  # *tim_sim+self.c

        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn
