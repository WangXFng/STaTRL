import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.2):
        super().__init__()

        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    # def forward(self, q, k, v, geo_, mask=None):
    def forward(self, q, k, v, inner_dis, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))  # [32, 8, 85, 512]  [32, 8, 85, 512]

        #
        # inner_dis = F.normalize(inner_dis, dim=-1)

        # # # print('==================')
        # # #
        for i in range(8):
            # print(attn[:, i:i+1, :, :].size(), inner_dis.size())
            # print(attn[:, i:i+1, :, :].max(), inner_dis.max())
            # print(torch.unsqueeze(inner_dis, 1))
            attn[:, i:i+1, :, :] *= torch.unsqueeze(inner_dis,1)

        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        # print(attn.max())
        # print(attn.min())
        #
        # print('====================')
        return output, attn
