import math
import torch.nn as nn

import transformer.Constants as Constants
import torch
# from reformer_pytorch.reformer_pytorch import Reformer
from transformer.Layers import EncoderLayer
import torch.nn.functional as F


def get_non_pad_mask(seq):
    """ Get the non-padding positions. """

    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q):
    """ For masking out the padding part of key sequence. """

    # expand to fit the shape of key query attention matrix
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask


def get_subsequent_mask(seq):
    """ For masking out the subsequent info, i.e., masked self-attention. """

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)  # 返回矩阵上三角部分，其余部分定义为0

    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask


class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self,
            num_types, d_model, d_inner,
            n_layers, n_head, d_k, d_v, dropout):
        super().__init__()

        self.d_model = d_model

        # position vector, used for temporal encoding
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)],
            device=torch.device('cuda'))

        # event type embedding (23 512) (K M)
        self.event_emb = nn.Embedding(num_types + 1, d_model, padding_idx=Constants.PAD)  # dding 0

        self.group_emb = nn.Embedding(38, d_model, padding_idx=Constants.PAD)  # dding 0

        # event type embedding (23 512) (K M)
        self.location_emb = nn.Linear(37, d_model)  # dding 0

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)  # 512 1024 4 512 512 M
            for _ in range(n_layers)])

        self.a = torch.nn.Parameter(torch.FloatTensor(d_model), requires_grad=True)
        self.b = torch.nn.Parameter(torch.FloatTensor(d_model), requires_grad=True)
        self.a.data.fill_(1e-5)
        self.b.data.fill_(1e-5)

        self.c = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.c.data.fill_(1e-5)


    def temporal_enc(self, time, non_pad_mask):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """

        # time [16, L]
        # time [16, L, 1]
        # self.position_vec [512]
        # (time.unsqueeze(-1) / self.position_vec) [16, L, 512]

        result = time.unsqueeze(-1) / self.position_vec
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        # (time.unsqueeze(-1) / self.position_vec) [16, L, 512]

        # non_pad_mask [16, L, 1]
        return result * non_pad_mask  # [16, L, 512]

    # def forward(self, event_type, event_time, non_pad_mask, geo_):
    def forward(self, event_type, event_score, non_pad_mask, lats, lngs, inner_dis):
        """ Encode event sequences via masked self-attention. """

        # prepare attention masks
        # slf_attn_mask is where we cannot look, i.e., the future and the padding
        slf_attn_mask_subseq = get_subsequent_mask(event_type)  # M * L * L
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=event_type, seq_q=event_type)  # M x lq x lk

        slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
        # change type of slf_attn_mask_keypad as the same as slf_attn_mask_subseq

        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
        # [16, L, L] + [16, L, L] -> [16, L, L]  bigger than 0, then 1 else 0

        # tem_enc = self.temporal_enc(event_time, non_pad_mask)  # event_type 1
        # return torch.Size([16, L, 512])

        # print(event_type.size())  # torch.Size([16, L])
        enc_output = self.event_emb(event_type)  # (K M)  event_emb: Embedding (23 512)

        max_len = enc_output.size()[1]
        position = torch.arange(0, max_len, device='cuda:0').unsqueeze(0)
        position = self.temporal_enc(position, non_pad_mask)
        # lat_ = self.temporal_enc(lats)
        # lng_ = self.temporal_enc(lngs)
        # lats = torch.unsqueeze(lats, dim=2)
        # lngs = torch.unsqueeze(lngs, dim=2)
        # # print(lats.size(), lngs.size())
        # location_ = self.location_emb(torch.cat((lats, lngs), dim=2))
        # location_ = torch.softmax(location_, dim=-1)

        # group_ = torch.unsqueeze(group_,dim=1)
        # group = self.group_emb(group_)
        # group = F.normalize(group, dim=-1)

        # enc_score = self.score_emb(event_score)  # (K M)  event_emb: Embedding (23 512)
        # enc_score = torch.unsqueeze(event_score, -1)
        # print(enc_output.size())  # torch.Size([16, L, 512])
        # inner_dis = inner_dis * self.c
        for enc_layer in self.layer_stack:
            # enc_output += self.b * enc_group  # self.a * tem_enc +
            # print(enc_output.size(), group.size(), enc_score.size())
            enc_output += position
            # enc_output *= enc_score
            # enc_output = enc_output * enc_score  # + location_  # + self.a * lat_ + self.b * lng_

            enc_output, _ = enc_layer(
                enc_output,
                inner_dis,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)

        return enc_output  # [64, 62, 1024]


class Predictor(nn.Module):
    """ Prediction of next event type. """

    def __init__(self, dim, num_types, batch_size, device):
        super().__init__()

        self.linear = nn.Linear(dim, num_types, bias=False)  # in_features: int dim, out_features: int num_types
        nn.init.xavier_normal_(self.linear.weight)

        self.batch_size = batch_size
        self.num_types = num_types
        self.device = device
        self.dim = dim
        # self.disc = disc

        self.a = torch.nn.Parameter(torch.DoubleTensor(1), requires_grad=True)
        self.b = torch.nn.Parameter(torch.DoubleTensor(1), requires_grad=True)
        self.c = torch.nn.Parameter(torch.DoubleTensor(1), requires_grad=True)

        self.d = torch.nn.Parameter(torch.DoubleTensor(1), requires_grad=True)
        self.e = torch.nn.Parameter(torch.DoubleTensor(1), requires_grad=True)
        self.f = torch.nn.Parameter(torch.DoubleTensor(1), requires_grad=True)

        # initialization
        self.a.data.fill_(1e-5)
        self.b.data.fill_(0)
        self.c.data.fill_(0)

        self.d.data.fill_(0)
        self.e.data.fill_(0)
        self.f.data.fill_(0)

        self.g = torch.nn.Parameter(torch.DoubleTensor(1), requires_grad=True)
        self.g.data.fill_(1)

        self.weight_ = torch.nn.Parameter(torch.FloatTensor(1, 128, 1), requires_grad=True)
        self.weight_.data.fill_(0.01)

        self.w_1 = nn.Linear(dim, dim)  # self-attention
        nn.init.xavier_uniform_(self.w_1.weight)

    def forward(self, enc_output, event_type):

        # data = enc_output.sum(1)
        # data = torch.squeeze(data[:,-1:,:], 1)

        # data = enc_output * self.weight_
        data = enc_output.sum(1)/enc_output.size()[1]

        # enc_output = self.w_1(enc_output)
        # a = torch.unsqueeze(torch.sum(enc_output, dim=-1),dim=1).transpose(1, 2)  # [32, 42, 1]
        # attn = torch.squeeze(torch.matmul((enc_output / self.dim ** 0.5).transpose(1, 2), a))
        # out = self.linear(attn)  # [16, 105, 512] -> [16, 105, 1]l

        #
        out = self.linear(data)  # [16, 105, 512] -> [16, 105, 1]l
        out = F.normalize(out, p=2, dim=-1, eps=1e-05)

        # if out.size()[0] == 1:

        target_ = torch.ones(event_type.size()[0], self.num_types, device=self.device, dtype=torch.double)

        for i,e in enumerate(event_type):
            e = e[e!=0]
            e = e-1

            target_[i][e] = 0

        # else:
        #     target_ = torch.ones(event_type.size()[0], self.num_types, device=self.device, dtype=torch.double)
        #     filter = torch.zeros(self.num_types, device=self.device, dtype=torch.double)
        #
        #     for i, e in enumerate(event_type):
        #         e = e[e != 0]
        #         e = e - 1
        #
        #         target_[i][e] = 0
        #         filter[e] = 1
        #
        #     target_ = target_ * torch.unsqueeze(filter, 0)

        out = torch.tanh(out)

        return out, target_


class RNN_layers(nn.Module):
    """
    Optional recurrent layers. This is inspired by the fact that adding
    recurrent layers on top of the Transformer helps language modeling.
    """

    def __init__(self, d_model, d_rnn):
        super().__init__()

        self.rnn = nn.LSTM(d_model, d_rnn, num_layers=1, batch_first=True)  # input_size: d_model, gate_size: 4 * d_rnn
        self.projection = nn.Linear(d_rnn, d_model)  # in_features: int d_rnn, out_features: int d_model

    def forward(self, data, non_pad_mask):

        lengths = non_pad_mask.squeeze(2).long().sum(1).cpu()
        pack_enc_output = nn.utils.rnn.pack_padded_sequence(
            data, lengths, batch_first=True, enforce_sorted=False)
        temp = self.rnn(pack_enc_output)[0]
        out = nn.utils.rnn.pad_packed_sequence(temp, batch_first=True)[0]

        out = self.projection(out)

        return out


class Transformer(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(
            self,
            num_types, d_model=256, d_rnn=128, d_inner=1024,
            n_layers=4, n_head=4, d_k=64, d_v=64, dropout=0.1, batch_size=32, device=0):
        super().__init__()

        self.encoder = Encoder(
            num_types=num_types, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout)
        # position embedding
        # event type embedding

        self.num_types = num_types

        # convert hidden vectors into a scalar
        self.linear = nn.Linear(d_model, num_types)  # in_features: int d_model, out_features: int num_types

        # parameter for the weight of time difference
        self.alpha = -0.1

        # OPTIONAL recurrent layer, this sometimes helps
        # self.rnn = RNN_layers(d_model, d_rnn)
        #   LSTM() # input_size: d_model, gate_size: 4 * d_rnn
        #   Linear() # in_features: int d_rnn, out_features: int d_model

        # # prediction of next time stamp
        # self.time_predictor = Predictor(d_model, 1, external, batch_size)
        # #   Linear() in_features: int dim, out_features: int num_types
        # #   nn.init.xavier_normal_ (Predictor.linear.weight): Normal distribution

        # prediction of next event type
        self.type_predictor = Predictor(d_model, num_types, batch_size, device)
        #   Linear() in_features: int dim, out_features: int num_types
        #   nn.init.xavier_normal_ (Predictor.linear.weight): Normal distribution

        self.a = torch.nn.Parameter(torch.DoubleTensor(1), requires_grad=True)
        self.b = torch.nn.Parameter(torch.DoubleTensor(1), requires_grad=True)
        self.c = torch.nn.Parameter(torch.DoubleTensor(1), requires_grad=True)

        # initialization
        self.a.data.fill_(1)
        self.b.data.fill_(1e-5)
        self.c.data.fill_(1e-5)

    def forward(self, event_type, score, lats, lngs, inner_dis):
        """
        Return the hidden representations and predictions.
        For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
        Input: event_type: batch*seq_len;
               event_time: batch*seq_len.
        Output: enc_output: batch*seq_len*model_dim;
                type_prediction: batch*seq_len*num_classes (not normalized);
                time_prediction: batch*seq_len.
        """

        inner_dis = self.a * inner_dis
        #  y_{ij} = a * x^{b} * exp(c * x)
        # inner_dis = torch.softmax(inner_dis, dim=-1)

        # inner_dis = F.normalize(inner_dis)
        # print(inner_dis.max())

        non_pad_mask = get_non_pad_mask(event_type)  # event_type 1

        # enc_output = self.encoder(event_type, event_time, non_pad_mask, geo)  # H(j,:)
        enc_output = self.encoder(event_type, score, non_pad_mask, lats, lngs, inner_dis)  # H(j,:)

        # enc_output = enc_output[:,-1:,:]
        # enc_output = torch.squeeze(enc_output, 1)

        # enc_output = self.rnn(enc_output, non_pad_mask)  # [16, 166, 512]

        # time_prediction = self.time_predictor(enc_output, non_pad_mask, event_type)  # [16, 105, 1]

        type_prediction, target_ = self.type_predictor(enc_output, event_type)  # [16, 105, 22]

        return enc_output, type_prediction, target_