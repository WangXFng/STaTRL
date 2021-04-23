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
            n_layers, n_head, d_k, d_v, dropout, n_dis):
        super().__init__()

        self.d_model = d_model

        # position vector, used for temporal encoding
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)],
            device=torch.device('cuda'))

        # event type embedding (23 512) (K M)
        self.event_emb = nn.Embedding(num_types + 1, d_model, padding_idx=Constants.PAD)  # dding 0

        # event type embedding (23 512) (K M)
        # self.score_emb = nn.Embedding(8, d_model, padding_idx=Constants.PAD)  # dding 0

        # # event type embedding (23 512) (K M)
        # self.location_emb = nn.Linear(37, d_model)  # dding 0

        # self.aspect_emb = nn.Linear(63, d_model)  # dding 0

        self.aspect_stack = nn.ModuleList([
            EncoderLayer(62, d_inner, n_head, d_k, d_v, n_dis, dropout=dropout)  # 512 1024 4 512 512 M
            for _ in range(n_layers)])

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, n_dis, dropout=dropout)  # 512 1024 4 512 512 M
            for _ in range(n_layers)])

        self.a = torch.nn.Parameter(torch.FloatTensor(d_model), requires_grad=True)
        self.b = torch.nn.Parameter(torch.FloatTensor(d_model), requires_grad=True)
        self.a.data.fill_(1e-5)
        self.b.data.fill_(1e-5)

        self.c = torch.nn.Parameter(torch.FloatTensor(3), requires_grad=True)
        self.c.data.fill_(1e-5)

        self.category_dict = ['hotelstravel', 'nightlife', 'food', 'active', 'arts', 'auto', 'shopping',
            'professional', 'physicians', 'pets', 'health', 'fitness', 'education', 'beautysvc']

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

    def getScore_(self, s):
        # s_ = torch.logical_and(s>5, s<95)
        # s[s_] = 6
        # s[s>95] = 7

        s_ = (s<=5)
        s[s_] = s[s_] / 10
        s_ = s>5
        s[s_] = 0.5 + 0.14 * torch.log(s[s_] - 4)
        return s

    # def forward(self, event_type, event_time, non_pad_mask, geo_):
    def forward(self, event_type, event_score, aspect, non_pad_mask, inner_dis):
        """ Encode event sequences via masked self-attention. """

        # prepare attention masks
        # slf_attn_mask is where we cannot look, i.e., the future and the padding
        slf_attn_mask_subseq = get_subsequent_mask(event_type)  # M * L * L
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=event_type, seq_q=event_type)  # M x lq x lk

        slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
        # change type of slf_attn_mask_keypad as the same as slf_attn_mask_subseq

        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
        # [16, L, L] + [16, L, L] -> [16, L, L]  bigger than 0, then 1 else 0

        # tem_enc = self.temporal_enc(event_score, non_pad_mask)  # event_type 1
        # return torch.Size([16, L, 512])

        # print(event_type.size())  # torch.Size([16, L])
        enc_event = self.event_emb(event_type)  # (K M)  event_emb: Embedding (23 512)

        # max_len = enc_output.size()[1]
        # position = torch.arange(0, max_len, device='cuda:0').unsqueeze(0)
        # position = self.temporal_enc(position, non_pad_mask)

        # event_score = self.getScore_(event_score)
        # enc_score = torch.unsqueeze(event_score, -1)
        # enc_score = self.score_emb(event_score)
        # print(enc_output.size())  # torch.Size([16, L, 512])
        #

        # # print('before', aspect)
        # pos = aspect[:, :, 0::2]
        # neg = aspect[:, :, 1::2]
        #
        # aspect[aspect<0.5]=0
        # n = torch.zeros((pos.size()), device="cuda:0")
        # n[pos>neg] = 1
        # n[neg>pos] = -1
        #
        # # food = aspect[:, :, 0:2]
        # # price = aspect[:, :, 2:4]
        # # service = aspect[:, :, 4:6]
        # #
        # # aspect = torch.cat((food, price, service), dim=-1)
        #
        # # print('after', n)

        # aspect[aspect<0.5]=0
        # enc_score = torch.unsqueeze(self.getScore_(event_score), -1)
        # # aspect = torch.cat((aspect, enc_score), dim=-1)

        # 1

        # enc_output = enc_event
        # enc_output = enc_event * enc_score
        # enc_output += position

        # enc_score = torch.unsqueeze(self.getScore_(event_score), dim=2)
        # aspect = torch.cat((aspect, enc_score), dim=-1)
        # aspect = self.aspect_emb(aspect)
        aspect_output = aspect
        for enc_layer in self.aspect_stack:
            aspect_output, _ = enc_layer(
                aspect_output,
                None,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)

        enc_output = enc_event  # + aspect  # + tem_enc
        # enc_output = enc_event
        for enc_layer in self.layer_stack:

            enc_output, _ = enc_layer(
                enc_output,
                inner_dis,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)

        # enc_output = torch.cat((aspect, enc_output), dim=-1)

        return enc_output, aspect_output  # [64, 62, 1024]


class Predictor(nn.Module):
    """ Prediction of next event type. """

    def __init__(self, dim, num_types, batch_size, device, poi_avg_aspect):
        super().__init__()

        self.rating_linear = nn.Linear(dim, num_types, bias=False)  # in_features: int dim, out_features: int num_types
        nn.init.xavier_normal_(self.rating_linear.weight)

        self.type_linear = nn.Linear(dim, num_types, bias=False)  # in_features: int dim, out_features: int num_types
        nn.init.xavier_normal_(self.type_linear.weight)

        self.batch_size = batch_size
        self.num_types = num_types
        self.device = device
        self.dim = dim
        self.poi_avg_aspect = torch.transpose(torch.tensor(poi_avg_aspect, device=device, dtype=torch.float), dim0=0, dim1=1)

        self.a = torch.nn.Parameter(torch.DoubleTensor(num_types), requires_grad=True)
        self.a.data.fill_(1)

        self.b = torch.nn.Parameter(torch.DoubleTensor(num_types), requires_grad=True)
        self.b.data.fill_(1e-5)

    def forward(self, enc_output, aspect_output, event_type):

        # data = torch.squeeze(data[:,-1:,:], 1)
        data = enc_output.sum(1)/enc_output.size()[1]
        aspect_output = aspect_output.sum(1)/aspect_output.size()[1]

        #
        rating_prediction = self.rating_linear(data)  # [16, 105, 512] -> [16, 105, 1]l
        rating_prediction = F.normalize(rating_prediction, p=2, dim=-1, eps=1e-05)

        aspect_output = torch.matmul(aspect_output, self.poi_avg_aspect)
        aspect_output = F.normalize(aspect_output, p=2, dim=-1, eps=1e-05)

        # print(rating_prediction.max(), rating_prediction.min(), aspect_output.max(), aspect_output.min())

        rating_prediction = rating_prediction + aspect_output * self.b

        type_prediction = self.type_linear(data)  # [16, 105, 512] -> [16, 105, 1]l
        type_prediction = F.normalize(type_prediction, p=2, dim=-1, eps=1e-05)
        # type_prediction = torch.softmax(type_prediction, dim=-1)

        out = self.a * rating_prediction + type_prediction
        # out = rating_prediction.detach() * type_prediction

        # out = F.normalize(out, p=2, dim=-1, eps=1e-05)
        # out = out + self.b * ingoing/100

        out = torch.tanh(out)

        target_ = torch.ones(event_type.size()[0], self.num_types, device=self.device, dtype=torch.double)
        for i,e in enumerate(event_type):
            e = e[e!=0] - 1
            target_[i][e] = 0

        return rating_prediction, out, target_


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
            num_types, d_model=256, d_rnn=128, d_inner=1024, n_layers=4, n_head=4, d_k=64, d_v=64, dropout=0.1,
            batch_size=32, device=0, ita=0.05, n_dis=4, poi_avg_aspect=[]):
        super().__init__()

        self.encoder = Encoder(
            num_types=num_types, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout, n_dis=n_dis)
        # position embedding
        # event type embedding
        self.ita = ita
        self.num_types = num_types

        # convert hidden vectors into a scalar
        self.linear = nn.Linear(d_model+0, num_types)  # in_features: int d_model, out_features: int num_types

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
        self.predictor = Predictor(d_model+0, num_types, batch_size, device, poi_avg_aspect)
        #   Linear() in_features: int dim, out_features: int num_types
        #   nn.init.xavier_normal_ (Predictor.linear.weight): Normal distribution

        self.a = torch.nn.Parameter(torch.DoubleTensor(1), requires_grad=True)

        # initialization
        self.a.data.fill_(1e-5)

    def grbf(self, d):
        n = self.ita
        d = torch.exp(-n * d)
        d[d < 0.125] = 0
        # print(a.max())
        # print(a.min())
        return d

    def forward(self, event_type, score, aspect, inner_dis):
        """
        Return the hidden representations and predictions.
        For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
        Input: event_type: batch*seq_len;
               event_time: batch*seq_len.
        Output: enc_output: batch*seq_len*model_dim;
                type_prediction: batch*seq_len*num_classes (not normalized);
                time_prediction: batch*seq_len.
        """
        # print(inner_dis.max())
        inner_dis = self.a * self.grbf(inner_dis)
        #  y_{ij} = a * x^{b} * exp(c * x)
        # inner_dis = torch.softmax(inner_dis, dim=-1)

        # inner_dis = F.normalize(inner_dis)
        # print(inner_dis.max())

        non_pad_mask = get_non_pad_mask(event_type)  # event_type 1

        enc_output, aspect_output = self.encoder(event_type, score, aspect, non_pad_mask, inner_dis)  # H(j,:)

        # enc_output = self.rnn(enc_output, non_pad_mask)  # [16, 166, 512]

        rating_prediction, type_prediction, target_ = self.predictor(enc_output, aspect_output, event_type)  # [16, 105, 22]

        return enc_output, rating_prediction, type_prediction, target_
