import argparse
import numpy as np
import pickle
import time

import torch
import torch.nn as nn
import torch.optim as optim

import optuna

import transformer.Constants as Constants
import Utils

from preprocess.Dataset import get_dataloader
from transformer.Models import Transformer
from tqdm import tqdm

import math


def prepare_dataloader(opt):
    """ Load data and prepare dataloader. """

    def load_data(name, dict_name):
        with open(name, 'rb') as f:
            data = pickle.load(f, encoding='latin-1')
            num_types = data['dim_process']

            data = data[dict_name]
            return data, int(num_types)

    print('[Info] Loading train data...')
    # las_vegas 31675  # toronto 20370 # Champaign 1327 # Charlotte 10429 # original length 18995
    train_data, num_types = load_data(opt.data + 'train_old_yelp.pkl', 'train')
    # print('[Info] Loading dev data...')
    # # dev_data, _ = load_data(opt.data + 'dev.pkl', 'dev')
    print('[Info] Loading test data...')
    test_data, _ = load_data(opt.data + 'test_old_yelp.pkl', 'test')

    trainloader = get_dataloader(train_data, opt.batch_size, shuffle=True)
    testloader = get_dataloader(test_data, opt.batch_size, shuffle=False)

    # print(train_data[0])  // [{'time_since_start': 1325.5123, 'time_since_last_event': 0.0, 'type_event': 3}]
    # print(len(train_data[0])) // 44

    return trainloader, testloader, num_types


def pre_rec_top(precision_s, recall_s, count, prediction, label, target_, a, alone_num, alone_total):
    prediction = prediction * target_

    top_ = torch.topk(prediction, Constants.TOP_N, -1)[1]  # (32, top_n)
    # top_10 = torch.tensor(np.random.randint(18995,size=(32,5)),device=opt.device)
    # when random precision@5:  0.00026, recall@5:  0.00015

    for top, l in zip(top_, label):
        l = l[l != 0]
        l = l - 1
        for j in l:
            # s = torch.max(top == j, dim=-1)[1]
            if top.__contains__(j):
                # print((top+1), j)
                # print(i, j)
                precision_s[count] += 1
                recall_s[count] += 1

        if precision_s[count] > Constants.TOP_N:
            print(top, l)

        if recall_s[count] > len(l):
            print('recall_s[count]>len(l)')

        if len(l) != 0:
            recall_s[count] /= len(l)
            precision_s[count] /= Constants.TOP_N
        # else:
        #     print(top, l)

        # # if precision_s[count] != 0:
        # print(a, count, top.cpu().numpy(), l.cpu().numpy(), precision_s[count].cpu().numpy(), recall_s[count].cpu().numpy())
        #
        # if top_.size()[0] == 1:
        #     alone_num += 1
        #     alone_total += precision_s[count].cpu().numpy()
        #     print('alone total',alone_total, '/', alone_num, '=', alone_total/alone_num)

        count += 1

    return precision_s, recall_s, count, alone_num, alone_total


def train_epoch(model, training_data, optimizer, pred_loss_func, opt):
    """ Epoch operation in training phase. """

    model.train()

    total_event_ll = 0  # cumulative event log-likelihood
    # total_time_se = 0  # cumulative time prediction squared-error
    # total_event_rate = 0  # cumulative number of correct prediction
    total_num_event = 0  # number of total events
    total_num_pred = 0  # number of predictions

    precision_s = torch.zeros(math.ceil(Constants.USER_NUMBER))
    recall_s = torch.zeros(math.ceil(Constants.USER_NUMBER))

    # total_ = Constants.USER_NUMBER * 0.8

    count = 0
    alone_num = 0
    alone_total = 0
    for batch in tqdm(training_data, mininterval=2,
                      desc='  - (Training)   ', leave=False):
        """ prepare data """
        # event_time, time_gap, event_type, in_out, distance = map(lambda x: x.to(opt.device), batch)
        event_type, score, lats, lngs, test_label, test_score, inner_dis = map(lambda x: x.to(opt.device), batch)

        """ forward """
        optimizer.zero_grad()
        # optimizer2.zero_grad()

        # event_type [16, L]  event_time [16, L]
        # enc_out, prediction = model(event_type, event_time, in_out, distance)  # X = (UY+Z) ^ T
        enc_out, prediction, target_ = model(event_type, score, lats, lngs, inner_dis)
        # enc_out [16, 174, 512]  # batch * seq_len * model_dim
        # prediction[0] [16, 133, 22]  type - event
        # prediction[1] [16, 133, 1]   time - event

        prediction = torch.squeeze(prediction, 1)

        ocount = count
        precision_s, recall_s, count, alone_num, alone_total = pre_rec_top(precision_s, recall_s, count, prediction, test_label, target_, "train1", alone_num, alone_total)
        """ backward """
        # negative log-likelihood    # [16, L-1]
        # event_ll, non_event_ll = Utils.log_likelihood(model, enc_out, event_time, event_type)
        # event_loss = -torch.sum(event_ll - non_event_ll)  # double value  l(s)

        # type prediction  # loss and correct_num  # event_type [16, L]  event_time [16, L]
        pred_loss = Utils.type_loss(prediction, event_type, score, test_label, test_score, pred_loss_func)

        loss = pred_loss  # + se / scale_time_loss
        loss.backward(retain_graph=True)

        """ update parameters """
        optimizer.step()

        """ note keeping """
        # total_event_ll += -event_loss.item()
        # total_event_rate += pred_num_event.item()
        total_num_event += event_type.ne(Constants.PAD).sum().item()
        # we do not predict the first event
        total_num_pred += event_type.ne(Constants.PAD).sum().item() - event_type.shape[0]
        # print(event_type.ne(Constants.PAD).sum().item(), event_time.shape[0], total_num_pred)

    # print('count', count)
    # rmse = np.sqrt(total_time_se / total_num_pred)
    # return total_event_ll / total_num_event, total_event_rate / total_num_pred, rmse

    # aaa = precision_s/Constants.TOP_N
    #
    # if torch.sum(aaa > 1) > 0:
    #     print('precision_s/Constants.TOP_N>0')

    precision = torch.sum(precision_s)
    recall = torch.sum(recall_s)
    return total_event_ll / total_num_event, precision.item() / count, recall.item() / count


def eval_epoch(model, validation_data, pred_loss_func, opt):
    """ Epoch operation in evaluation phase. """

    model.eval()

    total_event_ll = 0  # cumulative event log-likelihood
    # total_time_se = 0  # cumulative time prediction squared-error
    # total_event_rate = 0  # cumulative number of correct prediction
    total_num_event = 0  # number of total events
    total_num_pred = 0  # number of predictions

    precision_s = torch.zeros(Constants.USER_NUMBER, device=opt.device)
    recall_s = torch.zeros(Constants.USER_NUMBER, device=opt.device)

    alone_num = 0
    alone_total = 0
    # total_ = Constants.USER_NUMBER * 0.2

    count = 0

    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2,
                          desc='  - (Validation) ', leave=False):
            """ prepare data """
            #event_type, score, lats, lngs, test_label, test_score, inner_dis, group_ = map(lambda x: x.to(opt.device), batch)

            event_type, score, lats, lngs, test_label, test_score, inner_dis = map(lambda x: x.to(opt.device), batch)

            """ forward """
            # event_type [16, L]  event_time [16, L]
            # enc_out, prediction = model(event_type, event_time, in_out, distance)  # X = (UY+Z) ^ T
            enc_out, prediction, target_ = model(event_type, score, lats, lngs, inner_dis)
            # enc_out [16, 174, 512]  # batch * seq_len * model_dim
            # prediction[0] [16, 133, 22]  type - event
            # prediction[1] [16, 133, 1]   time - event

            prediction = torch.squeeze(prediction, 1)
            ocount = count
            precision_s, recall_s, count, alone_num, alone_total = pre_rec_top(precision_s, recall_s, count, prediction, test_label, target_, 'test1', alone_num, alone_total)

            """ compute loss """
            # event_ll, non_event_ll = Utils.log_likelihood(model, enc_out, event_time, event_type)
            # event_loss = -torch.sum(event_ll - non_event_ll)
            # # _, pred_num = Utils.type_loss(prediction[0], event_type, pred_loss_func)
            # # se = Utils.time_loss(prediction[1], event_time)
            #
            # """ note keeping """
            # total_event_ll += -event_loss.item()
            # total_time_se += se.item()
            # total_event_rate += pred_num.item()

            total_num_event += event_type.ne(Constants.PAD).sum().item()
            # we do not predict the first event
            total_num_pred += event_type.ne(Constants.PAD).sum().item() - event_type.shape[0]

    # print('count', count)
    # rmse = np.sqrt(total_time_se / total_num_pred)
    precision = torch.sum(precision_s)
    recall = torch.sum(recall_s)
    return total_event_ll / total_num_event, precision.item() / count, recall.item() / count


def train(model, traindata, testdata, optimizer, scheduler, pred_loss_func, opt):
    """ Start training. """

    valid_event_losses = []  # validation log-likelihood
    # valid_pred_losses = []  # validation event type prediction accuracy
    # valid_rmse = []  # validation event time prediction RMSE
    valid_precision_max = 0.0
    for epoch_i in range(opt.epoch):
        epoch = epoch_i + 1
        print('[ Epoch', epoch, ']')

        start = time.time()  # loglikelihood: {ll: 8.5f},
        train_event, precision, recall = train_epoch(model, traindata, optimizer, pred_loss_func, opt)
        print('   - (Training)    precision@{top_k:1d}: {precision: 8.5f},'
              ' recall@{top_k:1d}: {recall: 8.5f}, elapse: {elapse:3.3f} min'
              .format(ll=train_event, elapse=(time.time() - start) / 60,
                      precision=precision, recall=recall, top_k=Constants.TOP_N))

        start = time.time()  # loglikelihood: {ll: 8.5f},
        valid_event, valid_precision, valid_recall = eval_epoch(model, testdata, pred_loss_func, opt)
        print('   - (Testing)     precision@{top_k:1d}: {precision: 8.5f},'
              ' recall@{top_k:1d}: {recall: 8.5f}, elapse: {elapse:3.3f} min'
              .format(ll=valid_event, elapse=(time.time() - start) / 60,
                      precision=valid_precision, recall=valid_recall, top_k=Constants.TOP_N))

        # valid_event_losses += [valid_event]
        # print('  - [Info] Maximum ll: {event: 8.5f}'
        #       .format(event=max(valid_event_losses)))
        #
        # # logging
        # with open(opt.log, 'a') as f:
        #     f.write('{epoch}, {ll: 8.5f}\n'
        #             .format(epoch=epoch, ll=valid_event))

        scheduler.step()

        valid_precision_max = valid_precision_max if valid_precision_max > valid_precision else valid_precision

    return valid_precision_max


def main(trial):
    """ Main function. """


    parser = argparse.ArgumentParser()

    parser.add_argument('-data', required=True)

    parser.add_argument('-epoch', type=int, default=30)
    parser.add_argument('-batch_size', type=int, default=16)

    parser.add_argument('-d_model', type=int, default=64)
    parser.add_argument('-d_rnn', type=int, default=256)
    parser.add_argument('-d_inner_hid', type=int, default=128)
    parser.add_argument('-d_k', type=int, default=16)
    parser.add_argument('-d_v', type=int, default=16)

    parser.add_argument('-n_head', type=int, default=4)
    parser.add_argument('-n_layers', type=int, default=4)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument('-smooth', type=float, default=0.1)

    parser.add_argument('-log', type=str, default='log.txt')

    opt = parser.parse_args()

    import sys

    # print("Python Version {}".format(str(sys.version).replace('\n', '')))

    # print(torch.cuda.is_available())
    # default device is CUDA
    opt.device = torch.device('cuda')

    # disc = np.load("../data/Yelp/old_yelp_disc.npy")
    # disc = torch.tensor(disc, device=opt.device)

    # setup the log file
    with open(opt.log, 'w') as f:
        f.write('Epoch, Log-likelihood, Accuracy, RMSE\n')

    print('[Info] parameters: {}'.format(opt))

    """ prepare dataloader """
    trainloader, testloader, num_types = prepare_dataloader(opt)

    """ prepare model """
    # opt.n_layers = trial.suggest_int('n_layers', 1, 4)
    # opt.d_inner_hid = trial.suggest_int('n_hidden', 128, 1024, 128)
    #
    # opt.n_head = trial.suggest_int('n_head', 4, 16, 4)
    # # opt.d_rnn = trial.suggest_int('d_rnn', 128, 512, 128)
    # opt.d_model = trial.suggest_int('d_model', 512, 1024, 512)
    # # opt.d_model = 512
    # opt.dropout = trial.suggest_uniform('dropout_rate', 0.5, 0.7)
    # opt.smooth = trial.suggest_uniform('smooth', 1e-3, 1e-1)
    #
    # opt.lr = trial.suggest_uniform('learning_rate', 1e-5, 1e-4)

    opt.lr = 0.000085

    opt.n_layers = 2  # 2
    opt.d_inner_hid = 768  # 768
    # opt.d_rnn = 128
    opt.d_model = 1024
    opt.d_k = 768
    opt.d_v = 768
    opt.n_head = 12  # 8
    opt.dropout = 0.6801
    opt.smooth = 0.0997

    model = Transformer(
        num_types=num_types,
        d_model=opt.d_model,
        # disc=disc,
        d_rnn=opt.d_rnn,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        d_k=opt.d_k,
        d_v=opt.d_v,
        dropout=opt.dropout,
        batch_size=opt.batch_size,
        device=opt.device
    )
    model.to(opt.device)

    # model2 = Transformer(
    #     num_types=num_types,
    #     d_model=opt.d_model,
    #     # disc=disc,
    #     d_rnn=opt.d_rnn,
    #     d_inner=opt.d_inner_hid,
    #     n_layers=opt.n_layers,
    #     n_head=opt.n_head,
    #     d_k=opt.d_k,
    #     d_v=opt.d_v,
    #     dropout=opt.dropout,
    #     batch_size=opt.batch_size,
    #     device=opt.device
    # )
    # model2.to(opt.device)

    """ optimizer and scheduler """
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                           opt.lr, betas=(0.9, 0.999), eps=1e-05)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)

    """ prediction loss function, either cross entropy or label smoothing """
    if opt.smooth > 0:
        pred_loss_func = Utils.LabelSmoothingLoss(opt.smooth, num_types, opt.device, ignore_index=-1)
    else:
        pred_loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')

    """ number of parameters """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('[Info] Number of parameters: {}'.format(num_params))

    """ train the model """
    return train(model, trainloader, testloader, optimizer, scheduler, pred_loss_func, opt)


if __name__ == '__main__':
    # main()
    study = optuna.create_study(direction="maximize")
    study.optimize(main, n_trials=100)

    df = study.trials_dataframe()