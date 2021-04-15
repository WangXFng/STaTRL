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
    train_data, num_types = load_data(opt.data + 'train.pkl', 'train')
    # print('[Info] Loading dev data...')
    # # dev_data, _ = load_data(opt.data + 'dev.pkl', 'dev')
    print('[Info] Loading test data...')
    test_data, _ = load_data(opt.data + 'test.pkl', 'test')

    trainloader = get_dataloader(train_data, opt.batch_size, shuffle=True)
    testloader = get_dataloader(test_data, opt.batch_size, shuffle=False)

    # print(train_data[0])  // [{'time_since_start': 1325.5123, 'time_since_last_event': 0.0, 'type_event': 3}]
    # print(len(train_data[0])) // 44

    return trainloader, testloader, num_types


def vaild(prediction, label, top_n, precision_s, recall_s, count):
    top_ = torch.topk(prediction, top_n, -1)[1]  # (32, top_n)
    for top, l in zip(top_, label):
        l = l[l != 0]
        l = l - 1

        # if len(l) == 0:
        #     precision_s[count] = 1
        #     recall_s[count] = 1
        #     continue
        # else:
        #     print(set(l) & set(top))
        #     actual = len(set(l) & set(top))
        #     precision_s[count] = actual / len(top)
        #     recall_s[count] = actual / len(l)
        for j in l:
            # s = torch.max(top == j, dim=-1)[1]
            if top.__contains__(j):
                # print((top+1), j)
                # print(i, j)
                precision_s[count] += 1
                recall_s[count] += 1

        # if precision_s[count] > top_n:
        #     print(top, l)
        #
        # if recall_s[count] > len(l):
        #     print('recall_s[count]>len(l)')

        if len(l) != 0:
            recall_s[count] /= len(l)
            precision_s[count] /= top_n
        count += 1
    return precision_s, recall_s, count


def pre_rec_top(p_5, r_5, p_10, r_10, p_20, r_20, count, prediction, label, target_):
    prediction = prediction * target_
    original_count = count
    p_5, r_5, count = vaild(prediction, label, 5, p_5, r_5, count)

    count = original_count
    p_10, r_10, count = vaild(prediction, label, 10, p_10, r_10, count)

    count = original_count
    p_20, r_20, count = vaild(prediction, label, 20, p_20, r_20, count)

    return p_5, r_5, p_10, r_10, p_20, r_20, count


def train_epoch(model, training_data, optimizer, pred_loss_func, opt):
    """ Epoch operation in training phase. """

    model.train()

    n = Constants.USER_NUMBER
    p_5,r_5,p_10,r_10,p_20,r_20=torch.zeros(n),torch.zeros(n),torch.zeros(n),torch.zeros(n),torch.zeros(n),torch.zeros(n)

    count = 0
    for batch in tqdm(training_data, mininterval=2,
                      desc='  - (Training)   ', leave=False):
        """ prepare data """
        event_type, score, test_label, test_score, inner_dis = map(lambda x: x.to(opt.device), batch)

        """ forward """
        optimizer.zero_grad()

        # event_type [16, L]  event_time [16, L]
        enc_out, rating_prediction, prediction, target_ = model(event_type, score, inner_dis)  # X = (UY+Z) ^ T
        # enc_out [16, 174, 512]  # batch * seq_len * model_dim

        prediction = torch.squeeze(prediction, 1)
        p_5, r_s, p_10, r_10, p_20, r_20, count = pre_rec_top(p_5, r_5, p_10, r_10, p_20, r_20, count, prediction, test_label, target_)
        """ backward """
        # negative log-likelihood    # [16, L-1]
        # event_ll, non_event_ll = Utils.log_likelihood(model, enc_out, event_time, event_type)
        # event_loss = -torch.sum(event_ll - non_event_ll)  # double value  l(s)

        # event_type [16, L]  event_time [16, L]
        rating_loss = Utils.rating_loss(rating_prediction, event_type, test_label, pred_loss_func)
        # rating_loss.backward(retain_graph=True)

        pred_loss = Utils.type_loss(prediction, event_type, test_label, opt.smooth)
        loss = rating_loss + pred_loss
        loss.backward(retain_graph=True)

        """ update parameters """
        optimizer.step()

    return p_5.sum().item()/count, r_5.sum().item()/count, p_10.sum().item()/count, \
           r_10.sum().item()/count, p_20.sum().item()/count, r_20.sum().item()/count


def eval_epoch(model, validation_data, pred_loss_func, opt):
    """ Epoch operation in evaluation phase. """

    model.eval()

    n = Constants.USER_NUMBER
    p_5,r_5,p_10,r_10,p_20,r_20=torch.zeros(n),torch.zeros(n),torch.zeros(n),torch.zeros(n),torch.zeros(n),torch.zeros(n)

    count = 0

    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2,
                          desc='  - (Validation) ', leave=False):
            """ prepare data """
            event_type, score, test_label, test_score, inner_dis = map(lambda x: x.to(opt.device), batch)

            """ forward """
            # event_type [16, L]  event_time [16, L]
            enc_out, rating_prediction, prediction, target_ = model(event_type, score, inner_dis)  # X = (UY+Z) ^ T
            # enc_out [16, 174, 512]  # batch * seq_len * model_dim

            prediction = torch.squeeze(prediction, 1)
            p_5, r_s, p_10, r_10, p_20, r_20, count = pre_rec_top(p_5, r_5, p_10, r_10, p_20, r_20, count, prediction, test_label, target_)

    return p_5.sum().item() / count, r_5.sum().item() / count, p_10.sum().item() / count, \
               r_10.sum().item() / count, p_20.sum().item() / count, r_20.sum().item() / count


def train(model, traindata, testdata, optimizer, scheduler, pred_loss_func, opt):
    """ Start training. """

    valid_precision_max = 0.0
    for epoch_i in range(opt.epoch):
        epoch = epoch_i + 1
        print('[ Epoch', epoch, ']')

        start = time.time()  # loglikelihood: {ll: 8.5f},
        p_5, r_5, p_10, r_10, p_20, r_20 = train_epoch(model, traindata, optimizer, pred_loss_func, opt)
        print('\r (Training)    P@5:{p_5: 8.5f}, R@5:{r_5: 8.5f}, P@10:{p_10: 8.5f}, R@10:{r_10: 8.5f}, '
              'P@20:{p_20: 8.5f}, R@20:{r_20: 8.5f}, elapse:{elapse:3.3f} min'
              .format(elapse=(time.time() - start) / 60,
                      p_5=p_5, r_5=r_5, p_10=p_10, r_10=r_10, p_20=p_20, r_20=r_20))

        start = time.time()
        p_5, r_5, p_10, r_10, p_20, r_20 = eval_epoch(model, testdata, pred_loss_func, opt)
        print('\r (Test)        P@5:{p_5: 8.5f}, R@5:{r_5: 8.5f}, P@10:{p_10: 8.5f}, R@10:{r_10: 8.5f}, '
              'P@20:{p_20: 8.5f}, R@20:{r_20: 8.5f}, elapse:{elapse:3.3f} min'
              .format(elapse=(time.time() - start) / 60,
                      p_5=p_5, r_5=r_5, p_10=p_10, r_10=r_10, p_20=p_20, r_20=r_20))

        # valid_event_losses += [valid_event]
        # print('  - [Info] Maximum ll: {event: 8.5f}'
        #       .format(event=max(valid_event_losses)))
        #
        # # logging
        # with open(opt.log, 'a') as f:
        #     f.write('{epoch}, {ll: 8.5f}\n'
        #             .format(epoch=epoch, ll=valid_event))

        scheduler.step()

        valid_precision_max = valid_precision_max if valid_precision_max > p_5 else p_5

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

    parser.add_argument('-ita', type=float, default=0.05)

    opt = parser.parse_args()

    import sys

    # print("Python Version {}".format(str(sys.version).replace('\n', '')))

    # print(torch.cuda.is_available())
    # default device is CUDA
    opt.device = torch.device('cuda')

    # disc = np.load("../data/Yelp/old_yelp_disc.npy")
    # disc = torch.tensor(disc, device=opt.device)

    # # setup the log file
    # with open(opt.log, 'w') as f:
    #     f.write('Epoch, Log-likelihood, Accuracy, RMSE\n')

    """ prepare dataloader """
    trainloader, testloader, num_types = prepare_dataloader(opt)

    """ prepare model """
    # opt.n_layers = trial.suggest_int('n_layers', 2, 2)
    opt.d_inner_hid = trial.suggest_int('n_hidden', 512, 1024, 128)
    opt.d_k = trial.suggest_int('d_k', 512, 1024, 128)
    opt.d_v = trial.suggest_int('d_v', 512, 1024, 128)
    opt.n_head = trial.suggest_int('n_head', 8, 12, 2)
    opt.n_dis = trial.suggest_int('n_dis', 8, 12, 2)
    # opt.d_rnn = trial.suggest_int('d_rnn', 128, 512, 128)
    opt.d_model = trial.suggest_int('d_model', 1024, 1024, 512)
    opt.dropout = trial.suggest_uniform('dropout_rate', 0.5, 0.7)
    opt.smooth = trial.suggest_uniform('smooth', 1e-3, 1e-1)
    opt.lr = trial.suggest_uniform('learning_rate', 1e-5, 1e-4)

    opt.ita = trial.suggest_uniform('ita', 0.03, 0.06)
    opt.coefficient = trial.suggest_uniform('coefficient', 0.05, 0.15)

    # opt.lr = 0.000099
    # # #
    opt.n_layers = 2  # 2
    # opt.d_inner_hid = 1024  # 768
    # # opt.d_rnn = 128
    # opt.d_model = 1024
    # opt.d_k = 1024
    # opt.d_v = 896
    # opt.n_head = 12  # 8
    # opt.n_dis = 8
    # opt.dropout = 0.66203
    # opt.smooth = 0.05998
    # opt.ita = 0.037
    # opt.coefficient = 0.14

    print('[Info] parameters: {}'.format(opt))
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
        device=opt.device,
        ita=opt.ita,
        n_dis=opt.n_dis
    )
    model.to(opt.device)

    """ optimizer and scheduler """
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                           opt.lr, betas=(0.9, 0.999), eps=1e-05)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)

    """ prediction loss function, either cross entropy or label smoothing """
    if opt.smooth > 0:
        pred_loss_func = Utils.LabelSmoothingLoss(opt.smooth, num_types, opt.device, opt.coefficient, ignore_index=-1)
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

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))