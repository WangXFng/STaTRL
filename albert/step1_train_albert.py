from transformers import AlbertTokenizer, AlbertModel
import torch.nn as nn
import torch
from tqdm import tqdm
import time
import argparse
import torch.optim as optim
import torch.nn.functional as F
from albert_model import ALBertForClassification
import shutil
import os
import optuna
import random


def cal_acc(pred, label, corr_num, total_num):
    pred_ = torch.argmax(pred)
    return torch.sum(pred_ == label)+corr_num, total_num+pred.size()[0]


def read_training_data():
    train_data = open('./dataset/dataset/train_3_2.txt', 'r').readlines()
    random.shuffle(train_data)
    dataX, dataY = [], []
    for i, eachline in enumerate(train_data):
        # text, aspect, label = eachline.strip().replace("\n","").split("\t")
        # if i<int(len(train_data)*0.7):
        #     dataX.append([text, int(aspect), int(label)])
        # else:
        #     dataY.append([text, int(aspect), int(label)])

        text, aspect, polarity = eachline.strip().replace("\n","").split("\t")
        if i<int(len(train_data)*0.8):
            dataX.append([text, int(aspect), int(polarity)])
        else:
            dataY.append([text, int(aspect), int(polarity)])

    return dataX, dataY


def train_epoch(model, training_data, optimizer, opt):
    """ Epoch operation in training phase. """

    model.train()
    aspect_corr_num = 0
    aspect_num = 0
    polarity_corr_num = 0
    polarity_num = 0
    valid_precision_max = 0

    # print(len(training_data))
    t = tqdm(training_data, mininterval=2, desc='  - (Training)   ', leave=False)
    for i, batch in enumerate(t):
        """ prepare data """
        # text, aspect, label = batch  # map(lambda x: x.to(opt.device), batch)
        text, aspect, polarity = batch  # map(lambda x: x.to(opt.device), batch)

        """ forward """
        optimizer.zero_grad()
        aspect = torch.tensor([aspect], device=opt.device)
        polarity = torch.tensor([polarity], device=opt.device)

        loss, aspect_res, polarity_res = model(text, aspect, polarity)
        # loss, aspect_logits, polarity_logits = model(text, aspect, polarity)

        """ backward """
        loss.backward(retain_graph=True)

        """ update parameters """
        optimizer.step()

        aspect_corr_num, aspect_num = cal_acc(aspect_res, aspect, aspect_corr_num, aspect_num)
        polarity_corr_num, polarity_num = cal_acc(polarity_res, polarity, polarity_corr_num, polarity_num)

        if i % 50 == 0:
            # t.set_description('  - (Training)   loss: {loss: 8.5f} aspect_acc: {aspect_acc: 8.5f} '
            t.set_description('  - (Training)   loss: {loss: 8.5f} aspect_acc: {aspect_acc: 8.5f} '
                              'polarity_accuracy: {polarity_accuracy: 8.5f} '
                              .format(loss=loss, aspect_acc=aspect_corr_num / aspect_num,
                                      polarity_accuracy=polarity_corr_num / polarity_num))

    return torch.sum(loss).item(), aspect_corr_num / aspect_num, polarity_corr_num / polarity_num


def eval_epoch(model, test_data, optimizer, opt):
    """ Epoch operation in training phase. """
    model.eval()

    aspect_corr_num = 0
    aspect_num = 0
    polarity_corr_num = 0
    polarity_num = 0
    # print(len(training_data))

    with torch.no_grad():
        t = tqdm(test_data, mininterval=2, desc='  -     (Test)   ', leave=False)
        for i, batch in enumerate(t):

            """ prepare data """
            # text, aspect, polarity = batch  # map(lambda x: x.to(opt.device), batch)
            text, aspect, polarity = batch  # map(lambda x: x.to(opt.device), batch)
            """ forward """
            optimizer.zero_grad()

            aspect = torch.tensor([aspect], device=opt.device)
            polarity = torch.tensor([polarity], device=opt.device)

            loss, aspect_res, polarity_res = model(text, aspect, polarity)

            aspect_corr_num, aspect_num = cal_acc(aspect_res, aspect, aspect_corr_num, aspect_num)
            polarity_corr_num, polarity_num = cal_acc(polarity_res, polarity, polarity_corr_num, polarity_num)

            if i % 50 == 0:
                # t.set_description('  - (Training)   loss: {loss: 8.5f} aspect_acc: {aspect_acc: 8.5f} '
                t.set_description('  - (Training)   loss: {loss: 8.5f} aspect_acc: {aspect_acc: 8.5f} '
                                  'polarity_accuracy: {polarity_accuracy: 8.5f} '
                                  .format(loss=loss, aspect_acc=aspect_corr_num / aspect_num,
                                          polarity_accuracy=polarity_corr_num / polarity_num))
    return torch.sum(loss).item(), aspect_corr_num / aspect_num, polarity_corr_num / polarity_num


def train(model, training_data, test_data, optimizer, scheduler, opt):
    """ Start training. """

    aspect_acc_max = 0.0
    polarity_acc_max = 0.0
    max_name = ""
    for epoch_i in range(opt.epoch):
        epoch = epoch_i + 1
        print('\r[ Epoch', epoch, ']')

        start = time.time()  # loglikelihood: {ll: 8.5f},
        loss, aspect_acc, polarity_accuracy = train_epoch(model, training_data, optimizer, opt)
        print('\r (Training)    loss:{loss: 8.5f}, aspect_acc:{aspect_acc: 8.5f}, '
              'polarity_acc:{polarity_accuracy: 8.5f}, elapse:{elapse:3.3f} min'
              .format(elapse=(time.time() - start) / 60,loss=loss, aspect_acc=aspect_acc
                      , polarity_accuracy=polarity_accuracy))

        start = time.time()
        loss, aspect_acc, polarity_accuracy = eval_epoch(model, test_data, optimizer, opt)
        print('\r (Test)        loss:{loss: 8.5f}, aspect_acc:{aspect_acc: 8.5f}, '
              'polarity_acc:{polarity_accuracy: 8.5f}, elapse:{elapse:3.3f} min'
              .format(elapse=(time.time() - start) / 60,loss=loss, aspect_acc=aspect_acc
                      , polarity_accuracy=polarity_accuracy))

        scheduler.step()

        # model_dict = torch.load(PATH)
        # model_dict = model.load_state_dict(torch.load(PATH))
        # print(accuracy, valid_precision_max, accuracy > valid_precision_max)
        # if aspect_acc > aspect_acc_max and polarity_accuracy > polarity_acc_max:
        if polarity_accuracy > polarity_acc_max:
            aspect_acc_max, polarity_acc_max = aspect_acc, polarity_accuracy
            if os.path.exists('./dataset/model'):
                shutil.rmtree('./dataset/model')
            os.mkdir('./dataset/model')
            max_name = "{accuracy:8.5f}.pth.tar".format(accuracy=polarity_accuracy)
            torch.save(model, './dataset/model/'+max_name)

        if epoch == 50:
            os.mkdir('./dataset/model')
            shutil.move('./dataset/model/'+max_name, './model')

        # valid_precision_max = valid_precision_max if valid_precision_max > accuracy else accuracy
    # return valid_precision_max


def collate_fn(insts):
    """ Collate function, as required by PyTorch. """
    # print(list(zip(*insts)))
    (text, polarity) = list(zip(*insts))
    return text, polarity


def main():
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()
    opt.device = torch.device('cuda')
    opt.epoch = 50
    opt.lr = 0.000001
    # opt.d_inner_hid = trial.suggest_int('n_hidden', 512, 1024, 128)

    training_data, test_data = read_training_data()

    # dl = torch.utils.data.DataLoader(
    #     training_data,
    #     num_workers=2,
    #     batch_size=1,
    #     collate_fn=collate_fn,
    #     shuffle=False
    # )

    model = ALBertForClassification.from_pretrained("albert-base-v2", num_labels=3)
    model.cuda()

    """ optimizer and scheduler """
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                           opt.lr, betas=(0.9, 0.999), eps=1e-05)
    # parameters = [{'params': model.classifier.parameters(), 'lr': 0.00001},
    #               {'params': model.fc.parameters(), 'lr': 0.00001},
    #               {'params': model.albert.parameters(), 'lr': 0.000001}]
    # optimizer = torch.optim.Adam(parameters)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)

    return train(model, training_data, test_data, optimizer, scheduler, opt)


if __name__ == '__main__':
    # study = optuna.create_study(direction="maximize")
    # study.optimize(main, n_trials=100)
    #
    # df = study.trials_dataframe()
    main()

