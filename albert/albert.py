from transformers import AlbertTokenizer, AlbertModel
import torch.nn as nn
import torch
from tqdm import tqdm
import time
import argparse
import torch.optim as optim
import torch.nn.functional as F
from bert import ALBertForClassification


def cal_acc(pred, label):
    # print(pred, label)
    pred_ = torch.argmax(pred)
    # print(pred_)
    corr_num = torch.sum(pred_ == label)
    total_num = pred.size()[0]
    # print(corr_num, total_num, corr_num/total_num)
    return corr_num, total_num


def read_training_data():
    train_data = open('./dataset/dataset/train_6.txt', 'r').readlines()
    dataX, dataY = [], []
    for i, eachline in enumerate(train_data):
        text, aspect, label = eachline.strip().replace("\n","").split("\t")
        if i<int(len(train_data)*0.8):
            dataX.append([text, int(label)])
        else:
            dataY.append([text, int(label)])

    train_data = open('./dataset/dataset/res_train_6.txt', 'r').readlines()
    for i, eachline in enumerate(train_data):
        text, aspect, label = eachline.strip().replace("\n", "").split("\t")
        if i < int(len(train_data) * 0.8):
            dataX.append([text, int(label)])
        else:
            dataY.append([text, int(label)])
    return dataX, dataY


def train_epoch(tokenizer, model, training_data, optimizer, opt):
    """ Epoch operation in training phase. """

    model.train()
    total_corr_num = 0
    total_num = 0

    # print(len(training_data))
    t = tqdm(training_data, mininterval=2,
                      desc='  - (Training)   ', leave=False)
    for i, batch in enumerate(t):
        """ prepare data """
        # print(batch)
        text, label = batch  # map(lambda x: x.to(opt.device), batch)
        # print(batch)

        encoded_input = tokenizer(text, return_tensors='pt', add_special_tokens=True, padding=True)
        encoded_input = {key: tensor.to(opt.device) for key, tensor in encoded_input.items()}

        # output.cuda()

        """ forward """
        optimizer.zero_grad()
        label = torch.tensor([label], device=opt.device)
        loss, logits = model(encoded_input, label)

        """ backward """
        loss.backward(retain_graph=True)

        """ update parameters """
        optimizer.step()


        corr_num, num = cal_acc(logits, label)
        total_corr_num += corr_num
        total_num += num

        if i % 50 == 0:
            t.set_description('  - (Training)   loss: {loss: 8.5f} accuracy: {accuracy: 8.5f}'.format(loss=loss, accuracy=total_corr_num / total_num))

    t.set_description('')
    # print(total_corr_num, total_num, total_corr_num/total_num)

    return torch.sum(loss).item(), total_corr_num/total_num


def eval_epoch(tokenizer, model, test_data, optimizer, opt):
    """ Epoch operation in training phase. """
    model.eval()
    total_corr_num = 0
    total_num = 0
    # print(len(training_data))

    with torch.no_grad():
        t = tqdm(test_data, mininterval=2,
                          desc='  -     (Test)   ', leave=False)
        for i, batch in enumerate(t):

            """ prepare data """
            text, label = batch  # map(lambda x: x.to(opt.device), batch)

            encoded_input = tokenizer(text, return_tensors='pt', add_special_tokens=True, padding=True)
            encoded_input = {key: tensor.to(opt.device) for key, tensor in encoded_input.items()}

            """ forward """
            label = torch.tensor([label], device=opt.device)
            loss, logits = model(encoded_input, label)
            # print(loss, logits)

            corr_num, num = cal_acc(logits, label)
            total_corr_num += corr_num
            total_num += num
            if i % 50 == 0:
                t.set_description('  - (Test)   loss: {loss: 8.5f} accuracy: {accuracy: 8.5f}'.format(loss=loss, accuracy=total_corr_num / total_num))
        t.set_description('')
    # print(total_corr_num, total_num, total_corr_num/total_num)
    return torch.sum(loss).item(), total_corr_num / total_num


def train(tokenizer, model, training_data, test_data, optimizer, scheduler, opt):
    """ Start training. """

    valid_precision_max = 0.0
    for epoch_i in range(opt.epoch):
        epoch = epoch_i + 1
        print('\r[ Epoch', epoch, ']')

        start = time.time()  # loglikelihood: {ll: 8.5f},
        loss, accuracy = train_epoch(tokenizer, model, training_data, optimizer, opt)
        print('\r (Training)    loss:{loss: 8.5f}, accuracy:{accuracy: 8.5f}, elapse:{elapse:3.3f} min'
              .format(elapse=(time.time() - start) / 60,
                      loss=loss, accuracy=accuracy))

        start = time.time()
        loss, accuracy = eval_epoch(tokenizer, model, test_data, optimizer, opt)
        print('\r     (Test)    loss:{loss: 8.5f}, accuracy:{accuracy: 8.5f}, elapse:{elapse:3.3f} min'
              .format(elapse=(time.time() - start) / 60,
                      loss=loss, accuracy=accuracy))

        scheduler.step()

        valid_precision_max = valid_precision_max if valid_precision_max > accuracy else accuracy

    return valid_precision_max


def collate_fn(insts):
    """ Collate function, as required by PyTorch. """
    # print(list(zip(*insts)))
    (text, label) = list(zip(*insts))
    return text, label


def main():
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()
    opt.device = torch.device('cuda')
    opt.epoch = 30
    opt.lr = 0.00001

    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
    # print(tokenizer)
    # print(tokenizer.sep_token_id)

    training_data, test_data = read_training_data()
    # print(texts, labels)
    # training_data = zip(texts, labels)
    # print(training_data)

    # dl = torch.utils.data.DataLoader(
    #     training_data,
    #     num_workers=2,
    #     batch_size=1,
    #     collate_fn=collate_fn,
    #     shuffle=False
    # )

    model = ALBertForClassification.from_pretrained("albert-base-v2", num_labels=6)
    model.cuda()

    """ optimizer and scheduler """
    # optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.albert.parameters()),
    #                        opt.lr, betas=(0.9, 0.999), eps=1e-05)

    parameters = [{'params': model.classifier.parameters(), 'lr': 0.00001},
                  {'params': model.fc.parameters(), 'lr': 0.00001},
                  {'params': model.albert.parameters(), 'lr': 0.000001}]
    optimizer = torch.optim.Adam(parameters)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)

    # print(training_data)
    train(tokenizer, model, training_data, test_data, optimizer, scheduler, opt)

    # tokenizer = BertTokenizer.from_pretrained(args['bert_model'], do_lower_case=args['do_lower_case'])

    # print(output)


if __name__ == '__main__':
    main()

