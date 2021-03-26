from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from gcn.utils import load_data, accuracy
from gcn.models import GCN


def grbf(d):
    n = 0.125
    a = np.exp(-n * d)
    a[a < 0.125] = 0
    print(a.max())
    print(a.min())
    print(a)
    return a


print(grbf(np.array([1, 2, 3, 4, 5, 10, 15, 15, 17, 18, 19, 20, 30])))


# disc = np.load("../../data/Yelp/old_yelp_disc.npy")
#
# print(disc.min())
#
# A = grbf(disc)
# print(A.max())
#
# # Training settings
# parser = argparse.ArgumentParser()
# parser.add_argument('--no-cuda', action='store_true', default=False,
#                     help='Disables CUDA training.')
# parser.add_argument('--fastmode', action='store_true', default=False,
#                     help='Validate during training pass.')
# parser.add_argument('--seed', type=int, default=42, help='Random seed.')
# parser.add_argument('--epochs', type=int, default=200,
#                     help='Number of epochs to train.')
# parser.add_argument('--lr', type=float, default=0.000001,
#                     help='Initial learning rate.')
# parser.add_argument('--weight_decay', type=float, default=5e-4,
#                     help='Weight decay (L2 loss on parameters).')
# parser.add_argument('--hidden', type=int, default=1024,
#                     help='Number of hidden units.')
# parser.add_argument('--dropout', type=float, default=0.5,
#                     help='Dropout rate (1 - keep probability).')
#
# args = parser.parse_args()
# args.cuda = False  # not args.no_cuda and torch.cuda.is_available()
#
# np.random.seed(args.seed)
# torch.manual_seed(args.seed)
# if args.cuda:
#     torch.cuda.manual_seed(args.seed)
#
# # Load data
# # adj, features, labels, idx_train, idx_val, idx_test = load_data()
# # print(adj.size())
# # print(features.size())
# # print(labels.size())
# # print(idx_train.size())
# # print(idx_val.size())
# # print(idx_test.size())
#
# adj = torch.tensor(A, dtype=torch.double)
# features = torch.tensor(A, dtype=torch.double)
#
# # Model and optimizer
# model = GCN(nfeat=features.shape[1],
#             nhid=args.hidden,
#             nclass=1024,
#             dropout=args.dropout)
# optimizer = optim.Adam(model.parameters(),
#                        lr=args.lr, weight_decay=args.weight_decay)
#
# if args.cuda:
#     model.cuda()
#     features = features.cuda()
#     adj = adj.cuda()
#     # labels = labels.cuda()
#     # idx_train = idx_train.cuda()
#     # idx_val = idx_val.cuda()
#     # idx_test = idx_test.cuda()
#
#
# def train(epoch):
#     t = time.time()
#     model.train()
#     optimizer.zero_grad()
#     output = model(features, adj)
#     # print(output.size())
#     # print(output.T.size())
#     # print(torch.mm(output, output.T).size())
#     # # print((output.T * output).size())
#     # print(torch.mm(output, output.T).size(), A.size())
#     print('output.max()', output.max())
#     print('output.min()', output.min())
#     o = torch.mm(output, output.T)
#     o = torch.tanh(o)
#     # o = output
#     print('o.max()', o.max())
#     a = torch.tensor(A.copy(), dtype=torch.double)
#     loss_train = F.mse_loss(o, a)  # F.nll_loss(o, a)  # F.nll_loss(o, a)
#     print('loss_train', loss_train)
#     # acc_train = accuracy(output[idx_train], labels[idx_train])
#     loss_train.backward()
#     optimizer.step()
#
#     if not args.fastmode:
#         # Evaluate validation set performance separately,
#         # deactivates dropout during validation run.
#         model.eval()
#         output = model(features, adj)
#
#     # loss_val = F.nll_loss(output[idx_val], labels[idx_val])
#     # acc_val = accuracy(output[idx_val], labels[idx_val])
#     print('Epoch: {:04d}'.format(epoch+1),
#           'loss_train: {:.4f}'.format(loss_train.item()),
#           # 'acc_train: {:.4f}'.format(acc_train.item()),
#           # 'loss_val: {:.4f}'.format(loss_val.item()),
#           # 'acc_val: {:.4f}'.format(acc_val.item()),
#           'time: {:.4f}s'.format(time.time() - t))
#
#
# def test():
#     model.eval()
#     output = model(features, adj)
#     print('output')
#     print(output)
#     print(output.size())
#     # loss_test = F.nll_loss(output[idx_test], labels[idx_test])
#     # acc_test = accuracy(output[idx_test], labels[idx_test])
#     # print("Test set results:",
#     #       "loss= {:.4f}".format(loss_test.item()),
#     #       "accuracy= {:.4f}".format(acc_test.item()))
#
#
# # Train model
# t_total = time.time()
# for epoch in range(args.epochs):
#     train(epoch)
# print("Optimization Finished!")
# print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
#
# # Testing
# test()