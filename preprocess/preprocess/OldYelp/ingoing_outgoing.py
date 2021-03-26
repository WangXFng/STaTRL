import json
import re

# !/usr/bin/env python
# -*-encoding:UTF-8 -*-

import pymysql
import numpy as np
import ast

import database.Constants as Constants
from geopy.distance import geodesic

# print(geodesic((30.28708,120.12802999999997), (28.7427,115.86572000000001)).m) #计算两个坐标直线距离
# print(geodesic((30.28708,120.12802999999997), (28.7427,115.86572000000001)).km) #计算两个坐标直线距离

city = Constants.NOW_CITY  # Champaign # Charlotte

# Toronto Number of parameters: 43259282

count = 0

# ingoing outgoing
ingoing = np.zeros(Constants.POI_NUM)
outgoing = np.zeros(Constants.POI_NUM)
train_track = np.zeros((Constants.POI_NUM,Constants.POI_NUM))
test_track = np.zeros((Constants.POI_NUM,Constants.POI_NUM))

f = open('/home/g19tka20/Downloads/Yelp/Yelp_check_ins.txt', 'r')
line = f.readline()
user_index = 0
one_user_actions = []
while line:
    count += 1
    line = line.split("\n")[0]
    data = line.split("\t")
    # print(user_index,int(data[0] ))
    if user_index != int(data[0]):
        len_ = len(one_user_actions)
        # tuning_len = int(len_*0.7)
        # test_len = int(len_*0.9)
        for i,oua in enumerate(one_user_actions):
            if i == 0:
                continue
            for o in one_user_actions[:i]:
                test_track[o][oua] += 1
                train_track[o][oua] += 1

        # for i, oua in enumerate(one_user_actions[tuning_len:test_len]):
        #     if i == 0:
        #         continue
        #     for o in one_user_actions[:i]:
        #         test_track[o][oua] += 1
        #     # # ingoing[oua] += i
        #     # ingoing[oua] += 1
        #     # # outgoing[oua] += len(one_user_actions) - 1 - i
        #     # outgoing[oua] += 1

        user_index += 1
        one_user_actions = []

    one_user_actions.append(int(data[1]))

    if count % 1000 == 0:
        print(count)
    line = f.readline()

# print('ingoing.max()', ingoing.max())
# print('outgoing.max()', outgoing.max())
# print('track.max()', track.max())

# np.save("Yelp/%s_ingoing.npy" % city, ingoing / ingoing.max())
# np.save("Yelp/%s_outgoing.npy" % city, outgoing / outgoing.max())
np.save("Yelp/%s_train_track.npy" % city, train_track / train_track.max())
np.save("Yelp/%s_test_track.npy" % city, test_track / test_track.max())

print(train_track)
print(test_track)

# ingoing = np.load("Yelp/%s_ingoing.npy" % city)
# outgoing = np.load("Yelp/%s_outgoing.npy" % city)
# track = np.load("Yelp/%s_track.npy" % city)
#
# print('ingoing.max()',ingoing.max())
# print(ingoing)
#
# print('outgoing.max()',outgoing.max())
# print(outgoing)
#
# print('track.max()',track.max())
# print(track)

