import json
import re
import pickle
import numpy as np
import os
import math

# import transformer.Constants as Constants
# !/usr/bin/env python
# -*-encoding:UTF-8 -*-
import database.Constants as Constants

min_time = 1098028800.0
city = Constants.NOW_CITY

coors = np.load("Yelp/%s_coors.npy" % city)
lats = coors[0]
lngs = coors[1]

def grbf(d):
    n = 0.05
    a = np.exp(-n * d)
    a[a < 0.125] = 0
    # print(a.max())
    # print(a.min())
    return a


group = np.load("Yelp/%s_500_kmeans_g.npy" % city)
global disc
disc = np.load("Yelp/%s_disc.npy" % city)
disc = grbf(disc)

global poi_sim
poi_sim = np.load("Yelp/%s_poi_similarity.npy" % city)
# print(disc.max())

global group_
group_ = np.load("Yelp/%s_group.npy" % city)

with open("Yelp/%s_500_similarities.pkl" % city, 'rb') as f:
    similarities = pickle.load(f, encoding='latin-1')
# similarities = np.load("Yelp/%s_similarities.npy" % city)


def getScore_(s):
    if s < 5:
        s /= 10
    else:
        s = 0.5 + 0.35 * np.log(s-4)
    return s


# tune label and tune actions
tune_labels = [[] for i in range(Constants.USER_NUM)]
tune_scores = [[] for i in range(Constants.USER_NUM)]
tune_lats = [[] for i in range(Constants.USER_NUM)]
tune_lngs = [[] for i in range(Constants.USER_NUM)]
f = open(os.getcwd() + '/Yelp/Yelp_tune.txt', 'r')
line = f.readline()
while line:
    line = line.split("\n")[0]
    data = line.split("\t")
    user_id = int(data[0])

    tune_labels[user_id].append(int(data[1])+1)
    tune_scores[user_id].append(getScore_(int(data[2])))
    tune_lats[user_id].append(lats[int(data[1])])
    tune_lngs[user_id].append(lngs[int(data[1])])

    line = f.readline()

f.close()



# test label
f = open(os.getcwd() + '/Yelp/Yelp_test.txt', 'r')
line = f.readline()
test_labels = [[] for i in range(Constants.USER_NUM)]
test_scores = [[] for i in range(Constants.USER_NUM)]
while line:
    line = line.split("\n")[0]
    data = line.split("\t")

    user_id = int(data[0])
    test_labels[user_id].append(int(data[1])+1)
    test_scores[user_id].append(getScore_(int(data[2])))
    line = f.readline()

f.close()


def process(actions, score, lats, lngs, labels, scores, data):

    # [([9914, 7050, 4662, ..., 7051], [1, 4, 4,..., 4], [7970, 4662, 13853], [5, 3, 4], array(), array()
    # print(tune_labels[user_index])
    len_ = len(actions)
    inner_dis = np.zeros((len_, len_))
    inner_poi_sim = np.zeros((len_, len_))
    for i, a in enumerate(actions):
        for j, b in enumerate(actions):
            inner_dis[i][j] = inner_dis[j][i] = disc[a-1][b-1]
            inner_poi_sim[i][j] = inner_poi_sim[j][i] = poi_sim[a-1][b-1]

    if len(actions.copy()) != len(score.copy()):
        print('len(actions.copy()) != len(score.copy())')
        # print(actions, score)
        # print('<<<<<<<<<<<<<<<<<')

    # last_ = actions[0]
    # distance = disc[last_]
    # where_ = np.where(distance<15)[0]
    # distance = distance[where_]
    # print(len(where_), len(distance))
    #
    # where_2 = group_[group_==group_[last_]]
    #
    # w_ = where_.copy()
    # set(where_).intersection(set(where_2))
    #
    # print('where', len(where_), len(where_2), len(w_))

    # print(where_.tolist(), distance.tolist())


    data.append(((actions.copy(),), (score.copy(),),(lats.copy(),),(lngs.copy(),), (labels,), (scores,),
                  (inner_dis,), (inner_poi_sim,), ([group_[actions[0]] for i in actions],),), )


train_data = []
test_data = []

score = []
actions = []
lat = []
lng = []
f = open(os.getcwd() + '/Yelp/Yelp_train.txt', 'r')
line = f.readline()
user_index = 0
while line:
    line = line.split("\n")[0]
    data = line.split("\t")
    # print(user_index,int(data[0] ))
    if user_index != int(data[0]):
        len_ = len(actions)

        # if len_ >= 10:

        if len(tune_scores[user_index]) == 0:
            # print(actions, tune_labels[user_index])
            tune_labels[user_index].append(actions[len(actions)-1])
            tune_scores[user_index].append(score[len(actions)-1])
            actions = actions[:len(actions)-1]
            score = score[:len(score)-1]
            # print(actions, tune_labels[user_index])
            # print(score, tune_scores[user_index])

        process(actions, score, lat, lng,
                tune_labels[user_index], tune_scores[user_index],  train_data)

        # for i in range(6):
        #     action_s = int(i*len(actions)/8)
        #     action_e = int((i+1)*len(actions)/8)
        #
        #     label_e = int((i+2)*len(actions)/8)
        #     if action_e-action_s >= 5:
        #         actions_ = actions[action_s:action_e]
        #         tune_labels_ = actions[action_e:label_e]
        #
        #         score_ = score[action_s:action_e]
        #         tune_scores_ = score[action_e:label_e]
        #
        #         process(actions_, score_, [], [],
        #                 tune_labels_, tune_scores_,  train_data)

        #
        # user_group = int(group[user_index]) - 1
        # print(train_data[-1:])

        # test data
        actions.extend(tune_labels[user_index])
        score.extend(tune_scores[user_index])
        lat.extend(tune_lats[user_index])
        lng.extend(tune_lngs[user_index])
        process(actions, score, lat, lng,
                test_labels[user_index], test_scores[user_index], test_data)

        actions = []
        score = []
        lat = []
        lng = []

        # if user_index % 1000 == 0:
        #     print(user_index)
            # break
        user_index += 1

    # print(int(data[1]))
    # print((float(data[2])-min_time)/3600000+time_s)

    actions.append(int(data[1])+1)
    score.append(getScore_(int(data[2])))
    lat.append(lats[int(data[1])])
    lng.append(lngs[int(data[1])])
    # actions.append({'score': int(data[2]),
    #                          'type_event': int(data[1])
    #                          })
    # #                        'type_event': max_type if int(data[1]) > max_type else int(data[1])})


    line = f.readline()
        # break
    # if user_index == 1:
    #     print(train_data)
    #     break

# if len_ >= 10:
process(actions, score, lat, lng, tune_labels[user_index], tune_scores[user_index], train_data)

# test data
actions.extend(tune_labels[user_index])
score.extend(tune_scores[user_index])
lat.extend(tune_lats[user_index])
lng.extend(tune_lngs[user_index])
process(actions, score, lat, lng, test_labels[user_index], test_scores[user_index], test_data)

f.close()

# # for i in train_data:
# #     if len(i[0]) == 0:
# #         print(i)
#
# count = 0
#
# precision_s = np.zeros(math.ceil(30887))
# recall_s = np.zeros(math.ceil(30887))
#
# for i in test_data:
#     for j in i[0]:
#         score = np.zeros(18996)
#         # print(len(j[0][0]), len(j[1][0]), len(j[4][0]), len(j[5][0]))
#         ind = j[0][0].copy()
#         ind.extend(j[4][0])
#
#
#         sco = j[1][0].copy()
#         sco.extend(j[5][0])
#
#         score[ind] = sco
#         # print(j[0][0], ind, j[1][0])
#
#         target = np.ones(18996)
#         target[j[0][0]] = 0
#         # print(score, j[4][0], target)
#
#         precision_s, recall_s, count = pre_rec_top(precision_s, recall_s, count, score, j[4][0], target)
#
# precision = np.sum(precision_s)/count
# recall = np.sum(recall_s)/count
#
# print(precision.item(), recall.item())
# #
print('train_data', len(train_data))
print('test_data', len(test_data))

# np.random.shuffle(train_data)

with open("Yelp/%s_all_group.pkl" % city, 'rb') as f:

    data = {'dim_process': Constants.POI_NUM, 'train': train_data}

    data_output = open(os.getcwd() + '/Yelp/train_%s.pkl'%city,'wb')
    pickle.dump(data, data_output)
    data_output.close()

    data = {'dim_process': Constants.POI_NUM, 'test': test_data}

    data_output = open(os.getcwd() + '/Yelp/test_%s.pkl'%city,'wb')
    pickle.dump(data, data_output)
    data_output.close()

