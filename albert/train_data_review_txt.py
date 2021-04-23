import json
import re
import pickle
import numpy as np
import os
import ast

# !/usr/bin/env python
# -*-encoding:UTF-8 -*-

min_time = 1098028800

# path = '/home/g19tka20/Downloads/full_Yelp/yelp_dataset/11_0_4/dataset'
# city = Constants.NOW_CITY
user_num = 28038  # 119876
poi_num = 15745  # 62796

global disc
disc = np.load("./dataset/dataset/disc.npy")

with open("./dataset/dataset/business_id_2_top.pkl", 'rb') as f:
    category = pickle.load(f, encoding='latin-1')
    # num_types = data['dim_process']

top_parent = ['hotelstravel', 'nightlife', 'food', 'active', 'arts', 'auto', 'shopping',
            'professional', 'physicians', 'pets', 'health', 'fitness', 'education', 'beautysvc']

# print(category)

# tune label and tune actions
actions = [[] for i in range(user_num)]
times = [[] for i in range(user_num)]
scores = [[] for i in range(user_num)]
aspects = [[] for i in range(user_num)]

# def getScore(s):
#     if s < 5:
#         return s
#     else:
#         # s1 = 5
#         if s-5<100:
#             return 6
#         else:
#             return 7

# hotelstravel {'Food_neg': 0, 'Food_pos': 1, 'Price_neg': 2, 'Price_pos': 3, 'Service_neg': 4, 'Service_pos': 5 }
# nightlife
# food
# active {'Price_neg': 2, 'Price_pos': 3, 'Service_neg': 4, 'Service_pos': 5 }
# arts
# auto
# shopping
# professional
# physicians
# pets
# health
# fitness
# education
# beautysvc

def handle(str, poi):
    res = str.strip('][').replace('  ', ' ').replace('  ', ' ').split(' ')
    res = res[0:6]
    if len(res) != 6:
        print('len(res) : ', len(res), res)
        return 1/0

    res = [float(r) for r in res]

    for i in range(3):
        if res[i*2]>res[i*2+1]:
            res[i*2+1] = 0
        else:
            res[i*2] = 0
    # print(res)

    c = category[poi]
    nj = [0] * 62
    # print(nj)
    if c < 3:
        nj[c * 6: c * 6 + 6] = res
    else:
        nj[2 * c + 12: 2 * c + 16] = res[2:6]
    # print(top_parent[c], c, len(nj), nj)

    return nj


def demand(str, poi):
    res = str.strip('][').replace('  ', ' ').replace('  ', ' ').split(' ')
    res = res[0:6]
    if len(res) != 6:
        print('len(res) : ', len(res), res)
        return 1/0

    res = [float(r) for r in res]

    for i in range(3):
        res[i * 2] = 0
        res[i * 2 + 1] = max(res[i*2], res[i*2+1])
        # if res[i*2]>res[i*2+1]:
        #     res[i*2+1] = 0
        # else:
        #     res[i*2] = 0
    # print(res)

    c = category[poi]
    nj = [0] * 62
    # print(nj)
    if c < 3:
        nj[c * 6: c * 6 + 6] = res
    else:
        nj[2 * c + 12: 2 * c + 16] = res[2:6]
    # print(top_parent[c], c, len(nj), nj)

    return nj


poi_aspects = [[] for i in range(poi_num)]
f = open('./dataset/dataset/Yelp_reviews_test.txt', 'r')
line = f.readline()
count = 0
while line:
    count += 1
    line = line.split("\n")[0]
    data = line.split("\t")
    user_id = int(data[0])
    poi_id = int(data[1])

    actions[user_id].append(int(data[1])+1)
    scores[user_id].append(int(data[2]))
    times[user_id].append(int(data[3])-min_time)
    nj = demand(data[4], int(data[1]))
    aspects[user_id].append(nj)
    poi_aspects[poi_id].append(handle(data[4], int(data[1])))
    if int(data[3])-min_time<0:
        print("int(data[3])-min_time<0int(data[3])-min_time<0")
    line = f.readline()

    if count%10000==0:
        print(count)
        # break

f.close()

poi_avg_aspect = np.zeros((poi_num, 62))
for i in range(poi_num):
    poi_avg_aspect[i] = np.array(poi_aspects[i]).sum(0)/len(poi_aspects[i])
    # print(poi_avg_aspect[i])

# print(poi_avg_aspect[0:100])

def process(actions, score, time, aspect, labels, label_score, data):

    len_ = len(actions)
    inner_dis = np.zeros((len_, len_))
    for i, a in enumerate(actions):
        for j, b in enumerate(actions):
            inner_dis[i][j] = inner_dis[j][i] = disc[a-1][b-1]

    # print(time)
    # time_gap = [0 if i ==0 else time[i]-time[i-1] for i in range(len(time))]
    # print(time_gap)

    data.append(((actions.copy(),), (score.copy(),), (time.copy(),), (aspect.copy(),), (labels,),  (label_score.copy(),), (inner_dis,),), )




train_data = []
test_data = []
for i in range(user_num):
    action = actions[i]
    if len(action) <= 10:
        print("len(action) <= 10")
    score = scores[i]
    time = times[i]
    aspect = aspects[i]

    len_ = len(action)
    process(action[:int(len_*0.7)], score[:int(len_*0.7)], time[:int(len_*0.7)], aspect[:int(len_*0.7)], action[int(len_*0.7):int(len_*0.8)], score[int(len_*0.7):int(len_*0.8)], train_data)

    process(action[:int(len_*0.8)], score[:int(len_*0.8)], time[:int(len_*0.8)], aspect[:int(len_*0.8)], action[int(len_*0.8):], score[int(len_*0.8):], test_data)

    # print(train_data[len(train_data)-1])

    if i % 10000 == 0:
        print(i)

print('train_data', len(train_data))
print('test_data', len(test_data))

np.random.shuffle(train_data)


data = {'dim_process': poi_num, 'train': train_data, 'poi_avg_aspect': poi_avg_aspect}
data_output = open('./dataset/dataset/train_ta_s.pkl','wb')
pickle.dump(data, data_output)
data_output.close()

data = {'dim_process': poi_num, 'test': test_data, 'poi_avg_aspect': poi_avg_aspect}
data_output = open('./dataset/dataset/test_ta_s.pkl','wb')
pickle.dump(data, data_output)
data_output.close()

