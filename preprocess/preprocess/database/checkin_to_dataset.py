import json
import re
import pickle
import numpy as np
import os

import database.Constants as Constants
# !/usr/bin/env python
# -*-encoding:UTF-8 -*-
print(os.getcwd())
print(os.path.abspath(os.path.dirname(__file__)))


f = open(os.getcwd() + '/../data/Yelp/Yelp_check_ins.txt', 'r')
line = f.readline()
min_time = 999999999999999999.0
while line:
    line = line.split("\n")[0]
    data = line.split("\t")

    if float(data[2])<min_time:
        min_time = float(data[2])

    line = f.readline()

print(min_time)

f = open(os.getcwd() + '/../data/Yelp/Yelp_poi_categories.txt', 'r')
line = f.readline()
category_index = 0
categories = []
category = np.zeros(Constants.POI_NUM)
# category_id = []
# category_ids = []
while line:
    line = line.split("\n")[0]
    data = line.split("\t")

    if category_index != int(data[0]) or category_index == 18994:
        categories.append(category)
        # category_ids.append(category_id)
        category_index += 1
        category = np.zeros(Constants.POI_NUM)
        # category_id = []

    category[int(data[1])] = 1
    # category_id.append(int(data[1]))

    line = f.readline()
    # if category_index == 2:
    #     print(categories)
f.close()


# print(len(categories))
print('Generate category index successfully ！')

f = open(os.getcwd() + '/../data/Yelp/Yelp_check_ins.txt', 'r')
whole_data = []

line = f.readline()
user_index = 0
one_user_actions = []
time_s = 0
while line:
    line = line.split("\n")[0]
    data = line.split("\t")
    # print(user_index,int(data[0] ))
    if user_index != int(data[0]):
        whole_data.append(one_user_actions)
        user_index += 1
        category = np.zeros(Constants.POI_NUM)
        one_user_actions = []
        time_s = 0

    # print(int(data[1]))
    # print((float(data[2])-min_time)/3600000+time_s)
    one_user_actions.append({'time_since_start': (float(data[2])-min_time)/3600000+time_s,
                             'time_since_last_event': time_s if len(one_user_actions) == 1 else 0,
                             'type_event': categories[int(data[1])] })
    # print({'time_since_start': (float(data[2])-min_time)/3600000+time_s,
    #                          'time_since_last_event': time_s if len(one_user_actions) == 1 else 0,
    #                          'type_event': categories[int(data[1])] ,
    #                          'event_label': category_ids[int(data[1])] })

    time_s += 1

    line = f.readline()
    # if user_index == 5000:
    #     break
    # if user_index == 1:
    #     print(whole_data)
    #     break

f.close()

print(len(whole_data))

data = {'dim_process': Constants.POI_NUM, 'train': whole_data[:int(len(whole_data)*0.8)]}

data_output = open(os.getcwd() + '/../data/Yelp/train.pkl','wb')
pickle.dump(data, data_output)
data_output.close()

# data_output = open(os.getcwd() + '/../data/Yelp/dev.pkl','wb')
# pickle.dump(whole_data, data_output)
# data_output.close()

data = {'dim_process': Constants.POI_NUM, 'test': whole_data[int(len(whole_data)*0.8)+1:]}
data_output = open(os.getcwd() + '/../data/Yelp/test.pkl','wb')
pickle.dump(data, data_output)
data_output.close()

# error = []
# f = open(os.getcwd() + '/../data/Yelp/Yelp_check_ins_categories.txt', 'w')
# for er in error:
#     f.write(er+"\r\n")
#
# print(len(error))

