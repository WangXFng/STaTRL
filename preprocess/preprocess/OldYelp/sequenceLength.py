import json
import re
import pickle
import numpy as np
import os

# import transformer.Constants as Constants
# !/usr/bin/env python
# -*-encoding:UTF-8 -*-
import database.Constants as Constants

min_time = 1098028800.0

f = open(os.getcwd() + '/Yelp/Yelp_check_ins.txt', 'r')

city = Constants.NOW_CITY
ingoing = np.load("Yelp/%s_ingoing.npy" % city)
outgoing = np.load("Yelp/%s_outgoing.npy" % city)
disc = np.load("Yelp/%s_disc.npy" % city)


whole_data = []
# max_type = 999
line = f.readline()
user_index = 0
one_user_actions = []
time_s = 0

cc = np.zeros(200)

while line:
    line = line.split("\n")[0]
    data = line.split("\t")
    # print(user_index,int(data[0] ))
    if user_index != int(data[0]):
        # whole_data.append(one_user_actions)
        user_index += 1
        time_s = 0
        len_ = len(one_user_actions)

        if len_ > 199:
            cc[199] += 1
        else:
            cc[len_] += 1

        # if len_ >= 10:
        #     matrix = np.zeros((len_,len_))
        #     in_ = []
        #     out_ = []
        #     for i,a in enumerate(one_user_actions):
        #         for j,b in enumerate(one_user_actions):
        #             if i != j :
        #                 matrix[i][j] = matrix[j][i] = disc[a['type_event']][b['type_event']]
        #         in_.append(ingoing[a['type_event']])
        #         out_.append(outgoing[a['type_event']])
        #
        #     in_ = np.array(in_)
        #     out_ = np.array(out_)
        #
        #     a = np.reshape(in_, (1, -1))
        #     b = np.reshape(out_, (1, -1))
        #
        #     matrix[matrix == 0] = 0.0000001
        #
        #     geo_ = a.T * b
        #     geo_[geo_ == 0] = 0.0000001
        #
        #     # print(geo_.max())
        #     # print(geo_.min())
        #
        #     whole_data.append({'actions': one_user_actions, 'distance': matrix, 'in_out': geo_ })
        #
        one_user_actions = []

        if user_index % 1000 == 0:
            print(user_index)

    # print(int(data[1]))
    # print((float(data[2])-min_time)/3600000+time_s)

    one_user_actions.append({'time_since_start': (float(data[2])-min_time)/3600000+time_s,
                             'time_since_last_event': time_s if len(one_user_actions) != 0 else 0,
                             'type_event': int(data[1])})
    #                        'type_event': max_type if int(data[1]) > max_type else int(data[1])})

    # time_s += 1

    line = f.readline()
        # break
    # if user_index == 1:
    #     print(whole_data)
    #     break

f.close()

for i,o in enumerate(cc):
    print("%s:"%i, o)
# #
# print(len(whole_data))
# # print(whole_data)
#
# # data = {'dim_process': Constants.TYPE_DIMENSION, 'train': whole_data[:int(len(whole_data)*0.8)]}
#
# data = {'dim_process': Constants.POI_NUM, 'train': whole_data[:int(len(whole_data)*0.8)]}
#
# # print(data['train'].__len__())
#
# data_output = open(os.getcwd() + '/Yelp/train_%s.pkl'%city,'wb')
# pickle.dump(data, data_output)
# data_output.close()
#
# # data_output = open(os.getcwd() + '/data/Yelp/dev.pkl','wb')
# # pickle.dump(whole_data, data_output)
# # data_output.close()
#
# data = {'dim_process': Constants.POI_NUM, 'test': whole_data[int(len(whole_data)*0.8)+1:]}
#
# # print(data['train'].__len__())
#
# data_output = open(os.getcwd() + '/Yelp/test_%s.pkl'%city,'wb')
# pickle.dump(data, data_output)
# data_output.close()
# #
# # error = []
# # f = open(os.getcwd() + '/data/Yelp/Yelp_check_ins_categories.txt', 'w')
# # for er in error:
# #     f.write(er+"\r\n")
# #
# # print(len(error))

