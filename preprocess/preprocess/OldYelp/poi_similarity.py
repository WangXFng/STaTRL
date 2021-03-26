
import numpy as np
import pymysql
import json

import database.Constants as Constants

from math import radians, cos, sin, asin, sqrt

import os
count = 0

city = Constants.NOW_CITY

import os

sim_matrix = np.zeros((Constants.POI_NUM, Constants.USER_NUM))
f = open(os.getcwd() + '/Yelp/Yelp_check_ins.txt', 'r')
line = f.readline()
while line:
    line = line.split("\n")[0]
    data = line.split("\t")
    # print(user_index,int(data[0] ))

    sim_matrix[int(data[1])][int(data[0])] = 1

    line = f.readline()

f.close()

print('sim_matrix ready')

# similarity = np.zeros((Constants.POI_NUM, Constants.POI_NUM))
# for j, m in enumerate(sim_matrix):
#     for k, n in enumerate(sim_matrix):
#         if (j < k):
#             similarity[j][k] = np.sum(m * n) / (
#                 np.sqrt((m ** 2).sum() * (n ** 2).sum()))
#             # if similarity[j][k] != 0:
#             #     print(similarity[j][k])
#     # if j % 100 == 0:
#     print(j)

# np.save("Yelp/%s_poi_similarity.npy" % city, similarity)

similarity = np.load("Yelp/%s_poi_similarity.npy" % city)
print(similarity)