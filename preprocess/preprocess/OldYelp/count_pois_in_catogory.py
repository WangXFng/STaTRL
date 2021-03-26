
import numpy as np
import pymysql
import json

import database.Constants as Constants

from math import radians, cos, sin, asin, sqrt

import os
count = 0

city = Constants.NOW_CITY


cate_c = np.zeros(624)

# sim_matrix = np.zeros((Constants.POI_NUM, Constants.USER_NUM))
f = open(os.getcwd() + '/Yelp/Yelp_poi_categories.txt', 'r')
line = f.readline()
while line:
    line = line.split("\n")[0]
    data = line.split("\t")
    # print(user_index,int(data[0] ))

    cate_c[int(data[1])] += 1


    line = f.readline()

f.close()

print(cate_c[0], cate_c[int(len(cate_c)/2)], cate_c[len(cate_c)-1])
# for i in range(624):
#     print(cate_c[i])

ind = np.argsort(cate_c)

# print(ind)
# print(cate_c[2])
# print(cate_c.sum())

import math
poi_c = np.empty(Constants.POI_NUM)
poi_c.fill(-1)

f = open(os.getcwd() + '/Yelp/Yelp_poi_categories.txt', 'r')
line = f.readline()
while line:
    line = line.split("\n")[0]
    data = line.split("\t")
    # print(user_index,int(data[0] ))
    # print(int(data[1]), int(data[0]), int(poi_c[int(data[0])]))
    if cate_c[int(data[1])] > cate_c[int(poi_c[int(data[0])])]:
        poi_c[int(data[0])] = int(data[1])
    # poi_c[int(data[0])] = max(poi_c[int(data[0])], )

    line = f.readline()

f.close()

print(poi_c)
print(poi_c.max())

# np.save(os.getcwd() + '/Yelp/Yelp_poi_one_category.np', poi_c)