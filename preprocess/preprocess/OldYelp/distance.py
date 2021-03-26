
import numpy as np
import pymysql
import json

import database.Constants as Constants

from math import radians, cos, sin, asin, sqrt

#公式计算两点间距离（m）

def geodistance(lat1, lng1,lat2, lng2):
    #lng1,lat1,lng2,lat2 = (120.12802999999997,30.28708,115.86572000000001,28.7427)
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)]) # 经纬度转换成弧度
    dlon=lng2-lng1
    dlat=lat2-lat1
    a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    distance=2*asin(sqrt(a))*6371*1000 # 地球平均半径，6371km
    distance=round(distance/1000,3)

    # distance = int(distance)
    # return -1 if distance > 20 else distance
    return distance

# 返回 446.721 千米


count = 0

# ingoing outgoing
ingoing = np.zeros(Constants.POI_NUM)
outgoing = np.zeros(Constants.POI_NUM)

city = Constants.NOW_CITY


lats = []
lngs = []
f = open('/home/g19tka20/Downloads/Yelp/Yelp_poi_coos.txt', 'r')
line = f.readline()
while line:
    count += 1
    j = line.split("\t")
    lats.append(float(j[1]))
    lngs.append(float(j[2]))
    line = f.readline()

# np.save("Yelp/%s_coors.npy" % city, [lats, lngs])

# key = []
# value = []
#
#
# disc = np.zeros((Constants.POI_NUM,Constants.POI_NUM))
# for i in range(Constants.POI_NUM):
#     for j in range(Constants.POI_NUM):
#         if i<j:
#             disc[i][j] = disc[j][i] = geodistance(lats[i],lngs[i], lats[j],lngs[j])
#     if i % 100 == 0:
#         print(i)
#
# np.save("Yelp/%s_disc.npy" % city, disc)
#
# print('save disc', disc)
# #######################################
# disc = np.load("Yelp/%s_disc.npy" % city)
# #
# #
# import sys
# sys.setrecursionlimit(100000) #例如这里设置为十万
#
# disc_block = []
#
# def DFS(i, re, ug, gc, stack, c):
#
#     for j, r in enumerate(re[i]):
#         if r < 5 :
#             if ug[j] == 0:
#                 ug[j] = ug[i]
#                 c += 1
#                 print(c)
#                 ug, group_count, stack, c = DFS(j, re, ug, gc, stack, c)
#
#     return ug, gc, stack, c
#
#
# stack = []
# group_count = 0
# group = np.zeros(18995)
# c = 0
#
# for i in range(18995):
#     if group[i] == 0:
#         group_count += 1
#         group[i] = group_count
#
#         c += 1
#         print(c)
#         group, group_count, stack, c = DFS(i, disc, group, group_count, stack, c)
#
# np.save("Yelp/%s_group.npy" % city, group)
#

group = np.load("Yelp/%s_group.npy" % city)
for i in range(37):
    print(i, len(np.where(group == i)[0]), np.where(group == i))

# print('save group', group_count)
#
# # group = np.load("Yelp/%s_group.npy" % city)
# #
# # key_group = np.zeros(18995)
# # value_group = []
# # for index in range(group_count):
# #     l = np.where(group==index)[0]
# #     len_ = len(l)
# #     key_ = np.zeros(len_)
# #     value_ = np.zeros((len_, len_))
# #     for ii in range(len_):
# #         key_group[l[ii]] = ii
# #         for j in range(len_):
# #             if ii<j:
# #                 value_[ii][j] = disc[l[ii]][l[j]]
# #
# #     value_group.append(value_)
# #
# #
# # import pickle
# # import os
# # data_output = open('Yelp/%s_all_group.pkl'%city,'wb')
# # pickle.dump({'key_group':key_group,'value_group':value_group }, data_output)
# # data_output.close()
# #
# # # np.save("Yelp/%s_all_group.npy" % city, all_group)
# # #
# # # all_group = np.load("Yelp/%s_all_group.npy" % city)
# # # print(all_group)
# #
# # with open("Yelp/%s_all_group.pkl" % city, 'rb') as f:
# #     data = pickle.load(f, encoding='latin-1')
# #     print(data['key_group'])
# #     print(data['value_group'])
# # print(geodesic((30.28708,120.12802999999997), (28.7427,115.86572000000001)).m) #计算两个坐标直线距离
# # print(geodesic((30.28708,120.12802999999997), (28.7427,115.86572000000001)).km) #计算两个坐标直线距离



