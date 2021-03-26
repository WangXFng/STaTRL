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

# # ingoing outgoing
# similarity = []
#
# f = open('/home/g19tka20/Downloads/Yelp/Yelp_check_ins.txt', 'r')
# line = f.readline()
# user_index = 0
# one_user_actions = []
# while line:
#     count += 1
#     line = line.split("\n")[0]
#     data = line.split("\t")
#     # print(user_index,int(data[0] ))
#     if user_index != int(data[0]):
#         # tuning_len = int(len_*0.7)
#         # test_len = int(len_*0.9)
#         similarity.append(one_user_actions)
#
#         user_index += 1
#         one_user_actions = []
#
#     one_user_actions.append(int(data[1]))
#
#     # if count % 1000 == 0:
#     #     print(count)
#     line = f.readline()
#
# similarity.append(one_user_actions)
# print(count)
# np.save("Yelp/%s_similarity.npy" % city, Sij)

# Sij = []
# for i in range(Constants.USER_NUM):
#     for j in range(Constants.USER_NUM):
#         if i<j:
#             # print(len(similarity), i, j)
#             u = set(similarity[i]).intersection(set(similarity[j]))
#             if len(u)>0:
#                 s = len(u)/len(set(similarity[i]).union(set(similarity[j])))
#                 if s > 0.1:
#                     print(i,j,s)
#                 Sij.append((i,j,s))
#     if i%1000==0:
#         print(i)
#
# np.save("Yelp/%s_Sij.npy" % city, Sij)

# Sij = np.load("Yelp/%s_Sij.npy" % city)
# s_ = 0.1
# group_ = np.zeros(Constants.USER_NUM)
# group_count = 1
# group_[0] = group_count
# for (i,j,f) in Sij:
#     i, j, f = int(i), int(j), float(f)
#     print(i, j, f)
#     if f>s_:
#         group_[j] = group_[i]
#     else:
#         group_count += 1
#         group_[j] = group_count
#
# print(group_count)
# #
# # Sij = np.load("Yelp/%s_Sij_user.npy" % city)











#
#
#
# #
# import os
# sim_matrix = np.zeros((Constants.USER_NUM, Constants.POI_NUM))
# f = open(os.getcwd() + '/Yelp/Yelp_train.txt', 'r')
# line = f.readline()
# while line:
#     line = line.split("\n")[0]
#     data = line.split("\t")
#     user_id = int(data[0])
#     # print(user_index,int(data[0] ))
#
#     sim_matrix[int(data[0])][int(data[1])] = 1
#
#     line = f.readline()
#
# f.close()
#
# np.save("Yelp/%s_similarity_matrix.npy" % city, sim_matrix)
# print("save sim_matrix successfully")
sim_matrix = np.load("Yelp/%s_similarity_matrix.npy" % city)

from sklearn.cluster import KMeans
#
# type_numer = [500]
# print('start !!')
# for i in type_numer:
#     kmeans = KMeans(n_clusters=i, random_state=0).fit(sim_matrix)
#
#     np.save("Yelp/%s_%d_kmeans.npy" % (city,i), kmeans.labels_)
#     print(kmeans.labels_)
#
# print('end !!')
#
# # from sklearn.cluster import KMeans
# #
# # X = np.array([[1, 2], [1, 4], [1, 0],
# #               [4, 2], [4, 4], [4, 0]])
# #
# # kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
# #
# # print(kmeans.labels_)

# type_numer = [5, 10, 20, 50, 100]
# print('start !!')
# for i in type_numer:

# g = np.zeros(Constants.USER_NUM)
# kmeans = np.load("Yelp/%s_500_kmeans.npy" % city)
# count_ = 1
# c = 0
#
#
# def dfs(K_label, count_, l, c, n_cluster):
#     before = c
#     print(len(K_label), ' >>> ')
#     for j in range(n_cluster):
#         l2 = np.where(K_label == j)[0]
#         if len(l2) > 100:
#             n_cluster = int(len(l2)/100) + 1
#             n_s = sim_matrix[l[l2]]
#             k = KMeans(n_clusters=n_cluster, random_state=0).fit(n_s)
#             K_label2 = k.labels_
#
#             count_, c = dfs(K_label2, count_, l[l2], c, n_cluster)
#         else:
#             # print( l[l2])
#             if np.sum(g[l[l2]]>0)>0:
#                 print("repeat")
#             g[l[l2]] = count_
#             count_ += 1
#             c += len(l2)
#             # print(count_, '  ', len(l2), '  ', c)
#     print(c - before, ' <<< ')
#     print(c)
#     return count_, c
#
#
# for i in range(500):
#     # print(i, len())
#
#     l = np.where(kmeans == i)[0]
#
#     print(i, '   ', len(l),' start ================>>>>>>>>>>>>>>>>>>>>>>>')
#     if len(l)>100:
#         n_cluster = int(len(l)/100) + 1
#
#         n_s = sim_matrix[l]
#
#         k = KMeans(n_clusters=n_cluster, random_state=0).fit(n_s)
#         K_label = k.labels_
#         count_, c = dfs(K_label, count_, l, c, n_cluster)
#     else:
#         g[l] = count_
#         count_ += 1
#         c += len(l)
#         print(c)
#         # print(count_, len(l))
#     print(i, '   ', len(l),' over ================>>>>>>>>>>>>>>>>>>>>>>>')
#
# print(count_)
# np.save("Yelp/%s_500_kmeans_g.npy" % city, g)

g = np.load("Yelp/%s_500_kmeans_g.npy" % city)
# # for i in range(817):
# print(0, len(np.where(g == 0)[0]))

# kmeans = np.load("Yelp/%s_500_kmeans.npy" % city)
# for i in range(500):
#     print(i, len(np.where(kmeans == i)[0]))


user_re = np.load("Yelp/%s_user_re.npy" % city)

similarities = []
a_user = 0.25

for i in range(1, 2558): # 817 2558
    ps = np.where(g == i)[0]
    # print(ps)
    len_ = len(ps)
    similarity = np.zeros((len_,len_))
    for j,m in enumerate(ps):
        for k,n in enumerate(ps):
            if(j<k):
                # if (np.sqrt(sim_matrix[m] ** 2 * sim_matrix[n] ** 2)).sum() == 0:
                #     print(np.sum(sim_matrix[m]*sim_matrix[n])/(np.sqrt((sim_matrix[m] ** 2).sum() * (sim_matrix[n] ** 2).sum())))
                    # print(sim_matrix[m], max(sim_matrix[m]),  max(sim_matrix[n]))
                    # print(sim_matrix[n], max(sim_matrix[n]))
                similarity[j][k] = np.sum(sim_matrix[m]*sim_matrix[n])/(np.sqrt((sim_matrix[m] ** 2).sum() * (sim_matrix[n] ** 2).sum()))
                # if similarity[j][k] == 0:
                #     print(sim_matrix[m].sum(), sim_matrix[n].sum(), np.sum(sim_matrix[m]*sim_matrix[n]))
                similarity[k][j] = similarity[j][k] = (a_user + user_re[m][n])*similarity[j][k]/(1+a_user)
                # print(similarity[j][k])

    # print(i, len(np.where(kmeans == i)[0]))

    similarities.append(similarity)
    print(i)
    # print(similarity.max())

# np.save("Yelp/%s_similarities.npy" % city, similarities,dtype=object)

import pickle
data_output = open("Yelp/%s_500_similarities.pkl" % city, 'wb')
pickle.dump(similarities, data_output)
data_output.close()

