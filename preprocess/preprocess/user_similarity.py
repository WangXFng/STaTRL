import json
import re

# !/usr/bin/env python
# -*-encoding:UTF-8 -*-

import pymysql
import numpy as np
import ast

import database.Constants as Constants
from geopy.distance import geodesic

import ast
# Toronto Number of parameters: 43259282

count = 0

city = 'Charlotte'  # Champaign # Charlotte
#
# # 打开数据库连接
# db = pymysql.connect(user='root', password='root', host='localhost', database='business')
#
# # 使用cursor()方法获取操作游标
# cursor = db.cursor()
#
# # SQL 查询语句
# sql = "SELECT * FROM user_action_%s" % city
#
# print(sql)
#
# user_dic = {}
# similarity = []
# poi_num = 5485
# try:
#     # 执行SQL语句
#     cursor.execute(sql)
#     # 获取所有记录列表
#     results = cursor.fetchall()
#     for row in results:
#         user_dic[row[4]] = row[0]
#         indexes = ast.literal_eval(row[1])  # business_index
#         z = np.zeros(poi_num)
#         z[indexes] = 1
#         similarity.append(z)
# except:
#     print("Error: unable to fetch data")
# # 关闭数据库连接
# db.close()
#
# print(count)
# print('Reading data finished !!')
# print('=====================')
#
#
# np.save("Yelp/%s_similarity_matrix.npy" % city, similarity)
# similarity = np.load("Yelp/%s_similarity_matrix.npy" % city)
#
# from sklearn.cluster import KMeans
#
# type_numer = [50]
# print('start !!')
# for i in type_numer:
#     kmeans = KMeans(n_clusters=i, random_state=0).fit(similarity)
#
#     np.save("Yelp/%s_%d_kmeans.npy" % (city,i), kmeans.labels_)
#     print(kmeans.labels_)
#
# print('end !!')

# from sklearn.cluster import KMeans
#
# X = np.array([[1, 2], [1, 4], [1, 0],
#               [4, 2], [4, 4], [4, 0]])
#
# kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
#
# print(kmeans.labels_)

# type_numer = [5, 10, 20, 50, 100]
# print('start !!')
# for i in type_numer:
#

kmeans = np.load("Yelp/%s_50_kmeans.npy" % city)
for i in range(50):
    print(i, len(np.where(kmeans == i)[0]))


# kmeans = np.load("Yelp/%s_100_kmeans.npy" % city)
# for i in range(100):
#     print(i, len(np.where(kmeans == i)[0]))
#
# kmeans = np.load("Yelp/%s_300_kmeans.npy" % city)
# for i in range(300):
#     print(i, len(np.where(kmeans == i)[0]))
#
# kmeans = np.load("Yelp/%s_500_kmeans.npy" % city)
# for i in range(500):
#     print(i, len(np.where(kmeans == i)[0]))



