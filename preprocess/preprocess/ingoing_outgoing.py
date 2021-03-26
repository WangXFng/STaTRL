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
# 打开数据库连接
db = pymysql.connect(user='root', password='root', host='localhost', database='business')

# 使用cursor()方法获取操作游标
cursor = db.cursor()

# ingoing outgoing
ingoing = np.zeros(Constants.POI_NUM)
outgoing = np.zeros(Constants.POI_NUM)

# SQL 查询语句
sql = "SELECT * FROM user_action_%s" % city
print(sql)

ids = []
try:
    # 执行SQL语句
    cursor.execute(sql)
    # 获取所有记录列表
    results = cursor.fetchall()
    for count,row in enumerate(results):
        # ids.append(row[1])
        l = ast.literal_eval(row[1])  # business_index
        loca = ast.literal_eval(row[2])  # location
        for i,ll in enumerate(l):
            ingoing[ll] += i
            outgoing[ll] += len(l) - 1 - i
        # print(l)
        # if count == 1:
        #     break

except:
    print("Error: unable to fetch data")
# 关闭数据库连接
db.close()

# print('ingoing.max()',ingoing.max())
# print(ingoing)
#
# print('outgoing.max()',outgoing.max())
# print(outgoing)

np.save("Yelp/%s_ingoing.npy" % city, ingoing / ingoing.max())
np.save("Yelp/%s_outgoing.npy" % city, outgoing / outgoing.max())

ingoing = np.load("Yelp/%s_ingoing.npy" % city)
outgoing = np.load("Yelp/%s_outgoing.npy" % city)

print('ingoing.max()',ingoing.max())
print(ingoing)

print('outgoing.max()',outgoing.max())
print(outgoing)

