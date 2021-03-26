import json
import re
import pickle
import numpy as np
import os

import database.Constants as Constants
import pymysql
import ast

# !/usr/bin/env python
# -*-encoding:UTF-8 -*-
print(os.getcwd())
print(os.path.abspath(os.path.dirname(__file__)))

city = 'Charlotte'  # Champaign # Charlotte

# 打开数据库连接
db = pymysql.connect(user='root', password='root', host='localhost', database='business')

# 使用cursor()方法获取操作游标
cursor = db.cursor()

# SQL 查询语句
sql = "SELECT * FROM user_action_%s" % city

print(sql)

user_dic = {}
try:
    # 执行SQL语句
    cursor.execute(sql)
    # 获取所有记录列表
    results = cursor.fetchall()
    for row in results:
        user_dic[row[4]] = row[0]
except:
    print("Error: unable to fetch data")
# 关闭数据库连接
db.close()

min_time = 1098028800


count = 0
# 打开数据库连接
db = pymysql.Connect(
    host='localhost',
    user='root',
    passwd='root',
    db='business',
    charset='utf8'
)

# 使用cursor()方法获取操作游标
cursor = db.cursor()


# SQL 查询语句
sql = "SELECT * FROM user_action_%s" % city

print(sql)

# in_out = []

# ingoing = np.load("../Yelp/%s_ingoing.npy" % city)
# outgoing = np.load("../Yelp/%s_outgoing.npy" % city)
# disc = np.load("../Yelp/%s_disc.npy" % city)
# user_group = np.load("../Yelp/%s_user_group.npy" % city)

ids = []
train_data = []
test_data = []
try:
    # 执行SQL语句
    cursor.execute(sql)
    # 获取所有记录列表
    results = cursor.fetchall()
    count = 0
    for row in results:
        indexes = ast.literal_eval(row[1])  # business_index
        timestamps = ast.literal_eval(row[3])  # timestamp
        # print(type(indexes))
        # print(indexes)
        # print(type(timestamps))
        # print(timestamps)
        actions = []
        label = []
        len_ = len(indexes)
        
        tuning_len = int(len_*0.7)
        test_len = int(len_*0.9)

        # last = int(len_ * 0.8) + 1
        test_actions = []
        test_label = []

        for i,index in enumerate(indexes):
            if i < tuning_len:
                actions.append({'time_since_start': (float(timestamps[i]) - min_time) / 3600000 ,
                                     'time_since_last_event': float(indexes[i-1]) if i > 0 else 0,
                                     'type_event': int(index)})
                                   # ,'event_user_group': user_group[user_dic[row[4]]]})
                
                test_actions.append({'time_since_start': (float(timestamps[i]) - min_time) / 3600000 ,
                                     'time_since_last_event': float(indexes[i-1]) if i > 0 else 0,
                                     'type_event': int(index)})
            elif i < test_len:
                if not label.__contains__(int(index)):
                    label.append(int(index))
                test_actions.append({'time_since_start': (float(timestamps[i]) - min_time) / 3600000 ,
                                     'time_since_last_event': float(indexes[i-1]) if i > 0 else 0,
                                     'type_event': int(index)})
            else:
                if not test_label.__contains__(int(index)):
                    test_label.append(int(index))

        train_data.append({'actions': actions, 'label': label, 'group_': [], 'distance': [], 'track':[] })

        test_data.append({'actions': test_actions, 'label': test_label, 'group_': [], 'distance': [], 'track':[] })
        # 打印结果
        # print("fname=%s,id=%s,lat=%s,lng=%s" % \
        #       (fname, id, lat, lng))
        count += 1

        if count % 10000 == 0:
            print(count)
except:
    print("Error: unable to fetch data")
# 关闭数据库连接
db.close()


#
print(len(train_data))
# print(train_data)

# data = {'dim_process': Constants.TYPE_DIMENSION, 'train': train_data[:int(len(train_data)*0.8)]}

data = {'dim_process': Constants.POI_NUM, 'train': train_data,
            'num_groups': Constants.NUM_GROUP, 'key_group': [], 'value_group': [],
            'group':[], 'ingoing': [], 'outgoing': []}
# las_vegas 31675  # toronto 20370 # champaign 1327 # charlotte 10429

# print(data['train'].__len__())

data_output = open(os.getcwd() + '/../Yelp/train_%s.pkl'%city,'wb')
pickle.dump(data, data_output)
data_output.close()

# data_output = open(os.getcwd() + '/../data/Yelp/dev.pkl','wb')
# pickle.dump(train_data, data_output)
# data_output.close()

data = {'dim_process': Constants.POI_NUM, 'test': test_data,
            'num_groups': Constants.NUM_GROUP, 'key_group': [], 'value_group': [],
            'group':[], 'ingoing': [], 'outgoing': []}

# print(data['train'].__len__())

data_output = open(os.getcwd() + '/../Yelp/test_%s.pkl'%city,'wb')
pickle.dump(data, data_output)
data_output.close()
#
# error = []
# f = open(os.getcwd() + '/../data/Yelp/Yelp_check_ins_categories.txt', 'w')
# for er in error:
#     f.write(er+"\r\n")
#
# print(len(error))

