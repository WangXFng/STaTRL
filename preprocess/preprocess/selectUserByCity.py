import json
import re
import datetime
import time

# !/usr/bin/env python
# -*-encoding:UTF-8 -*-
import sys
# import chardet
#
# f = open('/home/g19tka20/Downloads/yelp_dataset/yelp_academic_dataset_business.json','rb')
# data = f.read()
# print(chardet.detect(data))


import pymysql

city = 'Charlotte'  # Las Vegas  # Champaign  # Toronto

count = 0
# 打开数据库连接
db = pymysql.connect(user='root', password='root', host='localhost', database='business')

# 使用cursor()方法获取操作游标
cursor = db.cursor()

# SQL 查询语句
# sql = "SELECT * FROM business \
#        WHERE city = '%s'" % city

sql = "SELECT * FROM business_Charlotte"
print(sql)

ids = []
location = []
try:
    # 执行SQL语句
    cursor.execute(sql)
    # 获取所有记录列表
    results = cursor.fetchall()
    for row in results:
        ids.append(row[1])
        location.append([row[2], row[3]])
        # fname = row[0]
        # id = row[1]
        # lat = row[2]
        # lng = row[3]
        # 打印结果
        # print("fname=%s,id=%s,lat=%s,lng=%s" % \
        #       (fname, id, lat, lng))
except:
    print("Error: unable to fetch data")
# 关闭数据库连接
db.close()


def getTime(str):
    d = datetime.datetime.strptime(str, "%Y-%m-%d %H:%M:%S")
    t = d.timetuple()
    return int(time.mktime(t))


user_ids = []
user_business_id_actions = []
user_location_actions = []
user_scores_actions = []
action_time = []
f = open('/home/g19tka20/Downloads/full Yelp/yelp_dataset/%s_review_useful.txt' % city, 'r')  # 385576  # 360632
line = f.readline()
while line:
    count += 1
    j = json.loads(line)

    if count%10000 == 0:
        print(count)
        # break

    if not user_ids.__contains__(j['user_id']):
        user_ids.append(j['user_id'])
        # print(j['business_id'])
        user_business_id_actions.append([ids.index(j['business_id'])])
        user_location_actions.append([location[ids.index(j['business_id'])]])
        action_time.append([getTime(j['date'])])
        user_scores_actions.append([j['stars']])
    else:
        # print(user_ids.index(j['user_id']))
        # print(len(user_ids))
        # print(ids.index(j['business_id']))
        # print(len(ids))
        user_business_id_actions[user_ids.index(j['user_id'])].append(ids.index(j['business_id']))
        user_location_actions[user_ids.index(j['user_id'])].append(location[ids.index(j['business_id'])])
        action_time[user_ids.index(j['user_id'])].append(getTime(j['date']))
        user_scores_actions[user_ids.index(j['user_id'])].append(j['stars'])
        # print(json.dumps(user_scores_actions[user_ids.index(j['user_id'])]))
    # city = j['city']
    # print(city)
    # if count%100 == 0:
    #     # print(count)
    #     # print(user_business_id_actions)
    #     # print(user_location_actions)
    #     # print(action_time)
    #     break

    line = f.readline()

print(count)
print("start to insert data !")

db = pymysql.connect(user='root', password='root', host='localhost', database='business')

sql = "INSERT INTO user_action_"+city+" (id, user_id, \
           business_index, location, timestamp, scores) \
           VALUES (%s,%s,%s,%s,%s,%s)"

# print(sql)

count = 0
user_count = 0
for i,line in enumerate(user_ids):

    count += 1

    if count%10000 == 0:
        print(count)

    # 使用 cursor() 方法创建一个游标对象 cursor
    cursor = db.cursor()

    # SQL 插入语句
    # 区别与单条插入数据，VALUES ('%s', '%s',  %s,  '%s', %s) 里面不用引号

    if len(user_location_actions[i])>=10:

        # print(json.dumps(user_scores_actions[i]))
        val = ((user_count, user_ids[i], json.dumps(user_business_id_actions[i]), json.dumps(user_location_actions[i]),
                json.dumps(action_time[i]), json.dumps(user_scores_actions[i])),)
        #        ('Bruse', 'Jerry', 30, 'F', 3000),
        #        ('Lee', 'Tomcat', 40, 'M', 4000),
        #        ('zhang', 'san', 18, 'M', 1500))

        try:
            # 执行sql语句
            cursor.executemany(sql, val)
            # 提交到数据库执行
            db.commit()
        except pymysql.Error as e:
            print(e.args[0], e.args[1])
            db.rollback()
            
        user_count += 1

print(user_count)  # Champaign 485  # Las Vegas 36459  # toronto 9701 # Charlotte 5860 5485


