import json
import re

# !/usr/bin/env python
# -*-encoding:UTF-8 -*-

import pymysql

city = 'Charlotte'  # Champaign # Charlotte

# Toronto Number of parameters: 43259282

count = 0
# 打开数据库连接
db = pymysql.connect(user='root', password='root', host='localhost', database='business')

# 使用cursor()方法获取操作游标
cursor = db.cursor()

# SQL 查询语句
# sql = "SELECT * FROM business \
#        WHERE city = '%s'" % city
# print(sql)

sql = "SELECT * FROM business_Charlotte"
print(sql)
ids = []
try:
    # 执行SQL语句
    cursor.execute(sql)
    # 获取所有记录列表
    results = cursor.fetchall()
    for row in results:
        ids.append(row[1])
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

print('finished importing: ' + str(count))


def addContent(whole_data):
    # f = open('/home/g19tka20/Downloads/full Yelp/yelp_dataset/%s_review.txt' % city,'a')
    f = open('/home/g19tka20/Downloads/full Yelp/yelp_dataset/%s_review_useful.txt' % city,'a')
    for er in whole_data:
        f.write(er)
    f.close()
    print(len(whole_data))


whole_data = []
f = open('/home/g19tka20/Downloads/full Yelp/yelp_dataset/%s_review.txt' % city, 'r')  # 8021122
# f = open('/home/g19tka20/Downloads/full Yelp/yelp_dataset/yelp_academic_dataset_review.json', 'r')  # 8021122
line = f.readline()
while line:
    count += 1
    j = json.loads(line)

    if ids.__contains__(j['business_id']):
        whole_data.append(line)
    # city = j['city']
    # print(city)
    if count%100000 == 0:
        # print(count)
        # break
        addContent(whole_data)
        whole_data = []

        print(count)

    line = f.readline()

addContent(whole_data)
print(count)




