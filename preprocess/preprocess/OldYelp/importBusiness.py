import json
import re

# !/usr/bin/env python
# -*-encoding:UTF-8 -*-
import sys
print(sys.getdefaultencoding())
# import chardet
#
# f = open('/home/g19tka20/Downloads/yelp_dataset/yelp_academic_dataset_business.json','rb')
# data = f.read()
# print(chardet.detect(data))


import pymysql

values = ()
count = 0
error = []


def insertDB(values, count):
    # 打开数据库连接
    db = pymysql.connect(user='root', password='root', host='localhost', database='transformer')

    # 使用 cursor() 方法创建一个游标对象 cursor
    cursor = db.cursor()

    # SQL 插入语句
    sql = "INSERT INTO business(id, \
           name, latitude, longitude, city) \
           VALUES (%s,%s,%s,%s,%s)"
    # 区别与单条插入数据，VALUES ('%s', '%s',  %s,  '%s', %s) 里面不用引号

    # val = (('li', 'si', 16, 'F', 1000),
    #        ('Bruse', 'Jerry', 30, 'F', 3000),
    #        ('Lee', 'Tomcat', 40, 'M', 4000),
    #        ('zhang', 'san', 18, 'M', 1500))

    print('start to import: ' + str(count))

    try:
        # 执行sql语句
        cursor.executemany(sql, values)
        # 提交到数据库执行
        db.commit()
    except pymysql.Error as e:
        print(e.args[0], e.args[1])
        print()
        db.rollback()

    # 关闭数据库连接
    db.close()

    print('finished importing: ' + str(count))


db = pymysql.connect(user='root', password='root', host='localhost', database='transformer')

sql = "INSERT INTO business(id, \
           name, latitude, longitude, city) \
           VALUES (%s,%s,%s,%s,%s)"

f = open('/home/g19tka20/Downloads/yelp_dataset/yelp_academic_dataset_business.json', 'r')
line = f.readline()
while line:
    count += 1
    j = json.loads(line)
    # city = j['city']
    # print(city)
    # print(city.encode('iso-8859-1'))
    # record = ((j['business_id'], j['name'], j['latitude'], j['longitude'], j['city']),)
    # values += record


    if count%10000 == 0:
        print(count)

    # 使用 cursor() 方法创建一个游标对象 cursor
    cursor = db.cursor()

    # SQL 插入语句
    # 区别与单条插入数据，VALUES ('%s', '%s',  %s,  '%s', %s) 里面不用引号

    val = ((j['business_id'], j['name'], j['latitude'], j['longitude'], j['city']), )
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
        error.append(line)

    line = f.readline()

# 关闭数据库连接
db.close()

f = open('/home/g19tka20/Downloads/yelp_dataset/taxi_review_without_hack.txt','w')
for er in error:
    f.write(er+"\r\n")

print(len(error))

