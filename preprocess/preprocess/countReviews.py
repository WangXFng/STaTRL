import numpy as np
import json

import pymysql


# def getIds(city):
#     # 打开数据库连接
#     db = pymysql.connect(user='root', password='root', host='localhost', database='business')
#
#     # 使用cursor()方法获取操作游标
#     cursor = db.cursor()
#
#     # SQL 查询语句
#     sql = "SELECT * FROM business \
#            WHERE city = '%s'" % city
#     print(sql)
#
#     ids = []
#     try:
#         # 执行SQL语句
#         cursor.execute(sql)
#         # 获取所有记录列表
#         results = cursor.fetchall()
#         for row in results:
#             ids.append(row[1])
#     except:
#         print("Error: unable to fetch data")
#     # 关闭数据库连接
#     db.close()
#
#     return ids
#
#
# city = 'Charlotte'  # Champaign # Charlotte
#
# Charlotte = getIds('Charlotte')
# Champaign = getIds('Champaign')
# Las_Vegas = getIds('Las Vegas')
# Toronto = getIds('Toronto')
#
# f = open('/home/g19tka20/Downloads/full Yelp/yelp_dataset/yelp_academic_dataset_review.json', 'r')
# line = f.readline()
# counts = np.zeros(4)  # 0 Las Vegas 1 Toronto 2 Charlotte 3 Champaign 4 Old Yelp
# count = 0
# while line:
#     count += 1
#     j = json.loads(line)
#     if Las_Vegas.__contains__(j['business_id']):
#         counts[0] += 1
#     elif Toronto.__contains__(j['business_id']):
#         counts[1] += 1
#     elif Charlotte.__contains__(j['business_id']):
#         counts[2] += 1
#     elif Champaign.__contains__(j['business_id']):
#         counts[3] += 1
#     if count % 10000 == 0:
#         print(count)
#     line = f.readline()
#
# print(count)
# print(counts)  # [2446350.  600614.  385576.   34846.]

# def getIds(city):
#     # 打开数据库连接
#     db = pymysql.connect(user='root', password='root', host='localhost', database='business')
#
#     # 使用cursor()方法获取操作游标
#     cursor = db.cursor()
#
#     # SQL 查询语句
#     sql = "SELECT * FROM business_Charlotte "
#     print(sql)
#
#     ids = []
#     try:
#         # 执行SQL语句
#         cursor.execute(sql)
#         # 获取所有记录列表
#         results = cursor.fetchall()
#         for row in results:
#             ids.append(row[1])
#     except:
#         print("Error: unable to fetch data")
#     # 关闭数据库连接
#     db.close()
#
#     return ids
#
# Charlotte = getIds('Charlotte')
# counts = np.zeros(10429)
#
# f = open('/home/g19tka20/Downloads/full Yelp/yelp_dataset/Charlotte_review.txt', 'r')
# line = f.readline()
# count = 0
# while line:
#     count += 1
#     j = json.loads(line)
#     index = Charlotte.index(j['business_id'])
#     counts[index] += 1
#     if count % 10000 == 0:
#         print(count)
#     line = f.readline()
# f.close()
#
# print(np.where(counts<10)[0])
# print(len(np.where(counts<10)[0]))
#
# np.save('/home/g19tka20/Downloads/full Yelp/yelp_dataset/business_Charlotte_useful.npy', counts)



# counts = np.load('/home/g19tka20/Downloads/full Yelp/yelp_dataset/business_Charlotte_useful.npy')
#
# db = pymysql.connect(user='root', password='root', host='localhost', database='business')
#
# # 使用cursor()方法获取操作游标
# cursor = db.cursor()
#
# w = np.where(counts<10)[0]
#
# w = w.tolist()
#
# print(w)
# # print(w)
# # s = ','.join(str(w))
# #
# # print(s)
#
# # SQL 查询语句
# sql = "update business_Charlotte set num_of_review='-1' where no in (%s)" % ','.join([str(i+1) for i in w])
# print(sql)
#
# ids = []
# try:
#     # 执行SQL语句
#     cursor.execute(sql, w)
#     print("1")
# except:
#     print("Error: unable to fetch data")
# # 关闭数据库连接
# db.close()


