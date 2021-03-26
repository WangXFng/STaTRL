import json
import re

# !/usr/bin/env python
# -*-encoding:UTF-8 -*-
import pymysql
import numpy as np

city = 'Charlotte'  # Champaign # Charlotte

# Toronto Number of parameters: 43259282

count = 0
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

user_re = np.zeros((5860, 5860))
whole_data = []
count = 0
l = open('/home/g19tka20/Downloads/full Yelp/yelp_dataset/Charlotte_users.txt', 'r')
line = l.readline()
while line:
    count += 1
    j = json.loads(line)
    u = user_dic[j['user_id']]

    # if user_group[u] == 0:
    #     user_group[u] = group_count
    #     group_count += 1

    for friend in j['friends'].split(", "):
        # print(friend)
        # print(user_dic.__contains__(friend))

        if user_dic.__contains__(friend):
            f = user_dic[friend]
            user_re[u][f] = user_re[f][u] = 1
            # user_group[f] = user_group[u]

    if count%10000 == 0:
        # print(count)
        # break
        whole_data = []

    line = l.readline()


def DFS(i, re, ug, group_count, stack):
    if ug[i] != 0:
        return ug, group_count, stack
    else:
        group_count += 1
    ug[i] = group_count
    stack.append(i)

    for j, r in enumerate(re[i]):
        if r == 1 :
            if j not in stack:
                if ug[j] == 0:
                    ug[j] = ug[i]
                    ug, group_count, stack = DFS(j, re, ug, group_count, stack)

            else:
                index = stack.index(j)
                print('Circle: ')
                for ii in stack[index:]:
                    print(ii, "->")
                print(j)
    stack.pop(-1)
    return ug, group_count, stack


stack = []
group_count = 1
user_group = np.zeros(5860)
for i in range(5860):
    user_group, group_count, stack = DFS(i, user_re, user_group, group_count, stack)

print(count)
print('group_count', group_count)  # 4194

for i in range(group_count):
    print(user_group == i)


np.save("Yelp/%s_user_group.npy" % city, user_group)
print(len(whole_data))

