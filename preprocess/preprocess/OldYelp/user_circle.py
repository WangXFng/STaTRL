import json
import re

# !/usr/bin/env python
# -*-encoding:UTF-8 -*-
import pymysql
import numpy as np

city = 'old_yelp'  # Champaign # Charlotte

# Toronto Number of parameters: 43259282

count = 0


user_re = np.zeros((30887, 30887))
whole_data = []
l = open('/home/g19tka20/PycharmProjects/Reviews/OldYelp/Yelp/Yelp_social_relations.txt', 'r')  # 265533
line = l.readline()
while line:
    count += 1
    j = line.split('\n')[0]
    j = j.split('\t')
    u = int(j[0])
    f = int(j[1])
    user_re[u][f] = user_re[f][u] = 1

    if count%10000 == 0:
        print(count)
        # break

    line = l.readline()

print(user_re)

import sys
sys.setrecursionlimit(100000) #例如这里设置为十万


def DFS(i, re, ug, group_count, stack, c):
    # if ug[i] != 0:
    #     return ug, group_count, stack

    stack.append(i)

    for j, r in enumerate(re[i]):
        if r == 1 :
            if j not in stack:
                if ug[j] == 0:
                    ug[j] = ug[i]
                    # print(j, i, ug[i])
                    c += 1
                    if c % 1000 == 0:
                        print(c)
                    ug, group_count, stack, c = DFS(j, re, ug, group_count, stack, c)

            else:
                group_count += 1
                # index = stack.index(j)
                # print('Circle: ', len(stack[index:])," " , stack[index:],  j)
                # for ii in stack[index:]:
                #     print(ii, "->")
                # print(j)
    stack.pop(-1)
    return ug, group_count, stack, c


stack = []
group_count = 0
user_group = np.zeros(30887)
c = 0

group1 = np.load("Yelp/%s_user_group.npy" % city)
print(group1)
print(np.where(group1 == 1))

# for i in range(30887):
    # print(i, "again", user_group[i])

for i in np.where(group1 == 1)[0]:
    # if user_group[i] == 0:
    #     group_count += 1
    #     user_group[i] = group_count
    c += 1
    if c % 1000 == 0:
        print(c)
    user_group, group_count, stack, c = DFS(i, user_re, user_group, group_count, stack, c)

print(count)
print('group_count', group_count)  # 8074


# user_group = np.load("Yelp/%s_user_group.npy" % city)

# for i in range(8074):
#     print(len(np.where(user_group == i)[0]), ": ",np.where(user_group == i))
#
# np.save("Yelp/%s_user_group.npy" % city, user_group)

# user_group = np.load("Yelp/%s_user_group.npy" % city)
#
# for i in range(8074):
#     print(user_group[user_group == i])