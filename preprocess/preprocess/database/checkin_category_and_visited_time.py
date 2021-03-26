import json
import re
import pickle
import numpy as np
import os

# import transformer.Constants as Constants
TYPE_DIMENSION = 624

# !/usr/bin/env python
# -*-encoding:UTF-8 -*-
print(os.getcwd())
print(os.path.abspath(os.path.dirname(__file__)))


f = open(os.getcwd() + '/../data/Yelp/Yelp_check_ins.txt', 'r')
line = f.readline()
min_time = 999999999999999999.0
while line:
    line = line.split("\n")[0]
    data = line.split("\t")

    if float(data[2])<min_time:
        min_time = float(data[2])

    line = f.readline()

print(min_time)

f = open(os.getcwd() + '/../data/Yelp/Yelp_poi_categories.txt', 'r')
line = f.readline()
category_index = 0
categories = []
category = []
while line:
    line = line.split("\n")[0]
    data = line.split("\t")

    if category_index != int(data[0]) or category_index == 18994:
        categories.append(category)
        category_index += 1
        category = []

    category.append(int(data[1]))

    line = f.readline()
    # if category_index == 2:
    #     print(categories)
f.close()


# print(len(categories))
print('Generate category index successfully ！')

f = open(os.getcwd() + '/../data/Yelp/Yelp_check_ins.txt', 'r')
whole_data = []

line = f.readline()
time_s = 0.0
count = 0
user_index = 0
arrived_poi = []
arrived_poi_time = []
others = []

type_length = []
type_visited_again = []

while line:
    line = line.split("\n")[0]
    data = line.split("\t")
    # print(user_index,int(data[0] ))

    whole_data.append(data[0]+"\t"+data[1]+"\t"+str(round((float(data[2])-min_time)/3600000.0+time_s,2))+"\t"+str(    [int(data[1])]))

    if user_index != int(data[0]):
        others.append('arrived_poi_type               ' + str(arrived_poi))
        others.append('arrived_type_visited_times ' + str(arrived_poi_time))
        user_index += 1
        category = np.zeros(TYPE_DIMENSION)
        time_s = 0
        type_length.append(arrived_poi.__len__())
        type_visited_again.append(np.array(arrived_poi_time).sum()-arrived_poi.__len__())
        arrived_poi = []
        arrived_poi_time = []

    for i in categories[int(data[1])]:

        if not arrived_poi.__contains__(i):
            arrived_poi.append(i)
            arrived_poi_time.append(1)
        else:
            arrived_poi_time[arrived_poi.index(i)] += 1

    time_s += 1.0

    line = f.readline()
    # if user_index == 1000:
    #     break
    count += 1
    # if count == 100:
    #     # print(whole_data)
    #     break

f.close()

print(len(whole_data))

# f = open(os.getcwd() + '/../data/Yelp/Yelp_check_ins_cate.txt', 'w')
# for s in whole_data:
#     f.write(s+"\r\n")
# f.close()
#
# f = open(os.getcwd() + '/../data/Yelp/Yelp_check_ins_cate_visited.txt', 'w')
# for s in others:
#     f.write(s+"\r\n")
# f.close()
#
# f = open(os.getcwd() + '/../data/Yelp/Yelp_check_ins_cate_visited_avg_count.txt', 'w')
# f.write(str(np.array(type_length).sum() / len(type_length))+" "+ str(np.array(type_visited_again).sum() / len(type_visited_again)) + "\r\n")
# f.close()

print(str(np.array(type_length).sum() / len(type_length))+" "+ str(np.array(type_visited_again).sum() / len(type_visited_again)))

# print(len(error))

