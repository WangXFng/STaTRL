import json
import re
import pickle
import numpy as np
import os

# import transformer.Constants as Constants
# !/usr/bin/env python
# -*-encoding:UTF-8 -*-
print(os.getcwd())
print(os.path.abspath(os.path.dirname(__file__)))

f = open(os.getcwd() + '/../data/Yelp/Yelp_check_ins.txt', 'r')
whole_data = []

line = f.readline()
time_s = 0.0
count = 0
user_index = 0
arrived_poi = []
arrived_poi_length = []
visited_again = []
visited_again_count = []
while line:
    line = line.split("\n")[0]
    data = line.split("\t")
    # print(user_index,int(data[0] ))

    if user_index != int(data[0]):
        user_index += 1
        print(len(arrived_poi)+np.array(visited_again).sum(), np.array(visited_again).sum())
        arrived_poi_length.append(len(arrived_poi)+np.array(visited_again).sum())
        visited_again_count.append(np.array(visited_again).sum())
        arrived_poi = []
        visited_again = []

    if not arrived_poi.__contains__(data[1]):
        arrived_poi.append(data[1])
        visited_again.append(0)
    else:
        visited_again[arrived_poi.index(data[1])] += 1

    # print(data[1])
    # arrived_poi.append(data[1])

    # print(arrived_poi)

    line = f.readline()
    # if user_index == 1000:
    #     break
    count += 1
    # if count == 100:
    #     # print(whole_data)
    #     break

f.close()

print(len(whole_data))

f = open(os.getcwd() + '/../data/Yelp/Yelp_check_ins_visited_count.txt', 'w')
f.write("arrived_poi_length visited_again_count\r\n")
f.write(str(np.array(arrived_poi_length).sum()/len(arrived_poi_length))+" " + str(np.array(visited_again_count).sum()/len(visited_again_count)))
# for s in whole_data:
#     f.write(s+"\r\n")
f.close()


# print(len(error))

