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


f = open(os.getcwd() + '/../data/Yelp/Yelp_poi_categories.txt', 'r')
line = f.readline()
counts = np.zeros(20)
index = 0
count = 0
while line:
    line = line.split("\n")[0]
    data = line.split("\t")

    if int(data[0]) != index:
        counts[count] += 1
        count = 0
        index += 1

    count += 1

    line = f.readline()

print(counts)

# [0  1    2    3    4    5   6    7  8 9]
# [0 94 7872 4542 3520 2038 803 1060 17 2 ]

