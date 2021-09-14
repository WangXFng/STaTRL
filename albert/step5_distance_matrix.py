import time
import numpy as np
import json

from math import radians, cos, sin, asin, sqrt
import math

# "latitude":35.4627242,"longitude":-80.8526119

users = np.load('/home/g19tka20/Downloads/full_Yelp/yelp_dataset/11_0_4/yelp_user_level_4_user_id.npy')
businesses = np.load('/home/g19tka20/Downloads/full_Yelp/yelp_dataset/11_0_4/yelp_business_level_4_business_id.npy')
# print([ businesses[i] for i in range(10)])

count = 0
f = open('/home/g19tka20/Downloads/full_Yelp/yelp_dataset/11_0_4/final_business.json', 'r')  # 3433618
line = f.readline()

t1 = time.time()
w = open('/home/g19tka20/Downloads/full_Yelp/yelp_dataset/11_0_4/dataset/Yelp_poi_coos.txt', 'w')
while line:
    j = json.loads(line)

    w.write(str(count+1)+'\t'+str(j['latitude'])+'\t'+str(j['longitude'])+'\n')

    if count % 100000 == 0:
        # print(whole_data)
        print(time.time()-t1)
        # break

    count += 1
    line = f.readline()

print(count)
w.close()

# def geodistance(lat1, lng1,lat2, lng2):
#     #lng1,lat1,lng2,lat2 = (125.12802999999997,30.28708,115.86572500000001,28.7427)
#     lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)]) # 经纬度转换成弧度
#     dlon=lng2-lng1
#     dlat=lat2-lat1
#     a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
#     distance=2*asin(sqrt(a))*6371*1000 # 地球平均半径，6371km
#     distance=round(distance/1000,3)
#     return distance
#
# count = 0
# lats = []
# lngs = []
# # f = open('/home/g19tka20/Downloads/full_Yelp/yelp_dataset/11_0_4/dataset/Yelp_poi_coos.txt', 'r')
# f = open('../Gowalla/Gowalla_poi_coos.txt', 'r')
# line = f.readline()
# while line:
#     count += 1
#     j = line.split("\t")
#     lats.append(float(j[1]))
#     lngs.append(float(j[2]))
#     line = f.readline()
#
# f.close()
# # G 18737	32510
# t1 = time.time()
# disc = np.zeros((32510, 32510))
# for i in range(32510):
#     for j in range(32510):
#         if i<j:
#             if abs(lats[i]-lats[j])>1 or abs(lngs[i]-lngs[j])>1:
#                 disc[i][j] = disc[j][i] = 999
#             else:
#                 disc[i][j] = disc[j][i] = geodistance(lats[i],lngs[i], lats[j],lngs[j])
#     if i % 100 == 0:
#         print(i, time.time()-t1)
#
# np.save('../Gowalla/disc.npy', disc)