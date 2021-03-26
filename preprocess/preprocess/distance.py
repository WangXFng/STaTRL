
import numpy as np
import pymysql
import database.Constants as Constants

from math import radians, cos, sin, asin, sqrt

#公式计算两点间距离（m）

def geodistance(lat1, lng1,lat2, lng2):
    #lng1,lat1,lng2,lat2 = (120.12802999999997,30.28708,115.86572000000001,28.7427)
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)]) # 经纬度转换成弧度
    dlon=lng2-lng1
    dlat=lat2-lat1
    a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    distance=2*asin(sqrt(a))*6371*1000 # 地球平均半径，6371km
    distance=round(distance/1000,3)
    return distance

# 返回 446.721 千米


count = 0
# 打开数据库连接
db = pymysql.connect(user='root', password='root', host='localhost', database='business')

# 使用cursor()方法获取操作游标
cursor = db.cursor()

# ingoing outgoing
ingoing = np.zeros(Constants.POI_NUM)
outgoing = np.zeros(Constants.POI_NUM)

city = Constants.NOW_CITY

# SQL 查询语句
sql = "SELECT * FROM business where city = '%s' " % city
print(sql)

lats = []
lngs = []
try:
    # 执行SQL语句
    cursor.execute(sql)
    # 获取所有记录列表
    results = cursor.fetchall()
    for count,row in enumerate(results):
        lats.append(row[2])
        lngs.append(row[3])
        # print(l)
        # if count == 1:
        #     break

except:
    print("Error: unable to fetch data")
# 关闭数据库连接
db.close()

disc = np.zeros((Constants.POI_NUM,Constants.POI_NUM))
for i in range(Constants.POI_NUM):
    for j in range(Constants.POI_NUM):
        if i!= j:
            disc[i][j] = geodistance(lats[i],lngs[i], lats[j],lngs[j])
            disc[i][j] = disc[i][j] if disc[i][j] < 10 else 10
            disc[j][i] = disc[i][j]
    if i % 100 == 0:
        print(i)
        # break

print(disc)
print(disc.max())

np.save("Yelp/%s_disc.npy" % city, disc / disc.max())
disc = np.load("Yelp/%s_disc.npy" % city)
print(disc)
print(disc.max())
# print(geodesic((30.28708,120.12802999999997), (28.7427,115.86572000000001)).m) #计算两个坐标直线距离
# print(geodesic((30.28708,120.12802999999997), (28.7427,115.86572000000001)).km) #计算两个坐标直线距离
