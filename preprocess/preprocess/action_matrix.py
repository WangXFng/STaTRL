#
# import numpy as np
# import pymysql
# import ast
#
# count = 0
# # 打开数据库连接
# db = pymysql.connect(user='root', password='root', host='localhost', database='business')
#
# # 使用cursor()方法获取操作游标
# cursor = db.cursor()
#
# # las_vegas 31675  # toronto 20370 # champaign 1327 # charlotte 10429
# POI_NUM = 10429
#
# import database.Constants as Constants
# city = Constants.NOW_CITY # Champaign # Charlotte
# # SQL 查询语句
# sql = "SELECT * FROM user_action_%s " % city
# print(sql)
#
# action_matrices = []
# in_out = []
#
# ingoing = np.load("Yelp/%s_ingoing.npy" % city)
# outgoing = np.load("Yelp/%s_outgoing.npy" % city)
# disc = np.load("Yelp/%s_disc.npy" % city)
#
# try:
#     # 执行SQL语句
#     cursor.execute(sql)
#     # 获取所有记录列表
#     results = cursor.fetchall()
#     for count,row in enumerate(results):
#         business_index = ast.literal_eval(row[1])
#         matrix = np.zeros((len(business_index),len(business_index)))
#         in_ = []
#         out_ = []
#         for i,a in enumerate(business_index):
#             for j,b in enumerate(business_index):
#                 if i != j :
#                     matrix[i][j] = matrix[j][i] = disc[a][b]
#             in_.append(ingoing[a])
#             out_.append(outgoing[a])
#
#         action_matrices.append(matrix)
#
#         in_ = np.array(in_)
#         out_ = np.array(out_)
#
#         a = np.reshape(in_, (1, -1))
#         b = np.reshape(out_, (1, -1))
#
#         in_out.append(a.T * b)
#
#         # print('matrix', len(matrix))
#         # print(matrix)
#         # print('a.T * b', (a.T * b).shape)
#         # print('a', type(a))
#         # print('b', type(b))
#         # print(a.T * b)
#         # print(l)
#         # if count == 1:
#         #     break
#
# except:
#     print("Error: unable to fetch data")
# # 关闭数据库连接
# db.close()
#
#
# print(disc)
# print(disc.max())
# # print(geodesic((30.28708,120.12802999999997), (28.7427,115.86572000000001)).m) #计算两个坐标直线距离
# # print(geodesic((30.28708,120.12802999999997), (28.7427,115.86572000000001)).km) #计算两个坐标直线距离
