import numpy as np
import time
import json
import datetime
import torch

np.set_printoptions(precision=4)  # 设置小数位置为3位

model = torch.load('./model/0.86858.pth.tar')
from nltk import sent_tokenize

with open('./dataset/dataset/categories.json','r') as load_f:
   load_dict = json.load(load_f)
   # print(load_dict)

top_parent = {}
for i,ld in enumerate(load_dict):
    if len(ld['parents']) == 0:
       top_parent[ld['title']] = ld['alias']
    else:
       top_parent[ld['title']] = ld['parents'][0]

# print(top_parent, len(top_parent))
# hotelstravel {'Food_neg': 0, 'Food_pos': 1, 'Price_neg': 2, 'Price_pos': 3, 'Service_neg': 4, 'Service_pos': 5 }
# nightlife
# food
# active {'Price_neg': 2, 'Price_pos': 3, 'Service_neg': 4, 'Service_pos': 5 }
# arts
# auto
# shopping
# professional
# physicians
# pets
# health
# fitness
# education
# beautysvc

type = ['hotelstravel', 'nightlife', 'active', 'arts', 'auto', 'food', 'shopping',
        'professional', 'physicians', 'pets', 'health', 'fitness', 'education', 'beautysvc']


# path = '/home/g19tka20/Downloads/full_Yelp/yelp_dataset/11_0_4/'

def string2timestamp(strValue):
    try:
        d = datetime.datetime.strptime(strValue, "%Y-%m-%d %H:%M:%S")  # 2515-12-05 03:18:11
        t = d.timetuple()
        timeStamp = int(time.mktime(t))
        timeStamp = float(str(timeStamp) + str("%06d" % d.microsecond)) / 1000000
        return int(timeStamp)
    except ValueError as e:
        print(e)


users = np.load('./dataset/dataset/yelp_user_level_4_user_id.npy')
businesses = np.load('./dataset/dataset/yelp_business_level_4_business_id.npy')

us,bs = {},{}
for i,u in enumerate(users):
    us[u] = i
for i,b in enumerate(businesses):
    bs[b] = i

actions = [[] for i in range(len(users))]
time_ = [[] for i in range(len(users))]
score = [[] for i in range(len(users))]
aspect = [[] for i in range(len(users))]


def processText(model, text):

    # print(text)
    text = text.replace("\n", " ")
    if len(text) > 1000:
        text = text[0:1000]

    aspect_res, logist = model(text)
    # print(text)
    aspect_res = torch.sigmoid(aspect_res)[0]
    logist = torch.sigmoid(logist)[0]
    # print(aspect_res, logist)

    #

    # for i in range(3):
    #
    #     if aspect_res[i]>0.5:
    #         if logist[i * 2]>logist[i * 2+1]:
    #             logist[i * 2] = 1
    #             logist[i * 2 + 1] = 0
    #         else:
    #             logist[i * 2 + 1] = 1
    #             logist[i * 2] = 0
    #     else:
    #         logist[i * 2] = logist[i * 2 + 1] = 0

    # aspect_res = aspect_res.detach().cpu().numpy().tolist()
    label = logist.detach().cpu().numpy()
    # print(label)
    # print(aspect_res, label.tolist())
    # print()

    return label.tolist()


count = 0
f = open('./dataset/dataset/final_review.json', 'r')  # 3433618  # 1733082 # 1624688
line = f.readline()
t1 = time.time()
while line:
    count += 1

    j = json.loads(line)

    user_id = us[j['user_id']]
    business_id = bs[j['business_id']]
    if not actions[user_id].__contains__(business_id):
        actions[user_id].append(business_id)
        time_[user_id].append(string2timestamp(j['date']))
        score[user_id].append(int(float(j['stars'])))
        try:
            aspect[user_id].append(processText(model, j['text']))
        except Exception as e:
            print(e.args)
            aspect[user_id].append([0, 0, 0, 0, 0, 0])
            print(j['review_id'])


    else:
        # print(score[user_id][actions[user_id].index(business_id)], int(float(j['stars'])))
        score[user_id][actions[user_id].index(business_id)] = int(float(j['stars']))
        # print(score[user_id][actions[user_id].index(business_id)])
        # print('========================')

    if count % 1000 == 0:
        print(count, time.time()-t1)
        # break

    line = f.readline()

print(count)
# print(aspect)

import os
#  # ========= Yelp_reviews.txt
if not os.path.exists('./dataset/dataset2/'):
    os.mkdir('./dataset/dataset2/')
f = open('dataset/dataset/Yelp_reviews_test.txt', 'w')
for i,(a,t,s,asp) in enumerate(zip(actions, time_, score, aspect)):
    a1_n, t1_n, s1_n, asp_n = np.array(a), np.array(t), np.array(s), np.array(asp)
    t1_index = np.argsort(t1_n)
    a, t, s, asp = a1_n[t1_index], t1_n[t1_index], s1_n[t1_index], asp_n[t1_index]
    for j,(a1,t1,s1,asp1) in enumerate(zip(a,t,s,asp)):
        f.write(str(i)+'\t'+str(a1)+'\t'+str(s1)+'\t'+str(t1)+'\t'+str(asp1)+'\n')
f.close()

# #  # ========= Yelp_checkins.txt Yelp_train.txt Yelp_tune.txt Yelp_test.txt
# checkins_f = open(path + 'dataset/Yelp_checkins.txt', 'w')
# train_f = open(path + 'dataset/Yelp_train.txt', 'w')
# tune_f = open(path + 'dataset/Yelp_tune.txt', 'w')
# test_f = open(path + 'dataset/Yelp_test.txt', 'w')
# for i,(a,t,s) in enumerate(zip(actions, time_, score)):
#     a1_n, t1_n, s1_n = np.array(a), np.array(t), np.array(s)
#     t1_index = np.argsort(t1_n)
#     a, t, s = a1_n[t1_index], t1_n[t1_index], s1_n[t1_index]
#     for j, (a1, t1, s1) in enumerate(zip(a, t, s)):
#         checkins_f.write(str(i)+'\t'+str(a1)+'\t'+str(t1)+'\n')
#
#         if j<int(len(a)*0.7):
#             train_f.write(str(i)+'\t'+str(a1)+'\t'+str(s1)+'\n')
#         elif j>=int(len(a)*0.7) and j<int(len(a)*0.8):
#             tune_f.write(str(i)+'\t'+str(a1)+'\t'+str(s1)+'\n')
#         else:
#             test_f.write(str(i)+'\t'+str(a1)+'\t'+str(s1)+'\n')
#
# checkins_f.close()
# train_f.close()
# tune_f.close()
# test_f.close()
# =========
#

