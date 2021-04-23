import numpy as np
import time
import json
import datetime
import torch
import pickle

np.set_printoptions(precision=4)  # 设置小数位置为3位

model = torch.load('./dataset/model/ 0.84827.pth.tar')
from nltk import sent_tokenize

with open('./dataset/dataset/categories.json','r') as load_f:
   load_dict = json.load(load_f)
   # print(load_dict)

type = ['hotelstravel', 'nightlife', 'food', 'active', 'arts', 'auto', 'shopping',
        'professional', 'physicians', 'pets', 'health', 'fitness', 'education', 'beautysvc']

with open('/home/g19tka20/Downloads/full_Yelp/yelp_dataset/11_0_4/business_id_2_top.pkl', 'rb') as f:
    business_id_2_top = pickle.load(f, encoding='latin-1')

# food neg 0 food pos 1 price nge 0 food pos 1 service neg 4 service pos 5

businesses_ = np.load('/home/g19tka20/Downloads/full_Yelp/yelp_dataset/11_0_4/yelp_business_level_4_business_id.npy')
bs = {}
for i,b in enumerate(businesses_):
    bs[b] = i


c = [ [[] for j in range(6)] for i in range(14)]
# print(c)
# for i in range(len(type)):
#     c[i] = {}
# # if i < 3:
#     c[i]['food-neg'] = []
#     c[i]['food-pos'] = []
#     c[i]['price-neg'] = []
#     c[i]['price-pos'] = []
#     c[i]['service-neg'] = []
#     c[i]['service-pos'] = []


def processText(model, text, c, top, is_full):

    # print(text)
    text = text.replace("\n", " ")
    if len(text) > 1000:
        text = text[0:1000]

    label = model(text)
    # print(text)
    label = torch.sigmoid(label)

    l = torch.argmax(label)
    if is_full[top][l] != 1:
        # list_ = label.detach().cpu().numpy().tolist()[0]

        c[top][l].append(str(len(c[top][l]) + 1) + " " + text)
        # if top<3:
        #     c[top][l].append(text+"\n"+"food(pos:{f_p: 8.5f} neg: {f_n: 8.5f}) price(pos:{p_p: 8.5f} "
        #                                "neg: {p_n: 8.5f}) service(pos:{s_p: 8.5f} neg:{s_n: 8.5f})".
        #                      format(f_p=list_[1],f_n=list_[0],p_p=list_[3],p_n=list_[2],s_p=list_[5],s_n=list_[4]))
        # else:
        #     c[top][l].append(text+"\n"+"price(pos:{p_p: 8.5f} "
        #                                "neg: {p_n: 8.5f}) service(pos:{s_p: 8.5f} neg:{s_n: 8.5f})".
        #                      format(p_p=list_[3],p_n=list_[2],s_p=list_[5],s_n=list_[4]))

        if len(c[top][l]) == 50:
            is_full[top][l] = 1


is_full = np.zeros((14, 6))
for i in range(3, 14):
    is_full[i][0:2] = 1


def w(c):
    for i in range(len(type)):
        f = open('./dataset/result/{}.txt'.format(type[i]), 'w')
        for j, cs in enumerate(c[i]):
            if j == 0:
                f.write(" ###### Food  negative ######\n")
            elif j == 1:
                f.write(" ###### Food  positive ######\n")
            elif j == 2:
                f.write(" ###### Price  negative ######\n")
            elif j == 3:
                f.write(" ###### Price  positive ######\n")
            elif j == 4:
                f.write(" ###### Service  negative ######\n")
            elif j == 5:
                f.write(" ###### Service  positive ######\n")

            # import random

            # random.shuffle(cs)

            for k in cs:
                f.write(k + "\n\n")

            f.write("\n")

        f.close()


count = 0
f = open('./dataset/dataset/final_review.json', 'r')  # 3433618  # 1733082 # 1624688
line = f.readline()
t1 = time.time()
while line:

    # if is_full.all() == 1 or count > 50000:
    #     break

    # if count > 150000:
    #     break

    count += 1

    j = json.loads(line)

    top = business_id_2_top[bs[j['business_id']]]
    try:
        processText(model, j['text'], c, top, is_full)
    except Exception as e:
        w(c)
        print(e.args)

    if count % 1000 == 0:
        print(count, time.time()-t1)
        # print(is_full)
        # print([ [len(j) for j in i] for i in c])
        # break

    line = f.readline()

print(count)
# print(aspect)

w(c)