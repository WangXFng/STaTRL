import json

import numpy as np

import time

# #########
count = 0
f = open('./data/yelp_academic_dataset_user.json', 'r')
line = f.readline()
whole_data = []
while line:
    count += 1
    j = json.loads(line)
    if int(j['review_count'])>10:
        whole_data.append(j['user_id'])
    if count % 100000 == 0:
        print(count, len(whole_data))
    line = f.readline()

print(count)

np.save('./data/11_0_4/yelp_user_level_1.npy', whole_data)  # 20 : 406925 390930 325871
#
#
#
#########
count = 0
f = open('./data/yelp_academic_dataset_business.json', 'r')
line = f.readline()
whole_data = []
while line:
    count += 1
    j = json.loads(line)
    if int(j['review_count'])>10:
        whole_data.append(j['business_id'])
    if count % 100000 == 0:
        print(count, len(whole_data))
    line = f.readline()
print(count)
np.save('./data/11_0_4/yelp_business_level_1.npy', whole_data)  # 63098 60932 52140


users = np.load('./data/11_0_4/yelp_user_level_1.npy')
businesses = np.load('./data/11_0_4/yelp_business_level_1.npy')

us,bs = {},{}
for i,u in enumerate(users):
    us[u] = i
for i,b in enumerate(businesses):
    bs[b] = i


def addContent(whole_data):
    # f = open('/home/g19tka20/Downloads/full Yelp/yelp_dataset/%s_review.txt' % city,'a')
    f = open('./data/11_0_4/yelp_academic_dataset_review_level_1.json', 'a')  # 3882631 3783977 3361021
    for er in whole_data:
        f.write(er)
    f.close()


count = 0
stop_c = int(8021122*0.4)
f = open('./data/yelp_academic_dataset_review.json', 'r')  # 8021122 4010561
line = f.readline()
whole_data = []
w_count = 0
t1 = time.time()
while line:
    count += 1
    if count>stop_c:
        break
    j = json.loads(line)
    if j['user_id'] in us and j['business_id'] in bs:
        whole_data.append(line)
    if count % 10000 == 0:
        w_count += len(whole_data)
        print(count, len(whole_data), time.time()-t1)
        addContent(whole_data)
        whole_data = []
    line = f.readline()

print(count)
w_count += len(whole_data)
print(w_count)
addContent(whole_data)


users = np.load('./data/11_0_4/yelp_user_level_1.npy')  # 688832  # 406925
businesses = np.load('./data/11_0_4/yelp_business_level_1.npy')  # 103803  # 63098

us,bs = {},{}
for i,u in enumerate(users):
    us[u] = i
for i,b in enumerate(businesses):
    bs[b] = i

u = np.zeros(len(users))
b = np.zeros(len(businesses))

u_a = [[] for i in range(len(users))]
b_a = [[] for i in range(len(businesses))]

t1 = time.time()
count = 0
f = open('./data/11_0_4/yelp_academic_dataset_review_level_1.json', 'r')  # 3882631
line = f.readline()
while line:
    count += 1
    j = json.loads(line)

    # if users.__contains__(user_id):
    user_no = us[j['user_id']]
    business_no = bs[j['business_id']]
    if not u_a[user_no].__contains__(business_no):
        u[user_no] += 1
        b[business_no] += 1

        u_a[user_no].append(business_no)
        b_a[business_no].append(user_no)

    if count % 10000 == 0:
        print(count, time.time()-t1)
        # addContent(whole_data)
        # whole_data = []
        # print(whole_data)
        # break

    line = f.readline()

np.save('./data/11_0_4/yelp_user_level_3_user_id_count.npy', u)  # 131291 34079 18721
np.save('./data/11_0_4/yelp_business_level_3_business_id_count.npy', b)  # 67621 14728

np.save('./data/11_0_4/yelp_user_level_3_user_array.npy', u_a)  # 131291 34079
np.save('./data/11_0_4/yelp_business_level_3_business_array.npy', b_a)  # 67621

print(count)

flag = True
while flag:
    where_u = np.where(u <= 10)[0]
    flag_u = False
    for u1 in where_u:
        if len(u_a[u1]) > 0:
            # print(u1, u[u1], u_a[u1], b)
            flag_u = True

            for t_b in u_a[u1]:
                b[t_b] -= 1

            u_a[u1] = []
            # print(u1, u[u1], u_a[u1], b)
    u[where_u] = 0

    where_b = np.where(b <= 10)[0]
    flag_b = False
    for b1 in where_b:
        if len(b_a[b1]) > 0:
            flag_b = True

            for t_u in b_a[b1]:
                u[t_u] -= 1

            b_a[b1] = []

    b[where_b] = 0

    flag = flag_b or flag_u

np.save('./data/11_0_4/yelp_user_level_4_user_id_count.npy', u)
np.save('./data/11_0_4/yelp_business_level_4_business_id_count.npy', b)

users = users[u!=0]
businesses = businesses[b!=0]
print(len(users), len(businesses))

np.save('./data/11_0_4/yelp_user_level_4_user_id.npy', users)  # 119876  # 34079 # 30808 28038
np.save('./data/11_0_4/yelp_business_level_4_business_id.npy', businesses)  # 62796 # 24406 # 22459 15745

f = open('./data/11_0_4/dataset/Yelp_data_size.txt', 'w')  # 22459
f.write(str(len(users))+"\t"+str(len(businesses)))
f.close()


# ############## get final users and reviews
users = np.load('./data/11_0_4/yelp_user_level_4_user_id.npy')
businesses = np.load('./data/11_0_4/yelp_business_level_4_business_id.npy')

us,bs = {},{}
for i,u in enumerate(users):
    us[u] = i
for i,b in enumerate(businesses):
    bs[b] = i


def addUser(whole_data):
    #  20 < user : 34079 poi: 24406 review: 1733082> 735092
    f = open('./data/11_0_4/final_user.json', 'a')  # 30808
    for er in whole_data:
        f.write(er)
    f.close()


r_count = 0
count = 0
f = open('./data/yelp_academic_dataset_user.json', 'r')  # 1968703
line = f.readline()
whole_data = []
t1 = time.time()
while line:
    count += 1
    j = json.loads(line)
    if j['user_id'] in us:
        whole_data.append(line)
    if count % 100000 == 0:
        r_count += len(whole_data)
        print(count, len(whole_data), time.time()-t1)
        addUser(whole_data)
        whole_data = []
    line = f.readline()
r_count += len(whole_data)
print(count, 'user:', r_count)
addUser(whole_data)


def addBusiness(whole_data):
    f = open('./data/11_0_4/final_business.json', 'a')  # 22459
    for er in whole_data:
        f.write(er)
    f.close()


r_count = 0
count = 0
f = open('./data/yelp_academic_dataset_business.json', 'r')  # 209393
line = f.readline()
whole_data = []
t1 = time.time()
while line:
    count += 1
    j = json.loads(line)
    if j['business_id'] in bs:
        whole_data.append(line)
    if count % 100000 == 0:
        r_count += len(whole_data)
        print(count, len(whole_data), time.time()-t1)
        addBusiness(whole_data)
        whole_data = []
    line = f.readline()
r_count += len(whole_data)
print(count, 'business', r_count)
addBusiness(whole_data)


def addReview(whole_data):
    f = open('./data/11_0_4/final_review.json', 'a')  # 1733082  # 1624688  # 1153324
    for er in whole_data:
        f.write(er)
    f.close()


r_count = 0
count = 0
f = open('./data/yelp_academic_dataset_review.json', 'r')  # 8021122
line = f.readline()
whole_data = []
t1 = time.time()
stop_c = int(8021122*0.4)
while line:
    count += 1
    if count>stop_c:
        break
    j = json.loads(line)
    if j['user_id'] in us and j['business_id'] in bs:
        whole_data.append(line)
    if count % 100000 == 0:
        r_count += len(whole_data)
        print(count, len(whole_data), time.time()-t1)
        addReview(whole_data)
        whole_data = []
    line = f.readline()
r_count += len(whole_data)
print(count, 'review: ', r_count)
addReview(whole_data)
#

