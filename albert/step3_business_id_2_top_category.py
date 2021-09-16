import numpy as np
import time
import json
import datetime

np.set_printoptions(suppress=True)

with open('./dataset/dataset/categories.json','r') as load_f:
   load_dict = json.load(load_f)
   # print(load_dict)

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

top_parent = ['hotelstravel', 'nightlife', 'food', 'active', 'arts', 'auto', 'shopping',
            'professional', 'physicians', 'pets', 'health', 'fitness', 'education', 'beautysvc']

# top_parent = ['active', 'arts', 'auto', 'restaurants', 'food', 'japanese', 'nightlife', 'shopping', 'professional', 'laywers'
#               , 'physicians', 'pets', 'hotelstravel', 'health', 'fitness', 'education', 'beautysvc']

category_parent = {}
for i,ld in enumerate(load_dict):
    if len(ld['parents']) == 0:
       category_parent[ld['title'].lower()] = ld['alias'].lower()
    else:
       category_parent[ld['title'].lower()] = ld['parents'][0].lower()

# category_parent['food'] = 'food'
category_parent['restaurants'] = 'food'
category_parent['Departments of Motor Vehicles'.lower()] = 'auto'
category_parent['Local Services'.lower()] = 'health'
category_parent['Home Services'.lower()] = 'health'
category_parent['Party Supplies'.lower()] = 'active'
category_parent['Laundry Services'.lower()] = 'active'
category_parent['Photographers'.lower()] = 'arts'
category_parent['Event Planning & Services'.lower()] = 'active'
category_parent['Public Services & Government'.lower()] = 'professional'
category_parent['Financial Services'.lower()] = 'professional'
category_parent['Local Flavor'.lower()] = 'active'
category_parent['Churches'.lower()] = 'active'
category_parent['Print Media'.lower()] = 'professional'

#
# category_parent['shopping'] = 'shopping'
#
# category_parent['nightlife'] = 'nightlife'
#
# category_parent['pets'] = 'pets'

print(top_parent, len(top_parent))
print(category_parent)


businesses_ = np.load('./data/11_0_4/yelp_business_level_4_business_id.npy')
bs = {}
for i,b in enumerate(businesses_):
    bs[b] = i

print('start business')
business_id_2_top = {}
count = 0
f = open('./data/11_0_4/final_business.json', 'r')
line = f.readline()
while line:
    count += 1
    # print(count)
    j = json.loads(line)
    if count % 100000 == 0:
        print('business', count)
        # break

    if j['business_id'] in bs and j['categories'] is not None:
       categories = j['categories'].split(', ')
       if len(categories) != 0:
           for c in categories:
               c = c.lower()
               # print(c)
               if c in category_parent:
                   category_top = category_parent[c]
                   if category_top in top_parent:
                       business_id_2_top[bs[j['business_id']]] = top_parent.index(category_top)

    if bs[j['business_id']] not in business_id_2_top:
       business_id_2_top[bs[j['business_id']]] = -1
       # print(j['categories'])

    line = f.readline()

print(business_id_2_top)
print(len(business_id_2_top.keys()))

import pickle
data_output = open('./data/11_0_4/business_id_2_top.pkl','wb')
pickle.dump(business_id_2_top, data_output)
data_output.close()
