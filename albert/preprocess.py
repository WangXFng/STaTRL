
# hotelstravel {'Price_pos': 0, 'Service_pos': 1, 'Food_pos': 2, 'Service_neg': 3, 'Food_neg': 4, 'Price_neg': 5}
# nightlife {'Food_pos': 6, 'Price_pos': 7, 'Service_pos': 8, 'Food_neg': 9, 'Price_neg': 10, 'Service_neg': 11}
# active {'Service_pos': 12, 'Price_pos': 13, 'Service_neg': 14, 'Price_neg': 15}
# arts {'Service_pos': 16, 'Price_pos': 17, 'Service_neg': 18, 'Price_neg': 19}
# auto {'Service_pos': 20, 'Price_pos': 21, 'Service_neg': 22, 'Price_neg': 23}
# food {'Food_pos': 24, 'Service_pos': 25, 'Price_pos': 26, 'Service_neg': 27, 'Food_neg': 28, 'Price_neg': 29}
# shopping {'Service_pos': 30, 'Price_pos': 31, 'Service_neg': 32, 'Price_neg': 33}
# professional {'Service_pos': 34, 'Price_pos': 35, 'Service_neg': 36, 'Price_neg': 37}
# physicians {'Price_pos': 38, 'Service_pos': 39, 'Service_neg': 40, 'Price_neg': 41}
# pets {'Service_pos': 42, 'Price_pos': 43, 'Service_neg': 44, 'Price_neg': 45}
# health {'Service_pos': 46, 'Price_pos': 47, 'Price_neg': 48, 'Service_neg': 49}
# fitness {'Service_pos': 50, 'Price_pos': 51, 'Price_neg': 52, 'Service_neg': 53}
# education {'Service_pos': 54, 'Price_pos': 55, 'Service_neg': 56, 'Price_neg': 57}
# beautysvc {'Price_pos': 58, 'Service_pos': 59, 'Service_neg': 60, 'Price_neg': 61}

# Price 0
# Service 1
# Food 2

# count = 0
# type = ['hotelstravel', 'nightlife', 'active', 'arts', 'auto', 'food', 'shopping',
#         'professional', 'physicians', 'pets', 'health', 'fitness', 'education', 'beautysvc']
#
# cate = {}
#
#
# def format(str):
#     j = 0
#     for i in range(len(str)):
#         # print(str[i: i+1])
#         if str[i: i+1] == ' ':
#             j += 1
#         else:
#             break
#
#     return str[j: len(str)]
#
#
# type_count = 0
# whole_data = []
# for t in type:
#     for i in ['pos', 'neg']:
#         f = open('./dataset/original/{}_train_{}.txt'.format(t, i), 'r')
#         print(' Processing ', './dataset/original/{}_train_{}.txt'.format(t, i))
#
#         line = f.readline()
#         while line and len(line) <= 5:
#             line = f.readline()
#         while line:
#             # if line and len(line) <= 5:
#             #     line = f.readline()
#             #     continue
#             # f2.write(format(line))
#             review_id = line
#             review_id = review_id.split(": ")[1].replace('\n','')
#             aspect = f.readline().split(": ")[1].replace('\n','')
#
#             # if t not in cate:
#             #     cate[t] = {}
#
#             # if '{}_{}'.format(aspect, i) not in cate[t]:
#             if '{}_{}'.format(aspect, i) not in cate:
#                 # cate[t]['{}_{}'.format(aspect, i)] = type_count
#                 cate['{}_{}'.format(aspect, i)] = type_count
#                 type_count += 1
#
#             text = f.readline().split(": ")[1].replace('\n','')
#             score = f.readline()
#             line = f.readline()
#
#             # f2.write(format(text))
#
#             # whole_data.append(format(text)+"\t"+format(aspect)+"\t"+str(cate[t][format('{}_{}'.format(aspect, i))]))
#             whole_data.append(format(text)+"\t"+format(aspect)+"\t"+str(cate['{}_{}'.format(aspect, i)]))
#
#             count += 1
#             # if count % 300 == 0:
#             #     break
#             while line and len(line) <= 5:
#                 line = f.readline()
#
#         f.close()
#
# for i in cate:
#     print(i, cate[i])
#
# # print(whole_data[0:10])
#
# f2 = open('./dataset/dataset/train_3.txt', 'w')
# for i in whole_data:
#     f2.write(i+"\n")
# f2.close()


# Price_pos 0
# Service_pos 1
# Food_pos 2
# Service_neg 3
# Food_neg 4
# Price_neg 5


#coding=utf-8
import  xml.dom.minidom

DOMTree = xml.dom.minidom.parse('./dataset/original/Restaurants_Train_v2.xml')
collection = DOMTree.documentElement
if collection.hasAttribute("shelf"):
   print ("Root element : %s" % collection.getAttribute("shelf"))
#打开xml文档

#得到文档元素对象

whole_data = []

sentences = collection.getElementsByTagName("sentence")
# print('len(sentences)', len(sentences))
for sentence in sentences:
    aspects = sentence.getElementsByTagName('aspectCategories')
    text = sentence.getElementsByTagName('text')[0].childNodes[0].data
    for aspect in aspects:
        sub_aspect = aspect.getElementsByTagName('aspectCategory')
        for sa in sub_aspect:
            # print(sa.getAttribute("category"))
            label = -1
            if sa.getAttribute("category") == 'price':
                if sa.getAttribute("polarity") == 'position':
                    label = 0
                else:
                    label = 5
            elif sa.getAttribute("category") == 'food':
                if sa.getAttribute("polarity") == 'position':
                    label = 2
                else:
                    label = 4
            elif sa.getAttribute("category") == 'service':
                if sa.getAttribute("polarity") == 'position':
                    label = 1
                else:
                    label = 3
            if label != -1:
                whole_data.append(text+"\t"+sa.getAttribute("category")+"\t"+str(label))


f2 = open('dataset/dataset/res_train_6.txt', 'w')
for i in whole_data:
    f2.write(i+"\n")
f2.close()

