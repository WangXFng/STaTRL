
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

count = 0
type = ['hotelstravel', 'nightlife', 'active', 'arts', 'auto', 'food', 'shopping',
        'professional', 'physicians', 'pets', 'health', 'fitness', 'education', 'beautysvc']


def format(str):
    j = 0
    for i in range(len(str)):
        # print(str[i: i+1])
        if str[i: i+1] == ' ':
            j += 1
        else:
            break

    return str[j: len(str)]


def append(aspect, score, text, data):
    aspect_ = -1
    if aspect == 'food' or aspect == 'Food':
        aspect_ = 0
    elif aspect == 'price' or aspect == 'Price':
        aspect_ = 2
    elif aspect == 'service' or aspect == 'Service':
        aspect_ = 4

    # + "\t" + str(s)
    if aspect_ != -1:
        s = 1 if int(score) == 1 else 0
        data.append(text + "\t" + str(aspect_+s))


whole_data = []
for t in type:
    for i in ['pos', 'neg']:
        f = open('./dataset/original/{}_train_{}.txt'.format(t, i), 'r')
        print(' Processing ', './dataset/original/{}_train_{}.txt'.format(t, i))

        exited_food = []
        exited_price = []
        exited_service = []
        line = f.readline()
        while line and len(line) <= 5:
            line = f.readline()
        while line:
            review_id = line.split(": ")[1].replace('\n', '')
            aspect = f.readline().split(": ")[1].replace('\n', '')
            text = f.readline().split(": ")[1].replace('\n', '')
            score = f.readline().replace('\n', '').replace(' ', '')
            # print(score)

            append(aspect, score, text, whole_data)

            line = f.readline()

            while line and len(line) <= 5:
                line = f.readline()

        f.close()

# #coding=utf-8
import  xml.dom.minidom

DOMTree = xml.dom.minidom.parse('./dataset/original/Restaurants_Train_v2.xml')
collection = DOMTree.documentElement
if collection.hasAttribute("shelf"):
   print ("Root element : %s" % collection.getAttribute("shelf"))

sentences = collection.getElementsByTagName("sentence")
# print('len(sentences)', len(sentences))
for sentence in sentences:
    aspects = sentence.getElementsByTagName('aspectCategories')
    text = sentence.getElementsByTagName('text')[0].childNodes[0].data
    for aspect in aspects:
        sub_aspect = aspect.getElementsByTagName('aspectCategory')
        for sa in sub_aspect:
            # print(sa.getAttribute("category"))

            aspect = sa.getAttribute("category")
            score = 1 if sa.getAttribute("polarity") == 'positive' else 0

            append(aspect, score, text, whole_data)


# dict = {'food': food_data, 'price': price_data, 'service': service_data}
# for i in dict:
f = open('./dataset/dataset/train_6.txt', 'w')
for i in whole_data:
    f.write(i+"\n")
f.close()