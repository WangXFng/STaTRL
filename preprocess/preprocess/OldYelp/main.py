import json
import re

# taxiDictionary = [' taxi ', ' taxi,', ' taxi.', 'Taxi ', ' cab ', ' cab,', ' cab.',  'Cab ',
#                   ' hack ', ' hack,', ' hack.', 'Hack ', ' taxicab ', ' taxicab,', ' taxicab.', 'Taxicab ',
#                   ]

taxiDictionary = [' taxi ', ' taxi,', ' taxi.', 'Taxi ',
                  ' taxis ', ' taxis,', ' taxis.', 'Taxis ',
                  ' taxies ', ' taxies,', ' taxies.', 'Taxies ',
                  ' cab ', ' cab,', ' cab.',  'Cab ',
                  ' cabs ', ' cabs,', ' cabs.',  'Cabs ',
                    ' taxicab ', ' taxicab,', ' taxicab.', 'Taxicab ',
                    ' taxicabs ', ' taxicabs,', ' taxicabs.', 'Taxicabs ',
                   ]

count = 0
result = []

with open('/home/g19tka20/Downloads/yelp_dataset/yelp_academic_dataset_review.json', 'r', encoding='utf-8') as f:
    for line in f:
        count += 1
        j = json.loads(line)
        for taxiName in taxiDictionary:
            if j['text'].find(taxiName) != -1:
                result.append([count,j['business_id'],j['text'],j['date']])
                break
        if count%100000 == 0 :
            print(count)


print(count)

f = open('/home/g19tka20/Downloads/yelp_dataset/taxi_review_without_hack.txt','w') #output.txt - 文件名称及格式 w - writing
    #以这种模式打开文件,原来文件内容会被新写入的内容覆盖,如文件不存在会自动创建
for r in result:
    f.write(json.dumps({'id': r[0], 'business_id':r[1], 'text': r[2], 'date':r[3]})+"\r\n")

print(len(result))

f.close()