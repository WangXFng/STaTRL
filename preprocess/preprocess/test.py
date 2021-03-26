import numpy as np

# a = np.array([1,1,1,1])
# b = np.array([1,2,3,4])
#
# a = np.reshape(a, (1, -1))
# b = np.reshape(b, (1, -1))
#
# print(a.T *b)
# print(a* (b.T))
#
# pad_width1 = ((0, 2), (0, 2))
# in_out = np.pad(a.T *b, pad_width=pad_width1, mode='constant', constant_values=0)
# print(in_out)
#
# c = np.array([[1,2,3,4],
#               [1,2,3,4]
#              ])
# print(c[c!=2])
#
# itemindex = np.where(c==2)
# print(itemindex)
#
# print('sdfd@{tip: 1d}'.format(tip = 31))
#
# aaa = np.zeros(10)
# b = np.array([1, 2, 4, 5])
# c = np.array([5, 4, 2, 1])
#
# aaa[b] = c
# print(aaa)


# a = np.array([[1,2,3,4,5,6,7,8,9,10],
#              [1,2,3,4,5,6,7,8,9,10],
#              [1,2,3,4,5,6,7,8,9,10],
#              [1,2,3,4,5,6,7,8,9,10],
#              [1,2,3,4,5,6,7,8,9,10],
#              [1,2,3,4,5,6,7,8,9,10],
#              [1,2,3,4,5,6,7,8,9,10],
#              [1,2,3,4,5,6,7,8,9,10]]
#              )
#
# b = np.array([1,2,3])
# c = np.array([4,5,6])
#
# print(a[b][1])


def score_(s):
    if s < 5:
        s /= 10
    else:
        s = 0.5 + 0.15 * np.log(s-4)
    return s

ss = [3, 4, 5, 6, 7, 8, 9, 19, 30, 35]

for i in ss:
    print(score_(i))


def grbf(d):
    n = 0.05
    a = np.exp(-n * d)
    a[a < 0.125] = 0
    # print(a.max())
    # print(a.min())
    return a

ss = np.array([0, 1, 5, 10, 15, 25, 30, 35])

print(grbf(ss))
