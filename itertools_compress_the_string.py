# # n = input()
# # l = {}
# # for i in n:
# #     a = n.count(i)
# #     if i not in l:
# #         l[i] = a
# # print(*l)
# from itertools import groupby
# i,j = groupby(input())
# print(*list(i))
# print(*j)
# # for i in n:
# #      print(len(list(i)))
# #      print(j)

from itertools import groupby
# here we are unpacking the items from the list comprehensiom
# we are distributing the outputs of groupby function in a and b
# here a is the element and b is the occurance of that number
# after that we are taking the len of list(b) and int(a)
print(*[(len(list(b)), int(a)) for a, b in groupby(input())])