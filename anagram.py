#this a random program found in the notifications

# s1 = input()
# s2 = input()
#
# set1 = set()
# set2 = set()
#
# for i in s1:
#     set1.add(i)
# for i in s2:
#     set2.add(i)
#
# diff = set1.symmetric_difference(set2)
# #print(diff)
# print(len(diff))
# # print(set1)
# # print(set2)

#Method 2:
s1 = input()
s2 = input()


total = 0
for letter in "abcdefghijklmnopqrstuvwxyz":
    total += abs(s1.count(letter) - s2.count(letter))
print(total)