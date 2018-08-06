K = int(input())
list1 = list(map(int,input().split()))

# for i in list1:
#     if list1.count(i) != K:
#         print(i)

set1 = set(list1)
# print(set1)
# print(sum(set1)*K)
# # ans = ((sum(set1)*K) - sum(list1)) // K-1
# # print(ans)
# print(sum(list1))
print((sum(set1)*K - sum(list1)) // (K-1))