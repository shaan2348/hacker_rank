k = list(map(int, input().split()))
N = k[0]
M = k[1]

set1 = list(map(int, input().split()))

A = set(map(int, input().split()))
B = set(map(int, input().split()))

temp = sum((i in A) - (i in B) for i in set1)
print(temp)