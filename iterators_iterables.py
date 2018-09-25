# for problem statement refer to the hackerrank site
from itertools import *
n = int(input())
l = input().split()
k = int(input())
count = 0
a = list(combinations(l,k))
for i in a:
    if 'a' in i:
        count+= 1
print(count/len(a))