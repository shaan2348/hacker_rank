# problem is to add elements from given list or input
# to a set and get the length of resultant set

N = int(input())
set1 = set()
#s = input().split()

for i in range(N):
    s = input()
    set1.add(s)
#print(set1)
print(len(set1))