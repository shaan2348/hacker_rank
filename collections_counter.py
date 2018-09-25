from collections import Counter

x = int(input())
# using counter to create a dictionary
# this dictonary will contain the count of each element of the string
d = Counter(map(int,input().split())) # for this we have to map our input

n = int(input())
l1 = []  # earning = 0

for i in range(n):
    a,b = map(int, input().split())

    # checking if it exists in the dictonary
    if d[a]>0:      # if a in d.keys() and d[a]>0:
        # adding money to total earning
        l1.append(b)
        # decrementing the value present by 1
        d[a]-= 1

print(sum(l1))  # print(earning)
