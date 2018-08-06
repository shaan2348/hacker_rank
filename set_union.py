# we take input for n number of students who have subscribed
#  for english newspaper and take input for their roll nos
n = int(input())
s1 = set(map(int, input().split()))

# we take input for b number of students who have subscribed
#  for french newspaper and take input for their roll nos
b = int(input())
s2 = set(map(int, input().split()))

# we take n and b's union and print its length
a = s1.union(s2)
print(len(a))

# we take n and b's intersection and print its length
b = s1.intersection(s2)
print(len(b))

c = s1.difference(s2)
print(len(c))

d = s1.symmetric_difference(s2)
print(len(d))
