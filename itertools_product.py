# we have to print the cartesian product of two sets
# using product function from itertools module
# "*" infront of any thing in print unpacks the list or tuple or set
from itertools import product

a = [int(x) for x in input().split()]
b = [int(x) for x in input().split()]

print(*product(a,b))