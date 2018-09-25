# we have to print permutations of given string of given length
from itertools import permutations

a,b = input().split()
# print(list(a))
# print(b)
# print(*permutations(list(a),int(b)), sep='\n')
x = sorted(list(permutations(list(a),int(b))))
# print(x)
for i in x:
    print("".join(i))