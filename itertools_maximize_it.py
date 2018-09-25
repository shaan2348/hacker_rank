# #my code but some test cases are wrong
# k,m = input().split()
# sum = 0
# for i in range(int(k)):
#     l = max([int(x) for x in input().split())
#     sum+= l*l
# print(sum%int(m))




from itertools import product

# taking input of k and m
K,M = map(int,input().split())

# taking input for k lines
# in each line the first element will be the number of elements in that line so we ignore it
N = (list(map(int, input().split()))[1:] for _ in range(K))

# here we are using list comprehension
# we are taking cartesian product of elementes of N list
# we then take each list form this list of cartesian product and take sum of their squares
#and store them in the results
results = [sum(num**2 for num in numbers) % M for numbers in product(*N)]
#results = map(lambda x: sum(i**2 for i in x)%M, product(*N))

# we are printing the max value from results list
print(max(results))