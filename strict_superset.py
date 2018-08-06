# we have to check if A is strict superset of other sets
# input for set A
A = set(map(int, input().split()))
N = int(input())

# here we are using all() method which returns true only if all are true
# we are taking input for N sets in the equation itself by  using comprehension
# also we are checking if A is superset here
temp = all(A.issuperset(set(map(int, input().split()))) for _ in range(N))
print(temp)