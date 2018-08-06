n = int(input())

t = tuple(map(int,input().split()))  # we are use map function here to convert the inputs from str to int because input() returns str

print(t)

print(hash(t))