# we have to perform remove , pop and discard operations on sets

# taking input for elements of sets
n = int(input())
s = set(map(int, input().split()))

# we will now take input for commands to be excuted

N = int(input()) # number of commands to execute

for i in range(N):
    eval('s.{0}({1})'.format(*input().split()+['']))
print(sum(s))
# print(''.join(s))

#
# 9
# 1 2 3 4 5 6 7 8 9
# 10
# pop
# remove 9
# discard 9
# discard 8
# remove 7
# pop
# discard 6
# remove 5
# pop
# discard 5