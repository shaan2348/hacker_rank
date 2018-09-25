N = int(input())
A = set(input().split())

M = int(input())
for i in range(M):
    x = input().split()
    y = input().split()
    eval('A.' + x[0] + '(y)' )
print(sum(map(int, A)))