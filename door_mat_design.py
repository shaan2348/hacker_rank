s = input().split()
N = int(s[0])
M = int(s[1])
c = '.|.'
for i in range(N//2):
    print((c*(2*i+1)).center(M,'-'))
print('WELCOME'.center(M,'-'))
for i in range(N//2,0,-1):
    print((c*(2*i-1)).center(M,'-'))