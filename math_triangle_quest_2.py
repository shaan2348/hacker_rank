# problem is to print palindromic string in format
# 1
# 121
# 12321
# 1234321
# 123454321
# for n = 5
# in this we can detect a pattern that each line is square of 1, 11, 111...
# that is it is square of increasing numbers of 1
# also input limit is 0-10
for x in range(1,int(input())+1):
    print(((10**x - 1)//9)**2)