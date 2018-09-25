# this ques asks to print this pattern
#1
# 22
# 333
# 4444
# 55555
# ......
# when N is given we will print upto N-1
# for example when N = 5 then we print upto N-1 i.e pattern is:
# 1
# 22
# 333
# 4444
for i in range(1,int(input())):
    print(((10**i)-1)//9 * i)