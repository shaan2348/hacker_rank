m = int(input())
my_set1 = set(map(int, input().split()))
n = int(input())
my_set2 = set(map(int, input().split()))

final_set = my_set1.symmetric_difference(my_set2)
x = sorted(final_set)

for i in x:
    print(i)

#
# 4
# 2 4 5 9
# 4          00
# 2 4 11 12=