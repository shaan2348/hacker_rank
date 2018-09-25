from itertools import combinations_with_replacement

a,b = input().split()

x = list(combinations_with_replacement(sorted(a),int(b)))
for i in x:
    print("".join(i))


# AA
# AC
# AH
# AK
# CC
# CH
# CK
# HH
# HK
# KK