n = int(input())

my_list = []
list2 = []

for i in range(0,n):
    my_list.append([input(), float(input())])
my_list.sort()

for i in range(len(my_list)):
    list2.append(my_list[i][1])

a = list(set(list2))
a.sort()

second_lowest = a[1]

for i in range(len(my_list)):
    if my_list[i][1] == second_lowest:
        print(my_list[i][0])



# 5
# Harry
# 37.21
# Berry
# 37.21
# Tina
# 37.2
# Akriti
# 41
# Harsh
# 39

#a = [['Harry', 37.21], ['Berry', 37.21], ['Tina', 37.2], ['Akriti', 41], ['Harsh', 39]]