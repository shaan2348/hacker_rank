n = int(input())

marksheet = {}

for _ in range(n):
    details = input().split()
    name = details[0]
    marks = details[1:]
    marksheet[name] = marks

    #marksheet[name] = [marks1, marks2, marks3] this is also a way to allocate multiple values to a single key in dictionary

#print(marksheet)

x = input()

l = marksheet[x]
a = len(l)
sum = 0
avg = 0

for i in range(a):
    sum = sum + float(l[i])

avg = sum/a

print(f'{avg:.2f}')  #applies only for py 3.6 or above otherwise use .format method
