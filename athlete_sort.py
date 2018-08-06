# we are asked to sort the list of given athletes based on Kth attribute
# we take input of number of athletes and attribute for each of them
N, M = map(int, input().split())
#input of details of athletes
rows = [input() for _ in range(N)]
print(rows)
# attribute on which we have to sort the details
K = int(input())

# here we use sorted method to sort the lists
# here in sorted we are passing our list of athlete and
# key is for passing the particular attribute on which we will be sorting our lists
for row in sorted(rows, key=lambda row: int(row.split()[K])):
    print(row)