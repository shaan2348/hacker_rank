# taking number of test cases
N = int(input())
list1 = []
# taking input for sets
for i in range(N):
    N2 = int(input())
    A = set(map(int, input().split()))
    N3 = int(input())
    B = set(map(int, input().split()))

    if A.issubset(B):
        list1.append('True')
    else:
        list1.append("False")
print('\n'.join(list1))
