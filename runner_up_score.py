n = int(input())

arr = list(map(int,input().split()))
arr.sort()

for i in range(len(arr)-1,-1,-1):
    if arr[i] != arr[i-1]:
        print(arr[i-1])
        break
    else:
        continue
