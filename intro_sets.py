def average(array):
    r1 = set(array)
    sum = 0
    for i in r1:
        sum = sum + i
    result = sum/len(r1)
    return result

if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().split()))
    result = average(arr)
    print(result)