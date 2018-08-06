def print_formatted(number):
    print(bin(number))
    a = len(bin(number)[2:])
    #a = len("{0:b}".format(number))
    #print(a)
    for num in range(1,number+1):
        print("{0:{a}} {0:{a}o} {0:{a}X} {0:{a}b}".format(num,a = a))

if __name__ == '__main__':
    n = int(input())
    print_formatted(n)