if __name__ == '__main__':
    x = int(input("Enter X:"))
    y = int(input("Enter Y:"))
    z = int(input("Enter Z:"))
    n = int(input("Enter N:"))

    my_list = [[i,j,k] for i in range(x+1) for j in range(y+1) for k in range(z+1) if i + j + k != n ]

    my_list.sort()

    print(my_list)


