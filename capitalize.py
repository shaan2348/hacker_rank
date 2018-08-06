def capitalize(string):
    a = []
    s = string.split()
    for i in s:
        a.append(i.capitalize())
    return (" ".join(a))

if __name__ == '__main__':
    string = input()
    capitalized_string = capitalize(string)
    print(capitalized_string)


# one liner:
# return (' '.join(i.capitalize() for i in string.split(' ')))