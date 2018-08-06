def split_and_join(line):
    s = line.split()

    result = '-'.join(s)
    return result

line = input()
result = split_and_join(line)
print(result)