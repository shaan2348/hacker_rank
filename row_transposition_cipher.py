# we have to implement row transposition cipher

# taking input for
key = [l for l in input("Enter Key:")]
plain_text = ''.join(input("Enter the message:").split())

print("Key is: ", key)
columns = int(max(key))
print('No of columns is ',columns)

# checking for nummber of rows:
if len(plain_text) % columns == 0:
    rows = len(plain_text)//columns
else:
    rows = (len(plain_text)//columns) + 1
print('No of rows is ',rows)

matrix = []

# for k in plain_text:
#     for i in range(rows):
#         for j in range(columns):
#             if k == '':
#                 matrix.append('x')
#             else:
#                 matrix.append(k)

for i in range(rows):
    for j in range(columns):
        if plain_text[i + j] == '':
            matrix.append('x')
        else:
            matrix.append(plain_text[i + j])

print(matrix)
