def encrypt():
    pass
def decrypt():
    pass

Alphabet = 'abcdefghijklmnopqrstuvwxyz'
list1 = list(Alphabet)
#print(list1)

shift_key = input("Enter the ciphertext alphabet:")
#print(shift_key)

#checking if the key is monoaplphabetic
if len(shift_key)>1:
    print("Error: Key to be only single letter.")

list2 = []
list2.append(shift_key.lower())
#print(list2)

a = list1.index(shift_key.lower())

for i in list1[a+1:]:
    list2.append(i.lower())
for i in list1[:a]:
    list2.append(i.lower())

#print(list2)
list3 = []
list4 = []

plain_text = input("Enter the Message:").split()

for i in plain_text:
    for j in i:
        #print(j)
        x = list1.index(j.lower())
        #print(x)
        list3.append(list2[x])
#print(list3)

for i in list3:
    x = list2.index(i.lower())
    list4.append(list1[x])

e_msg = ''.join(map(str, list3))
d_msg = ''.join(map(str, list4))
print(f"Encrypted message is: {e_msg}")
print(f"Decrypted message is: {d_msg}")
