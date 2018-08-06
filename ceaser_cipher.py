
def encrypt(text, key):
    result = ""

    for i in range(len(text)):
        char = text[i]

        if (char == ' '):
            result += ' '

        elif (char.isupper()):
            result += chr((ord(char) + key - 65) % 26 + 65)

        else:
            result += chr((ord(char) + key - 97) % 26 + 97)

    return result

def decrypt(result, key):
    d_res = ""

    for i in range(len(result)):
        char = result[i]

        if (char == ' '):
            d_res += ' '

        elif (char.isupper()):
            d_res += chr((ord(char) - key - 65) % 26 + 65)

        else:
            d_res += chr((ord(char) - key - 97) % 26 + 97)

    return d_res

p_text = input("Enter Plain text:")
key = int(input("Enter the key:"))
print('\n')
print("Text  : " + p_text)
print("Shift : " + str(key))
e_res = encrypt(p_text, key)
print("Cipher: " + e_res)
print("Decrypted: " + decrypt(e_res,key))