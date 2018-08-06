from itertools import cycle

ALPHA = 'abcdefghijklmnopqrstuvwxyz'


def encrypt( plaintext, key):
    """Encrypt the string and return the ciphertext"""
    pairs = zip(plaintext, cycle(key))
    result = ''

    for pair in pairs:
        total = reduce(lambda x, y: ALPHA.index(x) + ALPHA.index(y), pair)
        result += ALPHA[total % 26]

    return result.lower()


def decrypt(key, ciphertext):
    """Decrypt the string and return the plaintext"""
    pairs = zip(ciphertext, cycle(key))
    result = ''

    for pair in pairs:
        total = reduce(lambda x, y: ALPHA.index(x) - ALPHA.index(y), pair)
        result += ALPHA[total % 26]

    return result


def show_result(plaintext, key):
    """Generate a resulting cipher with elements shown"""
    encrypted = encrypt(key, plaintext)
    decrypted = decrypt(key, encrypted)

    print('Key: %s' % key)
    print('Plaintext: ' , plaintext)
    print('Encrytped: ' , encrypted)
    print('Decrytped: ', decrypted)

if __name__ == '__main__':
    p_text  = input("Enter your Message:")
    cipher_key = input("Enter your key here: ")
    show_result(p_text, cipher_key)