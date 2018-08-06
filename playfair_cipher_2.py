def matrix(key):
    m = []
    alphabet = "ABCDEFGHIKLMNOPQRSTUVWXYZ"

    for i in key.upper():
        if i not in m:
            m.append(i)

    for i in alphabet:
        if i not in m:
            m.append(i)

    m_group = []
    for i in range(5):
        m_group.append('')
    m_group[0] = m[0:5]
    m_group[1] = m[5:10]
    m_group[2] = m[10:15]
    m_group[3] = m[15:20]
    m_group[4] = m[20:25]

    return m_group

def groups(text):
    for i in range(len(text)):
        pass
def encrypt():
    pass
def decrypt():
    pass

text = input("Enter Your message here:")
key = input("Enter your key:")

print(matrix(key))
