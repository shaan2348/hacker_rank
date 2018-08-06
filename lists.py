n = int(input())  # this is for the number of inputs to be taken
l = []

for _ in range(n):
    cmd = input("Please Enter Your Command here:")
    s = cmd.split()  #splitting the command ex: if cmd is insert 1 2, so it becomes ['insert', '1' ,'2']

    cmd2 = s[0]
    arg = s[1:]

    if cmd2 != "print":
        cmd2 += "(" + ",".join(arg) + ")"  # here we rejoin the command and it becomes insert(1 2)
        eval("l." + cmd2)
    else:
        print(l)
