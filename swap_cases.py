def swap_cases(s):
    ans = []
    for i in s:
        if i.isupper() == True:  #checking if the letter is in upper case, if true then it we convert it to lowercase
            ans.append(i.lower())
        else:
            ans.append(i.upper())
    s1 = ''.join(ans)  # joining the elements of ans list to form a string
    print(s1)

swap_cases(input())