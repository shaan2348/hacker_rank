s = input()
for i in('isalnum()', 'isalpha()', 'isdigit()', 'islower()', 'isupper()'):
    print(any( eval('c.' + i) for c in s))


#METHOD 2:
# print (any(i.isalnum() for i in s))
# print (any(i.isalpha() for i in s))
# print (any(i.isdigit() for i in s))
# print (any(i.islower() for i in s))
# print (any(i.isupper() for i in s))
