def merge_the_tools(string, k):

    for part in zip(*[iter(string)] * k):
        print(part)
        d = dict()
        print(''.join([ d.setdefault(c, c) for c in part if c not in d ]))

if __name__ == '__main__':
    string, k = input(), int(input())
    merge_the_tools(string, k)

#Method 2:
# s=raw_input()
# k=int(raw_input())
# n=len(s)
#
# for x in xrange(0, n, k):
#     slicedStr = s[x : x+k]
#     uni =[]
#     for y in slicedStr:
#         if y not in uni:
#             uni.append(y)
#     print ''.join(uni)

# Method 3:
# ts = [string[ind:ind+k] for ind, s in enumerate(string) if ind % k == 0]
#     for s in ts:
#         print("".join([x for ind, x in enumerate(s) if x not in s[0:ind]]))