#!/bin/python3

import math
import os
import random
import re
import sys


# Complete the alternatingCharacters function below.
def alternatingCharacters(s):
    total = 0
    for i in range(len(s) - 1):
        if s[i] == s[i + 1]:
            total += 1
    return total


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    q = int(input())

    for q_itr in range(q):
        s = input()

        result = alternatingCharacters(s)

        fptr.write(str(result) + '\n')

    fptr.close()

# main_logic:
#
# s = input()
# total = 0
# for i in range(len(s)-1):
#     if s[i] == s[i + 1]:
#         total += 1
# print(total)