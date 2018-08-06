# a binary string is not beautiful if '010' is present as its substring
# ques asks to return minimum number of changes from 0 to 1 and vice versa

import math
import os
import random
import re
import sys

# Complete the beautifulBinaryString function below.
def beautifulBinaryString(b):
    # here what we are doing is print the number of occurances of '010'
    # because that will be the number of changes we will have to do
    return b.count('010')

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input())

    b = input()

    result = beautifulBinaryString(b)

    fptr.write(str(result) + '\n')

    fptr.close()