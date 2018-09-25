import cmath
# taking input as a complex number
n = complex(input())

# taking out two arguements from the complex number
r = abs(complex(n))
r2 = cmath.phase(complex(n))

print(f'{r:.3f}\n{r2:.3f}') # method not accepted by hackerrank
# print("{0:.3f}".format(r2)) # method accepted by hackerrank

