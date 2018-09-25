# got new thing to learn
import math
a = int(input())
b= int(input())

# atan2() intakes two arguments instead of one for atan()
# The gimmick is, the line that bisects the hypoteneuse will end up going
# through the other corner of the rectangle with sides a and b.
#  So we're just figuring atan(a,b)
print(str(int(round(math.degrees(math.atan2(a,b))))) + '°')