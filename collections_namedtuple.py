from collections import namedtuple

# taking input for number of students
n = int(input())
# taking names of different columns
fields = input().split()

total = 0 # total of marks
for i in range(n):
    # declaring a named tuple
    students = namedtuple('student',fields)
    # taking input for values of field
    field1, field2, field3,field4 = input().split()
    # placing this values in tuple
    student = students(field1,field2,field3,field4)
    # picking up the marks from tuple
    total += int(student.MARKS)
print('{0:.2f}'.format(total/n))

# shorter method
# here we only pick up the marks section from input
# stu, marks = int(input()), input().split().index("MARKS")
# print (sum([int(input().split()[marks]) for _ in range(stu)]) / stu)