import calendar
days = ['MONDAY','TUESDAY','WEDNESDAY','THURSDAY','FRIDAY','SATURDAY','SUNDAY']
month, day, year = map(int, input().split())

print(days[(calendar.weekday(year,month,day))])

# ONE LINE ANSWER
# m, d, y = map(int, input().split())
#print(list(calendar.day_name)[calendar.weekday(y, m, d)].upper())