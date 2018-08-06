def count_substring(string, sub_string):

    return(sum([1 for i in range(len(string) - len(sub_string) + 1)  # we move through the string here and check if we are able
                if string[i:i + len(sub_string)] == sub_string]))    # to find the substring in the string and generate 1 for each time
                                                                    # it is true and then we add all the 1's in the list
if __name__ == '__main__':
    string = input().strip()
    sub_string = input().strip()

    count = count_substring(string, sub_string)
    print(count)


#METHOD 2:
# def count_substring(string, sub_string):
#     count=0
#     #print(len(string),len(sub_string))
#     for i in range(0, len(string)-len(sub_string)+1):
#         if string[i] == sub_string[0]:
#             flag=1
#             for j in range (0, len(sub_string)):
#                 if string[i+j] != sub_string[j]:
#                     flag=0
#                     break
#             if flag==1:
#                 count += 1
#     return count



#METHOD 3:
    # count = 0
    # i = 0
    # while i < len(string):
    #     if string.find(sub_string, i) >= 0:
    #         i = string.find(sub_string, i) + 1
    #         count += 1
    #     else:
    #         break
