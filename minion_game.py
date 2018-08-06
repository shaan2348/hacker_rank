def minion_game(string):
    vowels = 'AEIOU'

    k_score = 0
    s_score = 0

    for i in range(len(string)):
        if string[i] in vowels:
            k_score += (len(s)-i)
        else:
            s_score += (len(s)-i)

    if s_score > k_score:
        print("Stuart", s_score)
    elif k_score > s_score:
        print("Kevin", k_score)
    else:
        print("Draw")
if __name__ == '__main__':
    s = input()
    minion_game(s)