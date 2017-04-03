with open('input.txt', 'r') as f_in:
    string = f_in.readline()

string = string.strip(' ')

if len(string) <= 1:
    print(-1)
else:
    ans = []

    flag_dot = False
    was_space = False
    was_space_after_dot = False

    for i in range(len(string)):
        if i == 0:
            ans.append(string[i].upper())
        else:
            if string[i] == '.':
                if i-1 >= 0 and string[i-1] == ' ':
                    ans.pop()
                ans.append('.')
                ans.append(' ')
                flag_dot = True
                was_space_after_dot = True
            elif string[i] == ' ':
                if was_space_after_dot is True:
                    continue
                elif was_space is False:
                    ans.append(' ')
                    was_space = True
                elif was_space is True:
                    continue
            elif flag_dot is False:
                ans.append(string[i].lower())
                was_space = False
            elif flag_dot is True:
                ans.append(string[i].upper())
                flag_dot = False
                was_space_after_dot = False
                was_space = False
    ans.pop()

    kek = (''.join(ans)).strip(' ')

    if '' in list(map(lambda x: x.strip(), kek.split('.')[:-1])) or not kek[-1] == '.':
        print(-1)
    else:
        print(kek)
