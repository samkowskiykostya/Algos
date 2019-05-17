from collections import Counter
num = 0
for i in range(10000):
    s = str(i).rjust(4, '0')
    c = Counter(s)
    if c.most_common(1)[0][-1] == 3:
       num += 1
       # print(s)
       continue
    mc = c.most_common(2)
    if len(mc) == 2 and  mc[0][-1] == mc[1][-1] == 2:
        num += 1
        # print(s)
        continue
    # if sum([int(a) for a in s]) == 10:
    #     num += 1
    #     print(s)
    #     continue
    inc = False
    for pair in range(1, 4):
        if s[0] == s[pair] or int(s[0]) + int(s[pair]) == 10:
            rest = []
            for k in range(1, 4):
                if k != pair:
                    rest.append(k)
            if s[rest[0]] == s[rest[1]] or int(s[rest[0]]) + int(s[rest[1]]) == 10:
                inc = True
                num += 1
                # print(s)
            break
    if inc:
        continue
    if (int(s[0]) - int(s[1])) == (int(s[2]) - int(s[3])):
    #     abs(int(s[0]) - int(s[3])) == abs(int(s[1]) - int(s[2])) or \
    #     abs(int(s[0]) - int(s[2])) == abs(int(s[1]) - int(s[3])):
        num += 1
        print(s)
        continue

print(num, float(num) / 10000, '%')
