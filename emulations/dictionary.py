with open('C:\\My\\Dictionary\\eng.txt', 'r') as f:
    m = dict()
    ls = f.read().split('\n')
    for l in ls:
        l = l.lower()
        if not any([ord(s) < 97 or ord(s) > 122 for s in l]):
            for i in range(len(l) - 1):
                m.setdefault(l[i], set()).add(l[i + 1])