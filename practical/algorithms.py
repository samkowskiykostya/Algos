import numpy as np
from collections import OrderedDict
import random
from collections import deque
from sortedcontainers import SortedList
from copy import copy
from functools import reduce

"""
reduce
"""
def savings(perc, init, terms):
    return reduce(lambda s,a: s*(1 + perc/100) + a, np.repeat(init, terms))
print(savings(5, 5000*12, 5))


"""
Brewers riddle

ele - 5,4,25
beer - 15,4,20
limits: 480, 160, 1190
prfit: ele - 13, beer - 23
"""
c = np.array([13, 23]) * -1
A= np.mat('5,15;4,4;35,20')
b = [480, 160, 1190]
res = optimize.linprog(c, A, b)
print('Best profit: ', -res.fun, ' Do ele and beer', res.x)


"""
Monte-Carlo

What it the probability of same birthday in 30 people group
"""
s = 0; N=10000; n=30
for _ in range(N):
    if len(np.unique(np.random.randint(1, 365, n))) < n: s +=1
s / N

"""
CSP
"""
def isUnique(vals, allvals):
    #Gives True or False for constraint. Second var -list of values which can be dropped
    # Input: [1,3,4,None]. Output: True, [None,None,None,[1,3,4]]
    b = not any([x != None and vals.count(x) > 1 for x in vals])
    return b, [[v for v in vals if v is not None and v != vs] for vs in vals] if b else None
def csp(values, links, rules):
    """
    d {el:[chosen, [possibleVals]]}
    links=[[el1,el2],..., [el3,el1]]
    rules: {-1: generic_rule_for_all_links, link_index: custom_rule}
    """
    def checkConstraints(i):
        acceptable = True
        ivals = [d[e][0] for e in links[i]]
        td = [[]] * len(ivals)
        if i in rules:
            a, td = rules[i](ivals, values)
            if td is None: td = []
            acceptable &= a
        if acceptable:
            if -1 in rules:
                a, td2 = rules[-1](ivals, values)
                acceptable &= a
                if acceptable:
                    if td2 is None: td2 = [[]] * len(ivals)
                    td = [np.unique(td[i] + td2[i]) for i in range(len(td))]
        return acceptable, td
    def backtrack(d):
        # print([(-1 if v[0] is None else v[0]) for k, v in d.items()])
        remainingUnass = [(k, len(d[k][1])) for k,v in d.items() if d[k][0] is None]
        if len(remainingUnass):
            nex = min(remainingUnass, key=lambda x: x[1])[0]
        else:
            return d
        for v in d[nex][1]:
            d[nex][0] = v
            # Check if no constaints violated
            applicable = True
            bk = dict()
            dq = deque([i for i, l in enumerate(links) if nex in l])
            while dq: #queu of links ids
                li = dq.popleft()
                a, td = checkConstraints(li)
                if not a:
                    applicable = False
                    break
                else:
                    for i,tdv in enumerate(td): #i - i in links[li], tdv - values to delete in d
                        if tdv is not None:
                            el = links[li][i]
                            tmp = copy(d[el][1])
                            d[el][1] = [a for a in d[el][1] if a not in tdv]
                            if d[el][1] != tmp:
                                bk[el] = tmp
                                for j in [i for i, l in enumerate(links) if el in l]:
                                    dq.append(j)
                            if not len(d[el][1]) or \
                                d[el][0] is not None and d[el][0] not in d[el][1]:
                                applicable = False
                                break
            if not applicable:
                d[nex][0] = None
                for k, v in bk.items():
                    d[k][1] = v
            else:
                res = backtrack(d)
                if res is not None:
                    return res
                else:
                    for k, v in bk.items():
                        d[k][1] = v
                    d[nex][0] = None
                    continue
        if d[nex][0] is None:
            return None
    if callable(rules): rules = {-1:rules}
    uniqueVals = set(np.concatenate(links))
    d = {k: [None, random.sample(values, len(values))] for k in uniqueVals}
    r = backtrack(d)
    return {k:v[0] for k,v in d.items()} if all([v[0] != None for k,v in d.items()]) else None

"""color map"""
# colors=['r','g','b']
# links=[(0,1),(0,2),(0,3),(1,2),(1,4),(1,5),(2,3),(2,4),(3,4),(3,5),(4,5)]
# print(csp(colors, links, isUnique))
"""sudoku"""
# bs = 4; hide = 80 #size of block, hide percentage
# bs2 = bs**2; bs3 = bs**3; bs4 = bs**4
# groups=[[a+bs2*i-1 for a in range(1,bs2+1)] for i in range(bs2)]+\
#       [[a+bs2*i-1 for i in range(0,bs2)] for a in range(1,bs2+1)]+\
#     list(np.concatenate([[np.concatenate([[x+bs2*i+bs*j+bs3*k-1 for x in range(1,bs+1)] for i in range(bs)]) for j in range(bs)] for k in range(bs)])) #hor, ver & diag
# vals = list(range(1,bs2+1))
# res = csp(vals, groups, isUnique)
# # print(res)
# res = [a[1] for a in sorted(res.items(), key=lambda x: x[0])]
# print(np.array(res).reshape(bs2,bs2))
# for i in range(bs4 * hide // 100): #Hide els
#     res[random.randint(0, bs4 - 1)] = ''
# res = np.array(res).reshape(bs2,bs2)
# print(np.core.defchararray.rjust(res, 2 ,' '))

"""SUN+FUN=SWIM"""
# groups=['nn1m','uu12i','sf2sw','sunfwim12']
# encodedVals = list(set(np.concatenate([list(a) for a in groups])))
# links = [[encodedVals.index(s) for s in g] for g in groups]
# def constraint(vals, allvals):
#     if len(vals) > 5:
#         return isUnique(vals, allvals)
#     else:
#         if not None in vals:
#             r = (sum(vals[:-2]) == vals[-2] * 10 + vals[-1])
#             return r, None
#         else: return True, None
# r = csp(list(range(10)), links, constraint)
# print([(encodedVals[k],v) for k,v in r.items()])
# print([(s, r[encodedVals.index(s)]) for s in 'sunfunswim'])





