# from functools import reduce
import copy, random, numpy as np
from collections import Counter
import functools
"""
num - number of elements in csp, with or without constraints
links - list 
possibleValues
"""
def isUnique(link, vals):
    return not any([x != None and vals.count(x) > 1 for x in vals])

class CSP:
    def __init__(self, links, possibleValues=[], f=isUnique):
        self.num = len(set(np.concatenate(links)))
        self.links = links
        self.possibleValues = possibleValues
        self.nodesOpened = 0
        self.isConstraintOk = f
    def backtrack(self, assignment):
        #Extract non-assigned yet nodes
        nonasssigned=[el for el in assignment if el[0] is None]
        self.nodesOpened += 1
        if not nonasssigned:
            #we're done, everything assigned
            return [el[0] for el in assignment]
        #Choose the node with minimal number of possible values left. Thta's the node we will assign in this recursive call
        elIndex = assignment.index(min(nonasssigned, key = lambda x:len(x[1])))
        #Iterate over possible values on chosen node.
        for possibleValue in assignment[elIndex][1]:
            assignment[elIndex][0]=possibleValue
            relatedLinks=[el for el in self.links if elIndex in el]
            #Go through all unassigned nodes and do forward remove of impossible values due to constraint function
            #Have a map for forvard removed values to be able to rollback changes. Due to this we don't need to make copy of whole csp on each recursive call
            forwardRemove={}
            #Check if chosen possible value is not being constrained by 'isConstraintOk' function
            if all([self.isConstraintOk(link, [assignment[i][0] for i in link]) for link in relatedLinks]):
                #possible value can be assigned - let's do the forward check and remove all constrained values on neighbour nodes (after this assignment)
                for link in relatedLinks:
                    for nodeIndex in link:
                        if nodeIndex != elIndex and assignment[nodeIndex][0] == None:
                            toRemove=[]
                            for forwardValue in assignment[nodeIndex][1]:
                                assignment[nodeIndex][0]=forwardValue
                                if not self.isConstraintOk(link,[assignment[index][0] for index in link]):
                                    toRemove.append(forwardValue)
                                assignment[nodeIndex][0]=None
                            assignment[nodeIndex][1] = [el for el in assignment[nodeIndex][1] if el not in toRemove]
                            if nodeIndex in forwardRemove:
                                forwardRemove[nodeIndex] += toRemove
                            else:
                                forwardRemove[nodeIndex] = toRemove
                #Now, when forward check was done - let's go deeper into search tree with new assignemnt
                res = self.backtrack(assignment)
                #This will return a valid complete assignment when algorithm will reach the full valid assignment and recurse back.
                if not res is None:
                    return res
            #Rollback changes done in this iteration (we get here only if the prvious path led us to conflict)
            assignment[elIndex][0]=None
            for el, removedItems in list(forwardRemove.items()):
                assignment[el][1]+=removedItems
        #There are no possible values for this node. Sad, but we went wrong way on search tree. Recurse back to previous tree leaf and try another path.
        return None
    #Prepare possible values for each of nodes in graph. Currently it is expected that all of them initially wil have the same set
    def solve(self):
        return self.backtrack([[None, self.possibleValues[:]] for l in range(self.num)])

"""GRAPH"""
# csp = CSP(6, ((0, 1), (0, 2), (0, 3), (1, 2), (1, 4), (1, 5), (2, 3), (2, 4), (3, 4), (3, 5), (4, 5)), ['r', 'b', 'g'])
# print(csp.solve())
# print('Nodes opened:',csp.nodesOpened)

"""SUDOKU"""
# Size of sudoku and how many % of numbers should be hidden
# bs, hide = 4, 70
# bs2, bs3, bs4=bs**2, bs**3, bs**4
# links=[[a+bs2*i-1 for a in range(1, bs2+1)] for i in range(bs2)]+\
#       [[a+bs2*i-1 for i in range(0, bs2)] for a in range(1, bs2+1)]+\
#     list(np.concatenate([[np.concatenate([[x+bs2*i+bs*j+bs3*k-1 for x in range(1, bs+1)] for i in range(bs)]) for j in range(bs)] for k in range(bs)]))
# vals = list(range(1, bs2+1))
# random.shuffle(vals)
# csp = CSP(links, vals)
# solvedSudoku = csp.solve()
# #Print res
# print((np.array(solvedSudoku).reshape(bs2, bs2)))
# for i in range(bs4*hide//100):
#     solvedSudoku[random.randint(0, bs4-1)]=''
# print(np.core.defchararray.rjust(np.array(solvedSudoku).reshape(bs2, bs2), 3 ,' '))
# print(('Nodes opened:',csp.nodesOpened))
