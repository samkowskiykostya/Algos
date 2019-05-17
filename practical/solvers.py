from sympy import *

init_session()

"""Solve any eq for any variable, evaluate numeric value"""
eq = Eq(x ** 2 + 2 * x + 6, y)
print(N(solveset(eq, x).subs(y, 12)))
print(N(solveset(eq, y).subs(x, 12)))
