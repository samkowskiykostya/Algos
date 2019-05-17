from scipy import optimize
import numpy as np

def f(p, *a):
    x,y,z = p
    return (x + y - z - 3) ** 2


x, fm, d = optimize.fmin_l_bfgs_b(f, np.array([5, 4, 3]),
                                  bounds=[(1,3),(2,5),(1,5)],
                                  epsilon=1,
                                  approx_grad=True, iprint=1)
print(x, fm)