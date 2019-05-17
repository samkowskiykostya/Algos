from scipy import special
from scipy import optimize as o
from scipy import integrate
from scipy.interpolate import interp1d
from scipy.interpolate import griddata
from matplotlib import pyplot as plt
import scipy
from socks import method

"""-----------------minimize----------------------------------"""
"""Simple convex function"""
o.minimize_scalar(lambda x: -np.exp(-(x - 0.7)**2), method='bounded')

"""Gradiend descent conjugate. Can pass jacobian"""
o.minimize(lambda x: .5*(1 - x[0])**2 + (x[1] - x[0]**2)**2, [2, -1], method='CG')

"""GD Newton. Need jacobian"""
o.minimize(lambda x: .5*(1 - x[0])**2 + (x[1] - x[0]**2)**2, [2, -1], method='Newton-CG', jac=lambda x: np.array((-2*.5*(1 - x[0]) - 4*x[0]*(x[1] - x[0]**2), 2*(x[1] - x[0]**2))))

"""prefer BFGS or L-BFGS"""
o.minimize(lambda x: .5*(1 - x[0])**2 + (x[1] - x[0]**2)**2, [2, -1], method='BFGS')

"""least squares - predict coefficients"""
x = np.array(range(10))
y = np.power(x, 2) - 2*x + 1
A = np.c_[np.power(x, 2), x, np.ones(len(x))]
np.linalg.lstsq(A, y) #Gives 1, -2, 1 as first arg

"""f root"""
o.fsolve(lambda x: x**2 - 4, 5)
o.fsolve(lambda x: (x-2)**2 - 4, 5)

"""-----------------integrate----------------------------------"""
"""Integration. Returns integral, error"""
f = lambda x: special.jv(2.5, x)
integrate.quad(f, 0, 10)

"""Integration by samples"""
x = np.linspace(0, 10, 10)
y = [f(xi) for xi in x]
integrate.simps(y, x)

"""-------------------interpolate--------------------------------"""
"""same can be done with splines"""
"""interpolation. Function by dots. Cubic best"""
y = [xi**2 + 2*xi + 15 for xi in x]
x2 = np.linspace(0, 10, 100)
f = interp1d(x, y)
f2 = interp1d(x, y, kind='cubic')
plt.plot(x, y, 'o', x2, f(x2), '-', x2, f2(x2), '--')
plt.legend(['data', 'linear', 'cubic'], loc='best')
plt.show()

"""interpolate multi-dim function (x,y)"""
f = lambda x,y: x*(1-x)*np.cos(4*np.pi*x) * np.sin(4*np.pi*y**2)**2
grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]
points = np.random.rand(1000, 2)
values = f(points[:,0], points[:,1])
plt.imshow(griddata(points, values, (grid_x, grid_y), method='cubic'))
plt.show()

"""------------------derivative-------------------------------"""
"""derivative"""
scipy.misc.derivative(lambda x: x**2 + 1, 1)

"""------------------graph------------------------------------"""
"""graph search distances between nodes (weights sum)"""
A = np.array([[0., 1., 0., 0., 0., 0., 0., 0.],
       [1., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 1., 1., 0., 0., 0.],
       [0., 0., 1., 0., 1., 1., 1., 0.],
       [0., 0., 1., 1., 0., 0., 0., 0.],
       [0., 0., 0., 1., 0., 0., 1., 0.],
       [0., 0., 0., 1., 0., 1., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0.]])
dists, pathsm = scipy.sparse.csgraph.shortest_path(A, return_predecessors=True)
"""dijkstra"""
num, pathsm = scipy.sparse.csgraph.dijkstra(A, return_predecessors=True)
"""components"""
num, elsIndexes = scipy.sparse.csgraph.connected_components(A)
def getPath(els, i, j):
       res = [j]
       while True:
           v = els[i][j]
           j = v
           if v < 0:
              break
           res.append(v)
       return list(reversed(res))


"""-------------------geometry----------------------------------"""
"""kd-tree"""
points = [[1, 1], [2, 1], [3, 3], [7, 2]]
t = scipy.spatial.KDTree(points)
t.count_neighbors(t, 1)
distance, closestNeighborIndex = t.query([[0,0]])
"""triangulation"""
scipy.spatial.Delaunay(points).simplices #triangles by dots indexes
"""convex hulls"""
scipy.spatial.ConvexHull(points).simplices #lines by indexes



