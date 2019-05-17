from util.plots import *
import math
import numpy as np

def plotLine(a,b): utilPlotFunc(lambda x: a*x+b, -5,5)

def lineByDots(x1,y1,x2,y2): return (y2-y1)/(x2-x1), (x2*y1 - x1*y2) / (x2-x1)

def intersect(a,b,c,d): return (d-b)/(a-c), (a*d-b*c)/(a-c)

def ortogonal(a,b,x,y): return -1/a, (a**2 + 1)*x/a + b

def middleLine(x1,y1,x2,y2): return (x1+x2)/2, (y1+y2)/2

def circleCenter(x1,y1,x2,y2,x3,y3):
    return intersect(*ortogonal(*lineByDots(x1,y1,x2,y2), *middleLine(x1,y1,x2,y2)), *ortogonal(*lineByDots(x2,y2,x3,y3), *middleLine(x2,y2,x3,y3)))

def distance(x1,y1,x2,y2): return math.sqrt((x1-x2)**2 + (y1-y2)**2)

def reflection(k,l, x1,y1):
    xi,yi = (x1 + k*(y1-l))/(k**2 + 1), (k**2*y1 + k*x1 + l)/(k**2 + 1)
    return xi + (xi - x1), yi + (yi - y1)

def rotate(cx,cy,px,py, th):
    tan = math.tan(th)
    k = (py + tan*px) / (px - py*tan)
    x = math.sqrt((px - cx)**2 + (py - cy)**2) * math.cos(math.atan(k))
    return x + cx, k*x + cy

def distance(a, b): return math.sqrt(sum((a-b)**2))

"""arg dots - np arrays with coordinates."""
def distanceLineToDot(l1, l2, d, onlyNormToL1L2 = False):
    """
    Check that d proection on l1-l2 line is between l1 and l2
    D - proection of d
    a - angle between l1-d and l1l2
    b - angle between l2-d and l1l2
    """
    def cos(angleDot, d1, d2):
        return np.dot(list(d1 - angleDot), list(d2 - angleDot)) / distance(d1, angleDot) / distance(d2, angleDot)
    cosa = cos(l1, d, l2)
    sina = math.sqrt(1 - cosa ** 2)
    h = distance(d, l1) * sina
    if onlyNormToL1L2:
        cosb = cos(l2, d, l1)
        l1D = cosa * distance(l1, d)
        l2D = cosb * distance(l2, d)
        if distance(l1, l2) < max(l1D, l2D):
            return np.inf
    return h