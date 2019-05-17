import sympy as s
from scipy import optimize
import math
from collections import OrderedDict
from matplotlib import pyplot as plt


def gain(n, ticket = 0):
    day_price = 27
    eat_price = 15
    return (2800 * 2 / 21) * ((n-3)//7*5 + 1 + min((n-3)%7, 5)) - n * day_price * 1.24 - n * eat_price - ticket

prices = {14:99,15:126,16:156,17:126,18:149,19:149,20:136,21:169,22:193,23:182,24:111}
m = OrderedDict(sorted(prices.items(), key=lambda t: t[0]))

print(7, gain(7) - 515)
print(10, gain(10) - 548)
print(16, gain(16) - 537)
print(17, gain(17) - 633)
print(18, gain(18) - 649)
print(19, gain(19) - 682)
print(20, gain(20) - 659)
print(21, gain(21) - 665)
print(22, gain(22) - 671)


plt.plot(list(m.keys()), [gain(k,v) for k, v in m.items()])
plt.show()