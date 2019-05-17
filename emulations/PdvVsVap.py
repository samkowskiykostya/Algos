import functools, numpy as np
pdv = 0.2
vt = 0.08
price = 100
gain = 0.15

def vtp(vt): return price * ((1 + gain) * (1 + vt)) ** link
def pdvp(pdv): return price * (1 + gain * (1 + pdv))**link

# print('vap', vapp(vap))
# print('pdv', pdvp(pdv))

print('vt in us(7-15%) vs pdv in ua(20%)')
for link in range(2, 10):
    print('link: ', link, ['%.2f' % (vtp(x) / pdvp(pdv)) for x in np.arange(0.07, 0.16, 0.01)])
