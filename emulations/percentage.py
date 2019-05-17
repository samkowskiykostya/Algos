import functools

def compound(years=10, start=10, percent=10):
    sum = start * (1 + percent / 100) ** years
    yearly = (sum - start) / years
    print(years, 'yrs',
          start, 'start',
          percent, '%',
          sum,
          'earn', sum - start,
          'yearly', '{0:.2f}'.format(yearly),
          'monthly', '{0:.2f}'.format(yearly / 12))
    return  sum

def compoundAdd(years = 10, start=10, inc = 10, percent = 10):
    sum = functools.reduce(lambda res, x: res*(1 + float(percent)/100) + x, [start] + [inc]*years)
    yearly = (sum - start) / years
    print(years, 'yrs',
          start, 'start',
          inc, 'inc',
          percent, '%',
          sum,
          'earn', sum - start,
          'yearly', '{0:.2f}'.format(yearly),
          'monthly', '{0:.2f}'.format(yearly / 12))
    return sum


compound(10, 100, 9)
compound(5, 50, 9)
print()
compoundAdd(10, 10, 10)
compoundAdd(10, 10, 15)
compoundAdd(10, 15, 15)
compoundAdd(20, 10, 10)
compoundAdd(20, 10, 15)
compoundAdd(20, 15, 10)
compoundAdd(20, 15, 15)
compoundAdd(5, 50, 1, 9)