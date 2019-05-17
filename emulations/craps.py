import random, csv, pyprind
from scipy import optimize

class Craps:
    def __init__(self, initialSum):
        self.mym = initialSum
        self.betfactor = {'pass':1, 'notpass':1, '6':1.25, '8':1.25, '5':1.6, '9':1.6, '4':2.2, '10':2.2, 'h6':9, 'h10':7, 'h8':9, 'h4':7, '3':15, '2':30, '12':30, '11':15, 'cany':7, '7':4}
        self.curbets = {}
        self.historybet = {}
        self.throws = 0
        self.on = None
    def doWin(self, bet):
        if bet in self.curbets:
            self.mym += (self.betfactor[bet] + 1) * self.curbets[bet]
            del self.curbets[bet]
    def doLost(self, bet):
        if bet in self.curbets:
            del self.curbets[bet]
    def bet(self, bet, m, check=True):
        if not self.on and not bet in ['pass', 'notpass'] or str(self.on) == bet:
            return
        if check and bet in self.curbets:
            return
        if self.mym >= m:
            self.curbets[bet] = m
            self.historybet[bet] = m
            self.mym -= m
    def withdrawBet(self, bet):
        if bet in self.curbets:
            self.mym += self.curbets[bet]
            del self.curbets[bet]
    def withdrawAllBets(self):
        for bet in self.betfactor:
            if bet not in ['pass', 'notpass']:
                self.withdrawBet(bet)
    def cycle(self):
        a = random.randint(1, 6)
        b = random.randint(1, 6)
        c = a + b
        self.throws += 1
        if self.on is None:
            if c in [7, 11]:
                self.doWin('pass')
                self.doLost('notpass')
                self.throws = 0
            elif c in [12, 2]:
                self.doLost('pass')
                self.throws = 0
            else:
                self.on = c
        else:
            if c == 7:
                self.doWin('notpass')
                self.curbets.clear()
                self.historybet.clear()
                self.on = None
            if c == self.on:
                self.doWin('pass')
            if 3 >= c >= 11:
                self.doWin('cany')
            self.doWin(str(c))
            if a == b:
                self.doWin('h' + str(c))
            self.doLost('h' + str(c))
            for b in ['2','3','11','12','cany']:
                self.doLost(b)
    def emulate(self, strategy, games=30):
        if strategy:
            i = 0
            while self.mym > 0 and i < games:
                strategy.strategy(self)
                self.cycle()
                if not self.on:
                    i += 1
            if i == games:
                return True, self.mym
            else:
                return False, i

def boundaryBet(bet, m, nonZero=False):
    mmin = 1
    if bet in ['pass', 'notpass', '6', '8', '5', '9', '4', '10']:
        mmin = 10
    if m < mmin:
        if nonZero:
            return mmin
        return 0
    return m

class Strategy:
    def __init__(self, vals, step=1):
        self.values = vals
        self.step = step
    def strategy(self, craps):
        def doBet(bet, m):
            m = boundaryBet(bet, m)
            if bet == 0:
                return
            if bet in craps.historybet:
                m = boundaryBet(bet, craps.historybet[bet] - self.step, nonZero=True)
            craps.bet(bet, m)
        if not craps.on:
            if 'pass' in self.values:
                doBet('pass', self.values['pass'])
            if 'notpass' in self.values:
                doBet('notpass', self.values['notpass'])
        else:
            if craps.throws == 5:
                craps.withdrawAllBets()
            else:
                for k,v in self.values.items():
                    if k not in ['pass', 'notpass']:
                        doBet(k, v)

random.seed(13)
N = 1000
startWith = 200

def findBestStrategies():
    earns = []
    step = 3
    # cany = 5
    bar = pyprind.ProgBar(214375, track_time=True, title='Craps data collection')
    with open('craps_data.csv', 'w') as file:
        writer = csv.writer(file, delimiter=',', lineterminator='\n')
        writer.writerow(['6','8','5','9','4','10','h4','h6','h8','h10','pass','notpass','cany','7','2','3','11','12','EARN'])
        for s68 in range(8, 22, step):
            for s59 in range(8, 22, step):
                for s410 in range(8, 22, step):
                    for pnp in range(8, 22, step):
                        for h in range(0, 6, 1):
                            for c in range(0, 6, 1):
                                for cany in range(0, 6, 1):
                                    # for c7 in range(0, 11, step):
                                    vals = {'6':s68, '8':s68, '5':s59, '9':s59, '4':s410,'10':s410, 'h4':h, 'h6':h, 'h8':h, 'h10':h, 'pass':pnp, 'notpass':0, 'cany': cany, '7': 0, '2':c, '3':c, '11':c, '12':c}
                                    s = Strategy(vals, step)
                                    totalWin = totalWinN = 0
                                    for _ in range(N):
                                        craps = Craps(startWith)
                                        win, sum = craps.emulate(s)
                                        totalWinN += win
                                        totalWin += win * sum
                                    if totalWinN == 0:
                                        earn = -startWith
                                    else:
                                        earn = (float(totalWinN) / N) * (totalWin / totalWinN - startWith)
                                    # earns.append([earn, {k:v for k,v in vals.items() if v not in [0,8]}, float(totalWinN) / N, totalWin / totalWinN - startWith])
                                    writer.writerow([s68, s68, s59, s59, s410, s410, h, h, h, h, pnp, 0, cany, 0, c, c, c, c, earn])
                                    bar.update()
        file.close()
    print(bar)
    # earns.sort(key=lambda x: x[0], reverse=True)
    # print(earns[:20])

def doTest(vals):
    s = Strategy(vals)
    totalWin = totalWinN = 0
    for _ in range(N):
        craps = Craps(startWith)
        win, sum = craps.emulate(s)
        totalWinN += win
        totalWin += win * sum
    return totalWin, totalWinN, float(totalWinN) / N, totalWin / totalWinN - startWith

def runStrategy():
    totalWin, totalWinN, rate, avgWin = doTest({'10': 20, 'h8': 1, 'h10': 1, 'h6': 1, 'h4': 1, '5': 20, '4': 20, '9': 20})
    print('Started with:', startWith, 'Won games:', rate, 'Average Earn:', avgWin)

def findMinFunc():
    def f(v):
        vals = dict(zip(betNames, v))
        print(vals)
        vals = {k:boundaryBet(k,v) for k,v in vals.items()}
        totalWin, totalWinN, rate, avgWin = doTest(vals)
        rate = 1 / rate
        if rate < 0: rate *= -1
        return -rate * avgWin
    betNames = ['6', '8', '5', '9', '4', '10', 'h4', 'h6', 'h8', 'h10', 'pass', 'notpass', 'cany', '7', '2', '3', '11', '12']
    x, fm, d = optimize.fmin_l_bfgs_b(f,
                                      [15] * 6 + [3] * 4 + [15] * 2 + [3] * 6,
                                      bounds=[[9, 21]] * 6 + [[0,6]] * 4 + [[9, 21]] * 2 + [[0,6]] * 6,
                                      epsilon=1,
                                      approx_grad=True, iprint=1)
    print(zip(betNames, x), fm)
    # print({betNames[i]:boundaryBet(betNames[i], m) for i,m in enumerate(fmin(f, [5]*18)) if boundaryBet(betNames[i], m)})
# runStrategy()
# findBestStrategies()
findMinFunc()
