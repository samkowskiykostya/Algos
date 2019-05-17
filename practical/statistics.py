from scipy import stats as s
from scipy import special as ss
import numpy as np
from statsmodels.stats import multicomp, anova
import pandas as pd

df = pd.read_csv('/datasets/genetherapy.csv')
df2 = pd.read_csv('/datasets/atherosclerosis.csv')

"""quartiles"""
np.percentile([1,2,3], 50)

"""t-distribution"""
stat, pval = s.ttest_1samp([1,2,3], 5)
s.norm.rvs(size=1000)

"""compare distributions"""
stat, p = s.levene([1,2,3], [2,3,4]) #test equal variance hypotesis.
stat, p = s.ttest_ind([1,2,3], [2,2,2]) #test if average equal. 1 - equal
stat, pval = s.shapiro([1,2,300]) #test norm. >.05 - norm

"""dispersion anlysis"""
f, p = s.f_oneway([1,2,3,4], (2,3,4,5)) #test equal M. if P<.05 => at least 2 groups differ
#df usage
gg = df.groupby('Therapy')
s.f_oneway(*[gg.get_group(g)['expr'].values for g in gg.groups.keys()])
df.groupby('Therapy').boxplot()

r = multicomp.pairwise_tukeyhsd(df['expr'], df['Therapy']) #tukey correction for multivariate dispersion analysis
r.plot_simultaneous()
print(r.summary())

mod = ols('expr ~ age*dose', data=df2).fit() #Multi-factor dispersion analysis
anova.anova_lm(mod, typ=2)

"""Correlation"""
c, p = s.pearsonr([1,2,3], [5,6,7])
c, p = s.spearmanr([4,5,2,3,1], [2,1,4,3,100]) #ranging values, works fine for ejections

"""Regression"""
s.stats.linregress([1,2,3],[4,5,6]) #slope, intercept, rvalue

"""Nominative variables (coin)"""
s.chisquare([795,705], [750,750]) #got, expected, dfx (k - 1 - dfx)=> chi**2, p. H0: no relation between ars
chi2, p, dof, exp = s.chi2_contingency([[15,9],[11,6]],  correction=True) #contingency matrix (coin throw case), f > 10. Yates correction 5 < f < 10
s.fisher_exact([[18,7],[6,13]]) #Small tables, f < 5

"""Combinatorics"""
ss.comb(1000,5) #N things taken k at a time
sm.perm(1000,5) #permutations of k in N