
import numpy as np
import random, copy, time
from scipy.stats import norm

def prob(a,b,kpd,std=11,homea=3):
    adjemdiff = ((float(kpd[a]['AdjEM']) - float(kpd[b]['AdjEM']))*
                (float(kpd[a]['AdjT']) + float(kpd[b]['AdjT']))/200.0) - homea
    bprob = norm.cdf(0,adjemdiff,std)
    aprob = 1.0-bprob
    return aprob,bprob
    
    
def loadkp(fname):
    keys = 'Rank	Team	Conf	W-L	AdjEM	AdjO	AdjD	AdjT	Luck	SOSAdjEM	SOSOppO	SOSOppD	NCSOSAdjEM'.split()
    f = open(fname).readlines()
    kpd = {}
    for line in f:
        lsp = line.split('\t')
        datanorank =  [lsp[i] for i in [0,1,2,3,4,5,7,9,11,13,15,17,19]]
        teamd = dict(zip(keys,datanorank))
        kpd[lsp[1]] = teamd
    return kpd
    
def loadbrack(fname):
    f = open(fname).read().split('\n')
    b = []
    for line in f:
        team = '.'.join(line.split('.')[1:]).strip()
        b.append(team)
    return b
    
class bsim:
    def __init__(self,b,kpd):
        self.b = b
        self.kpd = kpd
        self.finalbs = []
        self.wind = dict(zip([team for team in b[0]], [[0,0,0,0,0,0] for team in b[0]]))

    def sim(self,method='kp',std=11):
        thisb = copy.deepcopy(self.b)
        round = 0
        while len(thisb[-1]) > 1:
            thisround = []
            for i in range(0,len(thisb[-1]),2):
                teama, teamb = thisb[-1][i], thisb[-1][i+1]
                ap, bp = prob(teama, teamb, self.kpd, std=std)
                rn = random.random()
                if rn <= ap: 
                    thisround.append(teama)
                    self.wind[teama][round] += 1
                else:  
                    thisround.append(teamb)
                    self.wind[teamb][round] += 1
            thisb.append(thisround)
            round += 1
        self.finalbs.append(thisb)
    
    def simsum(self):
        for team in sorted(self.wind.keys()):
            winp = 100*np.array(self.wind[team])/float(len(self.finalbs))
            print '{0:20}\t{1}'.format(team, ' '.join(['{0:8.5f}'.format(val) for val in winp]))
                    
        
# d = loadkp('kp17.txt')  
# b = [loadbrack('b16.txt')]
# 
# kpb = bsim(b,d)
# t0 = time.time()
# for i in range(0,100000):
#     if i%10000 == 0:
#         print i
#     kpb.sim()
# print time.time() - t0
# 
# kpb.simsum()