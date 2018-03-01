
import numpy as np
import copy, re, random, time, datetime, pickle
from scipy.stats import norm


def kpprob(a,b,kpd,std=11,method='cdf',floor='neutral'):
    homeadv = 3.75
    if method == 'cdf':
        # new method using AdjEM
        adjemdiff = ((float(kpd[a]['AdjEM']) - float(kpd[b]['AdjEM']))*
                    (float(kpd[a]['AdjT']) + float(kpd[b]['AdjT']))/200.0)
        if floor == 'neutral':
            bprob = norm.cdf(0,adjemdiff,std)
        elif floor == a:
            bprob = norm.cdf(0,adjemdiff + homeadv,std)
        elif floor == b:
            bprob = norm.cdf(0,adjemdiff - homeadv,std)
        else:
            bprob = norm.cdf(0,adjemdiff,std)
        aprob = 1.0-bprob
        return aprob,bprob
    elif method == 'log5':
        # old method using log5 formula
        return  (a - a*b) / (a + b - 2.0*a*b)

def loadkp(fname):
    keys = 'Rank        Team    Conf    W-L     AdjEM   AdjO    AdjD    AdjT    Luck    SOSAdjEM        SOSOppO SOSOppD NCSOSAdjEM'.split()
    f = open(fname).readlines()
    kpd = {}
    for line in f:
        lsp = line.split('\t')
        if 'Strength' in lsp[0] or 'Rank' in lsp[0]: pass
        else:
            datanorank =  [lsp[i] for i in [0,1,2,3,4,5,7,9,11,13,15,17,19]]
            datanorank[1] = re.split('(\d+)',datanorank[1])[0].strip()
            teamd = dict(zip(keys,datanorank))
            kpd[datanorank[1]] = teamd
    return kpd
    
# def loadkp(fname,time='now'):
#     # read the kenpom data from a text file, kinda hardwired at the present time due to the team rankings in each category
#     # should consider pulling data automatically using curl or someething similar. 
#     if time == 'now':
#         keys = 'Rank	Team	Conf	W-L	AdjEM	AdjO	AdjD	AdjT	Luck	SOSAdjEM	SOSOppO	SOSOppD	NCSOSAdjEM'.split()
#         f = open(fname).readlines()
#         kpd = {}
#         for line in f:
#             lsp = line.split('\t')
#             datanorank =  [lsp[i] for i in [0,1,2,3,4,5,7,9,11,13,15,17,19]]
#             teamd = dict(zip(keys,datanorank))
#             kpd[lsp[1]] = teamd
#         return kpd
#     else:
#         keys = 'Rank	Team	Conf	W-L	AdjEM	AdjO	AdjD	AdjT	Luck	SOSAdjEM	SOSOppO	SOSOppD	NCSOSAdjEM'.split()
#         f = open(fname).readlines()
#         kpd = {}
#         for line in f:
#             lsp = line.split('\t')
#             if lsp[1].split()[-1].isdigit(): lsp[1] = ' '.join(lsp[1].split()[:-1])
#             datanorank =  [lsp[i] for i in [0,1,2,3,4,5,7,9,11,13,15,17,19]]
#             teamd = dict(zip(keys,datanorank))
#             kpd[lsp[1]] = teamd
#         return kpd
    
def loadbrack(fname):
    f = open(fname).read().split('\n')
    b = []
    for line in f:
        team = '.'.join(line.split('.')[1:]).strip()
        b.append(team)
    return b

def snscore(round,seed):
    # scoring method for main pool
    seedx = {1: 1.0, 2: 1.2, 3: 1.4, 4: 1.6, 5: 1.8, 6: 2.0, 7: 2.2, 8: 2.4, 9: 2.6, 10: 2.8, 11: 3.0, 12: 3.4, 13: 4.0, 14: 5.0, 15: 7.0, 16: 10.0}
    roundx = {1: 1.0, 2: 2.0, 3: 3.0, 4: 5.0, 5: 7.0, 6: 10.0}
    score = seedx[int(seed)] * roundx[int(round)]
    return score
  
def readbracket(fname):
    # read in the bracket for whatever year
    teamseadd, bracket = {}, []
    for line in open(fname).readlines():
        lsp = line.split()
        seed, team = int(float(lsp[0])), ' '.join(lsp[1:])
        teamseadd[team] = seed
        bracket.append(team)
    return [bracket], teamseadd

def bracketpredict(bracket, kenpomd, method='seed'):
    if len(bracket[-1]) == 1:
        return 
    else:
        nextround = []
        for i in range(0,len(bracket[-1]),2):
            t1, t2 = bracket[-1][i], bracket[-1][i+1]
            winner = gamepredict(t1, t2, kenpomd, method=method)
            nextround.append(winner)
        bracket.append(nextround)
        bracketpredict(bracket, kenpomd, method=method)

class ncaabracket:
    def __init__(self,bracket):
        self.bracket = bracket
        self.scores = []
    def calcscore(self,truebracket,kenpomd,method=None):
        thisscore = 0
        scorebracket = [[t for t in r if t in truebracket[i]] for i,r in enumerate(self.bracket)]
        for i,round in enumerate(scorebracket[1:]):
            for team in round:
                thisscore += spacenukescore(i+1,kenpomd[team])
        self.scores.append(thisscore)

class bsim:
    def __init__(self,b,kpd):
        self.b = b
        self.kpd = kpd
        self.tsd = seedd(b)
        self.finalbs = []
        self.wind = dict(zip([team for team in b[0]], [[0,0,0,0,0,0] for team in b[0]]))

    def sim(self,method='kpcdf',std=11, scoreweight=None):
        thisb = copy.deepcopy(self.b)
        round = 0
        while len(thisb[-1]) > 1:
            thisround = []
            for i in range(0,len(thisb[-1]),2):
                teama, teamb = thisb[-1][i], thisb[-1][i+1]
                winner = self.gamepredict(teama, teamb, method=method, kpstd=11, scoreweight=scoreweight, round=round+1)
                thisround.append(winner)
                self.wind[winner][round] += 1
            thisb.append(thisround)
            round += 1
        self.finalbs.append(thisb)
    
    def gamepredict(self,team1, team2, method='seed', kpstd=11, scoreweight=None,round=1):
        if method == 'seed':
            if self.kpd[team1] == self.kpd[team2]:  winner = np.random.choice([team1, team2],1)[0]
            elif self.kpd[team1] > self.kpd[team2]: winner = team2
            elif self.kpd[team1] < self.kpd[team2]: winner = team1
        elif method == 'upsets':
            if self.kpd[team1] == self.kpd[team2]:  winner = np.random.choice([team1, team2],1)[0]
            elif self.kpd[team1] > self.kpd[team2]: winner =  team1
            elif self.kpd[team1] < self.kpd[team2]: winner =  team2
        elif method == 'random':
            winner = np.random.choice([team1, team2],1)[0]
        elif method == 'kpcdf':
            aprob,bprob = kpprob(team1,team2,self.kpd,std=kpstd,method='cdf')
            if scoreweight == 'sn':
                eva, evb = aprob*snscore(round,self.tsd[team1]), bprob*snscore(round,self.tsd[team2])
                evt = eva + evb
                aprob, bprob = eva/evt, evb/evt
            else: pass
            winner = np.random.choice([team1,team2],1,p=[aprob,bprob])[0]
        return winner
    
    def simsum(self):
        windat, wind = [], {}
        
        for team in sorted(self.wind.keys()):
            winp = 100*np.array(self.wind[team])/float(len(self.finalbs))
            wind[team] = winp
            windat.append((team, np.sum(winp)))
        windat.sort(key=lambda x: x[1])
        print '{0:20}\t{1}'.format('round', ' '.join(['{0:8}'.format(val) for val in [64,32,16,8,4,2]]))
        for val in list(reversed(windat)):
            team = val[0]
            print '{0:20}\t{1}'.format(team, ' '.join(['{0:8.5f}'.format(val) for val in wind[team]]))
        return windat, wind 

def seedd(bracket):
    d = {}
    bracketorder = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]
    for i in range(0,64,16):
        for j in range(0,16):
            d[bracket[0][i+j]] = bracketorder[j]
    return d


# d = loadkp('./kpdata/kp17.txt', time='past')  
# b = [loadbrack('b17.txt')]
# teamseedd = seedd(b)
# 
# kpb = bsim(b,d)
# for i in range(0,10000):
#     if i%1000 == 0:
#         print i
#     kpb.sim()
# 
# kpb.simsum()

# Pickle results for later
# with open('kpb_standard.pickle', 'wb') as handle:
#     pickle.dump(kpb, handle, protocol=pickle.HIGHEST_PROTOCOL)






