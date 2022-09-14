from copy import copy
from logging.handlers import MemoryHandler
import random
import numpy as np
import matplotlib.pyplot as plt


class Player(object):
    
    def __init__(self,stats,name):
        self.hl=1
        self.hr=1
        self.stats=stats
        self.numBow=0
        self.numShield=0
        self.numArrow=0
        self.name=name
        self.score=0
    
    def combine(self,pl,pr, args):
        print(f"当前你手上的数字，左手{self.hl},右手{self.hr};对方手上的数字，左手{pl.hl}，右手{pl.hr}")
        
        myhd=input("请选择要去触碰对方的手,0表示左手，1表示右手：")
        myhd=self.verify("请选择要去触碰对方的手,0表示左手，1表示右手：",myhd)
        
           
        ophd=input("请选择触碰对方的手,0表示左手，1表示右手：")
        ophd=self.verify("请选择触碰对方的手,0表示左手，1表示右手：",ophd)

        return (int(myhd),int(ophd))
    
    def verify(self,text,val,acpval=('1','0')):
        #验证输入是否合法
        nl=True
        while nl:
            if val in acpval:
                nl=False
            else:
                val=input("错误，请输入。"+"或".join(tuple(acpval))+text) 
        return val

class RandomPlayer(Player):
    
    def __init__(self,stats,name):
        # super.__init__(stats,name)
        self.hl=1
        self.hr=1
        self.numBow=0
        self.numShield=0
        self.numArrow=0
        self.score=0
    
        
        self.stats=stats
        self.name=name
    
    def combine(self, *argv):
        return (random.choice([0,1]),random.choice([0,1]))


class AiPlayer(Player):
    
    def __init__(self,stats,name):
        # super.__init__(stats,name)
        self.hl=1
        self.hr=1
        self.numBow=0
        self.numShield=0
        self.numArrow=0
        self.score=0
    
        
        self.stats=stats
        self.name=name
    
    def getReward(self, *argv):

        inplayer, myhd, ophd = argv
        mNum = [self.hl, self.hr]
        oNum = [inplayer.hl, inplayer.hr]
        past = mNum.copy()
        mNum[myhd] = (mNum[myhd] + oNum[ophd]) % 10
        if mNum[myhd] == 0:
            mNum[myhd] = 1
        score = 0

        for idx in range(2):
            if (mNum[idx]==5)*(past[idx]!=5):
                score +=1
            elif (mNum[idx]==3)*(past[idx]!=3):
                score +=1
            elif (mNum[idx]==6)*(past[idx]!=6):
                score +=1

        if (self.numBow>=1)*(self.numArrow>=1):
            score += 3
            if inplayer.numShield <=1:
                score += 10
        
        return score


    
    def combine(self,inplayer,*args):
        hl, hr = random.choice([0,1]), random.choice([0,1])
        choices = [(m, n) for m in range(2) for n in range(2)]
        scores = np.array([self.getReward(inplayer,m, n) for m in range(2) for n in range(2)])
        return choices[scores.argmax()]



class GameController(object):
    
    def __init__(self):
        self.player1,self.player2=self.start()
        self.numround=0
        self.isEnd=False
    
    def start(self):
        # 均等概率初始化，决定谁先出。
        c1,c2=random.uniform(0,1),random.uniform(0,1)
        return AiPlayer(c1>=c2,"AiPlayer"),Player(c1<c2,"Player")
    
        
    def GameStart(self):
        # 抽签
        print("游戏开始")
        if self.player1.stats is True:
            activePlayer=self.player1
            inactivePlayer=self.player2
        else:
            activePlayer=self.player2
            inactivePlayer=self.player1
            
        print(f"根据抽签结果，请{activePlayer.name}先行动")
        # 控制游戏开始
        while not self.isEnd:
            # 设置最大回合数
            if self.numround >= 200:
                break
            self.numround+=1
            print(f"第{self.numround}回合，请{activePlayer.name}行动")
            # 活跃玩家行动，输入不活跃玩家对象，左手数字，右手数字
            # myhd就是当前玩家选择去碰的手 ophd是对方选择碰的手
            myhd,ophd=activePlayer.combine(inactivePlayer)
            # 处理结果， 返回所有信息
            self.process(activePlayer,inactivePlayer,myhd,ophd)
            if self.isEnd == True:
                # 如果某个玩家已经赢了，那么isEnd就会变成True。self.process 返回的是 赢的玩家名字,输得玩家名字
                pass
            else:
                # 如果没有玩家赢，那么self.process 返回的是 ai玩家的参数，真实玩家的参数
                pass
            tmp=activePlayer
            activePlayer=inactivePlayer
            inactivePlayer=tmp

        print("游戏结束",activePlayer.name,": ",activePlayer.score,inactivePlayer.name,": ",inactivePlayer.score)
        return [activePlayer, inactivePlayer][np.argmax([activePlayer.score, inactivePlayer.score])],self.numround
    
    

    def process(self,activePlayer,inactivePlayer,myhd,ophd):
        acpnum=[activePlayer.hl,activePlayer.hr]
        past=acpnum.copy()
        inacpnum=[inactivePlayer.hl,inactivePlayer.hr]
        acpnum[myhd]=(acpnum[myhd]+inacpnum[ophd])%10
        # 满10归1，避免死局出现
        if acpnum[myhd]==0:
            acpnum[myhd]=1
        
        
        for idx in range(2):
            if (acpnum[idx]==5)*(past[idx]!=5):
                activePlayer.numShield+=1
                activePlayer.score +=1
                # acpnum[acpnum.index(num)]=1
            elif acpnum[idx]==3*(past[idx]!=3):
                activePlayer.numBow+=1
                activePlayer.score +=1
                # acpnum[acpnum.index(num)]=1
            elif acpnum[idx]==6*(past[idx]!=6):
                activePlayer.numArrow+=1
                activePlayer.score +=1
                # acpnum[acpnum.index(num)]=1

        activePlayer.hl=acpnum[0]
        activePlayer.hr=acpnum[1]

        if (activePlayer.numBow>=1)*(activePlayer.numArrow>=1):
            activePlayer.numBow-=1
            activePlayer.numArrow-=1
            activePlayer.score += 3
            inactivePlayer.numShield-=1
        

        if inactivePlayer.numShield<0:
            # print(f"{activePlayer.name}赢了!")
            activePlayer.score += 10
            self.isEnd=True
            return activePlayer.name, inactivePlayer.name
        else:
            # player 1 是Ai player2 是玩家
            # print(f"第{self.numround}回合,结果为：")
            # print(f"当前{self.player1.name}手上的数字左手：{self.player1.hl},右手：{self.player1.hr}。")
            # print(f"弓个数{self.player1.numBow}，箭个数{self.player1.numArrow}，盾个数{self.player1.numShield}")
            # print(f"当前{self.player2.name}手上的数字左手：{self.player2.hl},右手：{self.player2.hr}。")
            # print(f"弓个数{self.player2.numBow}，箭个数{self.player2.numArrow}，盾个数{self.player2.numShield}")
            return (self.player1.hl, self.player1.hr, self.player1.numBow, self.player1.numArrow, self.player1.numShield),(self.player2.hl, self.player2.hr, self.player2.numBow, self.player2.numArrow, self.player2.numShield)
        


if __name__=="__main__":
    g = GameController()
    g.GameStart()
#   from CountGame import GameController,AiPlayer

# 贪心 Vs 随机

    # Nloop = 1000

    # numAi = 0
    # ls = []

    # for idx in range(Nloop):
    #     gm=GameController()
    #     winner, nound= gm.GameStart()
    #     ls.append(nound)
    #     if winner.name == "AiPlayer":
    #         numAi+=1

    # print(f"共{Nloop}回合， Ai玩家赢了{numAi}次")
    # plt.hist(nound)
    # plt.show()