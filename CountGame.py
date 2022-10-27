from multiprocessing.dummy import active_children
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


# 只需要调用该类即可
class GameController(object):
    
    def __init__(self):
        # player1是电脑选手, player2是人类玩家
        self.player1, self.player2 = self.start()
        self.numround = 0  # 标记游戏进行的轮数
        self.isEnd = False  # 标记游戏是否结束
        self.is_begin = False  # 标记游戏是否开始
        self.winner = ''  # 记录获得胜利的玩家
        self.activate_player   = None  # 当前正在活跃的选手
        self.inactivate_player = None  # 当前没有活跃的选手
    
    def start(self):
        # 均等概率初始化，决定谁先出。
        c1,c2=random.uniform(0,1),random.uniform(0,1)
        return AiPlayer(c1>=c2,"AiPlayer"), Player(c1<c2,"Player")
    
    def slient_start(self, myhd, ophd):
        '''
        myhd:玩家准备用于触碰的手
        ophd:电脑准备用于触碰的手
        return:游戏进行状态、玩家的数据和电脑的数据 例：True,[[[玩家左手数字,玩家右手数字],[技能1,技能2,技能3]],[[电脑左手数字,电脑右手数字],[技能1,技能2,技能3]]]、False,获胜者名称(技能顺序：箭、弓、盾)
        '''
        # 检测是否是第一次启动
        if not self.is_begin:
            self.activate_player   = self.player2
            self.inactivate_player = self.player1
            # if self.player1.stats is True:
            #     self.activate_player   = self.player1
            #     self.inactivate_player = self.player2
            # else:
            #     self.activate_player   = self.player2
            #     self.inactivate_player = self.player1
            # self.is_begin = True
        # 检测游戏是否结束
        if self.numround >= 200 or self.isEnd:
            self.isEnd = True
            print("游戏已结束")
            return False, self.winner
        self.numround+=1
        #print(f"第{self.numround}回合，请{self.activate_player.name}行动")
        # main 对于玩家，此处直接采用从前端界面传回的数据；对于电脑，继续沿用之前的方法
        for index in range(2):
            if self.activate_player == self.player1:
                myhd,ophd = self.activate_player.combine(self.inactivate_player,self.inactivate_player.hl,self.inactivate_player.hr)
        
            self.process(self.activate_player, self.inactivate_player, myhd, ophd)
            # 交换当前活跃的选手和静默的选手
            tmp = self.activate_player
            self.activate_player = self.inactivate_player
            self.inactivate_player = tmp
        print('\r游戏结束？',self.isEnd, end='')
        return True, [
            [
                [self.player2.hl, self.player2.hr],
                [self.player2.numArrow, self.player2.numBow, self.player2.numShield],
            ],
            [
                [self.player1.hl, self.player1.hr],
                [self.player1.numArrow, self.player1.numBow, self.player1.numShield],
            ],
        ]

    # bug 因为较大幅度的修改参数，可能该函数的基本功能已经失效
    def GameStart(self):
        print("游戏开始")
        if self.player1.stats is True:
            activePlayer=self.player1
            inactivePlayer=self.player2
        else:
            activePlayer=self.player2
            inactivePlayer=self.player1
            
        print(f"根据抽签结果，请{activePlayer.name}先行动")
        
        while not self.isEnd:
            if self.numround >= 200:
                break
            self.numround+=1
            print(f"第{self.numround}回合，请{activePlayer.name}行动")
            myhd,ophd=activePlayer.combine(inactivePlayer,inactivePlayer.hl,inactivePlayer.hr)
            self.process(activePlayer,inactivePlayer,myhd,ophd)
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
        
        
        for idx in [myhd]:
            if (acpnum[idx]==5)*(past[idx]!=5):
                activePlayer.numShield+=1
                activePlayer.score +=1
                # acpnum[acpnum.index(num)]=1
            elif acpnum[idx]==3*(past[idx]!=3):
                activePlayer.numArrow+=1
                activePlayer.score +=1
                # acpnum[acpnum.index(num)]=1
            elif acpnum[idx]==6*(past[idx]!=6):
                activePlayer.numBow+=1
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
            print(f"{activePlayer.name}赢了!")
            self.winner = activePlayer.name
            activePlayer.score += 10
            self.isEnd=True
        else:
            # main 停止打印状态，改为前端显示
            pass
            # print(f"第{self.numround}回合,结果为：")
            # print(f"当前{self.player1.name}手上的数字左手：{self.player1.hl},右手：{self.player1.hr}。")
            # print(f"弓个数{self.player1.numBow}，箭个数{self.player1.numArrow}，盾个数{self.player1.numShield}")
            # print(f"当前{self.player2.name}手上的数字左手：{self.player2.hl},右手：{self.player2.hr}。")
            # print(f"弓个数{self.player2.numBow}，箭个数{self.player2.numArrow}，盾个数{self.player2.numShield}")
        


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