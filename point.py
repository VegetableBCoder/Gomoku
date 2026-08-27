class Point:
    def __init__(self,record,x,y,color):
        self.record=[[0 for i in range(19)]for j in range(19)]
        for i in range(19):
            for j in range(19):
                self.record[i][j]=record[i][j]
        self.same=[0 for i in range(8)]
        self.x=x
        self.y=y
        self.color=color
        self.record[x][y]=color
    def moveWithDirection(self,dir,dis):#参数为方向和距离
        """
        :param dir: 方向
        0：左方   1：左上
        2：上方   3：右上
        4：右方   5：右下
        6：下方   7：左下
        :param dis: 距离
        :return: 序列a[]  a[0]:到达的点,a[1]到达的点之后的点，a[2]，再后面一个点  值：0:空     1/2棋子颜色    3：超出棋盘
        """
        x=self.x
        y=self.y
        if dir==0:
            xx= x
            xxx=x
            xxxx=x
            yy = y - dis
            yyy = yy - 1
            yyyy = yyy - 1
        elif dir==1:
            xx = x - dis
            xxx = xx - 1
            xxxx = xxx - 1
            yy = y - dis
            yyy = yy - 1
            yyyy = yyy - 1
        elif dir==2:
            yy = y
            yyy =yy
            yyyy = yyy
            xx = x - dis
            xxx = xx - 1
            xxxx = xxx - 1
        elif dir==3:
            xx = x - dis
            xxx = xx - 1
            xxxx = xxx - 1
            yy = y + dis
            yyy = yy + 1
            yyyy = yyy + 1
        elif dir==4:
            xx = x
            xxx = x
            xxxx = x
            yy = y +dis
            yyy = yy + 1
            yyyy = yyy + 1
        elif dir==5:
            xx = x + dis
            xxx = xx + 1
            xxxx = xxx + 1
            yy = y + dis
            yyy = yy + 1
            yyyy = yyy + 1
        elif dir==6:
            xx = x + dis
            xxx = xx + 1
            xxxx = xxx + 1
            yy = y
            yyy = yy
            yyyy = yyy
        else:
            xx = x + dis
            xxx = xx + 1
            xxxx = xxx + 1
            yy = y - dis
            yyy = yy - 1
            yyyy = yyy - 1
        ans=[0 for i in range(3)]
        if xx<0 or xx>=19 or yy<0 or yy>=19:
            return [3,3,3]
        ans[0]=self.record[xx][yy]
        if xxx<0 or xxx>=19 or yyy<0 or yyy>=19:
            return [ans[0],3,3]
        ans[1]=self.record[xxx][yyy]
        if xxxx<0 or xxxx>=19 or yyyy<0 or yyyy>=19:
            return [ans[0],ans[1],3]
        ans[2]=self.record[xxxx][yyyy]
        return ans
    def getLeft(self):
        """
        return: 左的与x,y同色的棋子数
        """
        i=self.y
        count=0
        while i>=0:
            if self.record[self.x][i]==self.color:
                count+=1
                i-=1
            else:
                return count
        return count
    def getLeftUp(self):#左上方
        i=self.x
        j=self.y
        count=0
        while i>=0 and j>=0:
            if self.record[i][j]==self.color:
                count+=1
                i-=1
                j-=1
            else:
                return count
        return count
    def getUp(self):
        i=self.x
        count=0
        while i>=0:
            if self.record[i][self.y]==self.color:
                count+=1
                i-=1
            else:
                return count
        return count
    def getRightUp(self):
        i=self.x
        j=self.y
        count=0
        while i>=0 and j<19:
            if self.record[i][j] == self.color:
                count+=1
                i-=1
                j+=1
            else:
                return count
        return count
    def getRight(self):
        i=self.y
        count=0
        while i<19:
            if self.record[self.x][i]==self.color:
                count+=1
                i+=1
            else:
                return count
        return count
    def getRightDown(self):
        i=self.x
        j=self.y
        count=0
        while i<19 and j<19:
            if self.record[i][j]==self.color:
                count+=1
                i+=1
                j+=1
            else:
                return count
        return count
    def getDown(self):
        i=self.x
        count=0
        while i<19:
            if self.record[i][self.y]==self.color:
                count+=1
                i+=1
            else:
                return count
        return count
    def getLeftDown(self):
        i=self.x
        j=self.y
        count=0
        while i<19 and j>=0:
            if self.record[i][j]==self.color:
                count+=1
                i+=1
                j-=1
            else:
                return count
        return count
    def setSame(self):
        self.same[0]=self.getLeft()
        self.same[1]=self.getLeftUp()
        self.same[2]=self.getUp()
        self.same[3]=self.getRightUp()
        self.same[4]=self.getRight()
        self.same[5]=self.getRightDown()
        self.same[6]=self.getDown()
        self.same[7]=self.getLeftDown()
    def getType(self):
        self.setSame()

        same=self.same
        """"
        连五：0, 
        活四：1, 
        死四：2，  
        连活三：3，       
        跳活三：4，      
        连死三：5，     
        跳死三：6
        连活二: 7
        跳活二：8
        连
        死二：9，  跳死二：10
        单子：11
        完全堵死：99
        """
        type=[99 for i in range(4)]
        for i in range(4):
            if same[i]+same[i+4]>5:
                type[i]=0
            elif same[i]+same[i+4]==5:
                ans=self.whichFour(i)
                if ans==0:
                    type[i]=1
                elif ans==1:
                    type[i]=2
                else:
                    type[i]=99
            elif same[i]+same[i+4]==4:
                #print("Three at %s %s"%(self.x,self.y))
                ans=self.whichThree(i)
                if ans==0:
                    type[i]=3
                elif ans==1:
                    type[i]=5
                else:
                    type[i]=99
                #print("type=%s"%type[i])
            elif same[i]+same[i+4]==3:
                ans=self.whichTwo(i)
                if ans==0:
                    type[i]=4
                elif ans==1:
                    type[i]=7
                elif ans==2:
                    type[i]=6
                elif ans==3:
                    type[i]=9
                else:
                    type[i]=99
            elif same[i] + same[i + 4] == 2:
                ans=self.single(i)
                if ans==0:
                    type[i]=8
                elif ans==1:
                    type[i]=10
                elif ans==2:
                    type[i]=11
                else:
                    type[i]=99
        #if self.x==7 and self.y==7:
         #   print("type=",type)
        return type
    def whichFour(self,dir):
        """
        param dir: 方向
        return: 0活四，1死四，2堵死
        """
        dis1=self.same[dir]
        a=self.moveWithDirection(dir,dis1)
        dis2=self.same[dir+4]
        b=self.moveWithDirection(dir+4,dis2)
        if a[0]==0 and b[0]==0:
            return 0
        elif a[0]==0 or b[0]==0:
            return 1
        else:
            return 2
    def whichThree(self,dir):
        """
        dir:方向
        return:0活三，1死三，2堵死
        """
        dis1=self.same[dir]
        dis2=self.same[dir+4]
        a=self.moveWithDirection(dir,dis1)
        b=self.moveWithDirection(dir+4,dis2)
        if a[0]==0 and b[0]==0:
            return 0
        elif a[0]==0 or b[0]==0:
            return 1
        else:
            return 2
    def whichTwo(self,dir):
        #//0:跳活三，1连活二，2跳死三，3.死二，4.堵死
        ans=0
        dis1 = self.same[dir]
        dis2 = self.same[dir + 4]
        a = self.moveWithDirection(dir, dis1)
        b = self.moveWithDirection(dir + 4, dis2)
        if a[0]==0 and b[0]==0:
            if self.isLiveJumpThree(dir):
                ans=0
            else:
                ans=1
        elif a[0]==0 or b[0]==0:
            if self.isDiedJumpThree(dir):
                ans=2
            else:
                ans=3
        else:
            ans=4
        return ans
    def isLiveJumpThree(self,dir):
        """
        True:跳活三，False:连活二
        """
        bool=False
        dis1 = self.same[dir]
        dis2 = self.same[dir + 4]
        a = self.moveWithDirection(dir, dis1)
        b = self.moveWithDirection(dir + 4, dis2)
        if a[1]==self.color:
            if a[2]==0:
                bool=True
        elif b[1]==self.color:
            if b[2]==0:
                bool=True
        return bool
    def isDiedJumpThree(self,dir):
        dis1 = self.same[dir]
        dis2 = self.same[dir + 4]
        a = self.moveWithDirection(dir, dis1)
        b = self.moveWithDirection(dir + 4, dis2)
        bool=False
        if a[0]==0:
            if a[1]==self.color:
                if a[2]==0:
                    bool=True
        elif b[0]==0:
            if b[1]==self.color:
                if b[2]==0:
                    bool=True
        return bool
    def single(self,dir):
        """
        param dir: 方向
        return: 0：跳活二，   1：跳死二，   2：单点，  3：堵死
        """
        dis1 = self.same[dir]
        dis2 = self.same[dir + 4]
        a = self.moveWithDirection(dir, dis1)
        b = self.moveWithDirection(dir + 4, dis2)
        ans=0
        if a[0]==0 and b[0]==0:
            if a[1]==self.color:
                if a[2]==0:
                    ans=0
            elif b[1]==self.color:
                if b[2]==0:
                    ans=0
            else:
                ans=2
        elif a[0]==0 or b[0]==0:
            if a[1]==self.color:
                if a[2]==0:
                    ans=1
                else:
                    ans=2
            elif b[1]==self.color:
                if b[2]==0:
                    ans=1
                else:
                    ans=1
            else:
                ans=2
        else:
            ans=3
        return ans
    def getGrade(self):
        type=self.getType()
        """"
                连五：0, 
                活四：1, 
                死四：2，  
                连活三：3，       
                跳活三：4，      
                连死三：5，     
                跳死三：6
                连活二: 7
                跳活二：8
                连
                死二：9，  跳死二：10
                单子：11
                完全堵死：99
                """
        ans=0
        type.sort()
        if type[0]==0:#连五
            ans=100000
        elif type[0]==1:#双活四
            if type[1]==1:
                ans=80000
            else:
                ans=50000
        elif type[0]==2:#双死四，死四活三
            if type[1]==2 or type[1]==3 or type[1]==4:
                ans=50000
            else:
                ans=20000
        elif type[0]==3:#连活三
            if type[1]==3 or type[1]==4:#双活三
                ans=30000
            else:#连活三
                ans=18000
        elif type[0]==4:
            if type[1]==4:
                ans=30000
            else:#跳活三
                ans=16000
        elif type[0]==5:
            if type[1]==5 or type[1]==6:
                ans=7000
            else:
                ans=3000
        elif type[0]==6:
            if type[1]==6:
                ans=7000
            else:
                ans=3000
        elif type[0]==7:
            if type[1]==7 or type[1]==8:
                ans=6000
            else:
                ans=2000
        elif type[0]==8:
            if type[1]==8:
                ans=6000
            else:
                ans=1500
        elif type[0]==9:
            ans=1000
        elif type[0]==10:
            ans=500
        elif type[0]==11:
            ans=100
        else:
            ans=0

        return ans
    def win(self):
        self.setSame()
        for i in range(4):
            if self.same[i]+self.same[i+4]>5:
                return True
        return False