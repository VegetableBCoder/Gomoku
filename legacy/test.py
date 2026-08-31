import tkinter as tk
import tkinter.messagebox
import PIL
from PIL import Image,ImageTk
from tkinter import *
import tkinter.filedialog as fileInput
import os
from point import *
import pickle
root=tkinter.Tk()
root.title("瓜皮五子棋")
root.geometry("720x780")

photo=PIL.Image.open("棋盘.png","r")
background=ImageTk.PhotoImage(photo)

white=PIL.Image.open("white.png","r").resize((24,24))
white=ImageTk.PhotoImage(white)

black=PIL.Image.open("black.png","r").resize((24,24))
black=ImageTk.PhotoImage(black)
position =[[(0,0) for i in range(19)]for j in range(19)]
record=[[0 for i in range(19)]for j in range(19)]#0空 1玩家，2电脑
rect=None
turn=""
v=StringVar()
v.set("电脑先手")
turn = "per"
def cacu_pos():
	position[0][0]=(36,37)
	for i in range(1,19):
		position[0][i]=(position[0][0][0]+i*36,position[0][0][1])
	for i in range(1,19):
		for j in range(19):
			position[i][j]=(position[0][j][0],position[0][j][1]+i*36)
def show_win(turn):
    if turn=="per":
        tkinter.messagebox.showinfo("恭喜","你赢了！")
        cv.pack()
        cv.unbind("<Button-1>")
    else:
        tkinter.messagebox.showinfo("提示", "电脑胜利！")
        cv.pack()
        cv.unbind("<Button-1>")
def back():
    cv.bind("<Button-1>",callback)
    record[x0][y0]=0
    record[x1][y1]=00
    resetxy()
    loadByRecord(record)
def resetxy():
    x0=0
    x1=0
    y0=0
    y1=0
def restart():
    resetxy()
    cv.bind("<Button-1>", callback)
    for i in range(19):
        for j in range(19):
            record[i][j]=0
    loadByRecord(record)
    if v.get()=="电脑先手":
        turn = "per"
        draw_circle(position[9][9], "com")
        record[9][9] = 2
    else:
        turn="per"
def save():
    filename = fileInput.askdirectory()
    if os.path.isdir(filename):
        filename=filename+"\\chess.msave"
    elif os.path.isfile(filename):
        pass
    else:
        tkinter.messagebox.showerror("错误", "文件路径不正确")
        return 0
    w_file=open(filename,"wb")
    pickle.dump(record,w_file)
    w_file.close()

def loadByRecord(record):
    load_bg()
    for i in range(19):
        for j in range(19):
            if record[i][j]==1:
                draw_circle(position[i][j],"per")
            elif record[i][j]==2:
                draw_circle(position[i][j], "com")
def load():
    resetxy()
    filename=fileInput.askopenfilename()
    if os.path.isfile(filename):
        r_file = open(filename, "rb")
        list0 = pickle.load(r_file)
        for i in range(19):
            for j in range(19):
                record[i][j]=list0[i][j]
        loadByRecord(list0)
    else:
        tkinter.messagebox.showerror("错误","文件路径不正确")
        restart()
def setTurn():
    if v.get()=="电脑先手":
        v.set("玩家先手")
    else:
        v.set("电脑先手")
    restart()
def load_bg():#加载棋盘
    cv.create_image(361,360,image=background)
    cv.pack()
def callback(event):
    x=event.x
    y=event.y
    global turn
    if turn=="per":
        if per_draw(x,y)==1:
            test = Point(record, x1, y1, 1)
            turn="com"
            if test.getGrade()==100000:
                show_win("per")
            else:#让电脑下一步棋
                com_draw()
                turn="per"
                test = Point(record, x1, y1, 2)

                if test.getGrade() == 100000:
                    show_win("com")


def drawrect(pos):
    global rect
    if rect!=None:
        cv.delete(rect)
    rect=cv.create_rectangle(pos[0]-14,pos[1]-14,pos[0]+14,pos[1]+14,outline="black")
    cv.pack()
cacu_pos()
x0=0
y0=0
x1=0
y1=0
cv=Canvas(root,width=722,height=720)
cv.bind("<Button-1>", callback)
load_bg()


def draw_circle(pos,turn):
    if turn=="com":
        img=black
    else:
        img=white
    cv.create_image(pos,image=img)
    cv.pack()

def per_draw(x,y):
    global x0, x1, y0, y1
    for i in range(19):
        for j in range(19):
            if (x - position[i][j][0] <= 10 and position[i][j][0] - x <= 10) and (
                    y - position[i][j][1] <= 10 and position[i][j][1] - y <= 10):
                if record[i][j] == 0:
                    draw_circle(position[i][j],"per")
                    record[i][j]=1
                    x0=x1
                    y0=y1
                    x1=i
                    y1=j
                    return 1
    return 0
def com_draw():
    global x0,x1,y0,y1
    max1=0
    max2=0
    x=0
    y=0
    p=None
    q=None
    pGrade=0
    qGrade=0
    for i in range(19):
        for j in range(19):
            if record[i][j]==0:
                p = Point(record,i,j,2)
                q = Point(record,i,j,1)
                pGrade=p.getGrade()
                qGrade=q.getGrade()
                grade=1.05*pGrade+qGrade
                if grade>max1+max2 or (grade==max1+max2 and abs(i-9)+abs(j-9)<abs(x-9)+abs(y-9)):
                    max1=pGrade
                    max2=qGrade
                    x=i
                    y=j
    print("Draw Black at %s %s"%(x,y))
    draw_circle(position[x][y],"com")
    drawrect(position[x][y])
    record[x][y] = 2
    x0 = x1
    y0 = y1
    x1 = x
    y1 = y

button_back=Button(root,text="悔棋",command=back)
button_restart=Button(root,text="重新开始",command=restart)
button_save=Button(root,text="保存棋局",command=save)
button_load=Button(root,text="加载棋局",command=load)
button_turn=Button(root,textvariable=v,command=setTurn)
button_turn.configure(width=20)
button_back.configure(width=20)
button_save.configure(width=20)
button_restart.configure(width=20)
button_load.configure(width=20)
button_save.configure(width=20)
button_back.pack(side="left")
button_restart.pack(side="left")
button_turn.pack(side="left")
button_save.pack(side="left")
button_load.pack(side="left")
if v.get()=="电脑先手":
    turn = "per"
    draw_circle(position[9][9], "com")
    record[9][9] = 2
root.mainloop()