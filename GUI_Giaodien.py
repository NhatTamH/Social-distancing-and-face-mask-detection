from tkinter import *
from PIL import Image
from PIL import ImageTk
import imutils
import time
from datetime import date
import cv2 
import numpy as np
root=Tk()
root.title("NHẬN DẠNG ĐEO KHẨU TRANG VÀ ĐO THÂN NHIỆT")

#Chia Frame cho giao diện
frame1=LabelFrame(root,text="NHIỆT ĐỘ ĐO ĐƯỢC",width=300,height=300)
frame1.grid(row=0,column=0,padx=20,pady=20,ipadx=20,ipady=20)

frame2=LabelFrame(root,text="CAMERA GIÁM SÁT",width=300,height=300)
frame2.grid(row=0,column=1,padx=20,pady=20,ipadx=20,ipady=20)

frame5=Frame(root,width=300,height=20)
frame5.grid(row=2,column=0,padx=20,pady=20,ipadx=20,ipady=20)

frame10=LabelFrame(root,text="Thời gian",width=340,height=100)
frame10.place(x=400,y=410)

def clock():
    hour=time.strftime("%H")
    min=time.strftime("%M")
    sec=time.strftime("%S")
    myt.config(text="Giờ:"+" "+hour+"h"+ min +" " + sec+ "s",font=("Arial",10))
    myt.after(1000,clock)
myt=Label(frame10,text="")
myt.place(x=10,y=10)
clock()

today=date.today()
today=str(today).replace("-",' ').split()
date=Label(frame10,text=str("Ngày:"+" "+today[2])+"/"+str(today[1])+"/"+str(today[0]),font=("Arial",10))
date.place(x=10,y=40)


#thông tin cá nhân
label1=Label(frame5,text="ĐỒ ÁN TỔNG HỢP",font=("Arial",18))
label1.grid(column=0,row=0)
label2=Label(frame5,text="TÊN: NGUYỄN LÊ NHẬT TÂM",font=("Arial",12))
label2.grid(column=0,row=1)
label3=Label(frame5,text="MSSV: 41702131",font=("Arial",12))
label3.grid(column=0,row=2)
label4=Label(frame5,text="LỚP: 17040201",font=("Arial",12))
label4.grid(column=0,row=3)
root.mainloop()

