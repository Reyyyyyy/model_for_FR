import shutil
import os
from tkinter import *
from tkinter.ttk import *
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import time
import sys  # 导入sys模块
sys.setrecursionlimit(5000)  # 将默认的递归深度修改为5000

class Cleaner(Tk):
    def __init__(self,Height=600,Width=750):
        super().__init__()
        #正样本目录和负样本目录
        self.pos_dir = 'D:\\核聚变课题组\\样本图片\\正样本'
        self.neg_dir = 'D:\\核聚变课题组\\样本图片\\负样本'
        #保存样本集目录列表和当前图片的指针索引，以及当前工作目录
        self.imgs = []
        self.img_idx  = 0     #从第一张开始检查
        self.work_dir = None
        #窗口尺寸
        self.Height = Height
        self.Width  = Width
        #设计标题
        self.title('数据清洗器')
        #调整大小，同时不能让用户调整尺寸
        self.geometry('{}x{}'.format(self.Width,self.Height))
        self.resizable(0,0)
        #创建一系列按钮并摆放好位置
        self.go_pos = Button(self, text="转移到正样本",command=self.go2pos)
        self.go_neg = Button(self, text="转移到负样本",command=self.go2neg)
        self.go_bin = Button(self, text="删除该样本",command=self.go2bin)
        self.go_dir = Button(self, text="导入样本",command=self.get_imgs)
        self.go_img = Button(self, text="切换到此索引",command=self.go2img)
        self.go_pos.place(x=self.Width//4+10, y=self.Height//5*4+60)
        self.go_neg.place(x=self.Width//4+155,y=self.Height//5*4+60)
        self.go_bin.place(x=self.Width//4+300,y=self.Height//5*4+60)
        self.go_dir.place(x=10,y=10)
        self.go_img.place(x=self.Height//2+90,y=10)
        #创建下拉框获取索引值，可以浏览任意索引的图片
        self.idx_getter = Combobox(self, width=6)
        self.idx_getter['values'] = (list(range(1,len(self.imgs))))
        self.idx_getter.place(x=self.Height//2+25,y=12)
        #放置标签
        self.img_box = Label(self)
        self.img_box.place(x=self.Width//6,y=self.Height//7)
        self.img_box.bind("<MouseWheel>",self.switch_img)
        self.idx_box = Label(self)
        self.idx_box.place(x=self.Height//2+50,y=self.Height//10)

    def go2img(self):
        self.img_idx = int(self.idx_getter.get())-1
        img = Image.fromarray(self.cut_img(np.array(Image.open(self.imgs[self.img_idx]))))
        img = ImageTk.PhotoImage(img.resize((500,420)))
        self.img_box.config(image=img)
        self.idx_box.config(text='{}/{}'.format(self.img_idx+1,len(self.imgs)))
        self.mainloop()
    
    def go2pos(self):
        try:
            if len(self.imgs) == 1:
                shutil.move(self.imgs[self.img_idx],self.pos_dir)
                img = Image.fromarray(self.cut_img(np.array(Image.open(r'D:\核聚变课题组\处理数据的脚本\数据清洗器\finish.png'))))
                img = ImageTk.PhotoImage(img.resize((500,420)))
                self.img_box.config(image=img)
                self.idx_box.config(text='0/0')
                self.mainloop()
            else:    
                shutil.move(self.imgs[self.img_idx],self.pos_dir)
                self.imgs = os.listdir(self.work_dir)
                if self.img_idx == len(self.imgs):
                    self.img_idx -= 1
                img = Image.fromarray(self.cut_img(np.array(Image.open(self.imgs[self.img_idx]))))
                img = ImageTk.PhotoImage(img.resize((500,420)))
                self.img_box.config(image=img)
                self.idx_box.config(text='{}/{}'.format(self.img_idx+1,len(self.imgs)))
                self.idx_getter['values'] = (list(range(1,len(self.imgs)+1)))
                self.mainloop()
        except:
            pass

    def go2neg(self):
        try:
            if len(self.imgs) == 1:
                shutil.move(self.imgs[self.img_idx],self.neg_dir)
                img = Image.fromarray(self.cut_img(np.array(Image.open(r'D:\核聚变课题组\处理数据的脚本\数据清洗器\finish.png'))))
                img = ImageTk.PhotoImage(img.resize((500,420)))
                self.img_box.config(image=img)
                self.idx_box.config(text='0/0')
                self.mainloop()
            else:    
                shutil.move(self.imgs[self.img_idx],self.neg_dir)
                self.imgs = os.listdir(self.work_dir)
                if self.img_idx == len(self.imgs):
                    self.img_idx -= 1
                img = Image.fromarray(self.cut_img(np.array(Image.open(self.imgs[self.img_idx]))))
                img = ImageTk.PhotoImage(img.resize((500,420)))
                self.img_box.config(image=img)
                self.idx_box.config(text='{}/{}'.format(self.img_idx+1,len(self.imgs)))
                self.idx_getter['values'] = (list(range(1,len(self.imgs)+1)))
                self.mainloop()
        except:
            pass

    def go2bin(self):
        try:
            if len(self.imgs) == 1:
                os.remove(self.imgs[self.img_idx])
                img = Image.fromarray(self.cut_img(np.array(Image.open(r'D:\核聚变课题组\处理数据的脚本\数据清洗器\finish.png'))))
                img = ImageTk.PhotoImage(img.resize((500,420)))
                self.img_box.config(image=img)
                self.idx_box.config(text='0/0')
                self.mainloop()
            else:    
                os.remove(self.imgs[self.img_idx])
                self.imgs = os.listdir(self.work_dir)
                if self.img_idx == len(self.imgs):
                    self.img_idx -= 1
                img = Image.fromarray(self.cut_img(np.array(Image.open(self.imgs[self.img_idx]))))
                img = ImageTk.PhotoImage(img.resize((500,420)))
                self.img_box.config(image=img)
                self.idx_box.config(text='{}/{}'.format(self.img_idx+1,len(self.imgs)))
                self.idx_getter['values'] = (list(range(1,len(self.imgs)+1)))
                self.mainloop()
        except:
            pass

    def cut_img(self,img_array):
        """基于对数组的切片算法"""
        if img_array.shape[0] ==1584 or img_array.shape[0] ==1581:
            img_array = img_array[196:1420,371:2612]
        if img_array.shape[0] ==2134 or img_array.shape[0] ==2145:
            img_array = img_array[240:1905,500:3524]
        return img_array

    def get_imgs(self):
        """获取样本集地址,并切换到工作目录"""
        self.work_dir = filedialog.askdirectory()
        self.imgs = os.listdir(self.work_dir)
        self.idx_getter['values'] = (list(range(1,len(self.imgs)+1)))
        os.chdir(self.work_dir)
        self.img_idx = 0
        self.show_img(self.imgs[self.img_idx])
        
    def show_img(self,img_dir):
        """对图片进行格式和大小的调整并且显示出来"""
        img = Image.fromarray(self.cut_img(np.array(Image.open(img_dir))))
        img = ImageTk.PhotoImage(img.resize((500,420)))
        self.img_box.config(image=img)
        self.idx_box.config(text='{}/{}'.format(self.img_idx+1,len(self.imgs)))
        self.mainloop()
        
    def switch_img(self,event):
        """根据鼠标滚轮事件改变当前指针索引"""
        if event.delta < 0 and self.img_idx < len(self.imgs)-1: #向下滑动
            self.img_idx += 1
        elif event.delta > 0 and self.img_idx > 0:              #向上滑动
            self.img_idx -= 1
        img = Image.fromarray(self.cut_img(np.array(Image.open(self.imgs[self.img_idx]))))
        img = ImageTk.PhotoImage(img.resize((500,420)))
        self.img_box.config(image=img)
        self.idx_box.config(text='{}/{}'.format(self.img_idx+1,len(self.imgs)))
        self.mainloop()
        
    def run(self):
        self.get_imgs()
        #放在各个函数最后，刷新事件循环，使其变成动态UI
        self.mainloop()

if __name__ == '__main__':
    cleaner = Cleaner()
    cleaner.run()  

