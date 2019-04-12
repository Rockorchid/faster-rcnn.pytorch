# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 09:41:08 2019

@author: shun
"""

import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

def computeFROC(FROCGTList, FROCProbList, totalNumberOfImages, excludeList):
    # Remove excluded candidates
    FROCGTList_local = []
    FROCProbList_local = []
    for i in range(len(excludeList)):
        if excludeList[i] == False:
            FROCGTList_local.append(FROCGTList[i])
            FROCProbList_local.append(FROCProbList[i])
    
    numberOfDetectedLesions = sum(FROCGTList_local)
    totalNumberOfLesions = sum(FROCGTList)
    totalNumberOfCandidates = len(FROCProbList_local)
    fpr, tpr, thresholds = metrics.roc_curve(FROCGTList_local, FROCProbList_local)
    if sum(FROCGTList) == len(FROCGTList): #  Handle border case when there are no false positives and ROC analysis give nan values.
      print("WARNING, this system has no false positives..")
      fps = np.zeros(len(fpr))
    else:
      fps = fpr * (totalNumberOfCandidates - numberOfDetectedLesions) / totalNumberOfImages
    sens = (tpr * numberOfDetectedLesions) / totalNumberOfLesions
    #sens=tpr
    
    return fps, sens, thresholds

def load(txtfile):
    '''
    读取检测结果或 groundtruth 的文档, 若为椭圆坐标, 转换为矩形坐标
    :param txtfile: 读入的.txt文件, 格式要求与FDDB相同
    :return imagelist: list, 每张图片的信息单独为一行, 第一列是图片名称, 第二列是人脸个数, 后面的列均为列表, 包含4个矩形坐标和1个分数
    :return num_allboxes: int, 矩形框的总个数
    '''
    imagelist = [] # 包含所有图片的信息的列表
    
    txtfile = open(txtfile, 'r')
    lines = txtfile.readlines() # 一次性全部读取, 得到一个list
    
    num_allboxes = 0
    i = 0
    while i < len(lines): # 在lines中循环一遍
        image = [] # 包含一张图片信息的列表
        image.append(lines[i].strip()) # 去掉首尾的空格和换行符, 向image中写入图片名称
        num_faces = int(lines[i + 1])
        num_allboxes = num_allboxes + num_faces
        image.append(num_faces) # 向image中写入个数
        
        if num_faces > 0:
            for num in range(num_faces):
                boundingbox = lines[i + 2 + num].strip() # 去掉首尾的空格和换行符
                boundingbox = boundingbox.split() # 按中间的空格分割成多个元素
                boundingbox = list(map(float, boundingbox)) # 转换成浮点数列表
                    
                image.append(boundingbox) # 向image中写入包含矩形坐标和分数的浮点数列表
                
        imagelist.append(image) # 向imagelist中写入一张图片的信息
        
        i = i + num_faces + 2 # 增加index至下张图片开始的行数
        
    txtfile.close()
    
    return imagelist, num_allboxes

def find(res_list,ground_list):
    xmin1=int(res_list[0])
    ymin1=int(res_list[1])
    w1=int(res_list[2])
    h1=int(res_list[3])
    xmin2=int(ground_list[0])
    ymin2=int(ground_list[1])
    w2=int(ground_list[2])
    h2=int(ground_list[3])
    med_x=xmin1+w1//2
    med_y=ymin1+h1//2
    zuo=xmin2
    you=xmin2+w2
    shang=ymin2
    xia=ymin2+h2
    if (med_x>=zuo) & (med_x<=you) & (med_y<=xia) & (med_y>=shang):
        return True
    else:
        return False
    
def make_list(results,groundtruth):
    assert len(results) == len(groundtruth), "数量不匹配: 标准答案中图片数量为%d, 而检测结果中图片数量为%d" % (len(groundtruth), len(results))
    img_num=len(results)
    FROCGTList = []
    FROCProbList = []
    excludeList=[]
    '''
    for i in range(len(results)):
        if groundtruth[i][1]==0:
            if results[i][1]!=0:
                for k in range(int(results[i][1])):
                    FROCGTList.append(0.)
                    excludeList.append(False)
                    p1=results[i][2+k][4]
                    FROCProbList.append(p1)
        else:
            
            for m in range(int(groundtruth[i][1])):
                p=0.
                FROCGTList.append(1.0)
                for j in range(int(results[i][1])):
                    if find(results[i][2+j],groundtruth[i][2+m]):
                        if results[i][2+j][4]>p:
                            p=results[i][2+j][4]
                print(p)
                FROCProbList.append(p) 
                if p==0.:
                    excludeList.append(True)
                else:
                    excludeList.append(False)
                
    '''  
    for i in range(len(results)):
        if results[i][1]==0:
            if groundtruth[i][1]!=0:
                for j in range(int(groundtruth[i][1])):
                    FROCGTList.append(1.0)
                    FROCProbList.append(0.)
                    excludeList.append(True)
                    print(groundtruth[i][2+j])
        else:
            if groundtruth[i][1]==0:
                for m in range(int(results[i][1])):
                    FROCGTList.append(0.0)
                    FROCProbList.append(results[i][2+m][4])
                    excludeList.append(False)
            else:
                
                for l in range(int(groundtruth[i][1])):
                    p=0.
                    FROCGTList.append(1.0)
                    for o in range(int(results[i][1])):
                        if find(results[i][2+o],groundtruth[i][2+l]):
                            if results[i][2+o][4]>p:
                                p=results[i][2+o][4]
                    FROCProbList.append(p) 
                    if p==0.:
                        #FROCGTList.append(1.0)
                        #FROCProbList.append(p) 
                        excludeList.append(True)
                        print(groundtruth[i][2+l])
                    else:
                        excludeList.append(False)
                        
                for m in range(int(results[i][1])):
                    f=False
                    for n in range(int(groundtruth[i][1])):
                        if find(results[i][2+m],groundtruth[i][2+n]):
                            f=True
                            '''
                            FROCGTList.append(1.0)
                            FROCProbList.append(results[i][2+m][4])
                            excludeList.append(False)
                            '''
                    if f==False:
                        FROCGTList.append(0.0)
                        FROCProbList.append(results[i][2+m][4])
                        excludeList.append(False)
                
        
    return FROCGTList, FROCProbList, img_num, excludeList


results, num_detectedbox = load("D:\\document\\workspace\\python\\duice\\ceshi\\dance_froc_data.txt")
groundtruth, num_groundtruthbox = load("D:\\document\\workspace\\python\\duice\\ceshi\\dance_groundtruth.txt")
FROCGTList, FROCProbList, img_num, excludeList=make_list(results,groundtruth)
fps, sens, thresholds=computeFROC(FROCGTList, FROCProbList, img_num, excludeList)

results1, num_detectedbox1 = load("D:\\document\\workspace\\python\\duice\\ceshi\\duice_froc_data.txt")
groundtruth1, num_groundtruthbox1 = load("D:\\document\\workspace\\python\\duice\\ceshi\\duice_groundtruth.txt")
FROCGTList1, FROCProbList1, img_num1, excludeList1=make_list(results1,groundtruth1)
fps1, sens1, thresholds=computeFROC(FROCGTList1, FROCProbList1, img_num1, excludeList1)

plt.figure()
plt.title('FROC')
plt.xlabel('False Positives per Image')
plt.ylabel('Sentivity')
#plt.plot(fps,sens)
plt.grid()

plt.semilogx(fps,sens,label = "resnet101", color='blue')
plt.semilogx(fps1,sens1,label = "vgg16", color='coral')
plt.rcParams['savefig.dpi'] = 1000 #图片像素
plt.rcParams['figure.dpi'] = 100
plt.legend(loc='lower right')


