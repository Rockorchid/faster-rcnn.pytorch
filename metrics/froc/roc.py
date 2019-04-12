# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 11:47:10 2019

@author: shun
"""
import xml.etree.ElementTree as ET
from sklearn import metrics
import matplotlib.pyplot as plt

def roc_anno(txtfile,path):
    txtfile = open(txtfile, 'r')
    lines = txtfile.readlines() 
    i=0
    prob=[]
    ground=[]
    while i<len(lines):
        try:
            tree = ET.parse(path + lines[i][:-5] + '.xml')
            #tree = ET.parse('D:\\document\\workspace\\python\\ddsm\\ddsm_xml'+'\\' + lines[i][:6] + '.xml')
            root = tree.getroot()
            node = root.findall("object")
            if node==[]:
                ground.append(0.)
            else:
                clas=node[0].find('name')
                if clas.text=='malignant':
                    ground.append(1.)
                else:
                    ground.append(0.)
        except:
            ground.append(0.)
        pro=float(lines[i+1][:-2])
        prob.append(pro)
        i=i+2
    return ground,prob

path1="D:\\document\\workspace\\python\\ddsm\\duice\\xml\\"
path2="D:\\document\\workspace\\python\\ddsm\\duice\\img_zuhe\\"

ground,prob=roc_anno("D:\\document\\workspace\\python\\duice\\ceshi\\dance_roc_data.txt",path1)
fpr, tpr, thresholds = metrics.roc_curve(ground,prob,drop_intermediate=False)
auc_roc=metrics.auc(fpr, tpr)

ground1,prob1=roc_anno("D:\\document\\workspace\\python\\duice\\ceshi\\duice_roc_data.txt",path2)
fpr1, tpr1, thresholds1 = metrics.roc_curve(ground1,prob1,drop_intermediate=False)
auc_roc1=metrics.auc(fpr1, tpr1)
#plt.rcParams['savefig.dpi'] = 1000 #图片像素
plt.rcParams['figure.dpi'] = 100
plt.grid()
plt.title('ROC')
plt.xlabel('False Positives rate')
plt.ylabel('Sentivity')
plt.plot(fpr,tpr,label = "resnet101", color='blue')
plt.plot(fpr1,tpr1,label = "vgg16", color='coral')
plt.legend(loc='lower right')
print("res101 auc=%f"%auc_roc)
print("vgg16 auc=%f"%auc_roc1)








