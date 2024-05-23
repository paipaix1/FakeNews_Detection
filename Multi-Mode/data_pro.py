import pandas as pd
import os
import csv
import numpy as np

# 读取图片名称
imgs=os.listdir('./data/images')


'''
功能：读取文本文件数据并整理格式，将其写入csv文件
文件中数据格式说明：一条数据分为三行
第一行：tweet id|user name|tweet url|user url|publish time| original?|retweet count|comment count|praise count|user id|userauthentication type|user fans count|user follow count|user tweet count|publish platform
第二行：image1 url|image2 url|null
第三行：tweet content
'''

def new_data(path,label,newpath):
    len_list=[]
    with open(path,'r',encoding='utf-8')as t1:
        t1=t1.readlines()

        if len(t1)%3==0:
            num=int(len(t1)/3)
            print('数据列数:',len(t1))
            print('数据条数(除以3):', num)
            for n in range(num):
                #a2为图片名，a3为文本
                a1,a2,a3=t1[n*3],t1[n*3+1],t1[n*3+2]
                a2=a2.strip()#取出换行符
                a3=a3.strip()
                text_len=len(a3)
                len_list.append(text_len)
                a2=a2.split('|')#分割图片
                a2=[x.split('/')[-1] for x in a2 if x!='null']#去除空数据并分割出图片路径

                a2 = ['./data/images/' + x for x in a2 if x in imgs]
                for m in a2:
                    all_info=m,a3,label
                    # print(all_info)
                    with open(newpath,'a',encoding='utf-8',newline='')as f:
                        writer=csv.writer(f)
                        writer.writerow(all_info)

        else:
            print('数据长度不合理')
    print('平均句子长度:',np.mean(len_list))

if os.path.exists('./data/train.csv'):
    os.remove('./data/train.csv')#如果存在就删除以免重复写入
with open('./data/train.csv', 'a', encoding='utf-8', newline='') as f:#写入列名
    writer = csv.writer(f)
    writer.writerow(('path','text','label'))

# 将训练集谣言信息tweets/train_rumor.txt的格式整理并导入train.csv
new_data('./data/tweets/train_rumor.txt',1,'./data/train.csv')      # 1表示给谣言数据添加标签1
# 将训练集非谣言信息tweets/train_nonrumor.txt的格式整理并导入train.csv
new_data('./data/tweets/train_nonrumor.txt',0,'./data/train.csv')   # 0表示给非谣言数据添加标签0


if os.path.exists('./data/test.csv'):
    os.remove('./data/test.csv')
with open('./data/test.csv', 'a', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(('path', 'text', 'label'))
new_data('./data/tweets/test_rumor.txt',1,'./data/test.csv')
new_data('./data/tweets/test_nonrumor.txt',0,'./data/test.csv')

df=pd.read_csv('./data/train.csv',encoding='utf-8')
val_df=df.sample(frac=0.1)                           # 10%数据交给验证集
train_df=df.drop(index=val_df.index.to_list())       # 90%数据作为训练集
print('训练集长度:',len(train_df))
print('测试集长度:',len(val_df))
val_df.to_csv('./data/val.csv',encoding='utf-8',index=None)
train_df.to_csv('./data/train.csv',encoding='utf-8',index=None)