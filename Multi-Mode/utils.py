import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms as transforms
import cv2
from transformers import BertTokenizer
from Config import Config


class My_Dataset(Dataset):
    '''
    功能：读取数据集并规整格式
    :param iftrain: 是否为训练模式
    __len__函数：创建DataLoader时，自动调用__len__方法获取数据集的大小，以便知道迭代多少批次
    __getitem___函数：创建DataLoader时，自动调用该方法，通过索引访问数据集中的样本，正是由该方法的return取出每一批次的样本和labels
    '''
    def __init__(self,path,config,iftrain):
        self.config=config
        self.iftrain=iftrain
        df = pd.read_csv(path).sample(frac=self.config.frac)
        self.img_path = df['path'].to_list() 
        self.text = df['text'].to_list()
        self.tokenizer = BertTokenizer.from_pretrained(self.config.bert_name)

        #启用训练模式，加载数据和标签
        if self.iftrain==1:
            self.labels=df['label'].to_list()


    def __getitem__(self, idx):
        '''
        1.读取图片转为numpy数组，opencv重塑数据格式并归一化
        2.读取文本，使用bert tokenizer进行分词，并使用bert进行编码
        :param idx: 数据索引(表示第几条数据)
        '''
        # 图片处理
        img=Image.open(self.img_path[idx])
        img=img.convert("RGB")
        img=np.array(img)
        img=cv2.resize(img,(224,224))
        img = img / 255.
        img=np.transpose(img,(2,0,1))                   # img.shape = [3,224,224]
        img = torch.tensor(img, dtype=torch.float32)
        
        # 文本处理
        text=self.text[idx]

        # 空值处理
        try:
            len(text)
        except:
            text=''



        '''
        text经过bert tokenizer进行分词和编码后变成一个字典
        input_ids：表示经过分词和编码后的 token ID 序列，包含特殊令牌（如 [CLS] 和 [SEP]）以及填充令牌（如 [PAD]）
        attention_mask：表示注意力掩码，其中1表示对应位置的 token 应被模型关注，0表示填充令牌，模型在计算注意力时会忽略这些位置
        '''
        text=self.tokenizer(text=text, add_special_tokens=True,
                  max_length=self.config.pad_size,  # 最大句子长度
                  padding='max_length',  # 补零到最大长度
                  truncation=True)
        input_id= torch.tensor(text['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(text['attention_mask'], dtype=torch.long)
        
        # 如果是训练模式，返回一个元组，元组的第一个数据是一个列表，列表包含图像数据、文本数据、注意力掩码矩阵，元组的第二个元素是一个标签数据,label.shape=torch.Size([32]),32是批量
        if self.iftrain==1:
            label=int(self.labels[idx])
            label = torch.tensor(label, dtype=torch.long)
            return (img.to(self.config.device),input_id.to(self.config.device),attention_mask.to(self.config.device)),label.to(self.config.device)
        # 如果不是训练模式，则仅返回图像数据、文本数据、注意力掩码矩阵
        else:
            return (img.to(self.config.device),input_id.to(self.config.device),attention_mask.to(self.config.device))

    def __len__(self):
        # 数据总长度
        return len(self.img_path)

def get_time_dif(start_time):

    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

if __name__=='__main__':
    config=Config()
    train_data=My_Dataset('./data/train.csv',config,1)
    train_iter = DataLoader(train_data, batch_size=32)

    # 查看一个batch取出的数据格式
    for batch,(trains,labels) in enumerate(train_iter):
        print(trains[0].shape)      # trains[0]是图像数据,torch.Size([32, 3, 224, 224])
        print(trains[1].shape)      # trains[1]是文本数据,torch.Size([32, 128])
        print(trains[2].shape)      # trains[2]是注意力掩码矩阵,torch.Size([32, 128])
        break

