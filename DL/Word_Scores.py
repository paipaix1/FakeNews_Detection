import torch  
import torch.nn as nn  
import torch.optim as optim  
from collections import Counter  
import time
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split 
import jieba
import random
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score ,roc_curve
import torch.nn.functional as F
import matplotlib.pyplot as plt


def get_data(path):  
    df = pd.read_csv(path)
    df_sample = df.sample(n=1000, random_state=42) #随机抽取10000个样本
    corpus = df_sample['Title'].tolist()
    labels = df_sample['label'].tolist()
    return corpus, labels  

def normalize_corpus(corpus):  
    normalized_corpus = []  
    stopwords = [sw.replace('\n', '') for sw in open('./Data/stopwords.txt',encoding='utf-8').readlines()]   
    for text in corpus:  
        filtered_tokens = []   
        tokens = jieba.lcut(text.replace('\n',''))  
        for token in tokens:  
            token = token.strip()  
            if token not in stopwords :  
                filtered_tokens.append(token)  
         
        normalized_corpus.append(filtered_tokens)  
    return normalized_corpus

# 构建词汇表  
def build_vocab(texts):  
    tokens = normalize_corpus(texts)
    flattened = [token for sublist in tokens for token in sublist]  
    vocab = set(flattened)  
    word2idx = {word: idx for idx, word in enumerate(vocab, 1)}  # 从1开始索引，0用作padding_idx  
    idx2word = {idx: word for word, idx in word2idx.items()}  
    return tokens,word2idx, idx2word  

# 将文本转换为索引列表  
def text_to_indices(token, word2idx, max_length):  
    # print(token)  
    indices = [word2idx.get(token_part, 0) for token_part in token]  # 未知单词用0表示  
    if max_length is not None:  
        indices = indices[:max_length]  
        indices += [0] * (max_length - len(indices))  # 用0填充至指定长度  
    return indices

# 找到二维列表中最长列表长度
def find_longest_length(nested_list):  
    longest_length = 0  
    for sublist in nested_list:  
        length = len(sublist)  
        if length > longest_length:  
            longest_length = length  
    return longest_length 

# 获取数据集
texts, labels = get_data('./Data/train.news.csv')     

# 构建词汇表  
tokens,word2idx, idx2word = build_vocab(texts) 
# print(tokens) 

# 构建词汇索引表
max_length = find_longest_length(normalize_corpus(texts))
text_indices = [text_to_indices(token, word2idx, max_length=max_length) for token in tokens]  
# print(text_indices)
  
# 将索引列表转换为PyTorch张量  
text_tensors = torch.tensor(text_indices, dtype=torch.long)  
labels_tensors = torch.tensor(labels, dtype=torch.float)  

# 定义模型参数  
embedding_dim = 32  
hidden_dim = 64  
output_dim = 1  
padding_idx = 0  
  
# 定义模型  
class FakeNewsRNN(nn.Module):  
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, padding_idx):  
        super(FakeNewsRNN, self).__init__()  
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)  
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)  
        self.fc = nn.Linear(hidden_dim, output_dim)  
  
    def forward(self, text):  
        embedded = self.embedding(text)  
        output,hidden = self.rnn(embedded)  
 
        hidden = hidden.squeeze(0)  # (1, batch_size, hidden_dim) -> (batch_size, hidden_dim)  
        prediction = self.fc(hidden)  
        return torch.sigmoid(prediction)  
  
# 初始化模型  
vocab_size = len(word2idx)+1    # 词汇表大小，包括padding_idx，因为词向量编码是词典的下表索引
model = FakeNewsRNN(vocab_size, embedding_dim, hidden_dim, output_dim, padding_idx)  
  
# 定义损失函数和优化器  
criterion = nn.BCELoss()  
optimizer = optim.Adam(model.parameters())  

# 训练模型（这里仅展示一个训练步骤）  
model.train()  
optimizer.zero_grad()  
outputs = model(text_tensors)  
print(text_tensors.shape)
loss = criterion(outputs.squeeze(), labels_tensors)  
loss.backward()  
optimizer.step()  

# 定义一个函数来计算词语重要性  
def compute_word_importance(model, text_tensor, label_tensor):  
    model.zero_grad()  
    model.eval()  # 使用评估模式，关闭dropout等  
      
    # 假设text_tensor是一个batch_size为1的tensor  
    outputs = model(text_tensor)    
    outputs = outputs.squeeze(0)  
      
    loss = criterion(outputs, label_tensor)   
    loss.backward()  
      
    # 获取嵌入层的梯度  
    embedding_grad = model.embedding.weight.grad  
    word_importance = torch.sum(embedding_grad, dim=1, keepdim=True) 
    # 删除padding_idx对应的词
    word_importance = word_importance[1:, :]

    return word_importance  

# 假设我们要计算第一个文本的重要性得分  
text_tensor = text_tensors[126].unsqueeze(0)  # 转换为batch_size为1的tensor,[1,seq_len]  
label_tensor = torch.tensor([labels[0]], dtype=torch.float)  # 转换为batch_size为1的tensor,[1]


# 计算词语重要性  
word_importances = compute_word_importance(model, text_tensor, label_tensor)  

# 打印词语重要性  
for idx, importance in enumerate(word_importances): 
    if importance.item() :  
        token = idx2word[idx + 1]  # 跳过padding_idx（0）
        print(f"{token}: {importance.item()}")  