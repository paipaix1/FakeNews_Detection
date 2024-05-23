## 数据描述

数据集是中文微信消息，包括微信消息的Official Account Name，Title，News Url，Image Url，Report Content，label。Title是微信消息的标题，label是消息的真假标签（0是real消息，1是fake消息）。训练数据保存在`train.news.csv`，测试数据保存在`test.news.csv`。

## 准备工作

##### 1.下载预训练词向量

从`https://github.com/Embedding/Chinese-Word-Vectors`当中找到搜狗新闻，选择`Word+Character+Ngram`这一类进行下载，之后解压，将得到的`sgns.sogounews.bigram-char`存入Data文件夹中。


