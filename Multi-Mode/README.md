## 数据描述

​	本实验数据集为多模态虚假新闻检测广泛使用的Weibo数据集，由Jin[28]等人提供。该数据集摘录了从2012年开始至2017年经过官方认证并公示的虚假新闻。

|      | Train | Val  | Test | All   |
| ---- | ----- | ---- | ---- | ----- |
| Real | 3689  | 414  | 1218 | 5321  |
| Fake | 6050  | 668  | 1235 | 7953  |
| All  | 9739  | 1082 | 2453 | 13274 |

**1.推文数据说明：**	

​	1.1推文分为训练集和测试集。对于每个集合，有两个文件分别存储谣言和非谣言推文。

​	1.2.每个txt文件的数据格式如下（一条数据存储三行）:
​		推特id|用户名|推特url|用户url|发布时间|原文?|转发数|评论数|好评数|用户id|用户认证类型|用户粉丝数|用户关注数|用户推文数|发布平台
​		Image1 url|image2 url|null
​		微博内容

**2..数据目录：**

​	data/images：存储新闻图片，名称与文本数据集的image url相对应

​	data/tweets：存储测试与训练的谣言、非谣言推文数据

## 实验配置

**1.下载预训练模型**

​	创建bert_model文件夹，在hugging-face上下载bert-base-chinese、chinese-bert-wwm-ext、minirbt-h256三个预训练模型存入bert_model文件夹，其中config.json、pytorch_model.bin、vocab.txt必备。

![image](https://github.com/paipaix1/FakeNews_Detection/assets/156734592/c6df479a-35f9-4f9f-b2cc-fc4a42f0a936)


**2.下载推文和图片数据集**

​	在data文件夹下创建tweets和images数据集，下载地址：https://drive.google.com/drive/folders/1SYHLEMwC8kCibzUYjOqiLUeFmP2a19CH?usp=sharing

![image](https://github.com/paipaix1/FakeNews_Detection/assets/156734592/208b82da-e29c-48bc-95fe-6bed2b7c7b50)


**3.自定义配置文件**
