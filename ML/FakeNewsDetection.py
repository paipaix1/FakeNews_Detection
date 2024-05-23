import pandas as pd
import numpy as np  
from sklearn.model_selection import train_test_split  
import re  
import jieba  
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier  
from sklearn.linear_model import LogisticRegression  
from sklearn import metrics 


def get_data():  

    df = pd.read_csv('news.csv')
    corpus = df['text']
    labels = df['label']
    return corpus, labels  

def remove_empty_docs(corpus, labels):  
    '''
    功能：重新整合邮件及标签，去除空白邮件
    zip(corpus, labels),控制同时遍历corpus和labels
    doc.strip(),去除文本首尾空格和换行符，如果有非空字符，返回真
    '''
    filtered_corpus = []  
    filtered_labels = []  
    for doc, label in zip(corpus, labels):  
        if doc.strip():  
            filtered_corpus.append(doc)  
            filtered_labels.append(label)  
  
    return filtered_corpus, filtered_labels 

def prepare_datasets(corpus, labels, test_data_proportion=0.3):  
    '''
    功能：划分数据集
    test_size=0.3，测试集30%，训练集70%
    random_state=42，设置随机种子，确保每次切割结果一致
    '''  
    train_X, test_X, train_Y, test_Y = train_test_split(corpus, labels, test_size = test_data_proportion, random_state=42)  
    return train_X, test_X, train_Y, test_Y

def normalize_corpus(corpus):  
    '''
    功能：导入停用词于stopwords列表,遍历corpus（即每一行）并分词，将每一行中不是停用词的词连成字符串（词与词用空格隔开），再存入normalized_corpus列表
    sw.replace('\n', '')，将sw通过readlines到的每一行内容中的换行符移除
    jieba.lcut(text.replace('\n','')),对字符串去除空格和换行符后，做分词处理放入列表中
    text = ' '.join(filtered_tokens)，将filterd_tokens列表元素拼接为字符串(用空格隔开元素)
    '''
    normalized_corpus = []  
    stopwords = [sw.replace('\n', '') for sw in open('Files\stopwords.txt',encoding='utf-8').readlines()]   
        
    for text in corpus:  
        filtered_tokens = []   
        tokens = jieba.lcut(text.replace('\n',''))  
  
        for token in tokens:  
            token = token.strip()  
            if token not in stopwords and len(token)>1:  
                filtered_tokens.append(token)  
          
        text = ' '.join(filtered_tokens)  
        normalized_corpus.append(text)  
    return normalized_corpus

def bow_extractor(normalized_corpus, ngram_range=(1, 1)):  
    '''
    功能：将normalized_corpus（分词处理后的）转化为BOW词袋模型
    min_df=1，表示包含至少在一个文档中（即每一行）出现的词汇才会被纳入词典
    ngram_range=(1, 1)，定义n-gram的范围，只提取单个词汇，不考虑组合词汇
    fit_transform(normalized_corpus)，文本数据转化为词频矩阵，每一行代表一个文档，每一列代表一个词汇项，矩阵中的元素表示相应词汇在该文档(这一行中)的出现次数
    features.shape = (len(normalized_corpus),dic_size)，dic_size为词典大小
    ''' 
    vectorizer = CountVectorizer(min_df=1, ngram_range=ngram_range)  
    features = vectorizer.fit_transform(normalized_corpus)  
    return vectorizer, features
    
def tfidf_extractor(normalized_corpus, ngram_range=(1, 1)):  
    '''
    功能：将normalized_corpus（分词处理后的）转化为TF-IDF表示形式
    norm = 'l2',在计算TF-IDF向量后，会对结果向量做L2范数归一化
    fit_transform(normalized_corpus),将文本数据转换为TF-IDF矩阵，每一行代表一个文档，每一列代表一个词汇项,矩阵中的元素表示相应词汇在文档中的TF-IDF值
    '''
    vectorizer = TfidfVectorizer(min_df=1,  
                                 norm='l2',  
                                 smooth_idf=True,       # 在IDF权重计算时引入平滑项，防止出现零概率问题
                                 use_idf=True,          # 启用IDF调整，即TF-IDF计算公式中的逆文档频率部分
                                 ngram_range=ngram_range)  
    features = vectorizer.fit_transform(normalized_corpus)  
    return vectorizer, features

def get_metrics(true_labels, predicted_labels): 
    '''
    功能：计算预测的各类指标
    ''' 
    acc = metrics.accuracy_score(true_labels,predicted_labels)  
    precision = metrics.precision_score(true_labels, predicted_labels, average = 'weighted')  
    recall = metrics.recall_score(true_labels, predicted_labels,average='weighted')  
    f1_score = metrics.f1_score(true_labels,predicted_labels,average='weighted')  
    print('准确率:%.4f' % acc)  
    print('精度:%.4f' % precision)  
    print('召回率:%.4f' % recall)  
    print('F1得分:%.4f' % f1_score)  
  
def train_predict(classifier, train_features, train_labels, test_features, test_labels):  
    '''
    功能：根据传入的训练分类器，进行模型训练
    '''
    classifier.fit(train_features, train_labels)  # 喂给模型训练矩阵和训练标签
    predictions = classifier.predict(test_features)  # 模型预测
    get_metrics(true_labels=test_labels, predicted_labels=predictions)  # 调用get_metrics函数,打印预测结果的各类指标
    return predictions

if __name__ == "__main__":  
    corpus, labels = get_data()  # 获取数据集  
    #print(corpus)
    #print(labels)
    print("总的数据量:", len(labels))   
    corpus, labels = remove_empty_docs(corpus, labels)  
    label_name_map = ["垃圾邮件", "正常邮件"]  
    # 对数据进行划分  
    train_corpus, test_corpus, train_labels, test_labels = prepare_datasets(corpus,  labels,  test_data_proportion=0.3)  
    print('训练集样本数量：%d，测试样本数量：%d'%(len(train_corpus),len(test_corpus)))  
    # 样本标准化  
    norm_train_corpus = normalize_corpus(train_corpus)  
    norm_test_corpus = normalize_corpus(test_corpus)  
    # 词袋模型特征  
    bow_vectorizer, bow_train_features = bow_extractor(norm_train_corpus)  
    bow_test_features = bow_vectorizer.transform(norm_test_corpus)  
    # tfidf 特征  
    tfidf_vectorizer, tfidf_train_features = tfidf_extractor(norm_train_corpus)  
    tfidf_test_features = tfidf_vectorizer.transform(norm_test_corpus)  
    # 分词  
    #tokenized_train = [jieba.lcut(text) for text in norm_train_corpus]  
    #tokenized_test = [jieba.lcut(text) for text in norm_test_corpus]  
    # Word2vec 模型  
    #model = gensim.models.Word2Vec(tokenized_train,  vector_size=500, window=100,  min_count=30,  sample=1e-3)  
 
    # 分类器  
    svm = SGDClassifier(loss='hinge')  
    lr = LogisticRegression()  
    # 基于词袋模型特征的逻辑回归  
    print("基于词袋模型特征的逻辑回归")  
    lr_bow_predictions = train_predict(classifier=lr,  
                                train_features=bow_train_features,  
                                train_labels=train_labels,  
                                test_features=bow_test_features,  
                                test_labels=test_labels)  
  
    # 基于词袋模型的支持向量机方法  
    print("基于词袋模型的支持向量机")  
    svm_bow_predictions = train_predict(classifier=svm,  
                                  train_features=bow_train_features,  
                                  train_labels=train_labels,  
                                  test_features=bow_test_features,  
                                  test_labels=test_labels)  
  
    # 基于tfidf的逻辑回归模型  
    print("基于tfidf的逻辑回归模型")  
    lr_tfidf_predictions = train_predict(classifier=lr,  
                                 train_features=tfidf_train_features,  
                                 train_labels=train_labels,  
                                 test_features=tfidf_test_features,  
                                 test_labels=test_labels)  
  
    # 基于tfidf的支持向量机模型  
    print("基于tfidf的支持向量机模型")  
    svm_tfidf_predictions = train_predict(classifier=svm,  
                                  train_features=tfidf_train_features,  
                                  train_labels=train_labels,  
                                  test_features=tfidf_test_features,  
                                  test_labels=test_labels)