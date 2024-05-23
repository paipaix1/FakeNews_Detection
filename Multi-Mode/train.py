# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from datetime import timedelta
import sys
import torch
import numpy as np
from models import Mynet,SupConLoss
from tensorboardX import SummaryWriter
from transformers import logging
from utils import My_Dataset,get_time_dif
from models import *
from Config import Config
from torch.utils.data import DataLoader



def train(config, model, train_iter, dev_iter, writer):
    start_time = time.time()            # 开始时间
    all_times = timedelta(seconds=0)    # 总时间

    model.train()

    # optimizer_grouped_parameters列表存储不同模型及其学习率
    # 根据模型参数名n和参数值p，为其分配提前设置好的响应学习率
    optimizer_grouped_parameters = [{'params': [p for n, p in model.named_parameters() if 'bert' in n],'lr': config.bert_learning_rate},#包含bert层学习率
                                    {'params': [p for n, p in model.named_parameters() if 'resnet' in n],'lr': config.resnet_learning_rate},#包含resnet层学习率
                                    {'params': [p for n, p in model.named_parameters() if 'resnet' not in n and 'bert' not in n]}]

    # 定义Adam优化器
    optimizer = torch.optim.Adam(optimizer_grouped_parameters , lr=config.other_learning_rate)
    # 定义学习率下降策略，每2个epoch学习率衰减为原来的一半
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,2, gamma=0.5, last_epoch=-1)

    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    print('*'*105)

    graph_nums = 1
    for epoch in range(config.num_epochs):
        loss_list=[]#承接每个batch的loss
        acc_list=[]
        print('Epoch [{}/{}]    Starting Training...'.format(epoch + 1, config.num_epochs))
        for batch_idx, (trains, labels) in enumerate(train_iter):

            # 模型图只画一次
            if graph_nums == 1:
                graph_nums = 0
                # writer.add_graph(model,input_to_model=(trains,))

            fea,outputs = model(trains)
            optimizer.zero_grad()
  
            if config.usesloss:
                bloss = F.cross_entropy(outputs, labels)
                sloss=SupConLoss()
                sloss=sloss(fea,labels=labels)
                loss=(bloss+sloss)/2
            else:
                loss = F.cross_entropy(outputs, labels)

            #print(bloss, sloss, loss)
            loss.backward()
            optimizer.step()

            true = labels.data.cpu()
            predic = torch.max(outputs.data, 1)[1].cpu()
            train_acc = metrics.accuracy_score(true, predic)
            
            # x轴是total_batch，y轴是每个批次loss
            writer.add_scalar('train/loss_batch', loss.item(),total_batch)
            # x轴是total_batch，y轴是每个批次acc
            writer.add_scalar('train/acc_batch',train_acc,total_batch)

            msg1 = 'Iter: {0:>4},  Train Loss: {1:>5.2},  Train Acc: {2:>7.2%}'
            if total_batch%20==0:
                # 20批次打印一次
                print(msg1.format(total_batch, loss.item(), train_acc))
            loss_list.append(loss.item())
            acc_list.append(train_acc)

            # 总批次+1
            total_batch += 1 
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过2000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break

        # 一个epoch结束后，就用模型对验证集进行一次评估，每次保存效果最好的模型
        dev_acc, dev_loss = evaluate(config, model, dev_iter)#model.eval()
        if dev_loss < dev_best_loss:
            dev_best_loss = dev_loss
            torch.save(model.state_dict(), config.save_path)
            improve = '*'
            last_improve = total_batch
        else:
            improve = ''
        time_dif = get_time_dif(start_time)
        all_times += time_dif                               # 时间累加
        start_time = time.time()                            # 时间重置
        epoch_loss=np.mean(loss_list)
        epoch_acc=np.mean(acc_list)
        msg2 = 'Average Train Loss: {0:>5.2},  Average Train Acc: {1:>7.2%},  Val Loss: {2:>5.2},  Val Acc: {3:>7.2%},  Time: {4} {5}'
        print(msg2.format(epoch_loss, epoch_acc, dev_loss, dev_acc, time_dif, improve))

        writer.add_scalar('train/loss_epoch',epoch_loss, epoch)     # x轴epoch
        writer.add_scalar('train/acc_epoch', epoch_acc, epoch)
        writer.add_scalar('val/loss_epoch', dev_loss, epoch)
        writer.add_scalar('val/acc_epoch', dev_acc, epoch)

        model.train()
        scheduler.step()
        print(f'bert_lr:  {scheduler.get_last_lr()[0]},  resnet_lr:  {scheduler.get_last_lr()[1]},  other_lr:  {scheduler.get_last_lr()[2]}')
        print('Epoch [{}/{}]    End of Training...'.format(epoch + 1, config.num_epochs))
        print('*'*105)

    print('All Training Time usage: ',all_times)



def test(config, model, test_iter):
    
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...") #精确率和召回率以及调和平均数
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("All Testing Time usage: ", time_dif)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            #print(texts)

            fea,outputs = model(texts)
            if config.usesloss:
                bloss = F.cross_entropy(outputs, labels)
                sloss=SupConLoss()
                sloss=sloss(fea,labels=labels)
                loss=(bloss+sloss)/2
            else:
                loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()  ###预测结果
            # print(outputs)    # [[0.9746, 0.0254],[0.9791, 0.0209],...,[0.0254, 0.9746]]
            # print(predic)     # [0, 0,..., 1]
            # print(labels)     # [0, 0,..., 1]
            # print('*************************')
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        # 纵轴解析：0和1:分类类别、accuracy:准确率、macro avg:宏平均值、weighted avg:加权平均值
        # 横轴解析：precision:精确率、recall:召回率、f1-score:f1-score、support:支持度
        report = metrics.classification_report(labels_all, predict_all, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)



if __name__ == '__main__':

    logging.set_verbosity_error() # 屏蔽警告信息
    config = Config()
    writer = SummaryWriter(log_dir=config.log_dir)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    print("Loading data...")

    train_data=My_Dataset('./data/train.csv',config,1)
    dev_data = My_Dataset('./data/val.csv',config,1)
    test_data = My_Dataset('./data/test.csv',config,1)

    print("Loading data finished!")

    train_iter = DataLoader(train_data, batch_size=config.batch_size,shuffle=True)    # 训练集迭代器
    dev_iter = DataLoader(dev_data, batch_size=config.batch_size,shuffle=True)        # 验证集迭代器
    test_iter = DataLoader(test_data, batch_size=config.batch_size,shuffle=True)      # 测试集迭代器
    # 训练
    mynet =Mynet(config)
    # 模型放入到GPU中去
    mynet= mynet.to(config.device)
    # print(mynet.parameters)


    train(config, mynet, train_iter, dev_iter, writer)
    test(config, mynet, test_iter)

#tensorboard --logdir=log/bert-base-chinese_resnet18 --port=6006