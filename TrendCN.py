import torch.nn as nn
import torch
import numpy as np
import math
import pickle
import time
from PoPnet_model import dataload, ChebyLSTM
import torch.utils.data as Data
import os
from torch.optim import  lr_scheduler


torch.manual_seed(6)

#*********data**********
observation = 3* 3600 -1
n_time_interval = 6 #
n_steps = 90# sequence 的数量
time_interval = math.ceil((observation+1)*1.0/n_time_interval)
batch_size =16

hidden_dim = (32,)
kernel_size = (2,)
num_layers = 1
input_dim = 6
node_feature =6
feature_dim =10
trend_dim = 5
dense1 = 32
dense2 = 16
lr = 0.01 #1e-2
Epoch = 41

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #"cuda:0" if torch.cuda.is_available() else
data_path ="E:\\\WDK_workshop\\TrendCN\\对比实验\\weibo\\ds\\ob=3\\sequ=90"
# E:\\学术相关\\TrendCN\\对比实验\\twitter\\ds_new\\ob=1 sequ=30
result_path ="E:\\\WDK_workshop\\TrendCN\\对比实验\\weibo\\ds\\ob=3\\sequ=90"

log_model = result_path +"\\trend=3\\0_0.01_casCN_180.pth"
batch_first = True
model = ChebyLSTM.MODEL(input_dim, node_feature, feature_dim, hidden_dim, kernel_size, num_layers,
                         batch_first, n_time_interval, trend_dim, dense1, dense2,device)

model.to(device)

criterion = nn.MSELoss(reduction= 'mean')


#opt_normal = torch.optim.Adam(model.parameters(),lr=lr)
opt_decay = torch.optim.SGD(model.parameters(),lr=lr, weight_decay=1e-2,momentum=0.09)
#scheduler = lr_scheduler.StepLR(opt_decay, step_size=5, gamma=0.1)

start = time.time()
max_try = 10
patience = max_try
best_val_loss =10000
best_test_loss =10000

#预加载模型
if os.path.exists(log_model):
    checkpoint = torch.load(log_model)
    model.load_state_dict(checkpoint['model'])
    opt_decay.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']+1
    print("load epoch {} success!".format(start_epoch))
else:
    start_epoch = 0
    print("start from epoch {}".format(start_epoch))


# print("Model's state_dict:")
# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd
# embedding = model.state_dict()['node_embedding.embedding']
# em_ay=embedding.cpu().numpy()
# print(type(em_ay))
#
# def draw_heatmap(data):
#     ylabels = data.columns.values.tolist()
#
#     plt.subplots(figsize=(6, 10)) # 设置画面大小
#     sns.heatmap(data, annot=True, vmax=1, square=True,yticklabels=ylabels,xticklabels=ylabels, cmap="RdBu")
#     plt.savefig('E:\\学术相关\\TrendCN\\对比实验\\weibo\\ob=3\\sequ=90\\trend=3\\em.jpg')
#     plt.show()
#
# data =pd.DataFrame(em_ay)
# draw_heatmap(data)
#
#
# for param_tensor in model.state_dict():
#     print(param_tensor, "\t", model.state_dict()[param_tensor].size())

def train(filepath, e, is_train):
    filelist = os.listdir(filepath)
    train_loss = 0
    train_step = 0
    for file in filelist:
        data_train = dataload.MyDataset(os.path.join(filepath, file), trend_dim)
        batch_data_train = dataload.DataloaderX(data_train, batch_size=batch_size, drop_last=True, shuffle=True)

        for id, batch in enumerate(batch_data_train):
            _,batch_x, batch_L, batch_y, batch_time_interval, batch_rnn_index, batch_trend = batch
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_L = batch_L.to(device)
            batch_time_interval =batch_time_interval.to(device)
            batch_rnn_index = batch_rnn_index.to(device)
            batch_trend = batch_trend.to(device)

            if not is_train =='train':
                model.eval()
                with torch.no_grad():
                    train_step += 1
                    pred = model(batch_x, batch_L, n_steps,
                                 hidden_dim, batch_rnn_index, batch_time_interval, batch_trend)

                    loss = criterion(pred.float(), batch_y.float())

                    train_loss += float(loss.mean())
                    print(str(is_train), train_loss / train_step)
                    file_tr_loss = open(result_path + '\\trend=3\\file_'+str(is_train)+'_loss' + str(e)+'_'+str(lr) + '.txt', 'a')
                    file_tr_loss.write(str(train_loss / train_step) + '\n')
            else:
                train_step += 1
                opt_decay.zero_grad()

                pred = model(batch_x, batch_L, n_steps,
                             hidden_dim, batch_rnn_index, batch_time_interval, batch_trend)

                loss = criterion(pred.float(), batch_y.float())

                train_loss += float(loss.mean())
                print(str(is_train), train_loss / train_step)
                file_tr_loss = open(result_path + '\\trend=3\\file_'+str(is_train)+'_loss'  + str(e)+ '_'+str(lr)+ '.txt', 'a')
                file_tr_loss.write(str(train_loss / train_step) + '\n')

                # opt1.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=5, norm_type=2)
                opt_decay.step()

    train_loss = train_loss/train_step
    if is_train == 'train':
        state = {'model': model.state_dict(), 'optimizer': opt_decay.state_dict(), 'epoch': e}
        torch.save(state, result_path + '\\trend=3\\' + str(e) + '_'+str(lr)+'_casCN_180.pth')
    del batch_x, batch_L, batch_y, batch_time_interval, batch_rnn_index, batch_trend
    return train_loss


best_val_loss = 10000
best_test_loss = 10000
for e in range(start_epoch, Epoch):


    filepath = data_path + '\\data_train\\'
    is_train = 'train'
    train_loss = train(filepath,e,is_train)
    torch.cuda.empty_cache()

    if e%2 == 0:
        is_train = 'val'
        filepath = data_path + '\\data_val\\'
        val_loss = train(filepath,e,is_train)

        is_train = 'test'
        filepath = data_path + '\\data_test\\'
        test_loss = train(filepath, e, is_train)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = max_try
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience = max_try

        print("#" + str(e) +
              ", Training Loss= " + "{:.6f}".format(train_loss) +
              ", Validation Loss= " + "{:.6f}".format(val_loss) +
              ", Best Valid Loss= " + "{:.6f}".format(best_val_loss)+
              ", Test Loss= " + "{:.6f}".format(test_loss) +
              ", Best Test Loss= " + "{:.6f}".format(best_test_loss)
              )
        with open(result_path+'\\trend=3\\result.txt','a') as result:
            result.write("#" + str(e) +
              ", Training Loss= " + "{:.6f}".format(train_loss) +
              ", Validation Loss= " + "{:.6f}".format(val_loss) +
              ", Best Valid Loss= " + "{:.6f}".format(best_val_loss)+
              ", Test Loss= " + "{:.6f}".format(test_loss) +
              ", Best Test Loss= " + "{:.6f}".format(best_test_loss)+'\n')


        model.train()
        patience -= 1
        if not patience:
            break



print("Finished!\n----------------------------------------------------------------")
print("Time:", time.time() - start)
print("Valid Loss:", best_val_loss)
print("Test Loss:", test_loss)
