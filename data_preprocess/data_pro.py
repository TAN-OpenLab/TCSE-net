#数据划分为一个文件96条数据，防止内存爆炸
import pickle
import math
import gc
# path ="G:\\学术相关\\TrendCN\\对比实验\\twitter\\ds_new\\ob=0.5 sequ=15"
# #"G:\\学术相关\\TrendCN\\对比实验\\twitter\\ob=1.5\\sequ=45"
# data = ['data_train', 'data_test', 'data_val']
path = 'E:\\学术相关\\TrendCN\\对比实验\\acc_graph\\weibo_3'
data=['trendCN_test_dlt']
for d in data:
    i=0
    id_train, x_train, L, y_train, sz_train, time_train, trend, vocabulary_size = pickle.load(
        open(path + '\\'+str(d)+'.pkl', 'rb'))
    step = 1
    filename = path+ '\\' + str(d)+'\\'+str(d)+'_'
    print(str(d),len(id_train))  # train_41975 test_8950 val_8941
    for i in range(math.floor(len(id_train) / 96)):
        pickle.dump((id_train[i * 96:(i + 1) * 96], x_train[i * 96:(i + 1) * 96], L[i * 96:(i + 1) * 96],
                     y_train[i * 96:(i + 1) * 96],
                     sz_train[i * 96:(i + 1) * 96], time_train[i * 96:(i + 1) * 96], trend[i * 96:(i + 1) * 96],
                     vocabulary_size),
                    open(filename + str(i) + '.pkl', 'wb'))
    if len(id_train) % 96 != 0:
        pickle.dump((id_train[i * 96:], x_train[i * 96:], L[i * 96:],
                     y_train[i * 96:],
                     sz_train[i * 96:], time_train[i * 96:], trend[i * 96:],
                     vocabulary_size), open(filename + str(i + 1) + '.pkl', 'wb'))
gc.collect()

# #数据划分为一个文件96条数据，防止内存爆炸
# import pickle
# import math
# import gc
# path ="G:\\学术相关\\TrendCN\\对比实验\\casca_weibo\\ob=2"
# data = ['data_train', 'data_test', 'data_val']
# for d in data:
#     id_train, x_train, L, y_train, sz_train, time_train, vocabulary_size = pickle.load(
#         open(path + '\\'+str(d)+'.pkl', 'rb'))
#     step = 1
#     filename = path + str(d)+'\\'+str(d)+'_'
#     print(len(id_train))  # train_41975 test_8950 val_8941
#     for i in range(math.floor(len(id_train) / 96)):
#         pickle.dump((id_train[i * 96:(i + 1) * 96], x_train[i * 96:(i + 1) * 96], L[i * 96:(i + 1) * 96],
#                      y_train[i * 96:(i + 1) * 96],
#                      sz_train[i * 96:(i + 1) * 96], time_train[i * 96:(i + 1) * 96],
#                      vocabulary_size),
#                     open(filename + str(i) + '.pkl', 'wb'))
#     if len(id_train) % 96 != 0:
#         pickle.dump((id_train[i * 96:], x_train[i * 96:], L[i * 96:],
#                      y_train[i * 96:],
#                      sz_train[i * 96:], time_train[i * 96:],
#                      vocabulary_size), open(filename + str(i + 1) + '.pkl', 'wb'))
# gc.collect()

