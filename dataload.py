# import torch.utils.data as Data
# import torch
# import pickle
# import numpy as np
# import math
# import gc
# from prefetch_generator import BackgroundGenerator
#
# class DataloaderX(Data.DataLoader):
#     def __iter__(self):
#         return BackgroundGenerator(super().__iter__())
#
#
# class MyDataset(Data.Dataset):
#     def __init__(self, filepath, n_time_interval):
#         # 获得训练数据的总行
#         _, x, L, y, sz, time, trend, _ = pickle.load(open(filepath, 'rb'))
#
#         self.number = len(x)
#         batch_x = []
#         batch_L = []
#         batch_y = np.zeros(shape=(len(x), 1))
#         rnn_index_sample = []
#
#         batch_time_interval_index_sample = []
#         for i in range(len(x)):
#             # x
#             temp_x = []
#             for k in range(len(x[i])):
#                 temp_x.append(x[i][k].todense().tolist())
#             batch_x.append(temp_x)
#             batch_L.append(L[i].todense().tolist())
#             batch_y[i,0] = y[i]
#             n_steps = len(x[i])
#             batch_time_interval_index_sample.append(time[i].tolist())
#             rnn_index_sample.append([time[i].sum(axis=1).astype(int)])
#
#
#         self.x = torch.tensor(batch_x,dtype=torch.float32)
#         self.L = torch.tensor(batch_L,dtype=torch.float32)
#         self.y = torch.tensor(batch_y,dtype=torch.float32)
#         self.time_interval_index = torch.tensor(batch_time_interval_index_sample,dtype=torch.float32)
#         self.rnn_index = torch.tensor(rnn_index_sample)
#         self.rnn_index = torch.reshape(self.rnn_index, (-1, n_steps))
#         trend = torch.tensor(trend,dtype=torch.float32)
#         trend = trend.permute(0,2,1)
#         self.trend = trend
#
#         gc.collect()
#
#     def __len__(self):
#         return self.number
#
#     def __getitem__(self, idx):
#         return self.x[idx], self.L[idx], self.y[idx], self.time_interval_index[idx], self.rnn_index[idx],self.trend[idx]

#trend_5
import torch.utils.data as Data
import torch
import pickle
import numpy as np
import math
import gc
from prefetch_generator import BackgroundGenerator

class DataloaderX(Data.DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

#filepath = 'D:\\学术相关\\007.CasCN-master\\dataset_weibo'
class MyDataset(Data.Dataset):
    def __init__(self, filepath, trend_dim):
        # 获得训练数据的总行
        id, x, L, y, sz, time, trend, _ = pickle.load(open(filepath, 'rb'))

        self.number = len(x)
        batch_x = []
        batch_L = []
        batch_y = np.zeros(shape=(len(x), 1))
        rnn_index_sample = []
        batch_trend=[]


        batch_time_interval_index_sample = []
        for i in range(len(x)):
            # x
            temp_x = []
            for k in range(len(x[i])):
                temp_x.append(x[i][k].todense().tolist())
            batch_x.append(temp_x)
            batch_L.append(L[i].todense().tolist())
            batch_y[i,0] = y[i]
            n_steps= len(x[0])

            batch_time_interval_index_sample.append(time[i].tolist())
            rnn_index_sample.append([time[i].sum(axis=1).astype(int)])
            padding = np.zeros(trend_dim-1,dtype=float).tolist()
            if len(trend[i]):
                trend_p = padding + trend[i][0]
                temp_trend = []
                for j in range(len(trend_p) - trend_dim + 1):
                    temp_trend.append(trend_p[j:j + trend_dim])
                batch_trend.append(temp_trend)
            else:
                batch_trend.append([])

        self.id = id
        self.x = torch.tensor(batch_x,dtype=torch.float32)
        del batch_x
        self.L = torch.tensor(batch_L,dtype=torch.float32)
        del batch_L
        self.y = torch.tensor(batch_y,dtype=torch.float32)
        del batch_y
        self.time_interval_index = torch.tensor(batch_time_interval_index_sample,dtype=torch.float32)
        del batch_time_interval_index_sample
        self.rnn_index = torch.tensor(rnn_index_sample)
        self.rnn_index = torch.reshape(self.rnn_index, (-1,n_steps))
        batch_trend = torch.tensor(batch_trend,dtype=torch.float32)
        #batch_trend = batch_trend.permute(0,2,1)
        self.trend = batch_trend
        del batch_trend

        gc.collect()

    def __len__(self):
        return self.number

    def __getitem__(self, idx):
        return self.id[idx],self.x[idx], self.L[idx], self.y[idx], self.time_interval_index[idx], self.rnn_index[idx],self.trend[idx]
