# import numpy as np
# import six.moves.cPickle as pickle
# from data_preprocess import config
# import networkx as nx
# from data_preprocess import Laplacian
# import scipy.sparse
# from collections import Counter
# import gc
# LABEL_NUM = 0
# import math
#
# # trans the original ids to 1~n
# class IndexDict:
#     def __init__(self, original_ids):
#         self.original_to_new = {}
#         self.new_to_original = []
#         cnt = 0
#         for i in original_ids:
#             new = self.original_to_new.get(i, cnt)  #get（key,default）,
#             if new == cnt:   #节点i为未加入o_t_n，为节点i编号放入o_t_n
#                 self.original_to_new[i] = cnt
#                 cnt += 1
#                 self.new_to_original.append(i)  #去重后的original，对应new_to_original
#
#     def new(self, original):
#         if type(original) is int:
#             return self.original_to_new[original]
#         else:
#             if type(original[0]) is int:
#                 return [self.original_to_new[i] for i in original]
#             else:
#                 return [[self.original_to_new[i] for i in l] for l in original]
#
#     def original(self, new):
#         if type(new) is int:
#             return self.new_to_original[new]
#         else:
#             if type(new[0]) is int:
#                 return [self.new_to_original[i] for i in new]
#             else:
#                 return [[self.new_to_original[i] for i in l] for l in new]
#
#     def length(self):
#         return len(self.new_to_original)
#
# #trainsform the sequence to list
# #graphs： 字典{级联id：【【种子】，时间】，【【节点1，节点2】，时间】，【节点3，节点4】，时间】】}
# def sequence2list(flename):
#     graphs = {}
#     with open(flename, 'r') as f:
#         for line in f:
#             walks = line.strip().split('\t')
#             graphs[walks[0]] = [] #walk[0] = cascadeID
#             for i in range(1, len(walks)):
#                 s = walks[i].split(":")[0] #node
#                 t = walks[i].split(":")[1] #time
#                 graphs[walks[0]].append([[str(xx) for xx in s.split(",")],int(t)])
#     return graphs
#
# #read label and size from cascade file
# # label： 字典{级联id:label} 级联在三小时后的增量
# # sizes：字典{级联id：级联的边数量} 级联在三小时内的转发量
# # 二者相加 为级联的总转发量
# def read_labelANDsize(filename):
#     labels = {}
#     sizes = {}
#     with open(filename, 'r') as f:
#         for line in f:
#             profile = line.split('\t')
#             labels[profile[0]] = profile[-1]
#             sizes[profile[0]] = int(profile[3])
#     return labels,sizes
#
# #original_ids:每个图的ID
# def get_original_ids(graphs):
#     original_ids = set()
#     for graph in graphs.keys():
#         for walk in graphs[graph]:
#             for i in walk[0]:
#                 original_ids.add(i)
#     print ("length of original ids:",len(original_ids))
#     return original_ids
#
# def get_nodes(graph):
#     nodes = {}
#     j = 0
#     for walk in graph:
#         for i in walk[0]:
#             if i not in nodes.keys():
#                 nodes[i] = j
#                 j = j+1
#     return nodes
#
# def get_max_deepth(graphs):
#     node_deepth = {}
#     max_deepth = 0
#     #max_leaf = 0
#     for graph in graphs.values():
#         for walk in graph:
#             if walk[1] == 0:
#                 node_deepth[walk[0][0]] = 1
#             else:
#                 for i in range(len(walk[0]) - 1):
#                     if walk[0][i] in node_deepth.keys():
#                         node_deepth[walk[0][i + 1]] = node_deepth.get(walk[0][i]) + 1
#                     else:
#                         node_deepth[walk[0][i]] = 0
#                         node_deepth[walk[0][i + 1]] = node_deepth.get(walk[0][i]) + 1
#         deepth = max(node_deepth.values())
#         max_deepth = max(max_deepth,deepth)
#         #leafs = max(Counter(node_deepth.values()).values())
#         #max_leaf = max(max_leaf,leafs)
#     return max_deepth#,max_leaf
#
# def BFS_label(G,root):
#     G_bfs_node = {}
#     G_BFS = nx.DiGraph()
#     node_ith_no = {}
#     i = 0
#     G_bfs_node[root] = i
#     node_ith_no[root] = 0
#     this_root= set()
#     this_root.add(root)
#     has_visit = set()
#     #has_visit.add(root)
#     G_BFS.add_node(G_bfs_node.get(root))
#     isolates = list(nx.isolates(G))
#     G.remove_nodes_from(list(nx.isolates(G)))
#     while len(has_visit) != nx.number_of_nodes(G):
#         next_root = set()
#         for n in this_root:
#             node = set()
#             if n not in has_visit:
#                 has_visit.add(n)
#                 for s in G.successors(n):
#                     node.add(s)
#                     next_root.add(s)
#             node = sorted(node, key=lambda d:G.nodes[d]['time'])
#             j = 0
#             for s in node:
#                 if s not in G_bfs_node.keys():
#                     i += 1
#                     G_bfs_node[s] = i
#                     node_ith_no[s] = j
#                     j += 1
#                 G_BFS.add_edge(G_bfs_node.get(n), G_bfs_node.get(s))
#         this_root = next_root
#     j = 0
#     for s in isolates:
#         i += 1
#         G_BFS[s] = i
#         G_BFS.add_node(G_bfs_node.get(s))
#         node_ith_no[s] = j
#         j += 1
#     return G_BFS, G_bfs_node,node_ith_no
#
# def G_time_ordered(G):
#     nodes = list(G.nodes)
#     nodes = sorted(nodes, key=lambda d: G.nodes[d]['time'])
#     nodes_times ={}
#     G_time = nx.DiGraph()
#     for (m,n) in G.edges():
#         nodes_times[m] = nodes.index(m)
#         nodes_times[n] = nodes.index(n)
#         G_time.add_node(nodes_times.get(m),time = G.nodes[m]['time'])
#         G_time.add_node(nodes_times.get(n), time=G.nodes[n]['time'])
#         G_time.add_edge(nodes_times.get(m),nodes_times.get(n))
#
#     return G_time, nodes_times
#
# #处理数据 将其转化为输入格式， 节点的embedding X【级联总数，num—sequence，max-num，max-num】，级联所在局部网络的拉普拉斯矩阵L，log（Y），每个级联内连接的发生时间
# def write_XYSIZE_data(graphs,labels,sizes,LEN_SEQUENCE,NUM_SEQUENCE,index,max_num, n_time_interval,filename):
#     #get the x,y,and size  data
#     id_data = []
#     x_data = []
#     y_data = []
#     sz_data = []
#     time_data = []
#     trend_data = []
#     Laplacian_data = []
#     l=0
#     maxdeep =0
#
#     for key,graph in graphs.items():
#         print(l)
#         l+=1
#         id = key
#         label = labels[key].split()
#         y = int(label[LABEL_NUM]) #label
#         temp = []
#         temp_time = np.zeros([NUM_SEQUENCE, n_time_interval],int)#store time
#         temp_trend = np.zeros([1,NUM_SEQUENCE],int)
#         size_temp = len(graph)
#         if size_temp !=  sizes[key]:
#             print (size_temp,sizes[key])
#
#         #nodes_items = get_nodes(graph)  #级联的所有节点{节点id：节点编号}
#         #nodes_list = nodes_items.values()
#         nx_G = nx.DiGraph()
#         #nx_G.add_nodes_from(nodes_list)
#         #将每个级联内部节点间的邻接矩阵（带有自环）
#         temp_dict = {}
#         node_deepth = {}
#         for walk in graph:
#             if walk[1] == 0:
#                 node_deepth[walk[0][0]] = 0
#                 nx_G.add_node(walk[0][0], time=walk[1])
#                 root = walk[0][0]
#             for i in range(len(walk[0]) - 1):
#                 nx_G.add_node(walk[0][i + 1], time=walk[1])
#                 nx_G.add_edge(walk[0][i], walk[0][i + 1])
#                 if walk[0][i] in node_deepth.keys():
#                     node_deepth[walk[0][i + 1]] = node_deepth.get(walk[0][i]) + 1
#                 else:
#                     node_deepth[walk[0][i]] = 0
#                     node_deepth[walk[0][i + 1]] = node_deepth.get(walk[0][i]) + 1
#
#
#             walk_time = math.floor(walk[1] / ((config.observation+1)/ NUM_SEQUENCE)) # 3*60 *60/180
#             time_interval = math.floor(walk[1] / ((config.observation+1)/ n_time_interval))#观察时长内划分六个时间间隔
#             temp_time[walk_time, time_interval] = 1
#             temp_trend[0, walk_time] +=1
#             if not temp_dict.get(walk_time):
#                 temp_dict[walk_time] = []
#             temp_dict[walk_time].append(walk[0])
#
#         #temp_dict = sorted(temp_dict.items(), key=lambda a: a[0])
#         #G_BFS, G_bfs_node, node_ith_no = BFS_label(nx_G, root)
#         G_time, node_times_orders = G_time_ordered(nx_G)
#         temp_emb = np.zeros(shape=(max_num, 50))
#         for i in range(NUM_SEQUENCE):
#             if i in temp_dict.keys():
#                 for value in temp_dict.get(i):
#                     for p in range(len(value)):
#                         d1 = node_deepth[value[p]] // 100 % 10
#                         d2 = node_deepth[value[p]] //10 % 10
#                         d3 = node_deepth[value[p]] // 1 % 10
#                         n1 = node_times_orders[value[p]] //10 % 10
#                         n2 = node_times_orders[value[p]] //1 % 10
#                         temp_emb[node_times_orders[value[p]]][10-1-d1] = 1
#                         temp_emb[node_times_orders[value[p]]][10*2 - 1 - d2] = 1
#                         temp_emb[node_times_orders[value[p]]][10*3 - 1 - d3] = 1
#                         temp_emb[node_times_orders[value[p]]][10*4 - 1 - n1] = 1
#                         temp_emb[node_times_orders[value[p]]][10*5 -1 - n2] = 1
#                 temp_s = scipy.sparse.coo_matrix(temp_emb, dtype=np.float32)
#                 temp.append(temp_s)
#             else:
#                 temp_s = temp[i-1]
#                 temp.append(temp_s)
#
#         deep =  max(list(node_deepth.values()))
#         maxdeep = max(deep,maxdeep)
#         #caculate laplacian
#         L = Laplacian.calculate_scaled_laplacian_dir(G_time, kind_of_laplacin= 'caslaplacian', lambda_max=None)
#         M, M = L.shape
#         M = int(M)
#         L = L.todense()
#         if M < max_num:
#             col_padding_L = np.zeros(shape=(M, max_num - M))
#             L_col_padding = np.column_stack((L, col_padding_L))
#             row_padding = np.zeros(shape=(max_num - M, max_num))
#             L_col_row_padding = np.row_stack((L_col_padding, row_padding))
#             Lapla = scipy.sparse.coo_matrix(L_col_row_padding, dtype=np.float32)
#         else:
#             Lapla = scipy.sparse.coo_matrix(L, dtype=np.float32)
#
#         time_data.append(temp_time)
#         trend_data.append((np.log(temp_trend+1.0)/np.log(2.0)).tolist())
#         id_data.append(id)
#         x_data.append(temp)
#         y_data.append(np.log(y+1.0)/np.log(2.0))
#         Laplacian_data.append(Lapla)
#         sz_data.append(size_temp)
#     gc.collect()
#     print('maxdeepth',maxdeep)
#     pickle.dump((id_data,x_data,Laplacian_data, y_data, sz_data, time_data, trend_data,index.length()), open(filename,'wb'))
#
# def get_maxsize(sizes):
#     max_size = 0
#     for cascadeID in sizes:
#         max_size = max(max_size,sizes[cascadeID])
#     gc.collect()
#     return max_size
#
# #级联的最大长度（级联中边的数量）
# def get_max_length(graphs):
#     len_sequence = 0
#     max_num = 0
#     for cascadeID in graphs:
#         max_num = max(max_num,len(graphs[cascadeID]))
#         for sequence in graphs[cascadeID]:
#             len_sequence = max(len_sequence,len(sequence[0]))
#     gc.collect()
#     return len_sequence
#
# def get_max_node_num(graphs):
#     max_num = 0
#     for key,graph in graphs.items():
#         nodes = get_nodes(graph)
#         max_num = max(max_num,len(nodes))
#     return max_num
#
# if __name__ == "__main__":
#
#     ### data set 数据转换，输入为list###
#     graphs_train = sequence2list(config.shortestpath_train)  #
#     graphs_val = sequence2list(config.shortestpath_val)
#     graphs_test = sequence2list(config.shortestpath_test)
#
#     # train_depth = get_max_deepth(graphs_train)
#     # test_depth = get_max_deepth(graphs_test)
#     # val_depth = get_max_deepth(graphs_val)
#     # max_depth = max(test_depth,train_depth,val_depth)
#     # #max_leaf = max(train_leafs,test_leafs,val_leafs)
#     # print('max_depth',max_depth)
#     #print('max_leaf',max_leaf)
#
#     # get Laplacian ##
#     cascade_train = config.cascade_train
#     cascade_test = config.cascade_test
#     cascade_val = config.cascade_val
#
#     ### get labels ###
#     labels_train, sizes_train = read_labelANDsize(config.cascade_train)  # labels是{id：观测时间后的转发量}以及sizes级联长度{id：级联总的转发数量}
#     labels_val, sizes_val = read_labelANDsize(config.cascade_val)
#     labels_test, sizes_test = read_labelANDsize(config.cascade_test)
#     # NUM_SEQUENCE = max(get_maxsize(sizes_train),get_maxsize(sizes_val),get_maxsize(sizes_test)) #三小时内，转发最多的级联的大小 884
#     NUM_SEQUENCE =config.num_squ
#     print(NUM_SEQUENCE)
#
#     # LEN_SEQUENCE_train = get_max_length(graphs_train)  #每个数据集内， 级联内某一传播链的最大长度
#     # LEN_SEQUENCE_val = get_max_length(graphs_val)
#     # LEN_SEQUENCE_test = get_max_length(graphs_test)
#     # LEN_SEQUENCE = max(LEN_SEQUENCE_train,LEN_SEQUENCE_val,LEN_SEQUENCE_test) #26
#     # print(LEN_SEQUENCE)
#     LEN_SEQUENCE =0
#     #
#     max_num_train = get_max_node_num(graphs_train)  #参与级联的最大节点数
#     max_num_test = get_max_node_num(graphs_test)
#     max_num_val = get_max_node_num(graphs_val)
#     max_num = max(max_num_train, max_num_test, max_num_val)
#     print(max_num) #100
#     #
#     # get the total original_ids and tranform the index from 0 ~n-1
#     original_ids = get_original_ids(graphs_train)\
#                     .union(get_original_ids(graphs_val))\
#                     .union(get_original_ids(graphs_test))
#
#     original_ids.add(-1)
#     ## index is new index
#     index = IndexDict(original_ids)  #字典{节点对id：节点对}
#
#     print("create train")
#     write_XYSIZE_data(graphs_train, labels_train,sizes_train,LEN_SEQUENCE,NUM_SEQUENCE,index,max_num,config.n_time_interval, config.train_pkl)
#     print("create val an test")
#     write_XYSIZE_data(graphs_val, labels_val, sizes_val,LEN_SEQUENCE,NUM_SEQUENCE,index,max_num,config.n_time_interval, config.val_pkl)
#     write_XYSIZE_data(graphs_test, labels_test, sizes_test,LEN_SEQUENCE,NUM_SEQUENCE,index,max_num,config.n_time_interval, config.test_pkl)
#     pickle.dump((len(original_ids),NUM_SEQUENCE,LEN_SEQUENCE), open(config.information,'wb'))
#     print("Finish!!!")

#teittwr
import numpy as np
import six.moves.cPickle as pickle
from data_preprocess import config
import networkx as nx
from data_preprocess import Laplacian
import scipy.sparse
from collections import Counter
import gc
LABEL_NUM = 0
import math

# trans the original ids to 1~n
class IndexDict:
    def __init__(self, original_ids):
        self.original_to_new = {}
        self.new_to_original = []
        cnt = 0
        for i in original_ids:
            new = self.original_to_new.get(i, cnt)  #get（key,default）,
            if new == cnt:   #节点i为未加入o_t_n，为节点i编号放入o_t_n
                self.original_to_new[i] = cnt
                cnt += 1
                self.new_to_original.append(i)  #去重后的original，对应new_to_original

    def new(self, original):
        if type(original) is int:
            return self.original_to_new[original]
        else:
            if type(original[0]) is int:
                return [self.original_to_new[i] for i in original]
            else:
                return [[self.original_to_new[i] for i in l] for l in original]

    def original(self, new):
        if type(new) is int:
            return self.new_to_original[new]
        else:
            if type(new[0]) is int:
                return [self.new_to_original[i] for i in new]
            else:
                return [[self.new_to_original[i] for i in l] for l in new]

    def length(self):
        return len(self.new_to_original)

#trainsform the sequence to list
#graphs： 字典{级联id：【【种子】，时间】，【【节点1，节点2】，时间】，【节点3，节点4】，时间】】}
def sequence2list(flename):
    graphs = {}
    with open(flename, 'r') as f:
        for line in f:
            walks = line.strip().split('\t')
            graphs[walks[0]] = [] #walk[0] = cascadeID
            for i in range(1, len(walks)):
                s = walks[i].split(":")[0] #node
                t = walks[i].split(":")[1] #time
                graphs[walks[0]].append([[str(xx) for xx in s.split(",")],int(t)])
    return graphs

#read label and size from cascade file
# label： 字典{级联id:label} 级联在三小时后的增量
# sizes：字典{级联id：级联的边数量} 级联在三小时内的转发量
# 二者相加 为级联的总转发量
def read_labelANDsize(filename):
    labels = {}
    sizes = {}
    with open(filename, 'r') as f:
        for line in f:
            profile = line.split('\t')
            labels[profile[0]] = profile[-1]
            sizes[profile[0]] = int(profile[3])
    return labels,sizes

#original_ids:每个图的ID
def get_original_ids(graphs):
    original_ids = set()
    for graph in graphs.keys():
        for walk in graphs[graph]:
            for i in walk[0]:
                original_ids.add(i)
    print ("length of original ids:",len(original_ids))
    return original_ids

def get_nodes(graph):
    nodes = {}
    j = 0
    for walk in graph:
        for i in walk[0]:
            if i not in nodes.keys():
                nodes[i] = j
                j = j+1
    return nodes

def get_max_deepth(graphs):
    node_deepth = {}
    max_deepth = 0
    #max_leaf = 0
    for graph in graphs.values():
        for walk in graph:
            if walk[1] == 0:
                node_deepth[walk[0][0]] = 0
            else:
                for i in range(len(walk[0]) - 1):
                    if walk[0][i] in node_deepth.keys():
                        node_deepth[walk[0][i + 1]] = node_deepth.get(walk[0][i]) + 1
                    else:
                        node_deepth[walk[0][i]] = 0
                        node_deepth[walk[0][i + 1]] = node_deepth.get(walk[0][i]) + 1
        deepth = max(node_deepth.values())
        max_deepth = max(max_deepth,deepth)
        #leafs = max(Counter(node_deepth.values()).values())
        #max_leaf = max(max_leaf,leafs)
    return max_deepth#,max_leaf

def BFS_label(G,root):
    G_bfs_node = {}
    G_BFS = nx.DiGraph()
    node_ith_no = {}
    i = 0
    G_bfs_node[root] = i
    node_ith_no[root] = 0
    this_root= set()
    this_root.add(root)
    has_visit = set()
    #has_visit.add(root)
    G_BFS.add_node(G_bfs_node.get(root))
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(list(nx.isolates(G)))
    while len(has_visit) != nx.number_of_nodes(G):
        next_root = set()
        for n in this_root:
            node = set()
            if n not in has_visit:
                has_visit.add(n)
                for s in G.successors(n):
                    node.add(s)
                    next_root.add(s)
            node = sorted(node, key=lambda d:G.nodes[d]['time'])
            j = 0
            for s in node:
                if s not in G_bfs_node.keys():
                    i += 1
                    G_bfs_node[s] = i
                    node_ith_no[s] = j
                    j += 1
                G_BFS.add_edge(G_bfs_node.get(n), G_bfs_node.get(s))
        this_root = next_root
    j = 0
    for s in isolates:
        i += 1
        G_BFS[s] = i
        G_BFS.add_node(G_bfs_node.get(s))
        node_ith_no[s] = j
        j += 1
    return G_BFS, G_bfs_node,node_ith_no

def G_time_ordered(G):
    nodes = list(G.nodes)
    nodes = sorted(nodes, key=lambda d: G.nodes[d]['time'])
    nodes_times ={}
    G_time = nx.DiGraph()
    for (m,n) in G.edges():
        nodes_times[m] = nodes.index(m)
        nodes_times[n] = nodes.index(n)
        G_time.add_node(nodes_times.get(m),time = G.nodes[m]['time'])
        G_time.add_node(nodes_times.get(n), time=G.nodes[n]['time'])
        G_time.add_edge(nodes_times.get(m),nodes_times.get(n))

    return G_time, nodes_times

#处理数据 将其转化为输入格式， 节点的embedding X【级联总数，num—sequence，max-num，max-num】，级联所在局部网络的拉普拉斯矩阵L，log（Y），每个级联内连接的发生时间
def write_XYSIZE_data(graphs,labels,sizes,LEN_SEQUENCE,NUM_SEQUENCE,index,max_num, n_time_interval,filename):
    #get the x,y,and size  data
    id_data = []
    x_data = []
    y_data = []
    sz_data = []
    time_data = []
    trend_data = []
    Laplacian_data = []
    l=0
    maxdeep =0

    for key,graph in graphs.items():
        print(l)
        l+=1
        id = key
        label = labels[key].split()
        y = int(label[LABEL_NUM]) #label
        temp = []
        temp_time = np.zeros([NUM_SEQUENCE, n_time_interval],int)#store time
        temp_trend = np.zeros([1,NUM_SEQUENCE],int)
        size_temp = len(graph)
        if size_temp !=  sizes[key]:
            print (size_temp,sizes[key])

        #nodes_items = get_nodes(graph)  #级联的所有节点{节点id：节点编号}
        #nodes_list = nodes_items.values()
        nx_G = nx.DiGraph()
        #nx_G.add_nodes_from(nodes_list)
        #将每个级联内部节点间的邻接矩阵（带有自环）
        temp_dict = {}
        node_deepth = {}
        for walk in graph:
            if walk[1] == 0:
                node_deepth[walk[0][0]] = 0
                nx_G.add_node(walk[0][0], time=walk[1])
                root = walk[0][0]
            for i in range(len(walk[0]) - 1):
                if walk[0][i] not in nx_G.nodes():
                    node_deepth[walk[0][0]] = 0
                    nx_G.add_node(walk[0][i], time=0)
                    nx_G.add_edge(walk[0][i], walk[0][i + 1])
                nx_G.add_node(walk[0][i + 1], time=walk[1])
                nx_G.add_edge(walk[0][i], walk[0][i + 1])
                if walk[0][i] in node_deepth.keys():
                    node_deepth[walk[0][i + 1]] = node_deepth.get(walk[0][i]) + 1
                else:
                    node_deepth[walk[0][i]] = 0
                    node_deepth[walk[0][i + 1]] = node_deepth.get(walk[0][i]) + 1


            walk_time = math.floor(walk[1] / ((config.observation+1)/ NUM_SEQUENCE)) # 3*60 *60/180
            time_interval = math.floor(walk[1] / ((config.observation+1)/ n_time_interval))#观察时长内划分六个时间间隔
            temp_time[walk_time, time_interval] = 1
            temp_trend[0, walk_time] +=1
            if not temp_dict.get(walk_time):
                temp_dict[walk_time] = []
            temp_dict[walk_time].append(walk[0])

        #temp_dict = sorted(temp_dict.items(), key=lambda a: a[0])
        #G_BFS, G_bfs_node, node_ith_no = BFS_label(nx_G, root)
        G_time, node_times_orders = G_time_ordered(nx_G)
        temp_emb = np.zeros(shape=(max_num, 50))
        for i in range(NUM_SEQUENCE):
            if i in temp_dict.keys():
                for value in temp_dict.get(i):
                    for p in range(len(value)):
                        d1 = node_deepth[value[p]] // 100 % 10
                        d2 = node_deepth[value[p]] //10 % 10
                        d3 = node_deepth[value[p]] // 1 % 10
                        n1 = node_times_orders[value[p]] //10 % 10
                        n2 = node_times_orders[value[p]] //1 % 10
                        temp_emb[node_times_orders[value[p]]][10-1-d1] = 1
                        temp_emb[node_times_orders[value[p]]][10*2 - 1 - d2] = 1
                        temp_emb[node_times_orders[value[p]]][10*3 - 1 - d3] = 1
                        temp_emb[node_times_orders[value[p]]][10*4 - 1 - n1] = 1
                        temp_emb[node_times_orders[value[p]]][10*5 -1 - n2] = 1
                temp_s = scipy.sparse.coo_matrix(temp_emb, dtype=np.float32)
                temp.append(temp_s)
            else:
                temp_s = temp[i-1]
                temp.append(temp_s)

        deep =  max(list(node_deepth.values()))
        maxdeep = max(deep,maxdeep)
        #caculate laplacian
        L = Laplacian.calculate_scaled_laplacian_dir(G_time, kind_of_laplacin= 'caslaplacian', lambda_max=None)
        M, M = L.shape
        M = int(M)
        L = L.todense()
        if M < max_num:
            col_padding_L = np.zeros(shape=(M, max_num - M))
            L_col_padding = np.column_stack((L, col_padding_L))
            row_padding = np.zeros(shape=(max_num - M, max_num))
            L_col_row_padding = np.row_stack((L_col_padding, row_padding))
            Lapla = scipy.sparse.coo_matrix(L_col_row_padding, dtype=np.float32)
        else:
            Lapla = scipy.sparse.coo_matrix(L, dtype=np.float32)

        time_data.append(temp_time)
        trend_data.append((np.log(temp_trend+1.0)/np.log(2.0)).tolist())
        id_data.append(id)
        x_data.append(temp)
        y_data.append(np.log(y+1.0)/np.log(2.0))
        Laplacian_data.append(Lapla)
        sz_data.append(size_temp)
    gc.collect()
    print('maxdeepth',maxdeep)
    pickle.dump((id_data,x_data,Laplacian_data, y_data, sz_data, time_data, trend_data,index.length()), open(filename,'wb'))

def get_maxsize(sizes):
    max_size = 0
    for cascadeID in sizes:
        max_size = max(max_size,sizes[cascadeID])
    gc.collect()
    return max_size

#级联的最大长度（级联中边的数量）
def get_max_length(graphs):
    len_sequence = 0
    max_num = 0
    for cascadeID in graphs:
        max_num = max(max_num,len(graphs[cascadeID]))
        for sequence in graphs[cascadeID]:
            len_sequence = max(len_sequence,len(sequence[0]))
    gc.collect()
    return len_sequence

def get_max_node_num(graphs):
    max_num = 0
    for key,graph in graphs.items():
        nodes = get_nodes(graph)
        max_num = max(max_num,len(nodes))
    return max_num

if __name__ == "__main__":

    ### data set 数据转换，输入为list###
    graphs_train = sequence2list(config.shortestpath_train)  #
    graphs_val = sequence2list(config.shortestpath_val)
    graphs_test = sequence2list(config.shortestpath_test)

    train_depth = get_max_deepth(graphs_train)
    test_depth = get_max_deepth(graphs_test)
    val_depth = get_max_deepth(graphs_val)
    max_depth = max(test_depth,train_depth,val_depth)
    #max_leaf = max(train_leafs,test_leafs,val_leafs)
    print('max_depth',max_depth)
    #print('max_leaf',max_leaf)

    # get Laplacian ##
    cascade_train = config.cascade_train
    cascade_test = config.cascade_test
    cascade_val = config.cascade_val

    ### get labels ###
    labels_train, sizes_train = read_labelANDsize(config.cascade_train)  # labels是{id：观测时间后的转发量}以及sizes级联长度{id：级联总的转发数量}
    labels_val, sizes_val = read_labelANDsize(config.cascade_val)
    labels_test, sizes_test = read_labelANDsize(config.cascade_test)
    NUM_SEQUENCE = max(get_maxsize(sizes_train),get_maxsize(sizes_val),get_maxsize(sizes_test)) #三小时内，转发最多的级联的大小 884
    #NUM_SEQUENCE =config.num_squ
    print('numsequence')
    print(get_maxsize(sizes_train))
    print(get_maxsize(sizes_val))
    print(get_maxsize(sizes_test))

    LEN_SEQUENCE_train = get_max_length(graphs_train)  #每个数据集内， 级联内某一传播链的最大长度
    LEN_SEQUENCE_val = get_max_length(graphs_val)
    LEN_SEQUENCE_test = get_max_length(graphs_test)
    LEN_SEQUENCE = max(LEN_SEQUENCE_train,LEN_SEQUENCE_val,LEN_SEQUENCE_test) #26
    print('LEN_SEQUENCE')
    print(LEN_SEQUENCE_train)
    print(LEN_SEQUENCE_val)
    print(LEN_SEQUENCE_test)
    #LEN_SEQUENCE =0
    #
    max_num_train = get_max_node_num(graphs_train)  #参与级联的最大节点数
    max_num_test = get_max_node_num(graphs_test)
    max_num_val = get_max_node_num(graphs_val)
    max_num = max(max_num_train, max_num_test, max_num_val)
    print(max_num) #100
    #
    # get the total original_ids and tranform the index from 0 ~n-1
    original_ids = get_original_ids(graphs_train)\
                    .union(get_original_ids(graphs_val))\
                    .union(get_original_ids(graphs_test))

    original_ids.add(-1)
    ## index is new index
    index = IndexDict(original_ids)  #字典{节点对id：节点对}

    # print("create train")
    # write_XYSIZE_data(graphs_train, labels_train,sizes_train,LEN_SEQUENCE,NUM_SEQUENCE,index,max_num,config.n_time_interval, config.train_pkl)
    # print("create val an test")
    # write_XYSIZE_data(graphs_val, labels_val, sizes_val,LEN_SEQUENCE,NUM_SEQUENCE,index,max_num,config.n_time_interval, config.val_pkl)
    # write_XYSIZE_data(graphs_test, labels_test, sizes_test,LEN_SEQUENCE,NUM_SEQUENCE,index,max_num,config.n_time_interval, config.test_pkl)
    # pickle.dump((len(original_ids),NUM_SEQUENCE,LEN_SEQUENCE), open(config.information,'wb'))
    # print("Finish!!!")