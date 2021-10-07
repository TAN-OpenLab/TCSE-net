# #weibo数据
# import networkx as nx
# import time
# from functools import cmp_to_key
# from data_preprocess import config
# import sys
# import random
# import math
#
#
# class IndexDict:
#     def __init__(self, original_ids):
#         self.original_to_new = {}
#         self.new_to_original = []
#         cnt = 0
#         for i in original_ids:
#             new = self.original_to_new.get(i, cnt)
#             if new == cnt:
#                 self.original_to_new[i] = cnt
#                 cnt += 1
#                 self.new_to_original.append(i)
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
#
# def gen_cascades_obser(observation_time,pre_times,filename):
#     cascades_total = dict()
#     cascades_type = dict()
#     with open(filename) as f:
#         for line in f:
#             parts = line.split("\t")
#             if len(parts) != 5:
#                 print('wrong format!')
#                 continue
#             cascadeID = parts[0]
#             n_nodes = int(parts[3])
#             path = parts[4].split(" ")
#             if n_nodes != len(path):
#                 print('wrong number of nodes', n_nodes, len(path))
#             msg_pub_time = parts[2]
#
#             msg_time = time.localtime(int(parts[2]))
#             hour = time.strftime("%H", msg_time)
#             hour = int(hour)
#             if hour <= 7 or hour >= 19:
#                 continue
#
#             observation_path = []
#             labels = []
#             edges = set()
#             for i in range(len(pre_times)):
#                 labels.append(0)
#             for p in path:
#                 nodes = p.split(":")[0].split("/")
#                 nodes_ok = True
#                 time_now = int(p.split(":")[1])
#                 if time_now < observation_time:
#                     observation_path.append(",".join(nodes) + ":" + str(time_now))
#                     for i in range(1, len(nodes)):
#                         edges.add(nodes[i - 1] + ":" + nodes[i] + ":1")
#                 for i in range(len(pre_times)):
#                     if time_now < pre_times[i]:
#                         labels[i] += 1
#             cascades_total[cascadeID] = msg_pub_time
#
#         n_total = len(cascades_total)
#         print('total:', n_total)
#
#         #按发布时间排序
#         # key = cmp_to_key(lambda x, y: int(x[1]) - int(y[1]))
#         # sorted_msg_time = sorted(cascades_total.items(), key=key)
#         # count = 0
#         # for (k, v) in sorted_msg_time:
#         #     if count < n_total * 1.0 / 20 * 14:
#         #         cascades_type[k] = 1
#         #     elif count < n_total * 1.0 / 20 * 17:
#         #         cascades_type[k] = 2
#         #     else:
#         #         cascades_type[k] = 3
#         #     count += 1
#         #随机排序
#         k = set(cascades_total.keys())
#         train = set(random.sample(k, math.floor(n_total * 1.0 / 20 * 14)))
#         test_val = k - train
#         test = set(random.sample(test_val,math.floor(n_total * 1.0 / 20 * 3)))
#         val = k - test - train
#         for i in train:
#             cascades_type[i] =1
#         for i in test:
#             cascades_type[i] = 2
#         for i in val:
#             cascades_type[i] = 3
#     return cascades_total,cascades_type
#
#
# def discard_cascade(observation_time,pre_times,filename):
#     discard_cascade_id=dict()
#     with open(filename) as f:
#         for line in f:
#             parts = line.split("\t")
#             if len(parts) != 5:
#                 print('wrong format!')
#                 continue
#             cascadeID = parts[0]
#             n_nodes = int(parts[3])
#             path = parts[4].split(" ")
#             if n_nodes != len(path):
#                 print('wrong number of nodes', n_nodes, len(path))
#             msg_pub_time = parts[2]
#
#             observation_path = []
#             edges = set()
#             for p in path:
#                 nodes = p.split(":")[0].split("/")
#                 time_now = int(p.split(":")[1])
#                 if time_now < observation_time:
#                     observation_path.append(",".join(nodes) + ":" + str(time_now))
#                     for i in range(1, len(nodes)):
#                         edges.add(nodes[i - 1] + ":" + nodes[i] + ":1")
#             nx_Cass = nx.DiGraph()
#             for i in edges:
#                 part = i.split(":")
#                 source = part[0]
#                 target = part[1]
#                 weight = part[2]
#                 nx_Cass.add_edge(source, target, weight=weight)
#             try:
#                 L = directed_laplacian_matrix(nx_Cass)
#             except:
#                 discard_cascade_id[cascadeID]=1
#                 s = sys.exc_info()
#             else:
#                 num = nx_Cass.number_of_nodes()
#                 if num>100:
#                     if cascadeID not in discard_cascade_id:
#                         discard_cascade_id[cascadeID] = 1
#                 else:
#                     discard_cascade_id[cascadeID]=0
#
#     return discard_cascade_id
#
# def directed_laplacian_matrix(G, nodelist=None, weight='weight',
#                               walk_type=None, alpha=0.95):
#     import scipy as sp
#     from scipy.sparse import identity, spdiags, linalg
#     if walk_type is None:
#         if nx.is_strongly_connected(G):
#             if nx.is_aperiodic(G):
#                 walk_type = "random"
#             else:
#                 walk_type = "lazy"
#         else:
#             walk_type = "pagerank"
#
#     M = nx.to_scipy_sparse_matrix(G, nodelist=nodelist, weight=weight,
#                                   dtype=float)
#     n, m = M.shape
#     if walk_type in ["random", "lazy"]:
#         DI = spdiags(1.0 / sp.array(M.sum(axis=1).flat), [0], n, n)
#         if walk_type == "random":
#             P = DI * M
#         else:
#             I = identity(n)
#             P = (I + DI * M) / 2.0
#
#     elif walk_type == "pagerank":
#         if not (0 < alpha < 1):
#             raise nx.NetworkXError('alpha must be between 0 and 1')
#         M = M.todense()
#         dangling = sp.where(M.sum(axis=1) == 0)
#         for d in dangling[0]:
#             M[d] = 1.0 / n
#         M = M / M.sum(axis=1)
#         P = alpha * M + (1 - alpha) / n
#     else:
#         raise nx.NetworkXError("walk_type must be random, lazy, or pagerank")
#
#     evals, evecs = linalg.eigs(P.T, k=1,tol=1E-2)
#     v = evecs.flatten().real
#     p = v / v.sum()
#     sqrtp = sp.sqrt(p)
#     Q = spdiags(sqrtp, [0], n, n) * P * spdiags(1.0 / sqrtp, [0], n, n)
#     I = sp.identity(len(G))
#     return I - (Q + Q.T) / 2.0
#
# def gen_cascade(observation_time, pre_times, filename, filename_ctrain, filename_cval,
#     filename_ctest, filename_strain, filename_sval, filename_stest, cascades_type, discard_cascade_id):
#     file = open(filename,"r")
#     file_ctrain = open(filename_ctrain, "w")
#     file_cval = open(filename_cval, "w")
#     file_ctest = open(filename_ctest, "w")
#     file_strain = open(filename_strain, "w")
#     file_sval = open(filename_sval, "w")
#     file_stest = open(filename_stest, "w")
#     for line in file:
#         parts = line.split("\t")
#         if len(parts) != 5:
#             print ('wrong format!')
#             continue
#         cascadeID = parts[0]
#         n_nodes = int(parts[3])
#         path = parts[4].split(" ")
#         if n_nodes !=len(path):
#             print ('wrong number of nodes',n_nodes,len(path))
#         msg_time = time.localtime(int(parts[2]))
#         hour = time.strftime("%H",msg_time)
#         hour = int(hour)
#         if hour <= 7 or hour >= 19:
#             continue
#
#         observation_path = []
#         labels = []
#         edges = set()
#         pretime_nodes = []
#         for i in range(len(pre_times)):
#             labels.append(0)
#             pretime_nodes.append(set())
#         for p in path:
#             nodes = p.split(":")[0].split("/")
#             time_now = int(p.split(":")[1])
#             observation_nodes = set()
#             if time_now < observation_time:
#                 observation_path.append(",".join(nodes) + ":" + str(time_now))
#                 for i in range(1, len(nodes)):
#                     observation_nodes.add(nodes[i - 1])
#                     observation_nodes.add(nodes[i])
#                     if (nodes[i - 1] + ":" + nodes[i] + ":" + str(time_now)) in edges:
#                         continue
#                     else:
#                         edges.add(nodes[i - 1] + ":" + nodes[i] + ":" + str(time_now))
#             for i in range(len(pre_times)):
#                 if time_now < pre_times[i]:
#                     for j in range(1, len(nodes)):
#                         pretime_nodes[i].add(nodes[j - 1])
#                         pretime_nodes[i].add(nodes[j])
#                 labels[i] = len(pretime_nodes[i])
#         for i in range(len(labels)):
#             labels[i] = str(labels[i] - len(observation_nodes))
#
#         #
#         # observation_path = []
#         # labels = []
#         # edges = set()
#         # for i in range(len(pre_times)):
#         #     labels.append(0)
#         # for p in path:
#         #     nodes = p.split(":")[0].split("/")
#         #     time_now = int(p.split(":")[1])
#         #     if time_now <observation_time:
#         #         observation_path.append(",".join(nodes)+":"+ str(time_now))
#         #         for i in range(1,len(nodes)):
#         #             if (nodes[i-1] +":"+ nodes[i] +":"+ str(time_now)) in edges:
#         #                 continue
#         #             else:
#         #                 edges.add(nodes[i-1]+":"+ nodes[i]+":"+ str(time_now))
#         #     for i in range(len(pre_times)):
#         #         if time_now <pre_times[i]:
#         #             labels[i] +=1
#         # for i in range(len(labels)):
#         #     labels[i] = str(labels[i]-len(observation_path))
#         if len(edges)<=1:
#             continue
#         if cascadeID in cascades_type and cascades_type[cascadeID] == 1 and discard_cascade_id[cascadeID]== 0:
#             file_strain.write(cascadeID + "\t" + "\t".join(observation_path) + "\n")#shortespath_train
#             file_ctrain.write(cascadeID+"\t"+parts[1]+"\t"+parts[2]+"\t"+str(len(observation_path))+"\t"+" ".join(edges)+"\t"+" ".join(labels)+"\n")#cascade_train part[1]-user_id parts[2]-publis_time observation_path "".join(edges) "".join(labels)
#         elif cascadeID in cascades_type and cascades_type[cascadeID] == 2 and discard_cascade_id[cascadeID]== 0:
#             file_sval.write(cascadeID + "\t" + "\t".join(observation_path) + "\n")
#             file_cval.write(cascadeID + "\t" + parts[1] + "\t" + parts[2] + "\t" + str(len(observation_path)) + "\t" + " ".join(edges) + "\t" + " ".join(labels) + "\n")
#         elif cascadeID in cascades_type and cascades_type[cascadeID] == 3 and discard_cascade_id[cascadeID]== 0:
#             file_stest.write(cascadeID + "\t" + "\t".join(observation_path) + "\n")
#             file_ctest.write(cascadeID + "\t" + parts[1] + "\t" + parts[2] + "\t" + str(len(observation_path)) + "\t" + " ".join(edges) + "\t" + " ".join(labels) + "\n")
#
#     file.close()
#     file_ctrain.close()
#     file_cval.close()
#     file_ctest.close()
#     file_strain.close()
#     file_sval.close()
#     file_stest.close()
#
# def get_original_ids(graphs):
#     original_ids = set()
#     for graph in graphs.keys():
#         for walk in graphs[graph]:
#             for i in walk[0]:
#                 original_ids.add(i)
#     print ("length of original isd:",len(original_ids))
#     return original_ids
# def sequence2list(flename):
#     graphs = {}
#     with open(flename, 'r') as f:
#         for line in f:
#             walks = line.strip().split('\t')
#             graphs[walks[0]] = [] #walk[0] = cascadeID
#             for i in range(1, len(walks)):
#                 s = walks[i].split(":")[0] #node
#                 t = walks[i].split(":")[1] #time
#                 graphs[walks[0]].append([[int(xx) for xx in s.split(",")],int(t)])
#     return graphs
# if __name__ =="__main__":
#     observation_time = config.observation
#     pre_times = config.pre_times
#
#     cascades_total, cascades_type = gen_cascades_obser(observation_time,pre_times,config.cascades)
#     discard_cascade_id= discard_cascade(observation_time,pre_times,config.cascades)
#
#     print("generate cascade new!!!")
#     gen_cascade(observation_time, pre_times, config.cascades, config.cascade_train,
#                   config.cascade_val, config.cascade_test,
#                   config.shortestpath_train, config.shortestpath_val,
#                   config.shortestpath_test,
#                   cascades_type, discard_cascade_id)
#
#





#twitter15 16数据处理
import networkx as nx
import time
from functools import cmp_to_key
from data_preprocess import config
import sys
import random
import math
import numpy as np

random.seed(10)

class IndexDict:
    def __init__(self, original_ids):
        self.original_to_new = {}
        self.new_to_original = []
        cnt = 0
        for i in original_ids:
            new = self.original_to_new.get(i, cnt)
            if new == cnt:
                self.original_to_new[i] = cnt
                cnt += 1
                self.new_to_original.append(i)

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


def gen_cascades_obser(observation_time,pre_times,filename):
    cascades_total = dict()
    cascades_type = dict()
    with open(filename) as f:
        for line in f:
            parts = line.split("\t")
            if len(parts) != 5:
                print('wrong format!')
                continue
            cascadeID = parts[0]
            # n_nodes = int(parts[3])
            # path = parts[4].split(" ")
            # if n_nodes != len(path):
            #     print('wrong number of nodes', n_nodes, len(path))
            msg_pub_time = parts[2]

            hour = math.floor((int(parts[2]))/60/60)
            if hour <= 3:
                continue

            # observation_path = []
            # labels = []
            # edges = set()
            # for i in range(len(pre_times)):
            #     labels.append(0)
            # for p in path:
            #     nodes = p.split(":")[0].split("/")
            #     nodes_ok = True
            #     time_now = int(p.split(":")[1])
            #     if time_now < observation_time:
            #         observation_path.append(",".join(nodes) + ":" + str(time_now))
            #         for i in range(1, len(nodes)):
            #             edges.add(nodes[i - 1] + ":" + nodes[i] + ":1")
            #     for i in range(len(pre_times)):
            #         if time_now < pre_times[i]:
            #             labels[i] += 1
            cascades_total[cascadeID] = msg_pub_time

        n_total = len(cascades_total)
        print('total:', n_total)

        #按发布时间排序
        # key = cmp_to_key(lambda x, y: int(x[1]) - int(y[1]))
        # sorted_msg_time = sorted(cascades_total.items(), key=key)
        # count = 0
        # for (k, v) in sorted_msg_time:
        #     if count < n_total * 1.0 / 20 * 14:
        #         cascades_type[k] = 1
        #     elif count < n_total * 1.0 / 20 * 17:
        #         cascades_type[k] = 2
        #     else:
        #         cascades_type[k] = 3
        #     count += 1
        #随机排序
        k = list(cascades_total.keys())
        random.shuffle(k)
        train = set(random.sample(k, math.floor(n_total * 1.0 / 20 * 14)))
        test_val = set(k) - train
        test = set(random.sample(test_val,math.floor(n_total * 1.0 / 20 * 3)))
        val = set(k) - test - train
        for i in train:
            cascades_type[i] =1
        for i in test:
            cascades_type[i] = 2
        for i in val:
            cascades_type[i] = 3
    return cascades_total,cascades_type


def discard_cascade(observation_time, pre_times,filename):
    discard_cascade_id=dict()
    with open(filename) as f:
        for line in f:
            parts = line.split("\t")
            if len(parts) != 5:
                print('wrong format!')
                continue
            cascadeID = parts[0]
            n_nodes = int(parts[3])
            path = parts[4].split(" ")
            if n_nodes != len(path):
                print('wrong number of nodes', n_nodes, len(path))

            observation_path = []
            edges = set()
            for p in path:
                nodes = p.split(":")[0].split("/")
                time_now = int(p.split(":")[1])
                if time_now < observation_time:
                    observation_path.append(",".join(nodes) + ":" + str(time_now))
                    for i in range(1, len(nodes)):
                        edges.add(nodes[i - 1] + ":" + nodes[i] + ":1")
            nx_Cass = nx.DiGraph()
            for i in edges:
                part = i.split(":")
                source = part[0]
                target = part[1]
                weight = part[2]
                nx_Cass.add_edge(source, target, weight=weight)
            try:
                L = directed_laplacian_matrix(nx_Cass)
            except:
                discard_cascade_id[cascadeID]=1
                s = sys.exc_info()
            else:
                num = nx_Cass.number_of_nodes()
                if num > 100:
                    if cascadeID not in discard_cascade_id:
                        discard_cascade_id[cascadeID] = 1
                else:
                    discard_cascade_id[cascadeID]=0

    return discard_cascade_id

def directed_laplacian_matrix(G, nodelist=None, weight='weight',
                              walk_type=None, alpha=0.95):
    import scipy as sp
    from scipy.sparse import identity, spdiags, linalg
    if walk_type is None:
        if nx.is_strongly_connected(G):
            if nx.is_aperiodic(G):
                walk_type = "random"
            else:
                walk_type = "lazy"
        else:
            walk_type = "pagerank"

    M = nx.to_scipy_sparse_matrix(G, nodelist=nodelist, weight=weight,
                                  dtype=float)
    n, m = M.shape
    if walk_type in ["random", "lazy"]:
        DI = spdiags(1.0 / sp.array(M.sum(axis=1).flat), [0], n, n)
        if walk_type == "random":
            P = DI * M
        else:
            I = identity(n)
            P = (I + DI * M) / 2.0

    elif walk_type == "pagerank":
        if not (0 < alpha < 1):
            raise nx.NetworkXError('alpha must be between 0 and 1')
        M = M.todense()
        dangling = sp.where(M.sum(axis=1) == 0)
        for d in dangling[0]:
            M[d] = 1.0 / n
        M = M / M.sum(axis=1)
        P = alpha * M + (1 - alpha) / n
    else:
        raise nx.NetworkXError("walk_type must be random, lazy, or pagerank")

    evals, evecs = linalg.eigs(P.T, k=1,tol=1E-2)
    v = evecs.flatten().real
    p = v / v.sum()
    sqrtp = sp.sqrt(p)
    Q = spdiags(sqrtp, [0], n, n) * P * spdiags(1.0 / sqrtp, [0], n, n)
    I = sp.identity(len(G))
    return I - (Q + Q.T) / 2.0

def gen_cascade(observation_time, pre_times, filename, filename_ctrain, filename_cval,
    filename_ctest, filename_strain, filename_sval, filename_stest, cascades_type, discard_cascade_id):
    file = open(filename,"r")
    file_ctrain = open(filename_ctrain, "w")
    file_cval = open(filename_cval, "w")
    file_ctest = open(filename_ctest, "w")
    file_strain = open(filename_strain, "w")
    file_sval = open(filename_sval, "w")
    file_stest = open(filename_stest, "w")
    for line in file:
        parts = line.split("\t")
        if len(parts) != 5:
            print ('wrong format!')
            continue
        cascadeID = parts[0]
        n_nodes = int(parts[3])
        path = parts[4].split(" ")
        # if n_nodes !=len(path):
        #     print ('wrong number of nodes',n_nodes,len(path))

        observation_path = []
        observation_nodes =set()
        labels = []
        edges = set()
        pretime_nodes = []
        for i in range(len(pre_times)):
            labels.append(0)
            pretime_nodes.append(set())
        for p in path:
            nodes = p.split(":")[0].split("/")
            time_now = int(p.split(":")[1])

            if time_now <observation_time :
                observation_path.append(",".join(nodes)+":"+ str(time_now))
                for i in range(1,len(nodes)):
                    observation_nodes.add(nodes[i-1])
                    observation_nodes.add(nodes[i])
                    if (nodes[i-1] +":"+ nodes[i] +":"+ str(time_now)) in edges:
                        continue
                    else:
                        edges.add(nodes[i-1]+":"+ nodes[i]+":"+ str(time_now))
            # popularity
            for i in range(len(pre_times)):
                if time_now<pre_times[i]:
                    labels[i]+=1
        for i in range(len(labels)):
            labels[i] = str(labels[i])
        if len(edges) <= 1:
            continue
        if cascadeID in cascades_type and cascades_type[cascadeID] == 1 and discard_cascade_id[cascadeID] == 0:
            file_strain.write(cascadeID + "\t" + "\t".join(observation_path) + "\n")  # shortespath_train
            file_ctrain.write(
                    cascadeID + "\t" + parts[1] + "\t" + parts[2] + "\t" + str(len(observation_path)) + "\t" + " ".join(
                        edges) + "\t" + " ".join(
                        labels) + "\n")  # cascade_train part[1]-user_id parts[2]-publis_time observation_path "".join(edges) "".join(labels)
        elif cascadeID in cascades_type and cascades_type[cascadeID] == 2 and discard_cascade_id[cascadeID] == 0:
            file_sval.write(cascadeID + "\t" + "\t".join(observation_path) + "\n")
            file_cval.write(
                    cascadeID + "\t" + parts[1] + "\t" + parts[2] + "\t" + str(len(observation_path)) + "\t" + " ".join(
                        edges) + "\t" + " ".join(labels) + "\n")
        elif cascadeID in cascades_type and cascades_type[cascadeID] == 3 and discard_cascade_id[cascadeID] == 0:
            file_stest.write(cascadeID + "\t" + "\t".join(observation_path) + "\n")
            file_ctest.write(
                    cascadeID + "\t" + parts[1] + "\t" + parts[2] + "\t" + str(len(observation_path)) + "\t" + " ".join(
                        edges) + "\t" + " ".join(labels) + "\n")

        # #   size
        #     for i in range(len(pre_times)):
        #         if time_now <pre_times[i]:
        #             for j in range(1, len(nodes)):
        #                 pretime_nodes[i].add(nodes[j - 1])
        #                 pretime_nodes[i].add(nodes[j])
        # for i in range(len(labels)):
        #     labels[i] = str(len(pretime_nodes[i])-len(observation_nodes))
        # if len(edges)<=1:
        #     continue
        # if cascadeID in cascades_type and cascades_type[cascadeID] == 1 and discard_cascade_id[cascadeID]== 0:
        #     file_strain.write(cascadeID + "\t" + "\t".join(observation_path) + "\n")#shortespath_train
        #     file_ctrain.write(cascadeID+"\t"+parts[1]+"\t"+parts[2]+"\t"+str(len(observation_path))+"\t"+" ".join(edges)+"\t"+" ".join(labels)+"\n")#cascade_train part[1]-user_id parts[2]-publis_time observation_path "".join(edges) "".join(labels)
        # elif cascadeID in cascades_type and cascades_type[cascadeID] == 2 and discard_cascade_id[cascadeID]== 0:
        #     file_sval.write(cascadeID + "\t" + "\t".join(observation_path) + "\n")
        #     file_cval.write(cascadeID + "\t" + parts[1] + "\t" + parts[2] + "\t" + str(len(observation_path)) + "\t" + " ".join(edges) + "\t" + " ".join(labels) + "\n")
        # elif cascadeID in cascades_type and cascades_type[cascadeID] == 3 and discard_cascade_id[cascadeID]== 0:
        #     file_stest.write(cascadeID + "\t" + "\t".join(observation_path) + "\n")
        #     file_ctest.write(cascadeID + "\t" + parts[1] + "\t" + parts[2] + "\t" + str(len(observation_path)) + "\t" + " ".join(edges) + "\t" + " ".join(labels) + "\n")

    file.close()
    file_ctrain.close()
    file_cval.close()
    file_ctest.close()
    file_strain.close()
    file_sval.close()
    file_stest.close()

def get_original_ids(graphs):
    original_ids = set()
    for graph in graphs.keys():
        for walk in graphs[graph]:
            for i in walk[0]:
                original_ids.add(i)
    print ("length of original isd:",len(original_ids))
    return original_ids
def sequence2list(flename):
    graphs = {}
    with open(flename, 'r') as f:
        for line in f:
            walks = line.strip().split('\t')
            graphs[walks[0]] = [] #walk[0] = cascadeID
            for i in range(1, len(walks)):
                s = walks[i].split(":")[0] #node
                t = walks[i].split(":")[1] #time
                graphs[walks[0]].append([[int(xx) for xx in s.split(",")],int(t)])
    return graphs
if __name__ =="__main__":
    observation_time = config.observation
    pre_times = config.pre_times

    cascades_total, cascades_type = gen_cascades_obser(observation_time,pre_times,config.cascades)
    discard_cascade_id= discard_cascade(observation_time,pre_times,config.cascades)

    print("generate cascade new!!!")
    gen_cascade(observation_time, pre_times, config.cascades, config.cascade_train,
                  config.cascade_val, config.cascade_test,
                  config.shortestpath_train, config.shortestpath_val,
                  config.shortestpath_test,
                  cascades_type, discard_cascade_id)






