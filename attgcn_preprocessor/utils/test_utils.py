
import math
import numpy as np
from scipy.special import softmax
import torch.nn.functional as F
import torch.nn as nn
from utils.plot_utils import plot_predictions_cluster
from config import *
from sklearn.utils.class_weight import compute_class_weight

from utils.process import *
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

from data.data_generator import tsp_instance_reader

from utils.tsplib import read_tsplib_coor, read_tsplib_opt, write_tsplib_prob


def partition_one_graph(coor, node_num=20, cluster_center=0, top_k=19, top_k_expand=19):

    coors = [coor]
    
    distA = pdist(coors[0], metric='euclidean')
    distB_raw = squareform(distA)
    distB = squareform(distA) + 10.0 * np.eye(N = node_num, M =node_num, dtype = np.float64)
    
    edges_probs = np.zeros(shape = (node_num, node_num), dtype = np.float64)
    
    pre_edges = np.ones(shape = (top_k + 1, top_k + 1), dtype = np.int32) + np.eye(N = top_k + 1, M = top_k + 1)
    pre_node = np.ones(shape = (top_k + 1, ))
    
    pre_node_target = np.arange(0, top_k + 1)
    pre_node_target = np.append(pre_node_target, 0)
    pre_edge_target = np.zeros(shape = (top_k + 1, top_k + 1)) 
    pre_edge_target[pre_node_target[:-1], pre_node_target[1:]] = 1
    pre_edge_target[pre_node_target[1:], pre_node_target[:-1]] = 1
    
    neighbor = np.argpartition(distB, kth = top_k, axis=1)
    
    neighbor_expand = np.argpartition(distB, kth=top_k_expand, axis=1)
    Omega_w = np.zeros(shape=(node_num, ), dtype = np.int32)
    Omega = np.zeros(shape=(node_num, node_num), dtype = np.int32)
    
    edges, edges_values = [], []
    nodes, nodes_coord = [], []
    edges_target, nodes_target = [], []
    meshs = []
    num_clusters = 0
    if node_num <= 20:
        num_clusters_threshold = 1
    else:
        num_clusters_threshold = math.ceil((node_num / (top_k+1) ) * 5)
    all_visited = False
    num_batch_size = 0
    
    while num_clusters < num_clusters_threshold or all_visited == False:
        if all_visited==False:
            
            cluster_center_neighbor = neighbor[cluster_center, :top_k]
            cluster_center_neighbor = np.insert(cluster_center_neighbor,
                                                0, cluster_center)
        else:
            np.random.shuffle(neighbor_expand[cluster_center, :top_k_expand])
            cluster_center_neighbor = neighbor_expand[cluster_center, :top_k]
            cluster_center_neighbor = np.insert(cluster_center_neighbor,
                                                0, cluster_center)
        
        Omega_w[cluster_center_neighbor] += 1

        # case 4
        node_coord = coors[0][cluster_center_neighbor]
        x_y_min = np.min(node_coord, axis=0)
        scale = 1.0 / np.max(np.max(node_coord, axis=0)-x_y_min)
        node_coord = node_coord - x_y_min
        node_coord *= scale
        nodes_coord.append(node_coord)

        # case 1-2
        edges.append(pre_edges)
        mesh = np.meshgrid(cluster_center_neighbor, cluster_center_neighbor)
        
        edges_value = distB_raw[mesh].copy()
        edges_value *= scale
        edges_values.append(edges_value)
        meshs.append(mesh)
        Omega[mesh] += 1

        # case 3
        nodes.append(pre_node)

        # case 5-6
        edges_target.append(pre_edge_target)
        nodes_target.append(pre_node_target[:-1])

        num_clusters += 1
        
        if 0 not in Omega_w:
            all_visited = True
        
        cluster_center = np.random.choice(np.where(Omega_w==np.min(Omega_w))[0])
    
    return edges, edges_values, nodes, nodes_coord, edges_target, nodes_target, meshs, Omega


def test_one_tsp(tsp_source, coor_buff, node_num=20, 
                 cluster_center = 0, top_k = 19, top_k_expand = 19):

    mean_rank_sum, mean_greater_zero_edges = 0, 0
    coor, opt = tsp_instance_reader(tspinstance=tsp_source,
                       buff = coor_buff, num_node=node_num)
    
    edges, edges_values, nodes, nodes_coord, edges_target, nodes_target, meshs, Omega = partition_one_graph(coor, 
                                                                                                            node_num=node_num, 
                                                                                                            cluster_center=cluster_center, 
                                                                                                            top_k=top_k, 
                                                                                                            top_k_expand=top_k_expand)
    
    return edges, edges_values, nodes, nodes_coord, edges_target, nodes_target, meshs, Omega, opt

def multiprocess(sub_prob, meshgrid, omega, node_num = 20):
    edges_probs = np.zeros(shape = (node_num, node_num), dtype = np.float32)
    for i in range(len(meshgrid)):
        edges_probs[list(meshgrid[i])] += sub_prob[i, :, :, 1]
    edges_probs = edges_probs / (omega + 1e-8)#[:, None]
    # normalize the probability in an instance 
    edges_probs = edges_probs + edges_probs.T
    edges_probs_norm = edges_probs / np.reshape(np.sum(edges_probs, axis=1), newshape=(node_num, 1))
    return edges_probs_norm

def multiprocess_write(sub_prob, meshgrid, omega, node_num = 20,
                       tsplib_name = './sample.txt', statiscs = False, opt = None):
    edges_probs = np.zeros(shape = (node_num, node_num), dtype = np.float32)
    for i in range(len(meshgrid)):
        edges_probs[list(meshgrid[i])] += sub_prob[i, :, :, 1]
    edges_probs = edges_probs / (omega + 1e-8)#[:, None]
    # normalize the probability in an instance 
    edges_probs = edges_probs + edges_probs.T
    edges_probs_norm = edges_probs/np.reshape(np.sum(edges_probs, axis=1),
                                              newshape=(node_num, 1))
    
    if statiscs:
        mean_rank = 0
        for i in range(node_num-1):
            mean_rank += len(np.where(edges_probs_norm[opt[i], :]>=edges_probs_norm[opt[i], opt[i+1]])[0]) 
        mean_rank /= (node_num-1)
        
        false_negative_edge = opt[np.where(edges_probs_norm[opt[:-1], opt[1:]]<1e-5)]
        # false negative edges in an instance
        num_fne = len(false_negative_edge)
        
        greater_zero_edges = len(np.where(edges_probs_norm>1e-6)[0])
        greater_zero_edges /= node_num
        
        write_tsplib_prob(tsplib_name, edge_prob = edges_probs_norm,
                  num_node=node_num, mean=mean_rank, fnn = num_fne, greater_zero=greater_zero_edges)
    else:
        write_tsplib_prob(tsplib_name, edge_prob = edges_probs_norm,
                          num_node=node_num, mean=0, fnn = 0, greater_zero=0)
    return mean_rank