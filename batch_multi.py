import numpy as np
from graph_model import *
from experiment_comp_spt_lbp import *
from multiprocessing import Pool, Manager
import os
from tqdm import tqdm
from other_sample_method import lbp, spt

from multiprocessing import Pool, Manager

import networkx as nx

import sys
import time

# sys.setrecursionlimit(100000000)

def complete_graph(n, seed=6):
    g = nx.complete_graph(n)
    gnodes = list(g.nodes)
    gedges = list(g.edges)
    
    # random state_nums
    states = [i for i in range(10, 16)]
    state_nums = {gnodes[i]: np.random.choice(states) for i in range(n)}

    # value for observe nodes    
    observations = {}
    
    for node in gnodes:
 
        states = [i for i in range(1, state_nums[node]+1)]
        observations[node] = np.random.choice(states)

    
    return state_nums, observations, g


def edge_prob(g):
    nodes = list(g.nodes)
    edges = list(g.edges)
    n, m = len(nodes), len(edges)
    
    probs = np.zeros((n, n))
    a = np.diag(np.ones(n))
    be = {}
    L = np.zeros((n, n))
    weights = {}
    
    for i in range(m):
        n1, n2 = edges[i]
        be[(n1, n2)] = a[:, n1] - a[:, n2]  
        L += np.outer(be[(n1, n2)], be[(n1, n2)])

    eig_val, eig_vec = np.linalg.eig(L)
    
    L_pinv = np.zeros((n, n))
    for i in range(len(eig_val)):
        if eig_val[i] < 0.00001:
            continue
        L_pinv = L_pinv + np.outer(eig_vec[:, i], eig_vec[:, i]) / eig_val[i]

    for e in edges:
        n1, n2 = e
        probs[n1, n2] = be[(n1, n2)].T @ L_pinv @ be[(n1, n2)]
        probs[n2, n1] = probs[n1, n2]
        weights[(min(n1, n2), max(n1, n2))] = 1 / probs[n1, n2]
    for i in range(probs.shape[0]):
        probs[i, :] /= sum(probs[i, :])

    return probs, weights

def edge_prob2(g):
    nodes = list(g.nodes)
    edges = list(g.edges)
    n, m = len(nodes), len(edges)
    
    probs = np.zeros((n, n))
    weights = {}
    
    for i in range(m):
        n1, n2 = edges[i]
        ef = nx.resistance_distance(g, n1, n2)
        probs[n1, n2] = ef
        probs[n2, n1] = ef
        weights[(min(n1, n2), max(n1, n2))] = 1 / ef
    for i in range(probs.shape[0]):
        probs[i, :] /= sum(probs[i, :])
    
    return probs, weights


def factorization_(tree, seed, nodes, pls, state_nums, observations, probs, weights, neighbors, marginals, num=None, FTree1=None):
    # spt = UST(g)
    # spt.sample(random_state=seed)
    # tree = spt.list_of_samples[0].edges
    
    if not tree:
        tree = ust_sampler_wilson(neighbors, probs, weights, root=None,
                       random_state=seed, num=num)
    
        FTree1 = factor_tree()
        for node in nodes:
            obs = node + pls
            FTree1.add_nodes([node, obs], [state_nums[node], 1], [0, observations[node]], [marginals[node], np.ones(1)])
            FTree1.add_edge([node, obs], 1)
        tree_weights = []
        for edge in tree:
            a, b = edge
            ee = (min(a,b), max(a,b))
            tree_weights.append(weights[ee])
            
        FTree1.add_edges(tree, tree_weights)

        FTree1.set_tree_root(0)
        
        FTree1.generate_node_tree()
        FTree1.build_factor_tree()
    else:
        for node in nodes:
            vnode = FTree1.vnodes[node]
            vnode.empty()
            vnode.marginal = marginals[node]   
        for fac in FTree1.factors:
            fac.empty()
            
    # FTree1.build()
    
    return FTree1, tree

def update_marginals2(marginals, marginals_rec):
    for i in range(len(marginals.keys())):
        a = np.zeros(marginals[i].shape)
        for v in marginals_rec[i]:
            a += v
        marginals[i] = marginals[i] * (a / len(marginals_rec[i]))
        marginals[i] = marginals[i] / np.sum(marginals[i])
    return marginals


def gibs_belief(marginals):
    values = []
    for i in range(len(marginals.keys())):
        marginal = marginals[i]
        v = np.random.choice([i for i in range(1, len(marginal)+1)], p=marginal)
        values.append(v)
    return values    


def TreeBP(tree, seed, gnodes, pls, state_nums, observations, 
                                            probs1, weights1, neighbors, marginals, FTree1):
    
    FT, tree = factorization_(tree, seed, gnodes, pls, state_nums, observations, 
                                            probs1, weights1, neighbors, marginals, FTree1=FTree1)
    
    r = FT.sum_product()
    marginal_res = []
    for node in gnodes:
        vnode = FT.vnodes[node]
        marginal_res.append(vnode.marginal)

    return FT, tree, marginal_res


def re_prop_parallel(batch_size, g, state_nums, observations, more, poolsize, complete=False):
    gnodes = list(g.nodes)
    gedges = list(g.edges)
    pls = len(gnodes)
    loss_ = []
    time_spt = []
    
    # probs1, weights1 = edge_prob(g)
    # weights1 = {}
    # for i in range(probs1.shape[0]):
    #     for j in range(i, probs1.shape[1]):
    #         if probs1[i, j] != 0:
    #             weights1[(i, j)] = 1 / probs1[i, j]
    # probs1, weights1 = select_prob(g)
    t1 = time.time()
    probs1, weights1 = edge_prob2(g)
    # print("done prob in", time.time() - t1)
    marginals = {i:np.ones(state_nums[i])/state_nums[i] for i in gnodes}
    neighbors = [[] for _ in gnodes]
    if complete:
        weights1 = {}
        for i in range(probs1.shape[0]):
            for j in range(i, probs1.shape[1]):
                if probs1[i, j] != 0:
                    weights1[(i, j)] = 1 
    
    t = 0
    for (a, b) in gedges:
        neighbors[a].append(b)
        neighbors[b].append(a)    
        
    cur_best = 999999999 
    tree_set = []
    FT_set = []

    pool = Pool(poolsize)
    
    for round in range(6, 6+more):
        marginal_rec = {i:[] for i in gnodes}

        for batch in range(batch_size // poolsize):
            params = []
            if round == 6:
                for i in range(poolsize):  
                    seed_ = round*batch_size + batch*poolsize + i
                    params.append((None, seed_, gnodes, pls, state_nums, observations, 
                                            probs1, weights1, neighbors, marginals, None))
                
            else:
                for i in range(poolsize):  
                    params.append((tree_set[batch*poolsize+i], 0, gnodes, pls, state_nums, observations, 
                                            probs1, weights1, neighbors, marginals, FT_set[batch*poolsize+i]))
               
            rtn = pool.starmap(TreeBP, params)
            
            if round == 6:
                for i in range(poolsize):
                    FT, tree, marginal_rec_ = rtn[i]
                    tree_set.append(tree)
                    FT_set.append(FT)
                    for node in gnodes:
                        marginal_rec[node].append(marginal_rec_[node])
            else:
                for i in range(poolsize):
                    FT, tree, marginal_rec_ = rtn[i]                   
                    for node in gnodes:
                        marginal_rec[node].append(marginal_rec_[node])
                        
        marginals = update_marginals2(marginals, marginal_rec)
        # for k,v in marginal_rec.items():
        #     print(k,v)
        r = gibs_belief(marginals)
        cur_loss = loss(observations, r, gedges)
        cur_best = min(cur_best, cur_loss)
        loss_.append(cur_best)

    return loss_


def re_prop(batch_size, g, state_nums, observations, more, rank):
    
    gnodes = list(g.nodes)
    gedges = list(g.edges)
    pls = len(gnodes)
    
    loss_ = []
    time_spt = []
    score = {edge:[] for edge in gedges}
    # probs1, weights1 = score_based_prob(gedges, score, pls)
    probs1, weights1 = select_prob(g)
    marginals = {i:np.ones(state_nums[i])/state_nums[i] for i in gnodes}
    neighbors = [[] for _ in gnodes]
    
    t = 0
    for (a, b) in gedges:
        neighbors[a].append(b)
        neighbors[b].append(a)    
        
    cur_best = 999999999 
    tree_set = []
    FT_set = []
    for round in range(6, 6+more):
        marginal_rec = {i:[] for i in gnodes}
        res = [[] for _ in range(pls)]
        
        for batch in range(batch_size):
            if round == 6:
                seed_ = round*batch_size + batch         
                FT, tree = factorization_(None, seed_, gnodes, pls, state_nums, observations, 
                                            probs1, weights1, neighbors, marginals)  
                tree_set.append(tree)
                FT_set.append(FT)
            else:
                FT, tree = factorization_(tree_set[batch], 0, gnodes, pls, state_nums, observations, 
                                            probs1, weights1, neighbors, marginals, FTree1=FT_set[batch])
                
            s = time.time()
            r = FT.sum_product()
            t += time.time() - s
            time_spt.append(t)
            
            for node in gnodes:
                vnode = FT.vnodes[node]
                marginal_rec[node].append(vnode.marginal)
            
            r = gibs_sampling(gnodes, FT, state_nums, False)
        
            for r_pair in r:
                i, rr = r_pair
                res[i].append(rr)
        
        marginals = update_marginals2(marginals, marginal_rec)
        r = gibs_belief(marginals)
        cur_loss = loss(observations, r, gedges)
        cur_best = min(cur_best, cur_loss)
        loss_.append(cur_best)
        
    # print(res)
    return loss_, time_spt, rank




# n = 40
# p0 = 10
# p1 = 10 / (n * (n + 1))
# seed = 6
# state_nums, observations, g = er_graph_obs(n*n, p=p1, seed=seed)
# state_nums, observations, g = complete_graph(n, seed=seed)
# state_nums, observations, g = grid_obs(n, n, seed=seed)
# print(len(list(g.edges)))
# # t0 = time.time()
# p = edge_prob(g)
# # print(p)
# print(batch_update2(20, g, state_nums, observations, 10, 1)[0])
# print(n, p0, seed)
# print(re_prop(100, g, state_nums, observations, 20, 1)[0])
# batch_size = 200
# round = 20
# poolsize = 50
# p1 = select_prob(g)[0]
# p2 = edge_prob(g)

# sum = 0
# maxdiff = 0
# for i in range(p1.shape[0]):
#     for j in range(p1.shape[1]):
#         diff = abs(p1[i, j] - p2[i, j])
#         maxdiff = max(maxdiff, diff)
#         sum += diff
# print("maxdiff", maxdiff)   
# print("sum", sum)
# print(re_prop_parallel(batch_size, g, state_nums, observations, round, poolsize))

# print(re_prop_parallel(batch_size, g, state_nums, observations, round, poolsize, True))
# print(lbp(g, state_nums, observations, 1))
# print(spt(g, state_nums, observations, 60, 1)[0])


title = "/home/wangyaomin/research/SPT/data/er(240203)/"
n = 30

for p0 in tqdm((0.5, 1, 2.5)):
    p1 = p0 * 4 / (n * (n + 1))
    for seed in [6, 7, 8, 9, 10]:
        state_nums, observations, g = er_graph_obs(n*n, p=p1, seed=seed)
        l1 = re_prop_parallel(200, g, state_nums, observations, 20, 100)
        l2 = lbp(g, state_nums, observations, 1)
        df = pd.DataFrame({'re_prop':l1})
        df2 = pd.DataFrame({'lbp':l2})
        df.to_csv(title + "re_prop_" + str(n) + "_"+ str(int(4*p0)) + "_" + str(seed) + ".csv",  index=False)
        df2.to_csv(title + "lbp_" + str(n) +"_"+ str(int(4*p0)) + "_" + str(seed) + ".csv", index=False)