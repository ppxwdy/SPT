## Each time we sample n trees, 
## these trees have the same start condition
## After this round, we update the start condition by using the 
## value choice for each node as the distribution of each node

## method2: using the average marginal as the marginal

import numpy as np
from graph_model import *
from experiment_comp_spt_lbp import *
from multiprocessing import Pool, Manager
import os
from tqdm import tqdm
from other_sample_method import lbp, spt
import networkx as nx
import time
import pandas as pd

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


def factorization_(g, seed, nodes, pls, state_nums, observations, probs, weights, neighbors, marginals, num=None):
    # spt = UST(g)
    # spt.sample(random_state=seed)
    # tree = spt.list_of_samples[0].edges
    
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
    
    # FTree1.build()
    
    return FTree1, tree


def factorization_2(tree, nodes, pls, state_nums, observations, weights, neighbors, marginals, num=None):
    # spt = UST(g)
    # spt.sample(random_state=seed)
    # tree = spt.list_of_samples[0].edges
    
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
    
    # FTree1.build()
    
    return FTree1, tree


def update_marginals(res, marginals, state_nums):
    for i in range(len(res)):
        count = [0 for _ in range(state_nums[i])]
        for v in range(1, state_nums[i]+1):
            count[v-1] = res[i].count(v)
        count = np.array(count)
        count = count / np.sum(count)
        marginals[i] = marginals[i] * count
    return marginals


def batch_update(batch_size, g, state_nums, observations, more, rank):
    
    gnodes = list(g.nodes)
    gedges = list(g.edges)
    pls = len(gnodes)
    
    loss_ = []
    time_spt = []
    score = {edge:[] for edge in gedges}
    probs1, weights1 = score_based_prob(gedges, score, pls)
    marginals = {i:np.ones(state_nums[i])/state_nums[i] for i in gnodes}
    neighbors = [[] for _ in gnodes]
    
    t = 0
    for (a, b) in gedges:
        neighbors[a].append(b)
        neighbors[b].append(a)    
        
    cur_best = 999999999 
    
    for round in range(6, 6+more):
        res = [[] for _ in range(pls)]
        for batch in range(batch_size):
            seed_ = round*batch_size + batch
            s = time.time()
            FT, tree = factorization_(g, seed_, gnodes, pls, state_nums, observations, 
                                      probs1, weights1, neighbors, marginals)  
            r = FT.sum_product()
            t += time.time() - s
            time_spt.append(t)
            
            r = gibs_sampling(gnodes, FT, state_nums, False)
        
            for r_pair in r:
                i, rr = r_pair
                res[i].append(rr)
            
        r = vote(res)
        # r = vote_avg(res)
        cur_loss = loss(observations, r, gedges)
        cur_best = min(cur_best, cur_loss)
        loss_.append(cur_best)
        
        marginals = update_marginals(res, marginals, state_nums)
    # print(res)
    return loss_, time_spt, rank

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

def batch_update2(batch_size, g, state_nums, observations, more, rank):
    
    gnodes = list(g.nodes)
    gedges = list(g.edges)
    pls = len(gnodes)
    
    loss_ = []
    time_spt = []
    score = {edge:[] for edge in gedges}
    probs1, weights1 = score_based_prob(gedges, score, pls)
    marginals = {i:np.ones(state_nums[i])/state_nums[i] for i in gnodes}
    neighbors = [[] for _ in gnodes]
    
    t = 0
    for (a, b) in gedges:
        neighbors[a].append(b)
        neighbors[b].append(a)    
        
    cur_best = 999999999 
    
    for round in range(6, 6+more):
        marginal_rec = {i:[] for i in gnodes}
        res = [[] for _ in range(pls)]
        
        for batch in range(batch_size):
            seed_ = round*batch_size + batch
            s = time.time()
            FT, tree = factorization_(g, seed_, gnodes, pls, state_nums, observations, 
                                        probs1, weights1, neighbors, marginals)  
            
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
            
        # r = vote(res)
        # # r = vote_avg(res)
        # cur_loss = loss(observations, r, gedges)
        # cur_best = min(cur_best, cur_loss)
        
        # loss_.append(cur_best)
        
        marginals = update_marginals2(marginals, marginal_rec)
        r = gibs_belief(marginals)
        cur_loss = loss(observations, r, gedges)
        cur_best = min(cur_best, cur_loss)
        loss_.append(cur_best)
        
    # print(res)
    return loss_, time_spt, rank

def re_prop(batch_size, g, state_nums, observations, more, rank):
    
    gnodes = list(g.nodes)
    gedges = list(g.edges)
    pls = len(gnodes)
    
    loss_ = []
    time_spt = []
    score = {edge:[] for edge in gedges}
    # probs1, weights1 = score_based_prob(gedges, score, pls)
    # probs1, weights1 = select_prob(g)
    probs1, weights1 = edge_prob2(g)
    marginals = {i:np.ones(state_nums[i])/state_nums[i] for i in gnodes}
    neighbors = [[] for _ in gnodes]
    
    t = 0
    for (a, b) in gedges:
        neighbors[a].append(b)
        neighbors[b].append(a)    
        
    cur_best = 999999999 
    tree_set = []
    for round in range(6, 6+more):
        marginal_rec = {i:[] for i in gnodes}
        res = [[] for _ in range(pls)]
        
        for batch in range(batch_size):
            if round == 6:
                seed_ = round*batch_size + batch
                s = time.time()
                FT, tree = factorization_(g, seed_, gnodes, pls, state_nums, observations, 
                                            probs1, weights1, neighbors, marginals)  
                
                r = FT.sum_product()
                t += time.time() - s
                time_spt.append(t)
                tree_set.append(tree)
            else:
                FT, tree = factorization_2(tree_set[batch], gnodes, pls, state_nums, observations, 
                                            weights1, neighbors, marginals)
                r = FT.sum_product()
            
            for node in gnodes:
                vnode = FT.vnodes[node]
                marginal_rec[node].append(vnode.marginal)
            
            r = gibs_sampling(gnodes, FT, state_nums, False)
        
            for r_pair in r:
                i, rr = r_pair
                res[i].append(rr)
            
        # r = vote(res)
        # # r = vote_avg(res)
        # cur_loss = loss(observations, r, gedges)
        # cur_best = min(cur_best, cur_loss)
        
        # loss_.append(cur_best)
        
        marginals = update_marginals2(marginals, marginal_rec)
        r = gibs_belief(marginals)
        cur_loss = loss(observations, r, gedges)
        cur_best = min(cur_best, cur_loss)
        loss_.append(cur_best)
        
    # print(res)
    return loss_, time_spt, rank


n = 30
p0 = 3
p1 =  10 / (n * (n + 1))
seed = 6
state_nums, observations, g = er_graph_obs(n*n, p=p1, seed=seed)
# state_nums, observations, g = grid_obs(n, n, seed=seed)
# print(len(list(g.edges)))
# # t0 = time.time()
# # p = edge_prob(g)
# # print(p)
# print(batch_update2(20, g, state_nums, observations, 10, 1)[0])
print(n, p0, seed)
t1 = time.time()
print(re_prop(200, g, state_nums, observations, 20, 1)[0])
print(time.time() - t1)
print(lbp(g, state_nums, observations, 1))
# print(spt(g, state_nums, observations, 60, 1)[0])

# title = "/home/wangyaomin/research/SPT/data/grid(240203)/"
# for n in tqdm([20, 30, 40]):
#     for seed in [6, 7, 8, 9, 10]:
#         state_nums, observations, g = grid_obs(n, n, seed)
#         l1 = re_prop(200, g, state_nums, observations, 20, 1)[0]
#         l2 = lbp(g, state_nums, observations, 1)
#         df = pd.DataFrame({'re_prop':l1})
#         df2 = pd.DataFrame({'lbp':l2})
#         df.to_csv(title + "re_prop_" + str(n) + "_" + str(seed) + ".csv",  index=False)
#         df2.to_csv(title + "lbp_" + str(n) + "_" + str(seed) + ".csv", index=False)
        
        
# title = "/home/wangyaomin/research/SPT/data/er(240203)/"
# n = 30
# p0 = 3 # 0.5, 1, 2, 3
# p1 = p0 * 4 / (n * (n + 1))
# for seed in [6, 7, 8, 9, 10]:
#     state_nums, observations, g = er_graph_obs(n*n, p=p1, seed=seed)
#     l1 = re_prop(200, g, state_nums, observations, 20, 1)[0]
#     l2 = lbp(g, state_nums, observations, 1)
#     df = pd.DataFrame({'re_prop':l1})
#     df2 = pd.DataFrame({'lbp':l2})
#     df.to_csv(title + "re_prop_" + str(n) + "_" + str(seed) + ".csv",  index=False)
#     df2.to_csv(title + "lbp_" + str(n) + "_" + str(seed) + ".csv", index=False)