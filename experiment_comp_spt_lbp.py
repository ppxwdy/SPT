from graph_model import *
# from BP_pure_version import *
import networkx as nx 
import numpy as np
import pandas as pd

# from dppy.exotic_dpps import UST
import factorgraph as fg
from tqdm import tqdm
import pandas as pd
from multiprocessing import Pool, Manager


from scipy.sparse.csgraph import laplacian
from scipy.sparse import * 
import matplotlib.pyplot as plt

from itertools import chain

import time
import os

import sympy as sp


def er_graph_obs(n, p=0.6, seed=6):
    np.random.seed(seed)
    g = nx.erdos_renyi_graph(n=n, p=p, seed=seed)
    
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

    S = [g.subgraph(c).copy() for c in nx.connected_components(g)]
    if len(S) > 1:
        nodee = []
        for i in range(len(S)):
            nodee.append(list(S[i].nodes)[0])
            
        for j in range(len(nodee)-1):
            n1, n2 = min(nodee[j], nodee[j+1]), max(nodee[j], nodee[j+1])
            g.add_edge(n1, n2)
    
    return state_nums, observations, g


def grid_obs(n,m, seed=6, p=1):
    g = nx.Graph()
    nodes = [i for i in range(n*m)]
    g.add_nodes_from(nodes)

    np.random.seed(seed)
    # random state_nums
    states = [i for i in range(10, 16)]
    state_nums = {nodes[i]: np.random.choice(states) for i in range(n*m)}
    observations = {}
    for row in range(n-1):
        for col in range(m-1):
            node = row*m + col
            right = node + 1
            below = node + m
            if np.random.rand() <= p:
                g.add_edge(node, right)
                g.add_edge(node, below)
            # observation
            states = [i for i in range(1, state_nums[node]+1)]
            observations[node] = np.random.choice(states)

   
    for row in range(n-1):
        node = (row + 1) * m - 1
        below = (node + m)
        if np.random.rand() <= p:
            g.add_edge(node, below)
        # observation
        states = range(1, state_nums[node]+1)
        observations[node] = np.random.choice(states)
    
    for col in range(m-1):
        node = (n-1)*m + col
        right = node + 1
        if np.random.rand() <= p:
            g.add_edge(node, right)
        # observation
        states = range(1, state_nums[node]+1)
        observations[node] = np.random.choice(states)
        
    if p < 1:
        S = [g.subgraph(c).copy() for c in nx.connected_components(g)]
        if len(S) > 1:
            nodee = []
            for i in range(len(S)):
                nodee.append(list(S[i].nodes)[0])
                
            for j in range(len(nodee)-1):
                n1, n2 = min(nodee[j], nodee[j+1]), max(nodee[j], nodee[j+1])
                g.add_edge(n1, n2)
        
    node = n*m - 1 
    states = range(1, state_nums[node]+1)
    observations[node] = np.random.choice(states)
    node = n*m - 1 
    states = range(1, state_nums[node]+1)
    observations[node] = np.random.choice(states)
    
    # print(observations)
    return state_nums, observations, g

def select_prob(g):
    adj = nx.adjacency_matrix(g)
    r = laplacian(adj).toarray()
    np.int = np.int64
    edges = list(g.edges)
    probs = np.zeros(adj.shape)
    weights = {}
    r1 = np.delete(r, 0, axis=0)
    r1 = np.delete(r1, 0, axis=1) 
    # denominator = int(np.round(np.linalg.det(r1)))
    

    denominator = np.linalg.det(r1)

    for n1, n2 in edges:
        L_tilde = np.delete(r, [n1, n2] , axis=0)
        L_tilde = np.delete(L_tilde, [n1, n2], axis=1)
        numerator = int(np.round(np.linalg.det(L_tilde)))


        probs[n1][n2] = numerator / denominator
        probs[n2][n1] = numerator / denominator
        # probs[(n1, n2)] = numerator / denominator
        weights[(n1, n2)] = denominator / numerator
    for i in range(probs.shape[0]):
        probs[i, :] /= sum(probs[i, :])

    return probs, weights


def quadratic(state1, state2):
    """quadratic energy function
    
    Args:
        state1 (int): the state number of up node
        state2 (int): the state number of sub node

    Returns:
        _type_: _description_
    """
    return np.exp(-(state1-state2)**2)
    
    
def cubic(state1, state2):
    """quadratic energy function
    
    Args:
        state1 (int): the state number of up node
        state2 (int): the state number of sub node

    Returns:
        _type_: _description_
    """
    return np.exp(-(state1-state2)**2)

def abs_(state1, state2):
    return np.exp(-abs(state1)-abs(state2))

def potential(state_nums):
    # all_states = []
    # for num in state_nums:
    #     all_states.append([i+1 for i in range(num)])

    num_states1, num_states2 = state_nums
    
    states1 = [i+1 for i in range(num_states1)]     
    states2 = [i+1 for i in range(num_states2)]    
    
    potentials = [[0 for i in range(num_states2)] for _ in range(num_states1)] 
        
    for i in range(num_states1):
        for j in range(num_states2):
            state1 = states1[i]
            state2 = states2[j]
            p = np.exp(-(state1-state2)**2)
            potentials[i][j] = p

    
    return potentials  


def potential_obs(state_nums, obs, func=quadratic):
    
    states1 = [i+1 for i in range(state_nums)]     
    
    potentials = [[0] for _ in range(state_nums)] 
        
    for i in range(state_nums):
        state1 = states1[i]
        p = func(state1, obs)
        potentials[i][0] = p

    return potentials

def factor_obs(edges, grid_nodes,  observations, states):
    # Create factor graph
    # fg = graphs.FactorGraph()
    fac_g = fg.Graph()
    # Create variable nodes

    for node in grid_nodes:
        fac_g.rv(f'{node}', states[node])
    
    for n1, n2 in edges:
        joint_p = potential([states[n1], states[n2]])
        fac_g.factor([f'{n1}', f'{n2}'], potential=np.array(joint_p))
        
        
    for node in grid_nodes:
        num_states1 = states[node]
        joint_p = potential_obs(num_states1, observations[node])
        fac_g.rv(f'obs{node}', 1)
        fac_g.factor([f'{node}', f'obs{node}'], potential=np.array(joint_p))
        
    iters, converged = fac_g.lbp(normalize=True)

    return fac_g.rv_marginals(normalize=True)



def gibs_sampling_grid_obs(gnodes, beliefs, state_nums):
    # values = []
    lp_beliefs = {}
    temp_values = {}
    for idx, node in enumerate(gnodes):
        lp_beliefs[node] = beliefs[idx][1].tolist()
        v = np.random.choice([i for i in range(1, state_nums[node]+1)], p=beliefs[idx][1])
        # values.append(v)
        temp_values[node] = v
    values = [[] for _ in range(max(temp_values.keys())+1)]
    for k,v in temp_values.items():
        values[k] = v
    # print("values lbp", values)
    return values, lp_beliefs

def loss(observations, values, edges):
    loss = 0
    nodes = set()
    for n1, n2 in edges:
        loss += (values[n1] - values[n2])**2
        nodes.add(n1)
        nodes.add(n2)
    nodes = list(nodes)
    for i in range(len(nodes)):
        n1 = nodes[i]
        loss += (values[n1] - observations[n1])**2
    return loss

def ust_sampler_wilson(list_of_neighbors, prob_mat, weight=None, root=None,
                       random_state=None, num=None):
    np.random.seed(random_state)
    # n = len(list_of_neighbors)

    # Initialize the tree
    nb_nodes = len(list_of_neighbors)
    num = num if num else nb_nodes
    # Initialize the root, if root not specified start from any node
    # n0 = root if root else np.random.choice([i for i in range(n)])  # size=1)[0]
    n0 = 0
    for i in range(len(list_of_neighbors)):
        if list_of_neighbors[i]:
            n0 = i
            break
    # -1 = not visited / 0 = in path / 1 = in tree
    state = -np.ones(nb_nodes, dtype=int)
    for i in range(len(list_of_neighbors)):
        lis = list_of_neighbors[i]
        if len(lis) == 0:
            state[i] = 10
            
    state[n0] = 1
    nb_nodes_in_tree = 1

    path, branches = [], []  # branches of tree, temporary path
    
    while nb_nodes_in_tree < num:  # |Tree| = |V| - 1

        # visit a neighbor of n0 uniformly at random
        prob = []
        # for n in list_of_neighbors[n0]:
        #     prob.append(prob_mat[n0, n])
        prob = prob_mat[n0, :][np.where(prob_mat[n0, :] != 0)]
    
        n1 = np.random.choice(list_of_neighbors[n0], p=prob)  # size=1)[0]
       

        if state[n1] == -1:  # not visited => continue the walk

            path.append(n1)  # add it to the path
            state[n1] = 0  # mark it as in the path
            n0 = n1  # continue the walk

        if state[n1] == 0:  # loop on the path => erase the loop

            knot = path.index(n1)  # find 1st appearence of n1 in the path
            nodes_loop = path[knot + 1:]  # identify nodes forming the loop
            del path[knot + 1:]  # erase the loop
            state[nodes_loop] = -1  # mark loopy nodes as not visited
            n0 = n1  # continue the walk

        elif state[n1] == 1:  # hits the tree => new branch

            if nb_nodes_in_tree == 1:
                branches.append([n1] + path)  # initial branch of the tree
            else:
                branches.append(path + [n1])  # path as a new branch

            state[path] = 1  # mark nodes in path as in the tree
            nb_nodes_in_tree += len(path)

            # Restart the walk from a random node among those not visited
            nodes_not_visited = np.where(state == -1)[0]
            if nodes_not_visited.size:
                n0 = np.random.choice(nodes_not_visited)  # size=1)[0]
                path = [n0]

    
    tree_edges = list(chain.from_iterable(map(lambda x: zip(x[:-1], x[1:]),
                                              branches)))
    # wilson_tree_graph.add_edges_from(tree_edges)

    return tree_edges



def factorization(g, seed, nodes, pls, state_nums, observations, probs, weights, neighbors, num=None):
    # spt = UST(g)
    # spt.sample(random_state=seed)
    # tree = spt.list_of_samples[0].edges
    
    tree = ust_sampler_wilson(neighbors, probs, weights, root=None,
                       random_state=seed, num=num)
    
    FTree1 = factor_tree()
    for node in nodes:
        obs = node + pls
        FTree1.add_nodes([node, obs], [state_nums[node], 1], [0, observations[node]])
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


def gibs_sampling(nodes, Tree, state_nums, key=True):
    if key:
        values = [0 for _ in range(len(nodes))]
        for node in nodes:
            vnode = Tree.vnodes[node]
            marginal = vnode.get_marginal()
            # print(marginal)
            try:
                val = np.random.choice([i for i in range(1, state_nums[node]+1)], p=marginal)
                values[node] = val
            except:
                print("Error in gibs sampling!")
                print(node)
                print(marginal)
    else:
        values = []
        for node in nodes:
            vnode = Tree.vnodes[node]
            marginal = vnode.get_marginal()
            # print(marginal)
            try:
                val = np.random.choice([i for i in range(1, state_nums[node]+1)], p=marginal)
                values.append([node, val])
            except:
                print("Error in gibs sampling!")
                print(node)
                print(marginal)
    # print("values", values)
    return values


def vote(all_res):
    ress = []
    for i in range(len(all_res)):
        res = all_res[i]
        counts = np.bincount(res)
        ress.append(np.argmax(counts))
    return ress


def rescore(tree, res, score):
    for a, b in tree:
        edge = (min(a, b), max(a, b))
        score[edge].append(res)
    return score

def rescore2(tree, res, score):
    for a, b in tree:
        edge = (min(a, b), max(a, b))
        score[edge] += 1
    return score
 
def score_based_prob2(edges, score, n, count, key=True):      
    prob_mat = np.zeros((n, n))
    weights = {}
    for edge in edges:
        a, b = min(edge), max(edge)
        edge = (a, b)
        
        if not score[edge] or not key:
            prob_mat[a][b] = 1
            prob_mat[b][a] = 1
            weights[edge] = 1
        #best n record
        else:
            prob_mat[a][b] = np.mean(score[edge])/count
            prob_mat[b][a] = 1/np.mean(score[edge])/count
            weights[edge] = count/np.mean(score[edge])

        
    for i in range(n):
        prob_mat[i, :] /= sum(prob_mat[i, :])

    return prob_mat, weights 
 
def score_based_prob(edges, score, n, key=True):      
    prob_mat = np.zeros((n, n))
    weights = {}
    for edge in edges:
        a, b = min(edge), max(edge)
        edge = (a, b)
        # prob_mat[a][b] = 1
        # prob_mat[b][a] = 1
        # weights[edge] = 1
        
        if not score[edge] or not key:
            prob_mat[a][b] = 1
            prob_mat[b][a] = 1
            weights[edge] = 1
        #best n record
        elif len(score[edge]) >= 5:
            
            s = np.mean(sorted(score[edge])[-n:])
            prob_mat[a][b] = 1/s
            prob_mat[b][a] = 1/s
            weights[edge] = s
        # all record
        else:
            
            prob_mat[a][b] = 1/np.mean(score[edge])
            prob_mat[b][a] = 1/np.mean(score[edge])
            weights[edge] = np.mean(score[edge])

        
    for i in range(n):
        prob_mat[i, :] /= sum(prob_mat[i, :])

    return prob_mat, weights
       
       
def experiment(n, more, seed=6, key=False):
    # print(f"START {n}!")
    m = n
    state_nums, observations, g = grid_obs(n, n, seed)
    gnodes = list(g.nodes)
    gedges = list(g.edges)

    t0 = time.time()
    res = factor_grid_obs(gedges, gnodes, observations, state_nums)
    time_lbp = time.time() - t0
    
    lp_values, lp_beliefs = gibs_sampling_grid_obs(gnodes, res[:m*n], state_nums)
    l1 = loss(observations, lp_values, gedges)

    # probs, weights = select_prob(g)
    pls = n*n
    all_res = [[] for _ in range(n*n)]
    l2s = []
    score = {edge:[] for edge in gedges}
    
    neighbors = [[] for _ in gnodes]
    
    for (a, b) in gedges:
        neighbors[a].append(b)
        neighbors[b].append(a)     
    # count = 0
    
    time_spt = []
    
    for seed_ in range(6, 6+more):

        probs, weights = score_based_prob(gedges, score, n*n)
        # if key:
        #     if count >= int(0.4*pls):
        #         probs, weights = score_based_prob(gedges, score, n*n)
        #     else:
        #         count += 1
        
        t1 = time.time()
        FT, tree = factorization(g, seed_, gnodes, pls, state_nums, observations, probs, weights, neighbors)
        res = FT.sum_product()
        t2 = time.time() - t1
        time_spt.append(t2)        

        res = gibs_sampling(gnodes, FT, state_nums)
        for i in range(n*n):
            all_res[i].append(res[i])
        
        res = vote(all_res)

        l2 = loss(observations, res, gedges)
        l2s.append(l2)
        
        score = rescore(tree, l2, score)
    # print(f"END {n}!")
    return  n, time_lbp, time_spt, l1, l2s


def experiment2(n, more, seed=6, p1=0.1, key=False, p2=1):
    print(f"START {n}, {seed}, {p1}!")
    m = n

    state_nums, observations, g = er_graph_obs(n*n, p=p1, seed=seed)
    # state_nums, observations, g = grid_obs(n,m, seed=seed, p=p2)

    gnodes = list(g.nodes)
    gedges = list(g.edges)
    time_lbp0 = 0
    l10 = 0
    try:
        t0 = time.time()
        res = factor_obs(gedges, gnodes, observations, state_nums)
        time_lbp0 = time.time() - t0
        
        lp_values, lp_beliefs = gibs_sampling_grid_obs(gnodes, res[:m*n], state_nums)

        l10 = loss(observations, lp_values, gedges)
    except:
        print(n, p1, "lbp failed!")
        time_lbp0 = 0
    

    
    # probs, weights = select_prob(g)
    pls = n*n
    all_res = [[] for _ in range(n*n)]
    l2s = []
    score = {edge:[] for edge in gedges}

    
    neighbors = [[] for _ in gnodes]
    
    for (a, b) in gedges:
        neighbors[a].append(b)
        neighbors[b].append(a)     
    # count = 0
    
    time_spt1 = []
    time_spt2 = []

    t2 = 0
    t3 = 0
    
    res = [[] for _ in range(n*n)]
    res2 = [[] for _ in range(n*n)]
    l2 = []
    l3 = []
    
    probs1, weights1 = score_based_prob(gedges, score, n*n)
    probs2, weights2 = score_based_prob(gedges, score, n*n)
    for seed_ in range(6, 6+more):

        s = time.time()
        FT, tree = factorization(g, seed_, gnodes, pls, state_nums, observations, probs1, weights1, neighbors)  
        r = FT.sum_product()
        t2 += time.time() - s
        time_spt1.append(t2)
        
        r = gibs_sampling(gnodes, FT, state_nums, False)
        
        for r_pair in r:
            i, rr = r_pair
            res[i].append(rr)
        
        r = vote(res)
        
        l2.append(loss(observations, r, gedges))
        
        
        s = time.time()
        FT, tree = factorization(g, seed_, gnodes, pls, state_nums, observations, probs2, weights2, neighbors)  
        r = FT.sum_product()
        t3 += time.time() - s
        time_spt2.append(t3)
        
        r = gibs_sampling(gnodes, FT, state_nums, False)
        
        for r_pair in r:
            i, rr = r_pair
            res2[i].append(rr)
        
        r = vote(res2)
        
        l = loss(observations, r, gedges)
        l3.append(l)
        score = rescore(tree, l, score)
        probs2, weights2 = score_based_prob(gedges, score, n*n)
        # print(score[(0,1)], score[(0, 5)])
        # print(probs2[0,:])
        # print(probs2)
        # l2s2.append(l3)
    print(f"END {n}, {seed}, {p1}!")
    return l10, l2, l3, time_lbp0, time_spt1, time_spt2, p1# 1 2

    # return  n, time_lbp0, time_spt, l10, l2s # 3\
    # return l1, l2s, loss(observations, res, gedges)
    # return  n,  l2 s, l2s2, time_spt, time_spt2, int(p*10) # 8\


# n = 20
# i = 1
# print(experiment2(30, 45, 6, 0.5*4/(30*31), False, 0.2)) #fit_model(3, 0.1)
# print((696-737)/737)

if __name__ == '__main__':

# # #     title = '/home/ymwang/code/Large_scale_MAP/SPT/SPTdata_10_1123(spt_lbp_grid_gap)/'

# # #    
    title = "/home/ymwang/code/Research/FromFiledsToTrees/SPTdata_30_1223(spt_lbp_rl_sparse_gap)/"
    
    isExists = os.path.exists(title)
    if not isExists:
        os.makedirs(title)  

    pool = Pool(processes=3)
    ns = [70, 80]
    # ps = [0.4, 0.5, 0.6, 0.7, 0.8]
    ps = [0.1, 0.2, 0.3]
    params = []
    
    for i in range(4):
        
        for n in ns:
            temp = [] 
            for p in ps:
                temp.append((n, int(0.1*n**2), 6+i, p*4/(n*(n+1)))) 
            params.append(temp)

    for i in tqdm(range(len(params))):
        param = params[i]
        n = param[0][0]
        rtn = pool.starmap(experiment2, param)
        p = param[0][-1]
        for j in range(len(rtn)):
            # print(len(rtn), len(param), "start")
            
            r = rtn[j]
            df = pd.DataFrame({'lbp':[r[0]]})
            df.to_csv(title+f"lbp_{n}_{str(ps[j])[:3]}_{i}.csv", index=False)
            df = pd.DataFrame({"l1":r[1], "l_rl":r[2], "t":r[4], "t_rl":r[5]})
            df.to_csv(title+f"spt_{n}_{str(ps[j])[:3]}_{i}.csv", index=False)
# # # 9. loss gap curve
# # ##############################################################################################
    # pool = Pool(processes=4)

    # params = []

    # n = 30
    # ps = [i*0.2 for i in range(1, 6)]

# #     for p in ps:
# #         temp = [] 
# #         for i in range(1, 5):
# #             temp.append((n, int(0.5*n**2), 6+i, 0.1, False, p)) 
# #         params.append(temp)

# #     for i in tqdm(range(len(params))):
# #         param = params[i]
# #         rtn = pool.starmap(experiment2, param)
# #         p = param[0][-1]
# #         for j in range(len(rtn)):
# #             # print(len(rtn), len(param), "start")
            
# #             r = rtn[j]

# #             df = pd.DataFrame({'l_1':r[0], "l_rl":r[1]})
# #             df.to_csv(title+f"{n}_{p}_{j}.csv", index=False)
            
# #     for p in ps:
# #         temp = [] 
# #         for i in range(1, 5):
# #             temp.append((n, int(0.5*n**2), 66+i, 0.1, False, p)) 
# #         params.append(temp)

# #     for i in tqdm(range(len(params))):
# #         param = params[i]
# #         rtn = pool.starmap(experiment2, param)
# #         p = param[0][-1]
# #         for j in range(len(rtn)):
# #             # print(len(rtn), len(param), "start")
            
# #             r = rtn[j]

# #             df = pd.DataFrame({'l_1':r[0], "l_rl":r[1]})
# #             df.to_csv(title+f"{n}_{p}_{4+j}.csv", index=False)

# # loss gap
    
#     ns = [50, 60]
#     p1 = [ 0.4, 0.5, 0.6, 0.7, 0.8]
    
    
    # t1 = time.time()
    # for n in tqdm(ns):
    #     for i in range(len(p1)):
    #         pp = p1[i]
    #         p = 4*pp / (n*(n+1))
            
    #         rtn = experiment2(n, int(0.1*n*n), 6, p)
    #         lbp_loss.append(rtn[0])
    #         lbp_time.append(rtn[2])
            
    #         file_lbp["loss"] = lbp_loss
    #         file_lbp["time"] = lbp_time
    #         Loss_spt[str(i)] = rtn[1]
    #         Time_spt[str(i)] = rtn[3]
            
    #         pd.DataFrame(file_lbp).to_csv(title+f"lbp_loss_time.csv", index=False)
    #         pd.DataFrame(Loss_spt).to_csv(title+f"spt_loss.csv", index=False)
    #         pd.DataFrame(Time_spt).to_csv(title+f"spt_time.csv", index=False)
            
    # print("time 1 is ", time.time() - t1)
    
    
#     pool = Pool(processes=5)
    
#     params = []
#     for seed in range(2, 8):
#         for n in ns:
#             temp = []
#             for i in range(len(p1)):
#                 temp.append((n, int(0.05*n*n), 6+seed, 4*p1[i] / (n*(n+1))))
#             params.append(temp)


#     count = 1
#     for i in tqdm(range(len(params))):
        
#         file_lbp = {}
#         Loss_spt = {}
#         Time_spt = {}
        
#         lbp_loss = []
#         lbp_time = []
            
#         n = params[i][0][0]
        
        
#         param = params[i] 
#         rtn = pool.starmap(experiment2, param)
#         rtn.sort(key=lambda x:x[-1])
#         for j in range(len(rtn)):
#             r = rtn[j]

#             lbp_loss.append(r[0])
#             lbp_time.append(r[2])
            
#             file_lbp["loss"] = lbp_loss
#             file_lbp["time"] = lbp_time
#             Loss_spt[str(j)] = r[1]
#             Time_spt[str(j)] = r[3]
            
#             pd.DataFrame(file_lbp).to_csv(title+f"lbp_loss_time_{n}_{count}.csv", index=False)
#             pd.DataFrame(Loss_spt).to_csv(title+f"spt_loss_{n}_{count}.csv", index=False)
#             pd.DataFrame(Time_spt).to_csv(title+f"spt_time_{n}_{count}.csv", index=False)
            
#         if n == 60:
#             count += 1
                
    