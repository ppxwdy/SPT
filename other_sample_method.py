import numpy as np
from graph_model import *
from experiment_comp_spt_lbp import *
from multiprocessing import Pool, Manager
import os
from tqdm import tqdm

import time

def edge_prob(g):
    nodes = list(g.nodes)
    edges = list(g.edges)
    n, m = len(nodes), len(edges)
    
    probs = np.zeros((n, n))
    a = np.diag(np.ones(n))
    be = {}
    L = np.zeros((n, n))
    
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
    for i in range(probs.shape[0]):
        probs[i, :] /= sum(probs[i, :])

    return probs

def select_prob(g):
    adj = nx.adjacency_matrix(g)
    r = laplacian(adj).toarray()
    r = r + np.ones(r.shape)/r.shape[0]
    edges = list(g.edges)
    probs = np.zeros((len(g.nodes), len(g.nodes)))
    r1 = np.delete(r, 0, axis=0)
    r1 = np.delete(r1, 0, axis=1) 
    denominator = int(np.round(np.linalg.det(r)))


    for n1, n2 in edges:
        L_tilde = np.delete(r, [n1, n2] , axis=0)
        L_tilde = np.delete(L_tilde, [n1, n2], axis=1)
        numerator = int(np.round(np.linalg.det(L_tilde)))
        probs[n1, n2] = numerator / denominator
        probs[n2, n1] = probs[n1, n2]
        
    for i in range(probs.shape[0]):
        probs[i, :] /= sum(probs[i, :])
    return probs



def prob_node(node_count, node_prob):
    for node in range(len(node_count)):
        node_prob[node] = 1 / node_count[node]
    return node_prob / sum(node_prob)


def prob_edge(edge_count, prob_mat, weights, edges):
    for a, b in edges:
        prob_mat[a, b] = 1 / edge_count[(a, b)]
        prob_mat[b, a] = 1 / edge_count[(a, b)]
        weights[(a,b)] = edge_count
        
    for i in range(prob_mat.shape[0]):
        prob_mat[i, :] /= sum(prob_mat[i, :])
        
    return prob_mat, weights

def gibs_sampling_random_walk(gnodes, beliefs, state_nums):
    # values = []
    lp_beliefs = {}
    temp_values = {}
    for idx, node in enumerate(gnodes):
        lp_beliefs[node] = beliefs[idx][1].tolist()
        v = np.random.choice([i for i in range(1, state_nums[node]+1)], p=beliefs[idx][1])
        # values.append(v)
        temp_values[node] = v
    values = []
    for k,v in temp_values.items():
        values.append((k,v))
    # print("values lbp", values)
    return values, lp_beliefs




def non_assign_vote(all_res, obs):
    ress = []
    for i in range(len(all_res)):
        if all_res[i] == []:
            ress.append(obs[i])
            continue
        res = all_res[i]
        counts = np.bincount(res)
        ress.append(np.argmax(counts))
    return ress

# def loss(observations, values, edges):
#     loss = 0
#     nodes = set()
#     for n1, n2 in edges:
#         loss += (values[n1] - values[n2])**2
#         nodes.add(n1)
#         nodes.add(n2)
#     nodes = list(nodes)
#     for i in range(len(nodes)):
#         n1 = nodes[i]
#         loss += (values[n1] - observations[n1])**2
#     return loss


def factorization_(tree, nodes, pls, state_nums, observations, weights):
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

    FTree1.set_tree_root(nodes[0])

    FTree1.generate_node_tree()
    FTree1.build_factor_tree()
    
    # FTree1.build()
    
    return FTree1


def sample_chain(list_of_neighbors, node_prob, prob_mat, node_num, node_count, edge_count, seed=6):
    start_node = np.random.choice(range(node_num), p=node_prob)
    n0 = start_node    
    
    visited = -1 * np.ones(node_num)
   
    edges = []
    nodes = []
    time = 1
    while True:
        visited[n0] = 1
        nodes.append(n0)
        
        node_count[n0] += 1
        prob = prob_mat[n0, :][np.where(prob_mat[n0, :] != 0)]
        n1 = np.random.choice(list_of_neighbors[n0], p=prob)
        
        if visited[n1] == 1:
            key = True
            for n in list_of_neighbors[n0]:
                if visited[n] == -1:
                    key = False
                    break
            
            if key:
                if time > 1:
                    break
                n0 = start_node
                time += 1
        else:
            edge = (min(n0, n1), max(n0, n1))
            edges.append(edge)
            n0 = n1
            edge_count[edge] += 1

    return edges, nodes, node_count, edge_count
        


def chain(g, state_nums, obs, more, rank, seed=6):
    gnodes = list(g.nodes)
    gedges = list(g.edges)
    
    neighbors = [[] for _ in gnodes]
    prob_mat = np.zeros((len(gnodes), len(gnodes)))
    node_prob = np.ones(len(gnodes)) / len(gnodes)
    weights = {}
    for (a, b) in gedges:
        neighbors[a].append(b)
        neighbors[b].append(a)    

        prob_mat[a][b] = 1
        prob_mat[b][a] = 1
        
        weights[(a,b)] = 1
        
    for i in range(prob_mat.shape[0]):
        prob_mat[i, :] /= sum(prob_mat[i, :])
        
    record = {node:[] for node in gnodes}
    
    node_count = {node: 0 for node in gnodes}
    edge_count = {edge: 0 for edge in gedges}
    times = []
    loss_ = []
    pls = len(gnodes)
    for seed_ in range(1, more+1):
        t0 = time.time()
        chain, nodes, node_count, edge_count = sample_chain(neighbors, node_prob, prob_mat, len(gnodes), node_count, edge_count, seed*seed_)
        # print(chain)
        FT = factorization_(chain, nodes, pls, state_nums, obs, weights)
        r = FT.sum_product()
        times.append(time.time() - t0)
        r = gibs_sampling(nodes, FT, state_nums, False)
        
        for r_pair in r:
            i, rr = r_pair
            record[i].append(rr)
        
        r = non_assign_vote(record, obs)
        l = loss(obs, r, gedges)
        loss_.append(l)
    return loss_, times, rank

        

def lbp(g, state_nums, observations, rank):
    gnodes = list(g.nodes)
    gedges = list(g.edges)
    time_lbp0 = 0
    l10 = 0
    try:
        t0 = time.time()
        res = factor_obs(gedges, gnodes, observations, state_nums)
        time_lbp0 = time.time() - t0
        
        lp_values, lp_beliefs = gibs_sampling_grid_obs(gnodes, res[:len(gnodes)], state_nums)
        l10 = loss(observations, lp_values, gedges)
    except:
        l10 = 0
        time_lbp0 = 0
        print("lbp fail", rank)

    return l10, time_lbp0, rank

def vote2(all_res):
    res = [0 for _ in range(len(all_res.keys()))]
    for k,v in all_res.items():
        l = []
        all_ = 0
        for k2, v2 in v.items():
            if v2 != []:
                all_ += np.mean(v2)
        for k2, v2 in v.items():
            if v2 != []:
                l.append((k2, len(v2)*all_/np.mean(v2)))#np.mean(v2)
        l.sort(key=lambda x:x[1], reverse=True)
        res[k] = l[0][0]
        
    return res

def vote_avg(all_res):
    ress = []
    for i in range(len(all_res)):
        res = all_res[i]
        # counts = np.bincount(res)
        ress.append(np.round(np.mean(res)))
    return ress

def spt(g, state_nums, observations, more, rank):
    gnodes = list(g.nodes)
    gedges = list(g.edges)
    pls = len(gnodes)
    # res = [[] for _ in range(pls)]
    res = {i:{v:[] for v in range(1, state_nums[i]+1)} for i in range(pls)}
    loss_ = []
    time_spt = []
    score = {edge:[] for edge in gedges}
    probs1, weights1 = score_based_prob(gedges, score, pls)
    
    neighbors = [[] for _ in gnodes]
    t = 0
    for (a, b) in gedges:
        neighbors[a].append(b)
        neighbors[b].append(a)    
        
    cur_best = 999999999 
    
    for seed_ in range(6, 6+more):

        s = time.time()
        FT, tree = factorization(g, seed_, gnodes, pls, state_nums, observations, probs1, weights1, neighbors)  
        r = FT.sum_product()
        t += time.time() - s
        time_spt.append(t)
        
        r = gibs_sampling(gnodes, FT, state_nums, False)
        
        # for r_pair in r:
        #     i, rr = r_pair
        #     res[i].append(rr)
        
        # r = vote(res)
        
        rr = [0 for i in range(pls)]
        for r_pair in r:
            rr[r_pair[0]] = r_pair[1]
        cur_loss = loss(observations, rr, gedges)
        for i in range(pls):
            res[i][rr[i]].append(cur_loss)
            
        r = vote2(res)
        # r = vote_avg(res)
        cur_loss = loss(observations, r, gedges)
        cur_best = min(cur_best, cur_loss)
        loss_.append(cur_best)
    # print(res)
    return loss_, time_spt, rank


def spt2(g, state_nums, observations, more, rank):
    gnodes = list(g.nodes)
    gedges = list(g.edges)
    pls = len(gnodes)
    res = [[] for _ in range(pls)]
    # res = {i:{v:0 for v in range(1, state_nums[i]+1)} for i in range(pls)}
    loss_ = []
    time_spt = []
    score = {edge:[] for edge in gedges}
    weights1 = {}
    probs1 = edge_prob(g)

    for i in range(probs1.shape[0]):
        for j in range(i, probs1.shape[1]):
            if probs1[i, j] != 0:
                weights1[(i, j)] = 1 / probs1[i, j]
    # probs1, weights0 = score_based_prob(gedges, score, pls)
    
    neighbors = [[] for _ in gnodes]
    t = 0
    for (a, b) in gedges:
        neighbors[a].append(b)
        neighbors[b].append(a)    
        
    cur_best = 999999999 
    
    for seed_ in range(6, 6+more):

        s = time.time()
        FT, tree = factorization(g, seed_, gnodes, pls, state_nums, observations, probs1, weights1, neighbors)  
        r = FT.sum_product()
        t += time.time() - s
        time_spt.append(t)
        
        r = gibs_sampling(gnodes, FT, state_nums, False)
        
        for r_pair in r:
            i, rr = r_pair
            res[i].append(rr)
        
        r = vote(res)
        cur_loss = loss(observations, r, gedges)
        cur_best = min(cur_best, cur_loss)
        loss_.append(cur_best)

    return loss_, time_spt, rank


def spt_rl(g, state_nums, observations, more, rank):
    gnodes = list(g.nodes)
    gedges = list(g.edges)
    pls = len(gnodes)
    res = [[] for _ in range(pls)]
    loss_ = []
    time_spt = []
    score = {edge:[] for edge in gedges}
    score2 = {edeg:0 for edeg in gedges}
    probs, weights = score_based_prob(gedges, score, pls)

    neighbors = [[] for _ in gnodes]
    t = 0
    
    for (a, b) in gedges:
        neighbors[a].append(b)
        neighbors[b].append(a)  
           
    cur_best = 999999999 
    
    for seed_ in range(6, 6+more):
        s = time.time()
        FT, tree = factorization(g, seed_, gnodes, pls, state_nums, observations, probs, weights, neighbors)  
        r = FT.sum_product()
        t += time.time() - s
        time_spt.append(t)
        
        r = gibs_sampling(gnodes, FT, state_nums, False)
        
        for r_pair in r:
            i, rr = r_pair
            res[i].append(rr)
        
        r = vote(res)
        
        cur_loss = loss(observations, r, gedges)
        cur_best = min(cur_best, cur_loss)
        loss_.append(cur_best)
        score = rescore(tree, cur_loss, score)
        probs, weights = score_based_prob(gedges, score, pls)
        # score2 = rescore2(tree, cur_loss, score2)
        # probs, weights = score_based_prob2(gedges, score2, pls, seed_- 5)
    # print(probs)
    return loss_, time_spt, rank


def spt_random_tree(g, state_nums, observations, more, num, rank):
    gnodes = list(g.nodes)
    gedges = list(g.edges)
    pls = len(gnodes)
    res = [[] for _ in range(pls)]
    loss_ = []
    time_spt = []
    score = {edge:[] for edge in gedges}
    probs, weights = score_based_prob(gedges, score, pls)
    
    neighbors = [[] for _ in gnodes]
    t = 0
    for (a, b) in gedges:
        neighbors[a].append(b)
        neighbors[b].append(a)     
    
    for seed_ in range(6, 6+more):

        s = time.time()
        FT, tree = factorization(g, seed_, gnodes, pls, state_nums, observations, probs, weights, neighbors, num)  
        r = FT.sum_product()
        t += time.time() - s
        time_spt.append(t)
        
        r = gibs_sampling(gnodes, FT, state_nums, False)
        
        for r_pair in r:
            i, rr = r_pair
            res[i].append(rr)
        
        r = non_assign_vote(res, observations)
        
        loss_.append(loss(observations, r, gedges))
        
    return loss_, time_spt, rank


def sample_random_walk(list_of_neighbors, node_prob, prob_mat, node_num, node_count, edge_count, step, seed=6):
    start_node = np.random.choice(range(node_num), p=node_prob)
    n0 = start_node    
    
    edges = set()
    nodes = set()
    count = 0
    while count < step:
        count += 1
        
        prob = prob_mat[n0, :][np.where(prob_mat[n0, :] != 0)]
        n1 = np.random.choice(list_of_neighbors[n0], p=prob)
        edge = (min(n0, n1), max(n0, n1))
        edges.add(edge)
        n0 = n1
    
    for edge in edges:
        edge_count[edge] += 1    
        nodes.add(edge[0])
        nodes.add(edge[1])
        
    for node in nodes:
        node_count[node] += 1
    

    return edges, nodes, node_count, edge_count
    

def random_walk(g, state_nums, observations, more, step, rank):
    gnodes = list(g.nodes)
    gedges = list(g.edges)
    
    neighbors = [[] for _ in gnodes]
    prob_mat = np.zeros((len(gnodes), len(gnodes)))
    node_prob = np.ones(len(gnodes)) / len(gnodes)
    weights = {}
    for (a, b) in gedges:
        neighbors[a].append(b)
        neighbors[b].append(a)    

        prob_mat[a][b] = 1
        prob_mat[b][a] = 1
        
        weights[(a,b)] = 1
        
    for i in range(prob_mat.shape[0]):
        prob_mat[i, :] /= sum(prob_mat[i, :])
        
    record = {node:[] for node in gnodes}
    
    node_count = {node: 0 for node in gnodes}
    edge_count = {edge: 0 for edge in gedges}
    times = []
    pls = len(gnodes)
    loss_ = []
    for seed_ in range(1, more+1):
        t0 = time.time()
        sub_g, sub_nodes, node_count, edge_count = sample_random_walk(neighbors, node_prob, prob_mat, pls, node_count, edge_count, step, seed_)
        # print(chain)
        res = factor_obs(sub_g, sub_nodes, observations, state_nums)
        times.append(time.time() - t0)
        r, lp_beliefs = gibs_sampling_random_walk(sub_nodes, res[:len(sub_nodes)], state_nums)
        
        for r_pair in r:
            i, rr = r_pair
            record[i].append(rr)
        
        r = non_assign_vote(record, observations)
        l = loss(observations, r, gedges)
        loss_.append(l)
    
    
    return loss_, times, rank

def example():
    g = nx.Graph()
    g.add_nodes_from([0, 1, 2, 3])
    g.add_edges_from([(0, 1), (0, 2), (0, 3)])
    state_nums = [3, 3, 3, 3]
    observations = [1, 3, 1, 1]
    return g, state_nums, observations

def example2():
    g = nx.Graph()
    g.add_nodes_from([0, 1, 2, 3])
    g.add_edges_from([(0, 1), (2, 3), (1, 3)])
    state_nums = [3, 3, 3, 3]
    observations = [1, 3, 1, 1]
    return g, state_nums, observations

def example3():
    g = nx.Graph()
    g.add_nodes_from([0, 1, 2, 3])
    g.add_edges_from([(0, 1), (0, 3), (0,2), (1, 2), (2, 3), (1, 3)])
    state_nums = [3, 3, 3, 3]
    observations = [1, 3, 1, 1]
    return g, state_nums, observations

# g, state_nums, observations = example()
# print(spt(g, state_nums, observations, 1, 1))
# g, state_nums, observations = example2()
# print(spt(g, state_nums, observations, 1, 1))

# g, state_nums, observations = example3()
n = 20
p1 = 0.7 * 4 / (n * (n + 1))
seed = 6
state_nums, observations, g = er_graph_obs(n*n, p=p1, seed=seed)
# print(len(list(g.edges)))
# # t0 = time.time()
# # p = edge_prob(g)
# # print(p)
# print(spt(g, state_nums, observations, 60, 1))
# print(spt2(g, state_nums, observations, 40, 1))
# print(select_prob(g))
# # print('+++++++++++++++++++++++++++++++++++')
# print(spt_rl(g, state_nums, observations, 40, 1))
      
# print(lbp(g, state_nums, observations, 1))

# print(lbp(g, state_nums, observations, 1))
# n = 10
# p1 = 0.1*4/(30*31)
# seed = 6
# state_nums, observations, g = er_graph_obs(n*n, p=p1, seed=seed)
# print("1---------------------------------")
# print(chain(g, state_nums, observations, 4, 1,seed=seed))
# # print("2---------------------------------")
# # print(lbp(g, state_nums, observations))
# # print("3---------------------------------")
# print(spt(g, state_nums, observations, 3, 1))
# # print("4---------------------------------")
# # print(spt_rl(g, state_nums, observations, 1))
# # print("5---------------------------------")
# # print(spt_random_tree(g, state_nums, observations, 1, 0.7*n*n))
# print("6---------------------------------")
# print(random_walk(g, state_nums, observations, 4, 200, 1))
# [[2], [2], [1], [1]]
# [[3], [1], [1], [1]]
# print(loss(observations, [1, 2, 1, 1], g.edges))
# print(loss(observations, [3, 1, 1, 1], g.edges))
# if __name__ == '__main__': 
    
    #################################################lbp+spt###########################################################
    # title = "/home/ymwang/code/Research/FromFiledsToTrees/results/SPTdata_24_1123(spt_lbp_sparse_gap)/"
    
    # isExists = os.path.exists(title)
    # if not isExists:
    #     os.makedirs(title)  

    # pool = Pool(processes=3)
    # ns = [10, 20, 30, 40, 50, 60]
    # ps = [0.1, 0.2, 0.3] #, 0.4, 0.5, 0.6, 0.7, 0.8]

    
    # for repeat in tqdm(range(7)):
    #     for n in ns:
    #         temp_lbp = []
    #         temp_chain = []
    #         temp_spt = []
    #         temp_rand_tree = []
    #         i = 0
    #         for p in ps:
    #             p1 = p*4/(n*(n+1))
    #             more = int(0.1*n**2)
    #             more2 = n*n

    #             state_nums, observations, g = er_graph_obs(n*n, p=p1, seed=6+repeat)
    #             temp_lbp.append((g, state_nums, observations, i))
    #             temp_spt.append((g, state_nums, observations, more, i))

    #             i += 1
                
    #         rtn = pool.starmap(lbp, temp_lbp)
    #         rtn.sort(key=lambda x:x[-1])
    #         res = {}
    #         for j in range(len(rtn)):
    #             r = rtn[j]
    #             res[f"l_{j}"] = [r[0]]
    #             res[f't_{j}'] = [r[1]]
    #         df = pd.DataFrame(res)
    #         df.to_csv(title+f"lbp_{n}_{repeat}.csv", index=False)
            
            
    #         rtn = pool.starmap(spt, temp_spt)
    #         rtn.sort(key=lambda x:x[-1])
    #         res = {}
    #         for j in range(len(rtn)):
    #             r = rtn[j]
    #             res[f"l_{j}"] = r[0]
    #             res[f't_{j}'] = r[1]
    #         df = pd.DataFrame(res)
    #         df.to_csv(title+f"spt_{n}_{repeat}.csv", index=False)
    
    ################################################topology###########################################################
            
    # title = "/home/ymwang/code/Research/FromFiledsToTrees/results/SPTdata_18_0124(rl_lbp_fixp)/"
    # isExists = os.path.exists(title)
    # if not isExists:
    #     os.makedirs(title)  
    # pool = Pool(processes=3)
    # ns = [20, 30, 40]
    # # # ns = [50, 60]
    # # # ps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    # # # ps = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    # # ps = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    # ps = [0.75]
    # # params = []
    # # params2 = []
    # # for repeat in tqdm(range(4)):
    # for re in range(1):
    #     for n in ns:
    #     # for p in ps:
    #         temp_lbp = []
    # #         temp_chain = []
    #         temp_spt = []
    # #         temp_rand_tree = []
    # #         temp_rand_walk = []
    #         i = 0
    #         for p in ps:
    #         # for n in ns:
    #             p1 = p*4/(n*(n+1))
    #             more = int(0.1*n**2)
    #             more2 = int(n**2)
    #             for repeat in range(4):
    #                 state_nums, observations, g = er_graph_obs(n*n, p=p1, seed=6+repeat)
    #                 temp_lbp.append((g, state_nums, observations, i))
    #                 # temp_chain.append((g, state_nums, observations, more2, i, 6+repeat))
    #                 temp_spt.append((g, state_nums, observations, more, i))
    #                 # temp_rand_tree.append((g, state_nums, observations, more, 0.7*n*n, i))
    #                 # temp_rand_walk.append((g, state_nums, observations, more, n*n, i))
    #                 i += 1
                
            # rtn = pool.starmap(lbp, temp_lbp)
            # rtn.sort(key=lambda x:x[-1])
            # res = {}
            # for j in range(len(rtn)):
            #     r = rtn[j]
            #     res[f"l_{j}"] = [r[0]]
            #     res[f't_{j}'] = [r[1]]
            # df = pd.DataFrame(res)
            # df.to_csv(title+f"lbp_deg3_{repeat}.csv", index=False)
            
            # rtn = pool.starmap(chain, temp_chain)
            # rtn.sort(key=lambda x:x[-1])
            # res = {}
            # for j in range(len(rtn)):
            #     r = rtn[j]
            #     res[f"l_{j}"] = r[0]
            #     res[f't_{j}'] = r[1]
            # df = pd.DataFrame(res)
            # df.to_csv(title+f"chain_{n}_{repeat}.csv", index=False)
            
            # rtn = pool.starmap(spt2, temp_spt)
            # rtn.sort(key=lambda x:x[-1])
            # res = {}
            # for j in range(len(rtn)):
            #     r = rtn[j]
            #     res[f"l_{j}"] = r[0]
            #     res[f't_{j}'] = r[1]
            # df = pd.DataFrame(res)
            # df.to_csv(title+f"spt_{n}_{repeat}2.csv", index=False)
            
            # rtn = pool.starmap(spt_rl, temp_spt)
            # rtn.sort(key=lambda x:x[-1])
            # res = {}
            # for j in range(len(rtn)):
            #     r = rtn[j]
            #     res[f"l_{j}"] = r[0]
            #     res[f't_{j}'] = r[1]
            # df = pd.DataFrame(res)
            # df.to_csv(title+f"sptrl_{n}_{repeat}.csv", index=False)
            
            
            # rtn = pool.starmap(spt_random_tree, temp_rand_tree)
            # rtn.sort(key=lambda x:x[-1])
            # res = {}
            # for j in range(len(rtn)):
            #     r = rtn[j]
            #     res[f"l_{j}"] = r[0]
            #     res[f't_{j}'] = r[1]
            # df = pd.DataFrame(res)
            # df.to_csv(title+f"randtree_{n}_{repeat}.csv", index=False)
            
            # rtn = pool.starmap(random_walk, temp_rand_walk)
            # rtn.sort(key=lambda x:x[-1])
            # res = {}
            # for j in range(len(rtn)):
            #     r = rtn[j]
            #     res[f"l_{j}"] = r[0]
            #     res[f't_{j}'] = r[1]
            # df = pd.DataFrame(res)
            # df.to_csv(title+f"randwalk_{n}_{repeat}_nn.csv", index=False)
                
                
                # print(chain(g, state_nums, observations, 1, seed=seed))
# print("2---------------------------------")
# print(lbp(g, state_nums, observations))
# print("3---------------------------------")
# print(spt(g, state_nums, observations, 1))
# print("4---------------------------------")
# print(spt_rl(g, state_nums, observations, 1))
# print("5---------------------------------")
# print(spt_random_tree(g, state_nums, observations, 1, 0.7*n*n))
