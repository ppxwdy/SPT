from graph_model_no_obs import *
import networkx as nx 
import numpy as np
import factorgraph as fg
from tqdm import tqdm
import time
import os
from experiment_comp_spt_lbp import ust_sampler_wilson

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


def er_graph(n, states=6, p=0.6, seed=6):
    np.random.seed(seed)
    g = nx.erdos_renyi_graph(n=n, p=p, seed=seed)
    
    gnodes = list(g.nodes)
    gedges = list(g.edges)
    
    S = [g.subgraph(c).copy() for c in nx.connected_components(g)]
    if len(S) > 1:
        nodee = []
        for i in range(len(S)):
            nodee.append(list(S[i].nodes)[0])
            
        for j in range(len(nodee)-1):
            n1, n2 = min(nodee[j], nodee[j+1]), max(nodee[j], nodee[j+1])
            g.add_edge(n1, n2)
    # random state_nums
    # each states generate a random number between 0 - 10
    Dxs = {gnodes[i]: np.random.randint(0, 10, states) for i in range(n)}

    # random_spanning_tree = nx.random_spanning_tree(g, seed=seed)
    # g = random_spanning_tree
    
    return Dxs, g


def saveModel(Dxs, g, states, id):
    loc = "/home/ymwang/code/Research/FromFiledsToTrees/TRW_model/"
    f = open(loc +f"model{id}.csv", "w")
    nodes = list(g.nodes())
    edges = list(g.edges())
    node_num = len(nodes)
    f.write(f"{node_num} {states}\n")
    
    for i in range(node_num):
        s = " ".join(str(element) for element in Dxs[i])
        f.write(s+"\n")
    
    f.write(f"{len(edges)}\n")
    for n1, n2 in edges:
        f.write(f"{n1} {n2}\n")
    f.close()


def potential(Dxs):
    # all_states = []
    # for num in state_nums:
    #     all_states.append([i+1 for i in range(num)])
    # Dxs = [Dx[x1], Dx[x2]]
    Dx1, Dx2 = Dxs
    l = len(Dx1)
    potentials = [[0 for i in range(l)] for _ in range(l)] 
    
    for i in range(l):
        d1 = Dx1[i]
        for j in range(l):
            d2 = Dx2[j]
            p = np.exp(- min( (i - j)**2, 10000000))#np.exp(- d1 - d2 - min(10 * (i - j)**2, 8))
            potentials[i][j] = p

    return potentials 


def loss(Dxs, values, edges):
    loss = 0
    nodes = set()
    for n1, n2 in edges:
        loss += (values[n1] - values[n2])**2
        nodes.add(n1)
        nodes.add(n2)
    nodes = list(nodes)
    for i in range(len(nodes)):
        n1 = nodes[i]
        # print(i, n1, values[n1])
        loss += Dxs[n1][int(values[n1])-1]
    return loss


# for lbp
def gibs_sampling_grid_obs(gnodes, beliefs, state_nums):
    # values = []
    lp_beliefs = {}
    temp_values = {}
    for idx, node in enumerate(gnodes):
        lp_beliefs[node] = beliefs[idx][1].tolist()
        v = np.random.choice([i for i in range(1, state_nums+1)], p=beliefs[idx][1])
        # values.append(v)
        temp_values[node] = v
    values = [[] for _ in range(max(temp_values.keys())+1)]
    for k,v in temp_values.items():
        values[k] = v
    # print("values lbp", values)
    return values, lp_beliefs


def factor_obs(edges, grid_nodes, Dxs, states):
    # Create factor graph
    # fg = graphs.FactorGraph()
    fac_g = fg.Graph()
    # Create variable nodes

    for node in grid_nodes:
        fac_g.rv(f'{node}', states)
    
    for n1, n2 in edges:
        joint_p = potential([Dxs[n1], Dxs[n2]])
        fac_g.factor([f'{n1}', f'{n2}'], potential=np.array(joint_p))
        
    iters, converged = fac_g.lbp(normalize=True)

    return fac_g.rv_marginals(normalize=True)



def factorization(g, seed, nodes, pls, state_nums, Dxs, probs, weights, neighbors, num=None):
    # spt = UST(g)
    # spt.sample(random_state=seed)
    # tree = spt.list_of_samples[0].edges
    
    tree = ust_sampler_wilson(neighbors, probs, weights, root=None,
                       random_state=seed, num=num)
    
    FTree1 = factor_tree()
    for node in nodes:
        FTree1.add_nodes([node], [Dxs[node]], [0])
       
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
                val = np.random.choice([i for i in range(1, state_nums+1)], p=marginal)
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
                val = np.random.choice([i for i in range(1, state_nums+1)], p=marginal)
                values.append([node, val])
            except:
                print("Error in gibs sampling!")
                print(node)
                print(marginal)
    # print("values", values)
    return values

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

def vote(all_res):
    ress = []
    for i in range(len(all_res)):
        res = all_res[i]
        # counts = np.bincount(res)
        # ress.append(np.argmax(counts))
        ress.append(np.round(np.mean(res)))
    return ress


def spt(g, state_nums, Dxs, more, rank):
    gnodes = list(g.nodes)
    gedges = list(g.edges)
    edges = []
    for edge in gedges:
        edges.append((min(edge), max(edge)))
    gedges = edges
    pls = len(gnodes)
    res = [[] for _ in range(pls)]
    # res = {i:{v:[] for v in range(1, state_nums+1)} for i in range(pls)}
    loss_ = []
    time_spt = []
    score = {edge:[] for edge in gedges}

    # probs1, weights1 = score_based_prob(gedges, score, pls)
    probs1 = edge_prob(g)
    weights1 = {}
    for i in range(probs1.shape[0]):
        for j in range(i, probs1.shape[1]):
            if probs1[i, j] != 0:
                weights1[(i, j)] = 1 / probs1[i, j]
    
    neighbors = [[] for _ in gnodes]
    t = 0
    for (a, b) in gedges:
        neighbors[a].append(b)
        neighbors[b].append(a)    
        
    cur_best = 999999999 
    
    for seed_ in range(6, 6+more):

        s = time.time()
        FT, tree = factorization(g, seed_, gnodes, pls, state_nums, Dxs, probs1, weights1, neighbors)  
        r = FT.sum_product()
        t += time.time() - s
        time_spt.append(t)
        
        r = gibs_sampling(gnodes, FT, state_nums, False)
        
        for r_pair in r:
            i, rr = r_pair
            res[i].append(rr)
        
        r = vote(res)
        
        # rr = [0 for i in range(pls)]
        # for r_pair in r:
        #     rr[r_pair[0]] = r_pair[1]
        # cur_loss = loss(Dxs, rr, gedges)
        # for i in range(pls):
        #     res[i][rr[i]].append(cur_loss)
            
        # r = vote2(res)
        cur_loss = loss(Dxs, r, gedges)
        cur_best = min(cur_best, cur_loss)
        loss_.append(cur_best)
        
    # print(res)
    return loss_, time_spt, rank


def lbp(Dxs, g):
    res = factor_obs(list(g.edges), list(g.nodes), Dxs, 6)
    vs, bs = gibs_sampling_grid_obs(list(g.nodes), res, 6)
    print(vs)
    print(bs)
    l = loss(Dxs, vs, list(g.edges))
    return l


def cal_marginal(nodes, beliefs, initials):
    rtn = {node: 0 for node in nodes}
    for node in nodes:
        bs = []
        for k,v in beliefs[node].items():
            if v != []:
                # v = np.sum(v, axis=0) / len(v)
                # bs.append(v)
                bs += v

        marginal = np.prod(initials[node]+bs, axis=0)
        marginal /= np.sum(abs(marginal))
        rtn[node] = marginal
        
    return rtn


def gibs_sampling2(nodes, beliefs, state_nums, key):
    values = []
    for node in nodes:
        marginal = beliefs[node]
        
        val = np.random.choice([i for i in range(1, state_nums+1)], p=marginal)
        values.append([node, val])
    return values


def belief_spt(g, state_nums, Dxs, more, rank):
    gnodes = list(g.nodes)
    gedges = list(g.edges)
    edges = []
    for edge in gedges:
        edges.append((min(edge), max(edge)))
    gedges = edges
    pls = len(gnodes)
    res = [[] for _ in range(pls)]
    # res = {i:{v:[] for v in range(1, state_nums+1)} for i in range(pls)}
    loss_ = []
    time_spt = []
    
    beliefs = {node:{} for node in gnodes}

    initial_beliefs = {node: np.ones(state_nums) / state_nums for node in gnodes}
    # probs1, weights1 = score_based_prob(gedges, score, pls)
    probs1 = edge_prob(g)
    weights1 = {}
    for i in range(probs1.shape[0]):
        for j in range(i, probs1.shape[1]):
            if probs1[i, j] != 0:
                weights1[(i, j)] = 1 / probs1[i, j]
    
    neighbors = [[] for _ in gnodes]
    t = 0
    for (a, b) in gedges:
        neighbors[a].append(b)
        neighbors[b].append(a)   
        beliefs[a][(a, b)] = []
        beliefs[b][(a, b)] = []

    cur_best = 999999999 
    
    for seed_ in range(6, 6+more):

        s = time.time()
        FT, tree = factorization(g, seed_, gnodes, pls, state_nums, Dxs, probs1, weights1, neighbors)  
        r = FT.sum_product()
        t += time.time() - s
        time_spt.append(t)
        
        for node in gnodes:
            vnode = FT.vnodes[node]
            for k,v in vnode.message_rec.items():
                nodes = [nodee.node for nodee in k.nodes]
                beliefs[node][(min(nodes), max(nodes))].append(v)
        
        cur_marginal = cal_marginal(gnodes, beliefs, initial_beliefs)
        r = gibs_sampling2(gnodes, cur_marginal, state_nums, False)
        
        # r = gibs_sampling(gnodes, FT, state_nums, False)
        print(cur_marginal)
        print(r)
        only_v = []
        for r_pair in r:
            i, rr = r_pair
            res[i].append(rr)
            only_v.append(rr)
        
        # r = vote(res)

        cur_loss = loss(Dxs, only_v, gedges)
        cur_best = min(cur_best, cur_loss)
        loss_.append(cur_best)

    return loss_, time_spt, rank

n = 3
num = n*n
Dxs, g = er_graph(num, states=6, p=4/(n*(n+1)), seed=6)

# print(loss(Dxs, v, list(g.edges)))
# print(len(list(g.edges)))
# print(2*len(list(g.edges))/num)
# saveModel(Dxs, g, 6, 10)
# print("Save over!")
# print(list(g.nodes))
print(belief_spt(g, 6, Dxs, 10, 1)[0])
print(lbp(Dxs, g))


# print(er_graph_obs(10, p=0.6, seed=6))