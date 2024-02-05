import numpy as np
import networkx as nx
import copy 
from enum import Enum
import sys

class NodeType(Enum):
    VarNode = 1
    FacNode = 2


class VNode():
    """
        This class defines the variable node in a factor graph.
    """
    
    def __init__(self, node, state_num, marginal=None) -> None:
        self.node = node                                # int, the original No.
        self.type = NodeType.VarNode
        self.message_rec = {}                           # who: message from who, FacNode: ndarray(num_of_states*1)
        self.num_rec = 0                                # number of received messages
        self.message_send = {}
        # print(node, marginal)# receiver: message send. FacNode: ndarray
        # if marginal:
        self.marginal = marginal 

        self.state_num = state_num
        
        self.subs = []                                  # a list of FacNode below the current node
        self.ups = []                                   # a list of VarNode above the current node
        
        self.sum_msg_rec = {}                           # the max message received from FacNode: message
        self.sum_msg_send = {}                          # the max message send to FacNode: message
        self.max_num_rec = 0                            # the number of received max info
        
        self.max_var = 0                                # the current maximum variable select
    
    
    def send(self, rec, test=False):
        """
            variable node send message to the specific receiver
        Args:
            rec (FacNode): The factor node that is connected with this Vnode
            test (bool, optional): _description_. Defaults to False.
        """
        
        message2 = np.prod([np.ones(self.state_num)]+[v for k,v in self.message_rec.items() if k != rec], axis=0)

        message2 = self.marginal * message2**1
        message2 /= sum(message2)
        rec.receive(self, message2/sum(message2))
        self.message_send[rec] = message2/sum(message2)

        
        if test:
            print(self.node,"->", rec.name, str(message2))
        
    def receive(self, who, message):
        """
            receive message from a Fnode
        Args:
            who (FacNode): the factor node that sends the message
            message (ndarray): state_nums * 1. _description_
        """
        self.message_rec[who] = message     
        self.num_rec += 1   

        
    # def add_neighbor(self, neighbor):
    #     self.neighbors.append(neighbor)
    #     self.num_of_neighbors += 1  
        
    def update(self):
        """
            Based on the received information to update self potential.
            Default to normalize. Otherwise when iterate more than about 5 times the prob will go wrong
            because the values are two dramatic.
        Args:
            normalize (bool, optional): _description_. Defaults to True.
        """
        # ? TODO: up and down need different treatment

        message = np.prod([np.ones(self.state_num)]+[v for v in self.message_rec.values()], axis=0)

        self.marginal = self.marginal * message
        self.marginal /= np.sum(abs(self.marginal))
        

    def get_marginal(self, normalize=True):
        return self.marginal / np.sum(abs(self.marginal)) if normalize else self.marginal
        
    # below are for max sum algorithm
    def send_max(self, rec):
        msg = np.zeros(self.state_num)
        if rec in self.ups:
            if self.subs: 
                msg = self.sum_msg_rec[self.subs[0]]

            
        rec.receive_max(self, msg)
        self.sum_msg_send[rec] = msg

    def empty(self):
        self.message_rec = {}
        self.message_send = {}
        self.num_rec = 0
    
class FNode():
    """
        This class defines the variable node in a factor graph.
    """
    
    def __init__(self) -> None:
        self.type = NodeType.FacNode
        self.message_rec = {}           # VarNode: Message
        self.num_rec = 0        
        self.message_send = {}          # VarNode: Message
        self.joint_dis = None           # node 1 rows, node 2 cols
        self.nodes = []                 # the node related to the factor  [VarNode_up, VarNode_down]
        
        self.weight = 1                 # the calibrated weight of this edge
        
        self.subs = []   
        self.ups = []
        
        self.sum_msg_rec = {}
        self.sum_msg_send = {}
        
        self.max_num_rec = 0
        
        self.name = None                # up_node, sub_node
    
    def set_name(self, name):
        self.name = name
        
        
    def receive_potential(self, potential, up=None, sub=None, weight=1):
        """
            This method is for the situation we initialize a factor node by input info manually.
            This might be a factor that lies in the lower layer to save one send and one rec between 
            the evidence node.
        Args:
            potential (ndarray): _description_
            up (VarNode, optional): _description_. Defaults to None.
            sub (VarNode, optional): _description_. Defaults to None.
        """
        name = ""
        if up:
            self.ups.append(up)
            up.subs.append(self)
            name += str(up.node)
            self.nodes = [up]
        # A factor must have a up, could not have a sub
        if sub and type(sub) != str:
            self.subs.append(sub)
            sub.ups.append(self)
            
            self.nodes = [up, sub]

        self.joint_dis = potential
        if type(sub) == str:
            name += ", " + sub
        self.name = name
        self.weight = weight
    
    def quadratic(state1, state2):
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
    
    @staticmethod
    def initialize_potentials(states1, states2):
        return np.exp(-(np.subtract.outer(states1, states2)) ** 2)

    def initialize(self, node1, node2, func=quadratic, weight=1):
        """Based on the input nodes info and the given potential functions to 
            calculate the initial joint distribution.

        Args:
            node1 (Vnode): father node
            node2 (Vnode): child node
        """
        num_states1, num_states2 = node1.state_num, node2.state_num
        
        states1 = np.arange(1, num_states1 + 1)
        if num_states2 == 1:
            states2 = np.array([node2.max_var])
        else:
            states2 = np.arange(1, num_states2 + 1)

        potentials = self.initialize_potentials(states1, states2)
  
        self.joint_dis = potentials   
        self.nodes = [node1, node2]

        self.name = f"f({node1.node}, {node2.node})"
        self.weight = weight
        
    def send(self, rec, test=False):
        """
            Sending message to the given receiver
        Args:
            rec (VarNode): _description_
        """
        duplicate = copy.deepcopy(self.joint_dis)
        out = np.zeros(rec.state_num)
        if len(self.nodes) == 1:
            out = duplicate.flatten()
        
        # based on the receiver to cal message
        # send to node 1, up 
        elif rec == self.nodes[0]:
            message = self.message_rec[self.nodes[1]]
            for i in range(len(message)):
                msg = message[i]
                duplicate[:, i] = duplicate[:, i] * msg
            for row in range(duplicate.shape[0]):
                out[row] = sum(duplicate[row, :])
            out.reshape(rec.state_num, 1)
        # send to node 2, down
        else:
            message = self.message_rec[self.nodes[0]]
            for i in range(len(message)):
                msg = message[i]
                duplicate[i, :] *= msg
            for col in range(duplicate.shape[1]):
                out[col] = sum(duplicate[:, col])
            out.reshape(rec.state_num, 1)
            
        out = out**self.weight
        
        final_message = out/sum(out)
        rec.receive(self, final_message)
        self.message_send[rec] = final_message
        
        if test:
            print(self.name,"->", rec.node, str(out))


    def receive(self, who, message):

        self.message_rec[who] = message
        self.num_rec += 1

            
    def update(self):
        
        message1 = self.message_rec[self.nodes[0]]
        message2 = self.message_rec[self.nodes[1]]

        if self.subs:
            self.joint_dis *= message1[:, np.newaxis]
            self.joint_dis *= message2[np.newaxis, :]

        else:
            self.joint_dis *= message1.reshape(-1, 1)

        self.joint_dis /= np.sum(self.joint_dis)
            
    def get_joint(self, normalize=True):
        return (self.nodes, self.joint_dis/sum(self.joint_dis)) if normalize else (self.nodes, self.joint_dis) 
        
    def empty(self):

        self.num_rec = 0
        self.message_rec = {}
        self.message_send = {}
        self.initialize(self.nodes[0], self.nodes[1], weight=self.weight)
    

class factor_tree():
    """
        The class that utilize FacNode and VarNode to build factor tree
            - Generate Factor Tree automatically
            - Generate Factor Tree manually
            - Sum Product
            - Max Sum
    """
    
    
    def __init__(self) -> None:
        self.normal_nodes = []          # a list contains all the original node in the tree
        self.vnodes = {}                # dict, node: VarNode
        self.num_of_nodes = 0           # number of normal nodes
        self.normal_edges = []          # a list contains all the original edges
        self.obs_key = False            # with or without observation nodes (Not functional!)
        self.g = nx.Graph()             # the netorkx graph object
        
        self.factor_dic = {}            # dict of dict up_normal_node: sub_normal_node: factor(up, sub)
        
        self.tree = None                # the normal node tree
        self.tree_root = 0              # the normal tree node
        
        self.factor_tree = None         # the factor tree (Not functional !)
        self.factor_tree_root = None    # the factor tree's root, a VarNode
        
        self.leaves = []                # all the leaf Fac/Var Nodes.
        self.factors = []               # all the FacNodes.
        self.weights = {}
    
    def get_Attr(self):
        """
            Output the basic info about the factor tree
        """
        print(self.normal_nodes)
        print(self.vnodes)
        print(self.num_of_nodes)
        print(self.normal_edges)
        print(self.obs_key)
        print(self.tree)
        print(self.tree_root)
        print(self.leaves)


    def add_node(self, node:int, state_num:int, max_var=0, marginal=None):
        """
            Add nodes to the factor tree, create a VarNode for all given normal nodes.
        Args:
            node (int): _description_
            state_num (int): _description_
            max_var (int, optional): _description_. Defaults to 0.
        """
        nod = VNode(node, state_num, marginal)
        self.vnodes[node] = nod
        self.num_of_nodes += 1
        self.normal_nodes.append(node)
        nod.max_var = max_var
        
    
    
    def add_nodes(self, nodes, state_nums, max_vars=[], marginals=None):
        """_summary_

        Args:
            nodes (_type_): list of int
            state_nums (_type_): list of int
        """
        if not marginals:
            marginals = [None for _ in range(len(nodes))]
        for node, state_num, max_var, marginal in zip(nodes, state_nums, max_vars, marginals):
            self.add_node(node, state_num, max_var, marginal)
    
    def set_tree_root(self,root:int):
        self.tree_root = root
    
    def set_factor_root(self):
        
        root = self.tree_root
        self.factor_tree_root = self.vnodes[root]
      
        
    def add_edge(self, edge:[int, int], weight=1):
        self.normal_edges.append(edge)
        self.weights[tuple(sorted(edge))] = weight
        
    def add_edges(self, egdes:list([int, int]), weights):
        for edge, weight in zip(egdes, weights):
            self.add_edge(edge, weight)
        
    
    def quadratic(state1, state2):
        return np.exp(-(state1-state2)**2)    
    
    def cubic(state1, state2):
        return np.exp(-(state1-state2)**3)
    
    def abs_(state1, state2):
        return np.exp(-abs(state1)-abs(state2))
    
    def build_factor_tree(self, Froot=None, root=None, get_leaves=True, func=quadratic):
        """
            Using DFS to travel over the normal Tree and create a factor node at each edge. 
            If the factor tree has some manually added nodes, we will set get_leaves False so that we can 
            add leaves using update after all nodes are inserted.
        Args:
            Froot (Vnode): The Vnode. Defaults to None.
            root (int): The number of the node in the regular tree. Defaults to None.
        """
      
        Froot = self.factor_tree_root if not Froot else Froot
        root = self.tree_root if (not root and root != 0) else root
        
        temp = []
       
        for child in self.tree[root]:
            temp.append(child)
           
            kid = self.vnodes[child]
            
            factor = FNode()
            factor.initialize(Froot, kid, func=func, weight=self.weights[tuple(sorted([root, child]))])
            
            Froot.subs.append(factor)
            factor.ups.append(Froot)
            
            factor.subs.append(kid)
            kid.ups.append(factor)
            
            self.factors.append(factor)
            self.build_factor_tree(kid, child, get_leaves)
            
        if get_leaves:
            if not self.tree[root]:
                self.leaves.append(Froot)
    
    
    def update(self):
        """
            If the factor tree has some manually added nodes, we need use update to find leaves.
        """
        cur = self.factor_tree_root
        wait = []
        wait += cur.subs

        cur =  wait.pop(0)
        
        while wait:
            if cur.subs:
                wait += cur.subs
            else:
                self.leaves.append(cur)
            cur = wait.pop(0)
        
    
        
    def __dfs1(self, root, record, traversed):
        for child in record[root]:
            if traversed[child]:
                traversed[child] = False
                self.tree[root].append(child)
                traversed = self.__dfs1(child, record, traversed)
        return traversed
        
        
    def generate_node_tree(self):
        """
            Based on the input edges to find the neighbors.
            Using the neighbors to do DFS to find the tree structure start 
            from the given root of the tree.
        """
        sys.setrecursionlimit(10000)
        record = {node:[] for node in self.normal_nodes}  
        traversed = {node:True for node in self.normal_nodes} 
        for n1, n2 in self.normal_edges:
            try:
                record[n1].append(n2)
                record[n2].append(n1)
            except:
                print("wrong !", n1, n2)
            
            
        self.tree = {node:[] for node in self.normal_nodes}
        traversed[self.tree_root] = False
        traversed = self.__dfs1(self.tree_root, record, traversed)

        self.set_factor_root()

    def sum_product(self, display=False):
        """
            The sum product algorithm.
            Using BFS to find the which will send next round.
        Args:
            display (bool, optional): _description_. Defaults to False.
        """
        this_round = self.leaves

        while this_round:
            # if in next round, means already receive all messages from subs.
            next_round = []
            for vnode in this_round:
                if display:
                    try:
                        print(vnode.name, vnode.joint_dis)
                    except:
                        print(vnode.node, vnode.marginal)
                        
                if not vnode.ups:
                    continue
                rec = vnode.ups[0]
                
                if display:
                    try:
                        print("sender is ", vnode.name, "receiver is ", rec.node)
                    except:
                        print("sender is ", vnode.node, "receiver is ", rec.name)
                        
                vnode.send(rec)
                if rec.num_rec == len(rec.subs):
                    next_round.append(rec)
 
            this_round = next_round


        this_round = [self.factor_tree_root]
        
        while this_round:
            # if in next round, means already receive all messages from both side, could update.
            next_round = []
            for vnode in this_round:
                if display:
                    try:
                        print(vnode.name, vnode.joint_dis)
                    except:
                        print(vnode.node, vnode.marginal)
                    print("sender is ", vnode.node)
                
                if not vnode.subs:
                    vnode.update()
                    continue
                for sub in vnode.subs:
                    rec = sub
                    vnode.send(rec)
                    next_round.append(rec)
                vnode.update()
            this_round = next_round
        
    
    def max_sum(self):
        """
            Using BFS to do the max sum algorithm, once reach the root node, means we find a set of configurations.
        """
        this_round = self.leaves
        # send to root
        while this_round:
            next_round = []
            for node in this_round:
                if not node.ups:
                    continue
                rec = node.ups[0]

                node.send_max(rec)
                if rec.max_num_rec == len(rec.subs):
                    next_round.append(rec)

            this_round = next_round
            
        self.factor_tree_root.update_max()

            
    def add_factor_nodes(self, nodes:list, potential:list, weight=1):
        """
            Add factor nodes manually.
        Args:
            nodes (list): edge pair of VarNodes, [father, son]
            potentials (list): _description_
        """
        n1, n2 = nodes
        
        fn = FNode()
        fn.receive_potential(potential, n1, n2, weight=weight)
        self.leaves.append(fn)
        
        if n1.node not in self.factor_dic:
            self.factor_dic[n1.node] = {}
        self.factor_dic[n1.node][n2] = fn
            
    
    
                        
        

